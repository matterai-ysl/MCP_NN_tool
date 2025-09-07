"""Task Queue Management System for Neural Network Training.

This module implements an asynchronous task queue system for managing long-running
neural network training tasks with status tracking and result management.
"""

import asyncio
import json
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Task type enumeration."""
    REGRESSION_TRAINING = "regression_training"
    CLASSIFICATION_TRAINING = "classification_training"
    PREDICTION = "prediction"


@dataclass
class TaskInfo:
    """Task information data class."""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    progress_message: str = "Initializing..."
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime handling."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'started_at', 'completed_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        # Convert enums to string values
        data['task_type'] = self.task_type.value
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskInfo':
        """Create TaskInfo from dictionary."""
        # Convert string datetime back to datetime objects
        for key in ['created_at', 'started_at', 'completed_at']:
            if data.get(key) is not None:
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert string enums back to enum objects
        if 'task_type' in data:
            data['task_type'] = TaskType(data['task_type'])
        if 'status' in data:
            data['status'] = TaskStatus(data['status'])
        
        return cls(**data)


class TaskQueue:
    """Asynchronous task queue manager for neural network training."""
    
    def __init__(self, max_concurrent_tasks: int = 2, 
                 persistence_file: str = "./trained_model/task_queue.json"):
        """Initialize task queue.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent training tasks
            persistence_file: File to persist task information
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.persistence_file = Path(persistence_file)
        self.tasks: Dict[str, TaskInfo] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Ensure persistence directory exists
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing tasks
        self._load_tasks()
        
        # Start background worker
        self._worker_task = None
        self._shutdown = False

    async def start(self):
        """Start the task queue worker."""
        if self._worker_task is None or self._worker_task.done():
            self._shutdown = False
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info("Task queue worker started")

    async def stop(self):
        """Stop the task queue worker."""
        self._shutdown = True
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            if not task.done():
                task.cancel()
                with self.task_lock:
                    if task_id in self.tasks:
                        self.tasks[task_id].status = TaskStatus.CANCELLED
        
        self.executor.shutdown(wait=True)
        self._save_tasks()
        logger.info("Task queue worker stopped")

    async def submit_task(self, 
                         task_type: TaskType,
                         task_function: Callable,
                         task_args: tuple = (),
                         task_kwargs: Optional[Dict[str, Any]] = None,
                         task_parameters: Optional[Dict[str, Any]] = None,
                         estimated_duration: Optional[float] = None) -> str:
        """Submit a task to the queue.
        
        Args:
            task_type: Type of the task
            task_function: Function to execute
            task_args: Arguments for the function
            task_kwargs: Keyword arguments for the function
            task_parameters: Task parameters for display/tracking
            estimated_duration: Estimated duration in seconds
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        if task_kwargs is None:
            task_kwargs = {}
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            parameters=task_parameters or {},
            estimated_duration=estimated_duration
        )
        
        with self.task_lock:
            self.tasks[task_id] = task_info
        
        # Create async task
        async_task = asyncio.create_task(
            self._execute_task(task_id, task_function, task_args, task_kwargs)
        )
        
        with self.task_lock:
            self.running_tasks[task_id] = async_task
        
        self._save_tasks()
        logger.info(f"Task {task_id} submitted to queue (type: {task_type.value})")
        
        # Start worker if not running
        await self.start()
        
        return task_id

    async def _execute_task(self, 
                           task_id: str, 
                           task_function: Callable, 
                           task_args: tuple, 
                           task_kwargs: Dict[str, Any]):
        """Execute a task with error handling and status updates."""
        try:
            # Update task status to running
            with self.task_lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.RUNNING
                    self.tasks[task_id].started_at = datetime.now(timezone.utc)
                    self.tasks[task_id].progress_message = "Task started"
            
            self._save_tasks()
            
            # Create progress callback for the task
            async def progress_callback(progress: float, message: str = ""):
                with self.task_lock:
                    if task_id in self.tasks:
                        self.tasks[task_id].progress = min(100.0, max(0.0, progress))
                        if message:
                            self.tasks[task_id].progress_message = message
                self._save_tasks()
            
            # Add progress callback to kwargs if not present
            if 'progress_callback' not in task_kwargs:
                task_kwargs['progress_callback'] = progress_callback
            
            # Execute the task function
            start_time = time.time()
            
            # Execute task in thread pool for CPU-intensive operations
            if asyncio.iscoroutinefunction(task_function):
                result = await task_function(*task_args, **task_kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor, 
                    lambda: task_function(*task_args, **task_kwargs)
                )
            
            execution_time = time.time() - start_time
            
            # Update task status to completed
            with self.task_lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.COMPLETED
                    self.tasks[task_id].completed_at = datetime.now(timezone.utc)
                    self.tasks[task_id].result = result
                    self.tasks[task_id].progress = 100.0
                    self.tasks[task_id].progress_message = "Task completed successfully"
                    self.tasks[task_id].actual_duration = execution_time
            
            logger.info(f"Task {task_id} completed successfully in {execution_time:.2f}s")
            
        except asyncio.CancelledError:
            # Handle task cancellation
            with self.task_lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.CANCELLED
                    self.tasks[task_id].completed_at = datetime.now(timezone.utc)
                    self.tasks[task_id].progress_message = "Task was cancelled"
            logger.info(f"Task {task_id} was cancelled")
            
        except Exception as e:
            # Handle task failure
            error_message = str(e)
            with self.task_lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.FAILED
                    self.tasks[task_id].completed_at = datetime.now(timezone.utc)
                    self.tasks[task_id].error_message = error_message
                    self.tasks[task_id].progress_message = f"Task failed: {error_message}"
            
            logger.error(f"Task {task_id} failed: {error_message}")
            
        finally:
            # Clean up running task reference
            with self.task_lock:
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            self._save_tasks()

    async def _worker_loop(self):
        """Background worker loop to manage task execution."""
        while not self._shutdown:
            try:
                # Clean up completed tasks from running_tasks dict
                completed_task_ids = []
                with self.task_lock:
                    for task_id, task in list(self.running_tasks.items()):
                        if task.done():
                            completed_task_ids.append(task_id)
                
                for task_id in completed_task_ids:
                    with self.task_lock:
                        if task_id in self.running_tasks:
                            del self.running_tasks[task_id]
                
                # Save tasks periodically
                self._save_tasks()
                
                # Wait before next iteration
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task queue worker: {e}")
                await asyncio.sleep(5)

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get status of a specific task."""
        with self.task_lock:
            return self.tasks.get(task_id)

    def list_tasks(self, 
                   status_filter: Optional[TaskStatus] = None,
                   task_type_filter: Optional[TaskType] = None,
                   limit: Optional[int] = None) -> List[TaskInfo]:
        """List tasks with optional filtering."""
        with self.task_lock:
            tasks = list(self.tasks.values())
        
        # Apply filters
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        
        if task_type_filter:
            tasks = [t for t in tasks if t.task_type == task_type_filter]
        
        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        # Apply limit
        if limit:
            tasks = tasks[:limit]
        
        return tasks

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        with self.task_lock:
            total_tasks = len(self.tasks)
            status_counts = {}
            
            for status in TaskStatus:
                status_counts[status.value] = sum(
                    1 for task in self.tasks.values() if task.status == status
                )
            
            running_count = len(self.running_tasks)
            
        return {
            "total_tasks": total_tasks,
            "running_tasks": running_count,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "status_counts": status_counts,
            "queue_active": not self._shutdown,
            "worker_running": self._worker_task is not None and not self._worker_task.done()
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        with self.task_lock:
            if task_id not in self.tasks:
                return False
            
            task_info = self.tasks[task_id]
            
            # Can only cancel pending or running tasks
            if task_info.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                return False
            
            # Cancel the async task if it's running
            if task_id in self.running_tasks:
                async_task = self.running_tasks[task_id]
                if not async_task.done():
                    async_task.cancel()
            
            # Update task status
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = datetime.now(timezone.utc)
            task_info.progress_message = "Task cancelled by user"
        
        self._save_tasks()
        logger.info(f"Task {task_id} cancelled")
        return True

    def delete_task(self, task_id: str) -> bool:
        """Delete a task from the queue."""
        with self.task_lock:
            if task_id not in self.tasks:
                return False
            
            task_info = self.tasks[task_id]
            
            # Can only delete completed, failed, or cancelled tasks
            if task_info.status in [TaskStatus.RUNNING, TaskStatus.PENDING]:
                return False
            
            del self.tasks[task_id]
        
        self._save_tasks()
        logger.info(f"Task {task_id} deleted")
        return True

    def _save_tasks(self):
        """Save task information to persistence file."""
        try:
            tasks_data = {}
            with self.task_lock:
                for task_id, task_info in self.tasks.items():
                    tasks_data[task_id] = task_info.to_dict()
            
            with open(self.persistence_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def _load_tasks(self):
        """Load task information from persistence file."""
        try:
            if not self.persistence_file.exists():
                return
            
            with open(self.persistence_file, 'r') as f:
                tasks_data = json.load(f)
            
            with self.task_lock:
                for task_id, task_data in tasks_data.items():
                    try:
                        task_info = TaskInfo.from_dict(task_data)
                        
                        # Reset running tasks to failed on startup (they were interrupted)
                        if task_info.status == TaskStatus.RUNNING:
                            task_info.status = TaskStatus.FAILED
                            task_info.error_message = "Task interrupted by system restart"
                            task_info.completed_at = datetime.now(timezone.utc)
                        
                        self.tasks[task_id] = task_info
                        
                    except Exception as e:
                        logger.error(f"Failed to load task {task_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to load tasks from {self.persistence_file}: {e}")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create the global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


async def initialize_task_queue():
    """Initialize and start the task queue."""
    task_queue = get_task_queue()
    await task_queue.start()
    return task_queue


async def shutdown_task_queue():
    """Shutdown the task queue."""
    global _task_queue
    if _task_queue is not None:
        await _task_queue.stop()
        _task_queue = None