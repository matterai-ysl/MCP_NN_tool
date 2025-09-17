"""MCP Server for Neural Network Tool.

This module implements an MCP server that provides neural network training and prediction
capabilities through a standardized interface using FastMCP, supporting both single and
multi-target predictions.
"""
import sys
import os
from pathlib import Path
print(str(Path(__file__).parent.parent.parent))
print(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from .config import BASE_URL,get_download_url,get_static_url
import asyncio
import json
from typing import Any, Dict, List, Optional, Union, Callable
import logging
import uuid
import numpy as np
import re
# FastMCP imports
from fastmcp import FastMCP, Context
import time

from mcp_nn_tool.data_utils import (
    preprocess_data, 
    read_data_file, 
    validate_data_format,
    extract_feature_target_info,
    preprocess_classification_data
)
from mcp_nn_tool import prediction
from mcp_nn_tool.training import (optimize_hyperparameters, 
                                  train_final_model,
                                  optimize_classification_hyperparameters,
                                  train_final_classification_model,
                                  generate_academic_report
                                  )
from mcp_nn_tool.model_manager import ModelManager
from mcp_nn_tool.task_queue import (
    get_task_queue, 
    initialize_task_queue,
    TaskType,
    TaskStatus
)
from mcp_nn_tool.nn_html_report_generator import NeuralNetworkHTMLReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("Neural-Network-MCP-Tool",
              instructions="Neural Network MCP Tool",
              version="0.1.0"
              )

# Global model manager removed - now using user-specific model managers


def get_user_models_dir(user_id: Optional[str] = None) -> str:
    """Get user-specific models directory with security validation.

    Args:
        user_id: User ID from MCP context, defaults to "default" if None

    Returns:
        Path to user-specific models directory
    """
    if user_id is None or user_id.strip() == "":
        user_id = "default"

    # Clean user ID to prevent path traversal attacks
    user_id = re.sub(r'[^\w\-_]', '_', user_id)

    user_dir = Path("trained_models") / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return str(user_dir)


def get_user_id(ctx: Optional[Context] = None) -> Optional[str]:
    """Extract user ID from MCP context.

    Args:
        ctx: MCP Context object

    Returns:
        User ID from request headers or None
    """
    if ctx is not None and hasattr(ctx, 'request_context') and ctx.request_context is not None:
        if hasattr(ctx.request_context, 'request') and ctx.request_context.request is not None:
            if hasattr(ctx.request_context.request, 'headers'):
                return ctx.request_context.request.headers.get("user_id", None)
    return None


async def _train_neural_network_regression_impl(
    training_file: str,
    target_columns: int = 1,
    n_trials: int = 100,
    cv_folds: int = 5,
    num_epochs: int = 500,
    algorithm: str = "TPE",
    loss_function: str = "MAE",
    progress_callback: Optional[Callable] = None,
    models_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train a neural network model with hyperparameter optimization.
    
    Args:
        training_file: Path to training data file (CSV/Excel)
        target_columns: Number of target columns at the end of the dataset (default: 1)
        n_trials: Number of hyperparameter optimization trials (default: 100)
        cv_folds: Number of cross-validation folds (default: 5)
        num_epochs: Number of training epochs per fold (default: 500)
        algorithm: Optimization algorithm - "TPE" or "GP" (default: "TPE")
        loss_function: Loss function to use - "MAE" or "MSE" (default: "MAE")
        
    Returns:
        Dictionary containing training results with model ID, best parameters, and performance metrics
    """
    logger.info(f"Starting neural network training with file: {training_file}, targets: {target_columns}")
    
    try:
        import time
        print("Starting neural network training")
        
        # Record training start time
        training_start_time = time.time()
        
        # Use provided models_dir or default
        if models_dir is None:
            models_dir = "./trained_model"

        # Create user-specific model manager
        user_model_manager = ModelManager(models_dir)

        model_id = str(uuid.uuid4())
        # Load and preprocess data
        print("training_file", training_file)
        print("开始预处理数据")
        train_data, _, full_scaler, feature_scaler = await preprocess_data(
            training_file, target_columns=target_columns
        )
        print("预处理数据完成")

        validate_data_format(train_data, target_columns=target_columns)
        print("数据格式验证完成")
        # Extract feature and target information
        feature_names, target_names, feature_number, target_number = extract_feature_target_info(
            train_data, target_columns
        )
        data_array = train_data.values
        
        # Progress callback for monitoring
        progress_updates = []
        async def internal_progress_callback(info):
            progress_updates.append(info)
            logger.info(f"Training progress: {info}")
            # Call external progress callback if provided
            if progress_callback:
                # Extract progress percentage from info
                if isinstance(info, dict) and 'progress' in info:
                    await progress_callback(info['progress'], info.get('message', ''))
                else:
                    await progress_callback(0, str(info))
        
        # Create temporary model folder for saving training artifacts
        temp_model_folder = f"{models_dir}/{model_id}"
        os.makedirs(temp_model_folder, exist_ok=True)
        
        # Optimize hyperparameters with saving
        optimization_start_time = time.time()
        best_params, best_loss, trials_df = await optimize_hyperparameters(
            data_array,
            feature_number,
            target_number,
            n_trials=n_trials,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            algorithm=algorithm,
            progress_callback=internal_progress_callback,
            save_dir=temp_model_folder,
            loss_function=loss_function
        )
        optimization_time = time.time() - optimization_start_time
        
        # Train final model with best parameters and detailed recording
        final_training_start_time = time.time()
        model_states, final_loss, cv_results = await train_final_model(
            best_params,
            data_array,
            feature_number,
            target_number,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            progress_callback=internal_progress_callback,
            save_dir=temp_model_folder,
            full_scaler=full_scaler,
            target_names=target_names,
            feature_names=feature_names,
            loss_function=loss_function
        )
        final_training_time = time.time() - final_training_start_time
        
        # Save model (auto-generate model ID)
        saved_model_id = await user_model_manager.save_model(
            model_states=model_states,
            best_params=best_params,
            best_mae=best_loss,
            feature_names=feature_names,
            target_names=target_names,
            full_scaler=full_scaler,
            feature_scaler=feature_scaler, # type: ignore
            cv_folds=cv_folds,
            training_epochs=num_epochs,
            model_id=model_id
        )
        
        # Get actual model folder path
        model_folder = user_model_manager.get_model_folder_path(saved_model_id)
        
        # Generate academic report
        total_training_time = time.time() - training_start_time
        data_info = {
            "data_shape": list(train_data.shape),
            "feature_number": feature_number,
            "target_number": target_number,
            "feature_names": feature_names,
            "target_names": target_names
        }
        
        try:
            report_path = generate_academic_report(
                model_folder=model_folder, # type: ignore
                model_id=saved_model_id,
                training_file=training_file,
                data_info=data_info,
                best_params=best_params,
                best_score=best_loss,
                trials_df=trials_df,
                cv_results=cv_results, # type: ignore
                average_mae=final_loss,
                optimization_time=optimization_time,
                training_time=final_training_time,
                n_trials=n_trials,
                cv_folds=cv_folds,
                num_epochs=num_epochs,
                algorithm=algorithm,
                loss_function=loss_function
            )
            logger.info(f"Academic report generated: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate academic report: {str(e)}")
            report_path = None
        
        report_url_path = get_download_url(report_path) # type: ignore
        model_folder_url_path = get_download_url(model_folder) # type: ignore
        result = {
            "status": "success",
            "model_id": saved_model_id,
            # "model_folder": model_folder,
            "best_parameters": best_params,
            "best_mae": float(best_loss),
            "feature_names": feature_names,
            "target_names": target_names,
            "training_summary": {
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "num_epochs": num_epochs,
                "hyperparameter_optimization_algorithm": algorithm,
                "loss_function": loss_function,
                "target_columns": target_columns,
                "data_shape": list(train_data.shape),
                # "optimization_time": optimization_time,
                # "training_time": final_training_time,
                # "total_time": total_training_time
            },
            "experiment_report": report_url_path,
            "training_artifacts_download_url": f"More training details, such as hyperparameter optimization, cross-validation scatter plots, and training logs, are available in the {model_folder_url_path}. The **experiment_report** at {report_url_path} provides comprehensive records and analysis for reproducibility and academic reference."
        }
        
        # Generate HTML report after result is created
        html_report_path = None
        try:
            html_generator = NeuralNetworkHTMLReportGenerator(output_dir=str(model_folder))
            html_report_path = html_generator.generate_training_report(
                model_directory=model_folder, # type: ignore
                training_results=result,
                include_visualizations=True
            )
            html_report_url = get_static_url(html_report_path)
            result["training_report_html_path"] = html_report_url
            logger.info(f"HTML training report generated: {html_report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML training report: {str(e)}")
        
        logger.info(f"Training completed successfully. Model ID: {saved_model_id}")
        logger.info(f"Model saved to: {model_folder}")
        logger.info(f"Academic report: {report_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise


@mcp.tool()
async def train_neural_network_regression(
    training_file: str,
    target_columns: int = 1,
    n_trials: int = 100,
    cv_folds: int = 5,
    num_epochs: int = 500,
    algorithm: str = "TPE",
    loss_function: str = "MAE",
    ctx: Context = None, # type: ignore
) -> Dict[str, Any]:
    """Submit a neural network regression training task to the queue.
    
    This function submits the training task to an asynchronous queue and returns 
    immediately with a task ID. Use get_training_results() to check progress and get results.
    
    Args:
        training_file: Path to training data file (CSV/Excel)
        target_columns: Number of target columns at the end of the dataset (default: 1)
        n_trials: Number of hyperparameter optimization trials (default: 100)
        cv_folds: Number of cross-validation folds (default: 5)
        num_epochs: Number of training epochs per fold (default: 500)
        algorithm: Optimization algorithm - "TPE" or "GP" (default: "TPE")
        loss_function: Loss function to use - "MAE" or "MSE" (default: "MAE")
        
    Returns:
        Dictionary containing task ID and submission status
    """
    try:
        # Get user-specific models directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Estimate training duration (rough estimation based on parameters)
        estimated_duration = (n_trials * cv_folds * num_epochs * 0.01) + 60  # seconds

        # Task parameters for tracking
        task_parameters = {
            "training_file": training_file,
            "target_columns": target_columns,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "num_epochs": num_epochs,
            "algorithm": algorithm,
            "loss_function": loss_function,
            "task_type": "regression",
            "models_dir": user_models_dir
        }
        
        # Submit task to queue
        task_queue = get_task_queue(user_models_dir)
        task_id = await task_queue.submit_task(
            task_type=TaskType.REGRESSION_TRAINING,
            task_function=_train_neural_network_regression_impl,
            task_args=(),
            task_kwargs={
                "training_file": training_file,
                "target_columns": target_columns,
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "num_epochs": num_epochs,
                "algorithm": algorithm,
                "loss_function": loss_function,
                "models_dir": user_models_dir
            },
            task_parameters=task_parameters,
            estimated_duration=estimated_duration
        )
        
        return {
            "status": "submitted",
            "task_id": task_id,
            "message": "Training task submitted to queue",
            "estimated_duration": estimated_duration,
            "parameters": task_parameters,
            "next_steps": "Use get_training_results(task_id) to check progress and get results when complete"
        }
        
    except Exception as e:
        logger.error(f"Error submitting training task: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to submit training task: {str(e)}"
        }


@mcp.tool()
async def predict_from_file_neural_network(
    model_id: str,
    prediction_file: str,
    generate_experiment_report: bool = True,
    ctx: Context = None, # type: ignore
) -> dict:
    """Make predictions on data from a file (supports both regression and classification).
    
    Automatically detects model type from model metadata and calls the appropriate prediction method.
    
    Args:
        model_id: ID of the trained model to use for prediction
        prediction_file: Path to CSV/Excel file containing prediction data
        generate_experiment_report: Whether to generate detailed experiment report (default: True)
        
    Returns:
        Dictionary containing prediction results and optional experiment report path
    """
    try:
        # Get user-specific models directory and create user model manager
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        user_model_manager = ModelManager(user_models_dir)

        # Load model components to determine task type
        model_components = await user_model_manager.load_model(model_id)
        task_type = model_components['metadata'].get('task_type', 'regression')
        
        logger.info(f"Model {model_id} detected as {task_type} task")
        
        if task_type == 'classification':
            # Use classification prediction
            results, experiment_report_path, basic_csv_path = await prediction.classify_from_file(
                prediction_file,
                model_components["model_folder"], # type: ignore
                generate_experiment_report=generate_experiment_report
            )
            
            # Format response for classification
            response = {
                'task_type': 'classification',
                'model_id': model_id,
                'results': results,
                'experiment_report_generated': generate_experiment_report,
                'basic_csv_path': basic_csv_path,
                "experiment_report_path": experiment_report_path,
                "prediction_details": f"Prediction task completed.More prediction details, such as the prediction results, can be found in the {basic_csv_path}. The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference."
            }
            
            # Generate HTML prediction report for classification
            if generate_experiment_report:
                try:
                    html_generator = NeuralNetworkHTMLReportGenerator()
                    html_report_path = html_generator.generate_prediction_report(
                        prediction_results=response,
                        model_info=model_components['metadata'],
                        include_visualizations=True
                    )
                    html_report_url = get_download_url(html_report_path)
                    response['html_prediction_report'] = html_report_url
                    logger.info(f"HTML prediction report generated: {html_report_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate HTML prediction report: {str(e)}")
            
        else:
            # Use regression prediction (default)
            result_df, raw_predictions, experiment_report_path, basic_csv_path = await prediction.predict_from_file(
                model_components, 
                prediction_file,
                generate_experiment_report=generate_experiment_report
            )
            experiment_realve_path = get_download_url(experiment_report_path) # type: ignore
            # Format response for regression
            response = {
                'task_type': 'regression',
                'model_id': model_id,
                'num_predictions': len(result_df),
                'predictions': result_df.to_dict('records'),
                'experiment_report_generated': generate_experiment_report,
                'basic_prediction_file': basic_csv_path,
                'experiment_report_path': experiment_realve_path,
                           }
            
            if experiment_report_path:
                
                # Get model folder path for regression
                model_base_path = os.path.join("trained_model", model_id)
                predictions_dir = os.path.join(model_base_path, "predictions")
                predictions_dir_realve_path = get_download_url(predictions_dir) # type: ignore
                response['output_directory'] = predictions_dir_realve_path
                response['prediction_details'] = f"Prediction task completed.More prediction details, such as the prediction results, can be found in the {predictions_dir}. The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference."
                
                # Generate HTML prediction report for regression
                if generate_experiment_report:
                    try:
                        html_generator = NeuralNetworkHTMLReportGenerator()
                        html_report_path = html_generator.generate_prediction_report(
                            prediction_results=response,
                            model_info=model_components['metadata'],
                            include_visualizations=True
                        )
                        html_report_url = get_download_url(html_report_path)
                        response['html_prediction_report'] = html_report_url
                        logger.info(f"HTML regression prediction report generated: {html_report_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate HTML regression prediction report: {str(e)}")
        
        # Add experiment report path if available
        if experiment_report_path:
            experiment_report_realve_path = get_download_url(experiment_report_path) # type: ignore
            response['experiment_report_path'] = experiment_report_realve_path
        
        logger.info(f"Prediction completed for {task_type} model {model_id}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'error': f"Prediction failed: {str(e)}"}


@mcp.tool()
async def predict_from_values_neural_network(
    model_id: str,
    feature_values: Union[list, List[list]],
    generate_experiment_report: bool = True,
    ctx: Context = None,
) -> dict:
    """Make prediction from feature values (supports both regression and classification).
    
    Automatically detects model type from model metadata and calls the appropriate prediction method.
    
    Supports both single prediction and batch prediction:
    - Single: feature_values = [1.0, 2.0, 3.0, 4.0]
    - Batch: feature_values = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    
    Args:
        model_id: ID of the trained model to use for prediction
        feature_values: List of numerical feature values (1D) or list of feature value lists (2D) for batch prediction
        generate_experiment_report: Whether to generate detailed experiment report (default: True)
        
    Returns:
        Dictionary containing prediction results and optional experiment report path
    """
    try:
        # Get user-specific models directory and create user model manager
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        user_model_manager = ModelManager(user_models_dir)

        # Load model components to determine task type
        model_components = await user_model_manager.load_model(model_id)
        task_type = model_components['metadata'].get('task_type', 'regression')
        
        logger.info(f"Model {model_id} detected as {task_type} task")

        if task_type == 'classification':
            # Use classification prediction
            results, experiment_report_path, basic_csv_path = prediction.classify_from_values(
                feature_values,
                model_components["model_folder"],
                generate_experiment_report=generate_experiment_report
            )
            
            # Format response for classification
            response = {
                'task_type': 'classification',
                'model_id': model_id,
                'results': results,
                'experiment_report':f"The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference.",
                'basic_csv_path': basic_csv_path
            }
            
            # Generate HTML prediction report for classification values
            if generate_experiment_report:
                try:
                    html_generator = NeuralNetworkHTMLReportGenerator()
                    html_report_path = html_generator.generate_prediction_report(
                        prediction_results=response,
                        model_info=model_components['metadata'],
                        include_visualizations=True
                    )
                    html_report_url = get_download_url(html_report_path)
                    response['html_prediction_report'] = html_report_url
                    logger.info(f"HTML classification values prediction report generated: {html_report_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate HTML classification values prediction report: {str(e)}")
            
        else:
            # Use regression prediction (default)
            results, experiment_report_path, basic_csv_path = await prediction.predict_from_values(
                model_components,
                feature_values,
                generate_experiment_report=generate_experiment_report
            )
            
            # Format response for regression
            response = {
                'task_type': 'regression',
                'model_id': model_id,
                'results': results,
                'experiment_report': f"The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference.",
                'basic_csv_path': basic_csv_path
            }
            
            if 'experiment_details' in results:
                response['output_directory'] = results['experiment_details']['output_directory']
            
            # Generate HTML prediction report for regression values
            if generate_experiment_report:
                try:
                    html_generator = NeuralNetworkHTMLReportGenerator()
                    html_report_path = html_generator.generate_prediction_report(
                        prediction_results=response,
                        model_info=model_components['metadata'],
                        include_visualizations=True
                    )
                    html_report_url = get_download_url(html_report_path)
                    response['html_prediction_report'] = html_report_url
                    logger.info(f"HTML regression values prediction report generated: {html_report_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate HTML regression values prediction report: {str(e)}")
        
        # Add experiment report path if available
        if experiment_report_path:
            experiment_report_realve_path = get_download_url(experiment_report_path) # type: ignore
            response['experiment_report_path'] = experiment_report_realve_path
        
        logger.info(f"Prediction completed for {task_type} model {model_id}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'error': f"Prediction failed: {str(e)}"}


@mcp.tool()
async def list_neural_network_models(ctx: Context = None) -> Dict[str, Any]:
    """List all saved neural network models.
    
    Returns:
        Dictionary containing list of model information
    """
    try:
        # Get user-specific models directory and create user model manager
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        user_model_manager = ModelManager(user_models_dir)

        models = await user_model_manager.list_models()

        # Add folder paths to model info
        for model in models:
            model["folder_path"] = user_model_manager.get_model_folder_path(model["model_id"])

        result = {
            "status": "success",
            "models": models,
            "total_models": len(models),
            "models_directory": user_models_dir
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise


@mcp.tool()
async def get_neural_network_model_info(model_id: str, ctx: Context = None) -> Dict[str, Any]:
    """Get detailed information about a specific model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        Dictionary containing detailed model metadata
    """
    try:
        # Get user-specific models directory and create user model manager
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        user_model_manager = ModelManager(user_models_dir)

        model_components = await user_model_manager.load_model(model_id)
        metadata = model_components['metadata']

        # Add folder information
        metadata["model_folder"] = model_components["model_folder"]
        metadata["folder_exists"] = os.path.exists(model_components["model_folder"])
        
        # Add file information
        folder_contents = []
        if metadata["folder_exists"]:
            for item in os.listdir(model_components["model_folder"]):
                item_path = os.path.join(model_components["model_folder"], item)
                folder_contents.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
        
        result = {
            "status": "success",
            "model_info": metadata,
            "folder_contents": folder_contents
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise


@mcp.tool()
async def delete_neural_network_model(model_id: str, ctx: Context = None) -> Dict[str, Any]:
    """Delete a saved neural network model and its folder.
    
    Args:
        model_id: ID of the model to delete
        
    Returns:
        Dictionary containing deletion status
    """
    try:
        # Get user-specific models directory and create user model manager
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        user_model_manager = ModelManager(user_models_dir)

        # Get folder path before deletion
        folder_path = user_model_manager.get_model_folder_path(model_id)

        success = await user_model_manager.delete_model(model_id)
        
        result = {
            "status": "success" if success else "failed",
            "model_id": model_id,
            "deleted": success,
            "folder_path": folder_path
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise


# ================== CLASSIFICATION TOOLS ==================

async def _train_classification_model_neural_network_impl(
    training_file: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    num_epochs: int = 300,
    algorithm: str = "TPE",
    progress_callback: Optional[Callable] = None,
    models_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Train a classification neural network model with hyperparameter optimization.
    
    Automatically detects class labels and converts them to numerical format.
    Supports both string labels ("cat", "dog") and numeric labels (0, 1, 2).
    
    Args:
        training_file: Path to training data file (CSV/Excel) 
        n_trials: Number of hyperparameter optimization trials (default: 50)
        cv_folds: Number of cross-validation folds (default: 5)
        num_epochs: Number of training epochs per fold (default: 300)
        algorithm: Optimization algorithm - "TPE" or "GP" (default: "TPE")
        
    Returns:
        Dictionary containing training results with model ID, best parameters, and performance metrics
    """
    logger.info(f"Starting classification model training with file: {training_file}")
    
    try:

        # Record training start time
        training_start_time = time.time()

        # Use provided models_dir or default
        if models_dir is None:
            models_dir = "./trained_model"

        # Create user-specific model manager
        user_model_manager = ModelManager(models_dir)

        model_id = str(uuid.uuid4())
        
        # Load and preprocess classification data with automatic label encoding
        processed_df, feature_scaler, label_info = await preprocess_classification_data(training_file)
        
        # Extract information from label_info
        num_classes = label_info['num_classes']
        class_names = label_info['classes']
        feature_names = label_info['feature_names']
        feature_number = label_info['feature_number']
        
        logger.info(f"Detected {num_classes} classes: {class_names}")
        logger.info(f"Features: {feature_number} columns")
        
        # Convert to numpy array for training
        data_array = processed_df.values
        
        # Progress callback for monitoring
        progress_updates = []
        async def internal_progress_callback(info):
            progress_updates.append(info)
            logger.info(f"Training progress: {info}")
            # Call external progress callback if provided
            if progress_callback:
                # Extract progress percentage from info
                if isinstance(info, dict) and 'progress' in info:
                    await progress_callback(info['progress'], info.get('message', ''))
                else:
                    await progress_callback(0, str(info))
        
        # Create temporary model folder for saving training artifacts
        temp_model_folder = f"{models_dir}/{model_id}"
        os.makedirs(temp_model_folder, exist_ok=True)
        
        # Optimize hyperparameters with saving
        optimization_start_time = time.time()
        best_params, best_score, trials_df = await optimize_classification_hyperparameters(
            data_array,
            feature_number,
            num_classes,
            n_trials=n_trials,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            algorithm=algorithm,
            progress_callback=internal_progress_callback,
            save_dir=temp_model_folder
        )
        optimization_time = time.time() - optimization_start_time
        
        # Train final model with best parameters
        final_training_start_time = time.time()
        model_states, final_accuracy, cv_results = await train_final_classification_model(
            best_params,
            data_array,
            feature_number,
            num_classes,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            progress_callback=internal_progress_callback,
            save_dir=temp_model_folder,
            feature_scaler=feature_scaler,
            class_names=class_names,
            feature_names=feature_names
        )
        final_training_time = time.time() - final_training_start_time
        
        # Save classification model with extended metadata including label info
        saved_model_id = await user_model_manager.save_classification_model(
            model_states=model_states,
            best_params=best_params,
            best_accuracy=final_accuracy,
            feature_names=feature_names,
            class_names=class_names,
            feature_scaler=feature_scaler,
            num_classes=num_classes,
            cv_folds=cv_folds,
            training_epochs=num_epochs,
            model_id=model_id,
            task_type="classification",
            label_info=label_info
        )
        
        # Get actual model folder path
        model_folder = user_model_manager.get_model_folder_path(saved_model_id)
        
        total_training_time = time.time() - training_start_time
        
        # Generate detailed academic report for classification
        from mcp_nn_tool.training import generate_classification_report
        
        data_info = {
            'data_shape': list(processed_df.shape),
            'feature_number': feature_number,
            'feature_names': feature_names
        }
        
        # Create label encoding information for the report using existing label_info
        # label_info already contains encoding information from preprocess_classification_data
        
        classification_report = generate_classification_report(
            model_folder=model_folder, # type: ignore
            model_id=saved_model_id,
            training_file=training_file,
            data_info=data_info,
            best_params=best_params,
            best_score=best_score,
            trials_df=trials_df,
            cv_results=cv_results, # type: ignore
            average_accuracy=final_accuracy,
            optimization_time=optimization_time,
            training_time=final_training_time,
            n_trials=n_trials,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            algorithm=algorithm,
            class_names=class_names,
            num_classes=num_classes,
            label_info=label_info
        )
        
        # Save the classification report
        report_path = os.path.join(model_folder, "CLASSIFICATION_TRAINING_REPORT.md") # type: ignore
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(classification_report)
        
        logger.info(f"Classification training report saved to: {report_path}")
        report_realve_path = get_download_url(report_path) # type: ignore
        result = {
            "status": "success",
            "task_type": "classification",
            "model_id": saved_model_id,
            "model_folder": model_folder,
            "best_parameters": best_params,
            "best_accuracy": float(final_accuracy),
            "best_score": float(1.0 - best_score),  # Convert back to accuracy
            "feature_names": feature_names,
            "class_names": class_names,
            "num_classes": num_classes,
            "classification_report_path": report_realve_path,
            "training_summary": {
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "num_epochs": num_epochs,
                "algorithm": algorithm,
                "data_shape": list(processed_df.shape),
                "feature_number": feature_number,
                "optimization_time": optimization_time,
                "final_training_time": final_training_time,
                "total_training_time": total_training_time,
            },
            "cv_results": {
                "fold_accuracies": cv_results["fold_accuracy_scores"], # type: ignore
                "average_accuracy": float(final_accuracy),
                "classification_report": cv_results.get("classification_report", {}), # type: ignore
                "auc_scores": cv_results.get("auc_scores", {}), # type: ignore
                "roc_data_path": cv_results.get("roc_data_path", ""), # type: ignore
                "roc_plot_path": cv_results.get("roc_plot_path", "") # type: ignore
            }
        }
        
        # Generate HTML report for classification
        html_report_path = None
        try:
            html_generator = NeuralNetworkHTMLReportGenerator(output_dir=str(model_folder))
            html_report_path = html_generator.generate_training_report(
                model_directory=model_folder, # type: ignore
                training_results=result,
                include_visualizations=True
            )
            html_report_url = get_static_url(html_report_path)
            result["training_report_html_path"] = html_report_url
            logger.info(f"HTML classification training report generated: {html_report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate HTML classification training report: {str(e)}")
        
        logger.info(f"Classification training completed. Model ID: {saved_model_id}, Accuracy: {final_accuracy:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Classification training failed: {str(e)}")
        raise


@mcp.tool()
async def train_classification_model_neural_network(
    training_file: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    num_epochs: int = 300,
    algorithm: str = "TPE",
    ctx: Context = None,
) -> Dict[str, Any]:
    """Submit a classification neural network training task to the queue.
    
    This function submits the training task to an asynchronous queue and returns 
    immediately with a task ID. Use get_training_results() to check progress and get results.
    
    Automatically detects class labels and converts them to numerical format.
    Supports both string labels ("cat", "dog") and numeric labels (0, 1, 2).
    
    Args:
        training_file: Path to training data file (CSV/Excel) 
        n_trials: Number of hyperparameter optimization trials (default: 50)
        cv_folds: Number of cross-validation folds (default: 5)
        num_epochs: Number of training epochs per fold (default: 300)
        algorithm: Optimization algorithm - "TPE" or "GP" (default: "TPE")
        
    Returns:
        Dictionary containing task ID and submission status
    """
    try:
        # Get user-specific models directory
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        # Estimate training duration (rough estimation based on parameters)
        estimated_duration = (n_trials * cv_folds * num_epochs * 0.008) + 45  # seconds, slightly faster than regression

        # Task parameters for tracking
        task_parameters = {
            "training_file": training_file,
            "n_trials": n_trials,
            "cv_folds": cv_folds,
            "num_epochs": num_epochs,
            "algorithm": algorithm,
            "task_type": "classification",
            "models_dir": user_models_dir
        }
        
        # Submit task to queue
        task_queue = get_task_queue(user_models_dir)
        task_id = await task_queue.submit_task(
            task_type=TaskType.CLASSIFICATION_TRAINING,
            task_function=_train_classification_model_neural_network_impl,
            task_args=(),
            task_kwargs={
                "training_file": training_file,
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "num_epochs": num_epochs,
                "algorithm": algorithm,
                "models_dir": user_models_dir
            },
            task_parameters=task_parameters,
            estimated_duration=estimated_duration
        )
        
        return {
            "status": "submitted",
            "task_id": task_id,
            "message": "Classification training task submitted to queue",
            "estimated_duration": estimated_duration,
            "parameters": task_parameters,
            "next_steps": "Use get_training_results(task_id) to check progress and get results when complete"
        }
        
    except Exception as e:
        logger.error(f"Error submitting classification training task: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to submit classification training task: {str(e)}"
        }


@mcp.tool()
async def generate_html_model_report(
    model_id: str,
    report_type: str = "training",
    ctx: Context = None,
) -> Dict[str, Any]:
    """Generate an HTML report for an existing trained model.
    
    Args:
        model_id: ID of the trained model
        report_type: Type of report to generate ("training" or "summary")
        
    Returns:
        Dictionary containing HTML report path and generation status
    """
    try:
        # Get user-specific models directory and create user model manager
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        user_model_manager = ModelManager(user_models_dir)

        # Load model components
        model_components = await user_model_manager.load_model(model_id)
        metadata = model_components['metadata']
        model_folder = model_components['model_folder']
        
        # Create HTML generator
        html_generator = NeuralNetworkHTMLReportGenerator(output_dir=str(model_folder))
        
        if report_type == "training":
            # Generate training report from metadata
            training_results = {
                "status": "success",
                "model_id": model_id,
                "model_folder": model_folder,
                "best_parameters": metadata.get('best_parameters', {}),
                "feature_names": metadata.get('feature_names', []),
                "target_names": metadata.get('target_names', []),
                "task_type": metadata.get('task_type', 'regression'),
                "training_summary": {
                    "cv_folds": metadata.get('cv_folds', 5),
                    "training_epochs": metadata.get('training_epochs', 500),
                    "total_time": 0  # Not available for existing models
                }
            }
            
            # Add performance metrics
            if metadata.get('task_type') == 'classification':
                training_results["best_accuracy"] = metadata.get('best_accuracy', 0)
            else:
                training_results["best_mae"] = metadata.get('best_mae', 0)
            
            html_report_path = html_generator.generate_training_report(
                model_directory=model_folder,
                training_results=training_results,
                include_visualizations=True
            )
        else:
            return {
                "status": "error",
                "message": f"Unsupported report type: {report_type}. Supported types: 'training'"
            }
        
        html_report_url = get_download_url(html_report_path)
        
        return {
            "status": "success",
            "model_id": model_id,
            "report_type": report_type,
            "html_report_path": html_report_url,
            "message": f"HTML {report_type} report generated successfully",
            "local_path": html_report_path
        }
        
    except Exception as e:
        logger.error(f"Error generating HTML model report: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to generate HTML model report: {str(e)}"
        }


# ================== TASK QUEUE MANAGEMENT TOOLS ==================

@mcp.tool()
async def get_training_results(task_id: str, ctx: Context = None) -> Dict[str, Any]:
    """Get training results and task status for a specific task.
    
    This unified tool provides both task status and training results in one call.
    
    Args:
        task_id: ID of the training task
        
    Returns:
        Dictionary containing task status, progress, and results if completed
    """
    try:
        # Get user-specific task queue
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        task_queue = get_task_queue(user_models_dir)
        task_info = task_queue.get_task_status(task_id)
        
        if not task_info:
            return {
                "status": "error",
                "message": f"Task {task_id} not found"
            }
        
        result = {
            "task_id": task_id,
            "task_type": task_info.task_type.value,
            "status": task_info.status.value,
            "progress": task_info.progress,
            "progress_message": task_info.progress_message,
            "created_at": task_info.created_at.isoformat() if task_info.created_at else None,
            "started_at": task_info.started_at.isoformat() if task_info.started_at else None,
            "completed_at": task_info.completed_at.isoformat() if task_info.completed_at else None,
            "parameters": task_info.parameters or {},
        }
        
        # Add duration information
        if task_info.estimated_duration:
            result["estimated_duration"] = task_info.estimated_duration
        if task_info.actual_duration:
            result["actual_duration"] = task_info.actual_duration
        
        # Add results if completed successfully
        if task_info.status == TaskStatus.COMPLETED and task_info.result:
            result["training_results"] = task_info.result
            
        # Add error information if failed
        if task_info.status == TaskStatus.FAILED and task_info.error_message:
            result["error_message"] = task_info.error_message
            
        return result
        
    except Exception as e:
        logger.error(f"Error getting training results: {str(e)}")
        return {
            "status": "error", 
            "message": f"Failed to get training results: {str(e)}"
        }


@mcp.tool()
async def list_training_tasks(
    status_filter: Optional[str] = None,
    task_type_filter: Optional[str] = None,
    limit: int = 20,
    ctx: Context = None,
) -> Dict[str, Any]:
    """List all training tasks with their status.
    
    Args:
        status_filter: Filter by task status (pending, running, completed, failed, cancelled)
        task_type_filter: Filter by task type (regression_training, classification_training, prediction)
        limit: Maximum number of tasks to return (default: 20)
        
    Returns:
        Dictionary containing list of tasks with their status information
    """
    try:
        # Get user-specific task queue
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        task_queue = get_task_queue(user_models_dir)
        
        # Parse filters
        status_enum = None
        if status_filter:
            try:
                status_enum = TaskStatus(status_filter.lower())
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Invalid status filter: {status_filter}. Valid values: {[s.value for s in TaskStatus]}"
                }
        
        task_type_enum = None
        if task_type_filter:
            try:
                task_type_enum = TaskType(task_type_filter.lower())
            except ValueError:
                return {
                    "status": "error", 
                    "message": f"Invalid task type filter: {task_type_filter}. Valid values: {[t.value for t in TaskType]}"
                }
        
        # Get filtered tasks
        tasks = task_queue.list_tasks(
            status_filter=status_enum,
            task_type_filter=task_type_enum,
            limit=limit
        )
        
        # Convert to dictionaries for JSON serialization
        task_list = []
        for task in tasks:
            task_dict = {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "status": task.status.value,
                "progress": task.progress,
                "progress_message": task.progress_message,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "parameters": task.parameters or {}
            }
            
            # Add duration info if available
            if task.estimated_duration:
                task_dict["estimated_duration"] = task.estimated_duration
            if task.actual_duration:
                task_dict["actual_duration"] = task.actual_duration
                
            # Add error message for failed tasks
            if task.status == TaskStatus.FAILED and task.error_message:
                task_dict["error_message"] = task.error_message
                
            task_list.append(task_dict)
        
        return {
            "status": "success",
            "tasks": task_list,
            "total_returned": len(task_list),
            "filters_applied": {
                "status": status_filter,
                "task_type": task_type_filter,
                "limit": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing training tasks: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to list training tasks: {str(e)}"
        }


@mcp.tool()
async def get_queue_status(ctx: Context = None) -> Dict[str, Any]:
    """Get overall training queue status and statistics.
    
    Returns:
        Dictionary containing queue status, task counts, and system information
    """
    try:
        # Get user-specific task queue
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        task_queue = get_task_queue(user_models_dir)
        queue_status = task_queue.get_queue_status()
        
        # Add additional system information
        result = {
            "status": "success",
            "queue_status": queue_status,
            "system_info": {
                "persistence_file": str(task_queue.persistence_file),
                "persistence_file_exists": task_queue.persistence_file.exists()
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to get queue status: {str(e)}"
        }


@mcp.tool()
async def cancel_training_task(task_id: str, ctx: Context = None) -> Dict[str, Any]:
    """Cancel a training task by task ID.
    
    Can only cancel tasks that are pending or currently running.
    Completed, failed, or already cancelled tasks cannot be cancelled.
    
    Args:
        task_id: ID of the task to cancel
        
    Returns:
        Dictionary containing cancellation status
    """
    try:
        # Get user-specific task queue
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)
        task_queue = get_task_queue(user_models_dir)
        
        # Check if task exists
        task_info = task_queue.get_task_status(task_id)
        if not task_info:
            return {
                "status": "error",
                "message": f"Task {task_id} not found"
            }
        
        # Attempt to cancel the task
        cancelled = await task_queue.cancel_task(task_id)
        
        if cancelled:
            return {
                "status": "success",
                "task_id": task_id,
                "message": "Task cancelled successfully",
                "previous_status": task_info.status.value
            }
        else:
            return {
                "status": "error",
                "task_id": task_id,
                "message": f"Cannot cancel task with status: {task_info.status.value}",
                "current_status": task_info.status.value
            }
        
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to cancel task: {str(e)}"
        }


@mcp.tool()
async def test_url_functionality(
    test_url: str = "http://47.99.180.80/file/uploads/SLM_2.xls",
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """Test URL data loading functionality in MCP environment.

    This function directly tests URL loading without going through training workflow
    to isolate and debug HTTP 404 issues in MCP server environment.

    Args:
        test_url: URL to test (default: http://47.99.180.80/file/uploads/SLM_2.xls)
        ctx: MCP context (optional)

    Returns:
        Dictionary with test results and diagnostics
    """
    try:
        logger.info(f"Testing URL functionality in MCP environment: {test_url}")

        # Get user context
        user_id = get_user_id(ctx)
        user_models_dir = get_user_models_dir(user_id)

        test_results = {
            "status": "running",
            "test_url": test_url,
            "user_id": user_id,
            "user_models_dir": user_models_dir,
            "tests": {},
            "environment_info": {}
        }

        # Test 1: Direct URL ping/accessibility
        try:
            import requests
            response = requests.head(test_url, timeout=10)
            test_results["tests"]["url_accessibility"] = {
                "status": "success",
                "http_status": response.status_code,
                "headers": dict(response.headers),
                "message": f"URL accessible, HTTP {response.status_code}"
            }
        except Exception as e:
            test_results["tests"]["url_accessibility"] = {
                "status": "failed",
                "error": str(e),
                "message": "URL not accessible via requests.head()"
            }

        # Test 2: Direct pandas read
        try:
            import pandas as pd
            data = pd.read_excel(test_url)
            test_results["tests"]["direct_pandas"] = {
                "status": "success",
                "data_shape": data.shape,
                "columns": list(data.columns),
                "message": f"Direct pandas.read_excel() successful, shape: {data.shape}"
            }
        except Exception as e:
            test_results["tests"]["direct_pandas"] = {
                "status": "failed",
                "error": str(e),
                "message": "Direct pandas.read_excel() failed"
            }

        # Test 3: Our custom read_data_file function
        try:
            data = await read_data_file(test_url, max_retries=3, retry_delay=1.0)
            test_results["tests"]["custom_read_data_file"] = {
                "status": "success",
                "data_shape": data.shape,
                "columns": list(data.columns),
                "message": f"Custom read_data_file() successful, shape: {data.shape}"
            }
        except Exception as e:
            test_results["tests"]["custom_read_data_file"] = {
                "status": "failed",
                "error": str(e),
                "message": "Custom read_data_file() failed"
            }

        # Test 4: Threading context test
        try:
            import concurrent.futures
            import threading

            def load_in_thread():
                return pd.read_excel(test_url)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(load_in_thread)
                thread_data = future.result(timeout=30)

            test_results["tests"]["threading_context"] = {
                "status": "success",
                "data_shape": thread_data.shape,
                "thread_id": threading.get_ident(),
                "message": f"Threading context successful, shape: {thread_data.shape}"
            }
        except Exception as e:
            test_results["tests"]["threading_context"] = {
                "status": "failed",
                "error": str(e),
                "message": "Threading context failed"
            }

        # Environment diagnostics
        test_results["environment_info"] = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "user_agent": getattr(requests.utils, 'default_user_agent', lambda: 'unknown')(),
            "ssl_context_available": hasattr(__import__('ssl'), 'create_default_context'),
            "pandas_version": pd.__version__,
            "current_time": time.time()
        }

        # Summary
        successful_tests = sum(1 for test in test_results["tests"].values() if test["status"] == "success")
        total_tests = len(test_results["tests"])

        test_results["status"] = "completed"
        test_results["summary"] = {
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "success_rate": f"{successful_tests}/{total_tests}",
            "overall_result": "success" if successful_tests == total_tests else "partial_failure"
        }

        logger.info(f"URL functionality test completed: {successful_tests}/{total_tests} tests passed")
        return test_results

    except Exception as e:
        logger.error(f"URL functionality test failed: {str(e)}")
        import traceback
        return {
            "status": "error",
            "test_url": test_url,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "message": "URL functionality test encountered an unexpected error"
        }


class NeuralNetworkMCPServer:
    """MCP Server for Neural Network operations using FastMCP."""
    
    def __init__(self, models_dir: str = "./trained_model"):
        """Initialize the Neural Network MCP Server.

        Args:
            models_dir: Directory to store trained models (default: './trained_model')
        """
        # Note: models_dir parameter is kept for compatibility but user-specific directories are now used
        self.mcp = mcp
        
    async def run(self, host: str = "localhost", port: int = 8000):
        """Run the MCP server.
        
        Args:
            host: Host to bind the server (not used in FastMCP STDIO mode)
            port: Port to bind the server (not used in FastMCP STDIO mode)
        """
        logger.info(f"Starting Neural Network MCP Server")
        logger.info(f"Models will be saved to user-specific directories under: trained_models/{{user_id}}")
        
        # Initialize task queue
        await initialize_task_queue()
        logger.info("Task queue initialized")
        
        try:
            # FastMCP typically runs in STDIO mode for MCP protocol
            await self.mcp.run() # type: ignore
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            # Cleanup task queue on shutdown
            from mcp_nn_tool.task_queue import shutdown_task_queue
            await shutdown_task_queue()
            logger.info("Task queue shutdown complete")


# Factory function for creating the server
def create_neural_network_server(models_dir: str = "./trained_model") -> NeuralNetworkMCPServer:
    """Create a Neural Network MCP Server instance.
    
    Args:
        models_dir: Directory to store trained models (default: './trained_model')
        
    Returns:
        Configured NeuralNetworkMCPServer instance
    """
    return NeuralNetworkMCPServer(models_dir)