# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-target Neural Network Training and Prediction System with MCP (Model Context Protocol) server integration. The project provides comprehensive neural network training capabilities for both regression and classification tasks, with advanced hyperparameter optimization and experiment reporting.

## Architecture

### Core Components

- **MCP Server** (`src/mcp_nn_tool/mcp_server.py`) - Main FastMCP server with 7 tool endpoints
- **Training Module** (`src/mcp_nn_tool/training.py`) - Hyperparameter optimization and cross-validation training
- **Prediction Module** (`src/mcp_nn_tool/prediction.py`) - Ensemble predictions with experiment reporting
- **Data Utils** (`src/mcp_nn_tool/data_utils.py`) - Data preprocessing and validation
- **Model Manager** (`src/mcp_nn_tool/model_manager.py`) - Model persistence and metadata management
- **Neural Network** (`src/mcp_nn_tool/neural_network.py`) - PyTorch neural network architecture

### Server Architecture

The project supports dual communication modes:
- **stdio mode** - Standard I/O for traditional MCP clients (Claude Desktop)
- **SSE mode** - Server-Sent Events for web clients (CherryStudio)

Multiple server entry points:
- `run_mcp_server.py` - Main production server with mode selection
- `hybrid_server.py` - Dual-mode server (stdio + SSE simultaneously)
- `single_port_server.py` - Single port SSE server
- `simple_mcp_server.py` - Basic stdio-only server

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -e .

# Install with optional dependencies
pip install -e ".[dev]"     # Development tools (pytest, black, isort, mypy)
pip install -e ".[gp]"      # Gaussian Process optimization

# Using uv (alternative)
uv sync
```

### Running the Server

```bash
# Default stdio mode
python run_mcp_server.py

# SSE mode for web clients
python run_mcp_server.py --mode sse

# Custom SSE configuration
python run_mcp_server.py --mode sse --host 127.0.0.1 --port 9000 --debug

# Hybrid mode (both stdio and SSE)
python hybrid_server.py --port 8080 --debug
```

### Code Quality

```bash
# Format code
black src/ --line-length 88
isort src/ --profile black

# Type checking
mypy src/

# Run tests
pytest
pytest tests/test_specific.py  # Single test file
```

## Data Flow Architecture

1. **Data Ingestion** - CSV/Excel files or raw CSV content
2. **Preprocessing** - StandardScaler normalization with cross-validation leak prevention
3. **Training** - Optuna hyperparameter optimization with k-fold cross-validation
4. **Model Storage** - UUID-based folder structure in `./trained_model/`
5. **Prediction** - Ensemble predictions using all CV fold models
6. **Reporting** - Academic-style reports with visualizations

## Model Storage Structure

```
trained_model/{model_id}/
├── model_metadata.json       # Configuration and performance metrics
├── model_states.pth          # All CV fold model weights
├── best_params.json          # Optimized hyperparameters
├── scalers.pkl              # Feature/target preprocessing scalers
├── column_names.json        # Feature and target column names
├── hyperparameter_optimization_trials.csv
├── training_history.csv     # Training metrics per epoch
├── training_curves.png      # Loss visualization
├── cv_predictions_scatter*.png  # Validation scatter plots
└── academic_report.md       # Comprehensive training report
```

## MCP Tools Available

### Training Tools (Asynchronous)
- `train_neural_network_regression` - Submit regression training to queue
- `train_classification_model_neural_network` - Submit classification training to queue

### Prediction Tools
- `predict_from_file_neural_network` - Batch predictions from files
- `predict_from_values_neural_network` - Single/multiple value predictions

### Model Management
- `list_neural_network_models` - List all trained models
- `get_neural_network_model_info` - Get detailed model information
- `delete_neural_network_model` - Remove model and its files
- `generate_html_model_report` - Generate HTML visualization reports for existing models

### Task Queue Management
- `get_training_results` - Get training results and task status (unified tool)
- `list_training_tasks` - List all training tasks with their status
- `get_queue_status` - Get overall training queue status
- `cancel_training_task` - Cancel a training task by task_id

## Key Technical Features

- **Asynchronous Task Queue** - Non-blocking training with progress tracking and status monitoring
- **Interactive HTML Reports** - Professional visualization reports for training and prediction analysis
- **Multi-target Support** - Single or multiple regression/classification targets
- **Hyperparameter Optimization** - TPE and Gaussian Process algorithms via Optuna
- **Cross-Validation** - Robust k-fold validation with ensemble predictions
- **Loss Functions** - MAE (L1) and MSE (L2) for regression
- **Experiment Reporting** - Academic-style reports with visualizations
- **Data Scaling** - Automatic standardization with inverse transformation
- **Model Persistence** - Complete model state and metadata preservation

## Task Queue System

The system now includes an asynchronous task queue for managing long-running training operations:

### Architecture
- **Task Manager** - Handles task submission, execution, and status tracking
- **Background Worker** - Executes training tasks without blocking the MCP server
- **Status Persistence** - Task information is saved to disk and restored on restart
- **Progress Tracking** - Real-time progress updates with percentage and status messages

### Usage Workflow

1. **Submit Training Task**
   ```python
   # Submit task and get immediate response with task_id
   result = await train_neural_network_regression("data.csv", n_trials=100)
   task_id = result["task_id"]
   ```

2. **Monitor Progress**
   ```python
   # Check task status and progress
   status = await get_training_results(task_id)
   # Returns: status, progress (%), progress_message, results (if complete)
   ```

3. **Task Management**
   ```python
   # List all tasks
   tasks = await list_training_tasks(status_filter="running", limit=10)
   
   # Get queue status
   queue_info = await get_queue_status()
   
   # Cancel a task
   cancel_result = await cancel_training_task(task_id)
   ```

### Task States
- **pending** - Task queued but not yet started
- **running** - Task currently executing with progress updates
- **completed** - Task finished successfully with results available
- **failed** - Task encountered an error (error message included)
- **cancelled** - Task was cancelled by user request

### Persistence
- Task queue state is automatically saved to `./trained_model/task_queue.json`
- Tasks interrupted by system restart are marked as failed
- Historical task information is preserved for analysis

## HTML Visualization Reports

The system generates professional, interactive HTML reports for comprehensive analysis:

### Training Reports
- **Model Architecture** - Visual representation of neural network layers and parameters
- **Training Progress** - Loss curves and convergence analysis with embedded visualizations
- **Hyperparameter Optimization** - Best performing parameter combinations from Optuna trials
- **Cross-Validation Results** - Performance across folds with scatter plots (regression) or ROC curves (classification)
- **Performance Metrics** - Detailed statistics and model evaluation
- **Recommendations** - Automated suggestions based on training results

### Prediction Reports
- **Prediction Summary** - Overview of prediction task and model information
- **Result Statistics** - Distribution analysis and statistical summaries
- **Prediction Details** - Tabular view of individual predictions
- **Model Information** - Architecture and feature details
- **Quality Assessment** - Confidence analysis and result validation

### Features
- **Professional Styling** - Modern, responsive design with gradient themes
- **Embedded Visualizations** - Base64-encoded images for self-contained reports
- **Interactive Elements** - Hover effects and responsive layout
- **Comprehensive Coverage** - All training artifacts and prediction results included
- **Download Ready** - Self-contained HTML files with all assets embedded

### Usage
```python
# Automatic generation during training/prediction
result = await train_neural_network_regression("data.csv")
# result["html_training_report"] contains URL to HTML report

# Manual generation for existing models
html_report = await generate_html_model_report(model_id, "training")
# html_report["html_report_path"] contains URL to generated report
```

## Configuration

- **Python Version** - Requires Python 3.11+
- **Core Dependencies** - PyTorch, scikit-learn, pandas, optuna, FastMCP
- **Optional GPU** - CUDA support for faster training
- **Code Style** - Black formatter (line length 88), isort with black profile
- **Type Checking** - mypy with strict untyped definitions disabled

## Testing Strategy

- Use `pytest` for unit tests
- Include `pytest-asyncio` for async function testing
- Test files should follow pattern `test_*.py`
- Run specific tests with `pytest tests/test_filename.py`