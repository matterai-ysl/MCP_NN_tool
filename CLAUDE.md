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

### Training Tools
- `train_neural_network` - File-based regression training
- `train_neural_network_from_content` - Content-based regression training
- `train_classification_model` - File-based classification training  
- `train_classification_from_content` - Content-based classification training

### Prediction Tools
- `predict_from_file` - Batch predictions from files
- `predict_from_content` - Batch predictions from CSV content
- `predict_from_values` - Single/multiple value predictions

### Model Management
- `list_models` - List all trained models
- `get_model_info` - Get detailed model information
- `delete_model` - Remove model and its files

## Key Technical Features

- **Multi-target Support** - Single or multiple regression/classification targets
- **Hyperparameter Optimization** - TPE and Gaussian Process algorithms via Optuna
- **Cross-Validation** - Robust k-fold validation with ensemble predictions
- **Loss Functions** - MAE (L1) and MSE (L2) for regression
- **Experiment Reporting** - Academic-style reports with visualizations
- **Data Scaling** - Automatic standardization with inverse transformation
- **Model Persistence** - Complete model state and metadata preservation

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