# MCP Neural Network Tool

Enhanced Multi-Target Neural Network Training and Prediction System with comprehensive experiment reporting capabilities.

## Features

### Core Functionality
- **Multi-target regression** support (single or multiple target variables)
- **Hyperparameter optimization** using Optuna (TPE/GP algorithms)
- **Cross-validation training** with detailed performance tracking
- **Ensemble predictions** using CV-fold models
- **FastMCP server** integration for easy API access
- **Comprehensive experiment reporting** for both training and prediction

### Advanced Reporting
- **Training Reports**: Academic-style reports with data processing details, hyperparameter optimization, training curves, and cross-validation results
- **Prediction Reports**: Detailed experiment documentation with original scale features and predictions
- **Multi-target visualization**: Individual scatter plots for each target
- **Data preservation**: Both scaled and original scale data saved

### 🚀 Model Training
- **Automated Hyperparameter Optimization**: Uses Optuna for Bayesian optimization (TPE or Gaussian Process)
- **Cross-Validation**: Built-in k-fold cross-validation for robust model evaluation
- **Multi-Target Support**: Train models for single or multiple target variables simultaneously
- **Automatic Model ID Generation**: Each model gets a unique UUID for easy identification
- **Organized Model Storage**: Models saved in `./trained_model` directory with dedicated folders
- **Complete Configuration Saving**: All training parameters and preprocessing information preserved
- **Data Preprocessing**: Automatic standardization and preprocessing of input data

### 🔮 Prediction Capabilities
- **File-based Predictions**: Make predictions on data from CSV/Excel files
- **Value-based Predictions**: Direct prediction from feature value lists
- **Batch Predictions**: Efficient batch processing of multiple predictions
- **Multi-Target Predictions**: Support for predicting multiple targets simultaneously
- **Ensemble Predictions**: Averages predictions from all cross-validation folds
- **Confidence Intervals**: Optional prediction statistics and confidence intervals

### 📊 Model Management
- **Folder-Based Organization**: Each model gets its own folder in `./trained_model/`
- **Model Registry**: List, view, and manage all trained models
- **Model Metadata**: Detailed information about model architecture and performance
- **Prediction History**: Save prediction results to model folders automatically
- **Model Deletion**: Clean removal of unwanted models and their folders
- **Automatic Scaling**: Preserves data preprocessing for consistent predictions

## Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (optional, for faster training)

### Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# For Gaussian Process optimization (optional)
pip install -e ".[gp]"

# For development tools (optional)
pip install -e ".[dev]"
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv
uv sync
```

### Basic Usage

1. **Training a Model**:
```python
result = await train_neural_network(
    training_file="data.csv",
    target_columns=1,  # Number of target columns
    n_trials=100,
    cv_folds=5,
    num_epochs=500,
    algorithm="TPE",
    loss_function="MAE"
)
```

2. **Making Predictions with Experiment Report**:
```python
result = await predict_with_experiment_report(
    prediction_file="test_data.csv",
    model_id="your-model-id",
    output_dir=None,  # Auto-generated
    experiment_name="Production Experiment"
)
```

## API Reference

### Training Tools

#### `train_neural_network`
Train a neural network with hyperparameter optimization and comprehensive reporting.

**Parameters:**
- `training_file` (str): Path to training data (CSV/Excel)
- `target_columns` (int): Number of target columns (default: 1)
- `n_trials` (int): Hyperparameter optimization trials (default: 100)
- `cv_folds` (int): Cross-validation folds (default: 5)
- `num_epochs` (int): Training epochs per fold (default: 500)
- `algorithm` (str): "TPE" or "GP" (default: "TPE")
- `loss_function` (str): "MAE" or "MSE" (default: "MAE")

**Returns:** Training results with model ID and performance metrics

### Prediction Tools

#### `predict_from_file`
Make predictions on data from a file with optional experiment reporting.

**Parameters:**
- `model_id` (str): ID of the trained model
- `prediction_file` (str): Path to CSV file containing prediction data  
- `generate_experiment_report` (bool, optional): Generate detailed experiment report (default: False)

**Returns:** Dictionary containing prediction results and file paths

**Generated Files:**
- **Basic Mode** (always): Single CSV with original scale features and predictions
- **Experiment Mode** (when `generate_experiment_report=True`):
  - `*_scaled.csv`: Preprocessed features + scaled predictions
  - `*_original.csv`: Original scale features + original scale predictions  
  - `*_detailed.csv`: Comprehensive results with statistics
  - `*_report.md`: Detailed experiment documentation
  - `*_metadata.json`: Experiment configuration and statistics

**Example:**
```python
# Basic prediction (saves 1 CSV file)
result = await client.call_tool("predict_from_file", {
    "model_id": "your_model_id",
    "prediction_file": "prediction_data.csv"
})

# Prediction with detailed experiment report (saves 5 files)
detailed_result = await client.call_tool("predict_from_file", {
    "model_id": "your_model_id", 
    "prediction_file": "prediction_data.csv",
    "generate_experiment_report": True
})
```

#### `predict_from_values`
Make prediction from feature values with optional experiment reporting.

**Parameters:**
- `model_id` (str): ID of the trained model
- `feature_values` (list): List of numerical feature values  
- `generate_experiment_report` (bool, optional): Generate detailed experiment report (default: False)

**Returns:** Dictionary containing prediction results and file paths

**Generated Files:**
- **Basic Mode** (always): Single CSV with original scale features and predictions
- **Experiment Mode** (when `generate_experiment_report=True`):
  - `*_scaled.csv`: Preprocessed features + scaled predictions
  - `*_original.csv`: Original scale features + original scale predictions
  - `*_detailed.csv`: Comprehensive results with statistics
  - `*_report.md`: Detailed experiment documentation
  - `*_metadata.json`: Experiment configuration and statistics

**Example:**
```python
# Basic single prediction (saves 1 CSV file)
result = await client.call_tool("predict_from_values", {
    "model_id": "your_model_id",
    "feature_values": [1.5, 2.5, 3.5, 4.5]
})

# Single prediction with detailed experiment report (saves 5 files)
detailed_result = await client.call_tool("predict_from_values", {
    "model_id": "your_model_id",
    "feature_values": [1.5, 2.5, 3.5, 4.5],
    "generate_experiment_report": True
})
```

### Model Management

#### `list_models`
List all saved models with metadata.

#### `get_model_info`
Get detailed information about a specific model.

#### `delete_model`
Delete a saved model and its files.

## Data Format

### Training Data
CSV/Excel files with:
- Feature columns first
- Target columns last
- Numerical data only
- No missing values

Example for single target:
```csv
feature_1,feature_2,feature_3,target
1.0,2.5,3.2,10.5
2.1,3.6,4.1,12.3
...
```

Example for multi-target:
```csv
feature_1,feature_2,target_1,target_2
1.0,2.5,10.5,20.1
2.1,3.6,12.3,22.8
...
```

### Prediction Data
Same format as training data but without target columns:
```csv
feature_1,feature_2,feature_3
1.5,2.8,3.5
2.3,4.1,4.8
...
```

## Experiment Reports

### Training Report Features
- **Data Processing**: Detailed preprocessing pipeline documentation
- **Hyperparameter Optimization**: Search space and optimization results
- **Cross-Validation**: Fold-wise performance and ensemble statistics  
- **Training Curves**: Loss progression with dynamic labeling based on loss function
- **Multi-target Support**: Individual performance metrics and visualizations
- **Reproducibility**: Complete experimental setup documentation

### Prediction Report Features
- **Comprehensive Results**: Original and scaled features/predictions
- **Uncertainty Quantification**: Model agreement and confidence intervals
- **Statistical Analysis**: Prediction distributions and summary statistics
- **Quality Assurance**: Data validation and processing verification
- **Multi-target Visualization**: Individual analysis for each target variable

## File Structure

### Training Output
```
trained_model/{model_id}/
├── model_metadata.json              # Model configuration
├── model_states.pth                 # CV fold model weights
├── best_params.json                 # Optimized hyperparameters  
├── scalers.pkl                      # Preprocessing scalers
├── column_names.json                # Feature/target names
├── hyperparameter_optimization_trials.csv  # Optimization history
├── training_history.csv             # Training metrics per epoch
├── training_curves.png              # Training visualization
├── cv_predictions_scatter*.png      # Cross-validation scatter plots
└── academic_report.md               # Comprehensive training report
```

### Prediction Output
```
prediction_experiment_{timestamp}/
├── prediction_results_detailed.csv  # Complete results with original scale
├── experiment_metadata.json         # Experiment configuration
└── prediction_experiment_report.md  # Detailed experiment documentation
```

## Advanced Features

### Multi-Target Regression
- Supports any number of target variables
- Individual performance metrics per target
- Separate visualization for each target
- Target-specific confidence intervals

### Data Processing
- **StandardScaler normalization** for features and targets
- **Cross-validation leak prevention** with fold-specific scaling
- **Inverse transformation** to original scale for interpretability
- **Data quality validation** with comprehensive checks

### Ensemble Prediction
- Uses all cross-validation fold models
- Simple averaging for robust predictions
- Uncertainty quantification via model agreement
- Confidence intervals based on ensemble variance

### Loss Function Support
- **MAE (L1Loss)**: Mean Absolute Error for robust training
- **MSE (MSELoss)**: Mean Squared Error for traditional regression
- **Dynamic reporting**: Training curves adapt to selected loss function

## Examples

See `examples/example_usage.py` for comprehensive usage examples including:
- Basic training workflow
- Prediction with experiment reporting
- Multi-target regression
- FastMCP server usage

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- optuna
- matplotlib
- seaborn
- mcp (FastMCP)

## License

[Add your license information here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows Google-style documentation
5. Submit a pull request

## Support

For issues, questions, or contributions, please open an issue on the project repository.
