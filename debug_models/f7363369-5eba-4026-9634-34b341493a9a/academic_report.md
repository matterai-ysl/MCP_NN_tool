# Neural Network Training Report

**Generated on:** 2025-09-18 00:58:41  
**Model ID:** `f7363369-5eba-4026-9634-34b341493a9a`  
**Model Folder:** `./debug_models/f7363369-5eba-4026-9634-34b341493a9a`

## Executive Summary

This report documents a comprehensive neural network training experiment conducted for academic research and reproducibility purposes. The experiment involved hyperparameter optimization using Optuna, followed by cross-validated model training with detailed performance analysis.

### Key Results
- **Final Cross-Validation MAE:** 0.523131
- **Best Hyperparameters Found:** {
  "lr": 0.00043628123748669037,
  "batch_size": 64,
  "drop_out": 0.10898268683922359,
  "unit": 80,
  "layber_number": 1
}
- **Optimization Time:** 0.58 seconds
- **Training Time:** 0.77 seconds

---

## 1. Experimental Setup

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| Data File | `http://47.99.180.80/file/uploads/SLM_2.xls` |
| Data Shape | [2703, 5] |
| Number of Features | 4 |
| Number of Targets | 1 |
| Total Samples | 2703 |

**Feature Names:** layer_thickness, hatch_distance, laser_power, laser_velocity
**Target Names:** relative_density

### 1.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Hyperparameter Optimization Algorithm | TPE (TPE: Tree-structured Parzen Estimator, GP: Gaussian Process) |
| Number of Trials | 2 |
| Cross-Validation Folds | 2 |
| Training Epochs per Fold | 3 |
| Loss Function | L1Loss (Mean Absolute Error) |
| Optimizer | Adam |

### 1.3 Hardware and Software Environment

- **Python Version:** 3.8+
- **Deep Learning Framework:** PyTorch
- **Optimization Library:** Optuna
- **Device:** CPU

---

## 2. Data Processing and Preprocessing

### 2.1 Data Loading and Initial Inspection

The training data was loaded from `http://47.99.180.80/file/uploads/SLM_2.xls` and underwent comprehensive preprocessing to ensure model compatibility and optimal performance.

**Input Features (4 columns):**
`layer_thickness`, `hatch_distance`, `laser_power`, `laser_velocity`

**Target Variables (1 column):**
`relative_density`

### 2.2 Data Preprocessing Pipeline

The preprocessing pipeline implements a standardized approach to ensure consistent data transformation across training and prediction phases.

#### 2.2.1 Feature Normalization

**Normalization Method**: StandardScaler (Z-score normalization)

The StandardScaler transformation is applied to both features and targets:

```python
# Feature transformation: X_scaled = (X - μ) / σ
# Where μ = mean, σ = standard deviation
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
y_scaled = (y - y.mean(axis=0)) / y.std(axis=0)
```

**Normalization Benefits:**
- **Gradient Optimization**: Ensures stable gradient descent by normalizing feature scales
- **Learning Rate Efficiency**: Allows optimal learning rate selection across all features
- **Numerical Stability**: Prevents overflow/underflow issues in neural network computations
- **Activation Function Performance**: Optimizes activation function behavior (ReLU, etc.)

#### 2.2.2 Data Transformation Details

| Component | Transformation | Purpose |
|-----------|---------------|---------|
| **Input Features** | StandardScaler fit_transform | Normalize feature distributions to μ=0, σ=1 |
| **Target Variables** | StandardScaler fit_transform | Normalize target distributions for stable training |
| **Cross-Validation** | Separate fit per fold | Prevent data leakage between train/validation sets |
| **Prediction Phase** | Saved scaler transform | Ensure consistent preprocessing for new predictions |

#### 2.2.3 Scaler Persistence

**Scaler Storage Strategy:**
- **Feature Scaler**: Saved as `scalers.pkl` in model directory
- **Target Scaler**: Included in the same pickle file for inverse transformation
- **Metadata**: Scaler parameters stored in `column_names.json` for reference
- **Validation**: Scaler compatibility checked during prediction phase

### 2.4 Cross-Validation Data Splitting

#### 2.4.1 Splitting Strategy

**Method**: K-Fold Cross-Validation with 2 folds
- **Randomization**: `random_state=42` for reproducible splits
- **Shuffle**: Data is shuffled before splitting to ensure representative folds
- **Stratification**: Not applicable for regression tasks

#### 2.4.2 Fold Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Number of Folds** | 2 | Each fold uses ~50.0% for training, ~50.0% for validation |
| **Training Samples per Fold** | ~1351 | Approximate number of training samples |
| **Validation Samples per Fold** | ~1351 | Approximate number of validation samples |
| **Random Seed** | 42 | Ensures reproducible train/validation splits |

#### 2.4.3 Data Leakage Prevention

**Key Safeguards:**
- **Separate Scaling**: Each CV fold fits scaler only on training data
- **Independent Validation**: Validation sets never used for preprocessing parameter estimation


### 2.5 Data Transformation for Neural Networks

#### 2.5.1 Tensor Conversion

**PyTorch Integration:**
- **Data Type**: Convert to `torch.FloatTensor` for GPU compatibility
- **Batch Processing**: Data organized into batches of size 64
- **Memory Management**: Efficient tensor operations for large datasets
- **Device Placement**: Automatic CPU/GPU tensor placement based on hardware

#### 2.5.2 Batch Processing Configuration

| Parameter | Value | Impact |
|-----------|-------|--------|
| **Batch Size** | 64 | Optimized for memory usage and gradient stability |
| **Shuffle Training** | True | Randomizes sample order each epoch |
| **Shuffle Validation** | False | Maintains consistent validation order |
| **Drop Last Batch** | False | Includes partial batches for complete data coverage |

### 2.6 Preprocessing Metadata

**Stored Information:**
- **Original Data Statistics**: Mean, standard deviation, min/max values
- **Transformation Parameters**: Scaler coefficients for each feature/target
- **Column Mapping**: Feature and target name preservation
- **Preprocessing Version**: Compatibility tracking for future predictions

### 2.7 Data Processing Summary

**Key Preprocessing Achievements:**
1. **Consistent Scaling**: All features and targets normalized using StandardScaler
2. **Leak-Free CV**: Independent preprocessing for each cross-validation fold
3. **Invertible Transforms**: Full scaler preservation for prediction phase
4. **Quality Assurance**: Comprehensive validation of preprocessing pipeline
5. **Reproducible Process**: Fixed random seeds and deterministic transformations

**Files Generated:**
- `scalers.pkl`: Complete scaler objects for features and targets
- `column_names.json`: Feature and target name mappings
- `model_metadata.json`: Preprocessing configuration details

---

## 3. Hyperparameter Optimization

### 3.1 Search Space

The hyperparameter optimization was conducted using TPE algorithm with the following search space:

| Hyperparameter | Type | Range | Distribution |
|----------------|------|-------|--------------|
| Learning Rate (`lr`) | Float | [1e-5, 1e-2] | Log-uniform |
| Batch Size (`batch_size`) | Integer | [16, 128] | Step=16 |
| Dropout Rate (`drop_out`) | Float | [0.0, 0.3] | Uniform |
| Hidden Units (`unit`) | Integer | [16, 128] | Step=16 |
| Number of Layers (`layber_number`) | Integer | [1, 8] | Uniform |

### 3.2 Optimization Results

**Best Hyperparameters:**
```json
{
  "lr": 0.00043628123748669037,
  "batch_size": 64,
  "drop_out": 0.10898268683922359,
  "unit": 80,
  "layber_number": 1
}
```

**Best Cross-Validation Score:** 0.507296

### 3.3 Hyperparameter Optimization Trials

The complete hyperparameter optimization results are saved in:
- **File:** `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/hyperparameter_optimization_trials.csv`
- **Total Trials:** 2
 
## 4. Final Model Training

### 4.1 Cross-Validation Training

The final model was trained using 2-fold cross-validation with the best hyperparameters. Training history and validation metrics were recorded for each epoch.

### 4.2 Training Results

| Metric | Value |
|--------|-------|
| Average CV MAE | 0.523131 |
| CV Standard Deviation | 0.014503 |
| Best Fold MAE | 0.508628 |
| Worst Fold MAE | 0.537634 |

#### Fold-wise Results

| Fold | MAE |
|------|-----|
| 1 | 0.508628 |
| 2 | 0.537634 |


### 4.3 Training Curves

Training progress and validation metrics are visualized in the following plots:

#### 4.3.1 Training and Validation Loss
![Training Curves](./debug_models/f7363369-5eba-4026-9634-34b341493a9a/training_curves.png)

**File:** `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/training_curves.png`

This figure shows:
- **Subplot 1:** Average training and validation loss across all folds
- **Subplot 2:** Average validation MAE progression
- **Subplot 3:** Average validation R² progression  
- **Subplot 4:** Individual fold training curves

#### 4.3.2 Training History Data
**File:** `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/training_history.csv`

The complete training history including epoch-by-epoch metrics for all folds is saved in CSV format for further analysis.

### 4.4 Cross-Validation Predictions

#### 4.4.1 Prediction Scatter Plot
![CV Predictions](./debug_models/f7363369-5eba-4026-9634-34b341493a9a/cv_predictions_scatter.png)

**File:** `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/cv_predictions_scatter.png`

This scatter plot shows actual vs predicted values from cross-validation, including:
- Perfect prediction line (red dashed)
- Performance statistics (MAE, R²)
- Sample distribution

#### 4.4.2 Prediction Statistics

| Metric | Value |
|--------|-------|
| Cross-Validation MAE | 0.523126 |
| Cross-Validation R² | 0.342312 |
| Number of Predictions | 2703 |
| Prediction Range | [-0.757, 1.877] |
| Actual Range | [-3.624, 1.505] |

**Prediction Data Files:**
- `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/cv_predictions.csv` - Basic predictions data


---

## 5. Model Architecture and Implementation

### 5.1 Neural Network Architecture

The final model uses a Multi-Layer Perceptron (MLP) architecture with the following specifications:

| Component | Configuration |
|-----------|---------------|
| Input Layer | 4 features |
| Hidden Layers | 1 layers |
| Hidden Units per Layer | 80 |
| Activation Function | ReLU |
| Dropout Rate | 0.10898268683922359 |
| Output Layer | 1 target(s) |
| Loss Function | L1Loss (Mean Absolute Error) |

### 5.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 4.36e-04 |
| Batch Size | 64 |
| Loss Function | L1Loss (Mean Absolute Error) |
| Device | CPU |

---

## 6. Conclusions and Future Work

### 6.1 Key Findings

1. **Model Performance**: The optimized neural network achieved a cross-validation MAE of 0.523131
2. **Hyperparameter Sensitivity**: The optimization process explored 2 different configurations
3. **Training Stability**: Cross-validation results show consistent performance across 2 folds

### 6.2 Reproducibility

This experiment is fully reproducible using the following artifacts:
- **Model Weights**: Saved in `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/`
- **Hyperparameters**: `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/model_metadata.json`
- **Training Data**: `http://47.99.180.80/file/uploads/SLM_2.xls`
- **Training History**: `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/training_history.csv`
- **Optimization Trials**: `./debug_models/f7363369-5eba-4026-9634-34b341493a9a/hyperparameter_optimization_trials.csv`

### 6.3 Technical Implementation

- **Framework**: PyTorch for neural network implementation
- **Optimization**: Optuna with TPE sampler for hyperparameter search
- **Cross-Validation**: 2-fold stratified cross-validation
- **Loss Function**: L1Loss (Mean Absolute Error)
- **Data Processing**: StandardScaler for feature normalization

---

## Appendix

### A.1 System Information

- **Generation Time**: 2025-09-18 00:58:41
- **Model ID**: `f7363369-5eba-4026-9634-34b341493a9a`
- **Training System**: Enhanced Multi-Target Neural Network Training System
- **Report Version**: 2.0 (with MAE Loss Function Support)

### A.2 File Structure

```
./debug_models/f7363369-5eba-4026-9634-34b341493a9a/
├── model_metadata.json              # Model configuration and metadata
├── fold_X_model.pth                 # Trained model weights for each fold
├── hyperparameter_optimization_trials.csv  # Optimization history
├── training_history.csv             # Epoch-by-epoch training metrics
├── cv_predictions.csv               # Cross-validation predictions
├── training_curves.png              # Training progress visualization
├── cv_predictions_scatter.png       # Prediction accuracy visualization
└── academic_report.md               # This report
```

---

*This report was automatically generated by the Enhanced Multi-Target Neural Network Training System with MAE loss function support.*
