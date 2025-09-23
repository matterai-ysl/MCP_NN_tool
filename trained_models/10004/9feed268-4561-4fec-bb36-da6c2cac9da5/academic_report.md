# Neural Network Training Report

**Generated on:** 2025-09-22 17:43:43  
**Model ID:** `9feed268-4561-4fec-bb36-da6c2cac9da5`  
**Model Folder:** `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5`

## Executive Summary

This report documents a comprehensive neural network training experiment conducted for academic research and reproducibility purposes. The experiment involved hyperparameter optimization using Optuna, followed by cross-validated model training with detailed performance analysis.

### Key Results
- **Final Cross-Validation MAE:** 0.347791
- **Best Hyperparameters Found:** {
  "lr": 0.002278442088992486,
  "batch_size": 96,
  "drop_out": 0.1902810079806919,
  "unit": 96,
  "layber_number": 2
}
- **Optimization Time:** 38.62 seconds
- **Training Time:** 3.19 seconds

---

## 1. Experimental Setup

### 1.1 Dataset Information

| Parameter | Value |
|-----------|-------|
| Data File | `http://47.99.180.80/file/uploads/lgm_4.xlsx` |
| Data Shape | [192, 12] |
| Number of Features | 11 |
| Number of Targets | 1 |
| Total Samples | 192 |

**Feature Names:** BPA, HBPA, Epoxy, Ph, CYH, DPh, DCYH, Amine, Isobutyl, Isooctyl, PCL
**Target Names:** PSD

### 1.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Hyperparameter Optimization Algorithm | TPE (TPE: Tree-structured Parzen Estimator, GP: Gaussian Process) |
| Number of Trials | 10 |
| Cross-Validation Folds | 5 |
| Training Epochs per Fold | 500 |
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

The training data was loaded from `http://47.99.180.80/file/uploads/lgm_4.xlsx` and underwent comprehensive preprocessing to ensure model compatibility and optimal performance.

**Input Features (11 columns):**
`BPA`, `HBPA`, `Epoxy`, `Ph`, `CYH`, `DPh`, `DCYH`, `Amine`, `Isobutyl`, `Isooctyl`, `PCL`

**Target Variables (1 column):**
`PSD`

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

**Method**: K-Fold Cross-Validation with 5 folds
- **Randomization**: `random_state=42` for reproducible splits
- **Shuffle**: Data is shuffled before splitting to ensure representative folds
- **Stratification**: Not applicable for regression tasks

#### 2.4.2 Fold Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Number of Folds** | 5 | Each fold uses ~80.0% for training, ~20.0% for validation |
| **Training Samples per Fold** | ~153 | Approximate number of training samples |
| **Validation Samples per Fold** | ~38 | Approximate number of validation samples |
| **Random Seed** | 42 | Ensures reproducible train/validation splits |

#### 2.4.3 Data Leakage Prevention

**Key Safeguards:**
- **Separate Scaling**: Each CV fold fits scaler only on training data
- **Independent Validation**: Validation sets never used for preprocessing parameter estimation


### 2.5 Data Transformation for Neural Networks

#### 2.5.1 Tensor Conversion

**PyTorch Integration:**
- **Data Type**: Convert to `torch.FloatTensor` for GPU compatibility
- **Batch Processing**: Data organized into batches of size 96
- **Memory Management**: Efficient tensor operations for large datasets
- **Device Placement**: Automatic CPU/GPU tensor placement based on hardware

#### 2.5.2 Batch Processing Configuration

| Parameter | Value | Impact |
|-----------|-------|--------|
| **Batch Size** | 96 | Optimized for memory usage and gradient stability |
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
  "lr": 0.002278442088992486,
  "batch_size": 96,
  "drop_out": 0.1902810079806919,
  "unit": 96,
  "layber_number": 2
}
```

**Best Cross-Validation Score:** 0.370673

### 3.3 Hyperparameter Optimization Trials

The complete hyperparameter optimization results are saved in:
- **File:** `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/hyperparameter_optimization_trials.csv`
- **Total Trials:** 10
 
## 4. Final Model Training

### 4.1 Cross-Validation Training

The final model was trained using 5-fold cross-validation with the best hyperparameters. Training history and validation metrics were recorded for each epoch.

### 4.2 Training Results

| Metric | Value |
|--------|-------|
| Average CV MAE | 0.347791 |
| CV MSE | 0.292149 |
| CV R² Score | 0.707851 |
| CV Standard Deviation | 0.048315 |
| Best Fold MAE | 0.263085 |
| Worst Fold MAE | 0.398752 |

#### Fold-wise Results

| Fold | MAE |
|------|-----|
| 1 | 0.327761 |
| 2 | 0.263085 |
| 3 | 0.379802 |
| 4 | 0.369554 |
| 5 | 0.398752 |


### 4.3 Training Curves

Training progress and validation metrics are visualized in the following plots:

#### 4.3.1 Training and Validation Loss
![Training Curves](trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/training_curves.png)

**File:** `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/training_curves.png`

This figure shows:
- **Subplot 1:** Average training and validation loss across all folds
- **Subplot 2:** Average validation MAE progression
- **Subplot 3:** Average validation R² progression  
- **Subplot 4:** Individual fold training curves

#### 4.3.2 Training History Data
**File:** `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/training_history.csv`

The complete training history including epoch-by-epoch metrics for all folds is saved in CSV format for further analysis.

### 4.4 Cross-Validation Predictions

#### 4.4.1 Prediction Scatter Plot
![CV Predictions](trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/cv_predictions_scatter.png)

**File:** `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/cv_predictions_scatter.png`

This scatter plot shows actual vs predicted values from cross-validation, including:
- Perfect prediction line (red dashed)
- Performance statistics (MAE, R²)
- Sample distribution

#### 4.4.2 Prediction Statistics

| Metric | Value |
|--------|-------|
| Cross-Validation MAE | 0.347245 |
| Cross-Validation R² | 0.707851 |
| Number of Predictions | 192 |
| Prediction Range | [-1.053, 2.894] |
| Actual Range | [-1.030, 2.951] |

**Prediction Data Files:**
- `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/cv_predictions.csv` - Basic predictions data


---

## 5. Model Architecture and Implementation

### 5.1 Neural Network Architecture

The final model uses a Multi-Layer Perceptron (MLP) architecture with the following specifications:

| Component | Configuration |
|-----------|---------------|
| Input Layer | 11 features |
| Hidden Layers | 2 layers |
| Hidden Units per Layer | 96 |
| Activation Function | ReLU |
| Dropout Rate | 0.1902810079806919 |
| Output Layer | 1 target(s) |
| Loss Function | L1Loss (Mean Absolute Error) |

### 5.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2.28e-03 |
| Batch Size | 96 |
| Loss Function | L1Loss (Mean Absolute Error) |
| Device | CPU |

---

## 6. Conclusions and Future Work

### 6.1 Key Findings

1. **Model Performance**: The optimized neural network achieved a cross-validation MAE of 0.347791
2. **Hyperparameter Sensitivity**: The optimization process explored 10 different configurations
3. **Training Stability**: Cross-validation results show consistent performance across 5 folds

### 6.2 Reproducibility

This experiment is fully reproducible using the following artifacts:
- **Model Weights**: Saved in `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/`
- **Hyperparameters**: `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/model_metadata.json`
- **Training Data**: `http://47.99.180.80/file/uploads/lgm_4.xlsx`
- **Training History**: `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/training_history.csv`
- **Optimization Trials**: `trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/hyperparameter_optimization_trials.csv`

### 6.3 Technical Implementation

- **Framework**: PyTorch for neural network implementation
- **Optimization**: Optuna with TPE sampler for hyperparameter search
- **Cross-Validation**: 5-fold stratified cross-validation
- **Loss Function**: L1Loss (Mean Absolute Error)
- **Data Processing**: StandardScaler for feature normalization

---

## Appendix

### A.1 System Information

- **Generation Time**: 2025-09-22 17:43:43
- **Model ID**: `9feed268-4561-4fec-bb36-da6c2cac9da5`
- **Training System**: Enhanced Multi-Target Neural Network Training System
- **Report Version**: 2.0 (with MAE Loss Function Support)

### A.2 File Structure

```
trained_models/10004/9feed268-4561-4fec-bb36-da6c2cac9da5/
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
