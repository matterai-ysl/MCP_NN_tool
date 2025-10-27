# Prediction Experiment Report

**Generated on:** 2025-10-27 21:25:29  
**Experiment Name:** SingleValue_20251027_212529  
**Model ID:** `d3d7d308-c8e4-4efd-8ce8-4ca4123c5582`  
**Output Directory:** `trained_models/10004/d3d7d308-c8e4-4efd-8ce8-4ca4123c5582/predictions`

## Executive Summary

This report documents a comprehensive neural network prediction experiment conducted using a pre-trained ensemble model. The experiment involved preprocessing input data, making predictions with multiple cross-validation models, and providing detailed statistical analysis of the prediction results.

### Key Results
- **Number of Predictions:** 1
- **Feature Count:** 4
- **Target Count:** 1
- **Ensemble Models Used:** 5 (from cross-validation)

---

## 1. Experiment Setup

### 1.1 Input Data Information

| Parameter | Value |
|-----------|-------|
| Input File | `Feature Values (Manual Input - Single)` |
| Number of Samples | 1 |
| Number of Features | 4 |
| Number of Targets | 1 |
| Data Type | Numerical (floating-point) |

### 1.2 Feature Information

**Input Features (4 columns):**
`layer_thickness`, `hatch_distance`, `laser_power`, `laser_velocity`

**Target Variables (1 column):**
`relative_density`

### 1.3 Model Information

| Component | Details |
|-----------|---------|
| **Model Type** | Multi-Layer Perceptron (MLP) Ensemble |
| **Ensemble Size** | 5 models (from cross-validation) |
| **Model ID** | `d3d7d308-c8e4-4efd-8ce8-4ca4123c5582` |
| **Training Framework** | PyTorch |
| **Prediction Method** | Ensemble averaging |

---

## 2. Prediction Results

### 2.1 Prediction Statistics


#### Single Target Prediction Statistics

**Target: relative_density**

| Statistic | Value |
|-----------|-------|
| Mean Prediction | 96.779088 |
| Standard Deviation | 0.000000 |
| Minimum Prediction | 96.779088 |
| Maximum Prediction | 96.779088 |
| Prediction Range | 0.000000 |

**Ensemble Uncertainty:**
- **Model Agreement (Std)**: 0.022741
- **95% Confidence Interval**: [0.871323, 0.960467]


---

## 3. Generated Files

| File | Description |
|------|-------------|
| `SingleValue_20251027_212529_scaled.csv` | Preprocessed features + scaled predictions |
| `SingleValue_20251027_212529_original.csv` | Original scale features + predictions |
| `SingleValue_20251027_212529_detailed.csv` | Comprehensive results with statistics |
| `SingleValue_20251027_212529_report.md` | This report |

---

*Report generated on 2025-10-27 21:25:29*
