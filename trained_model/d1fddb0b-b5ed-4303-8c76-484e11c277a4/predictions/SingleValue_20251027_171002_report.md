# Prediction Experiment Report

**Generated on:** 2025-10-27 17:10:02  
**Experiment Name:** SingleValue_20251027_171002  
**Model ID:** `d1fddb0b-b5ed-4303-8c76-484e11c277a4`  
**Output Directory:** `trained_model/d1fddb0b-b5ed-4303-8c76-484e11c277a4/predictions`

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
| **Model ID** | `d1fddb0b-b5ed-4303-8c76-484e11c277a4` |
| **Training Framework** | PyTorch |
| **Prediction Method** | Ensemble averaging |

---

## 2. Prediction Results

### 2.1 Prediction Statistics


#### Single Target Prediction Statistics

**Target: relative_density**

| Statistic | Value |
|-----------|-------|
| Mean Prediction | 97.326871 |
| Standard Deviation | 0.000000 |
| Minimum Prediction | 97.326871 |
| Maximum Prediction | 97.326871 |
| Prediction Range | 0.000000 |

**Ensemble Uncertainty:**
- **Model Agreement (Std)**: 0.019231
- **95% Confidence Interval**: [0.996730, 1.072116]


---

## 3. Generated Files

| File | Description |
|------|-------------|
| `SingleValue_20251027_171002_scaled.csv` | Preprocessed features + scaled predictions |
| `SingleValue_20251027_171002_original.csv` | Original scale features + predictions |
| `SingleValue_20251027_171002_detailed.csv` | Comprehensive results with statistics |
| `SingleValue_20251027_171002_report.md` | This report |

---

*Report generated on 2025-10-27 17:10:02*
