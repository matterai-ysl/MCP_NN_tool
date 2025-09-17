"""Data utilities for the MCP NN Tool.

This module contains data loading, preprocessing, and transformation utilities.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from typing import Optional, Tuple, Union, List, Dict, Any
import os
from urllib.parse import urlparse
from pathlib import Path


def _detect_url_format(url: str) -> str:
    """Detect file format from URL path.

    Args:
        url: URL to analyze

    Returns:
        File format ('csv', 'excel', 'tsv')
    """
    parsed = urlparse(url)
    path = parsed.path.lower()

    if path.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif path.endswith('.tsv'):
        return 'tsv'
    elif path.endswith(('.csv', '.txt')):
        return 'csv'
    else:
        return 'csv'  # Default to CSV


def _is_url(path: str) -> bool:
    """Check if path is a URL.

    Args:
        path: Path or URL to check

    Returns:
        True if path is a URL
    """
    return path.startswith(('http://', 'https://'))


async def read_data_file(file_path: str) -> pd.DataFrame:
    """Read data from various file formats including URLs.

    Args:
        file_path: Path to the data file or URL (supports .csv, .xls, .xlsx, URLs)

    Returns:
        DataFrame containing the loaded data

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If local file does not exist
    """
    is_url = _is_url(file_path)

    if is_url:
        # Handle URL data source
        file_format = _detect_url_format(file_path)

        if file_format == 'excel':
            data = pd.read_excel(file_path)
        elif file_format == 'tsv':
            data = pd.read_csv(file_path, sep='\t')
        else:  # csv or default
            data = pd.read_csv(file_path)
    else:
        # Handle local file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith(".csv"):
            data = pd.read_csv(file_path)
        elif file_path.endswith((".xls", ".xlsx")):
            data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    return data


async def preprocess_data(
    train_path: str, 
    test_path: Optional[str] = None,
    target_columns: int = 1,
    save_transformed: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], preprocessing.StandardScaler, Optional[preprocessing.StandardScaler]]:
    """Preprocess data with standardization supporting multiple targets.
    
    Args:
        train_path: Path to training data file
        test_path: Optional path to test data file
        target_columns: Number of target columns at the end of the dataset (default: 1)
        save_transformed: Whether to save transformed data to files
        
    Returns:
        Tuple containing:
            - Transformed training data
            - Transformed test data (if test_path provided)
            - Scaler fitted on full training data
            - Feature scaler fitted on training features only (for prediction)
    """
    # Load training data
    train_data = await read_data_file(train_path)
    
    # Validate target columns
    if target_columns < 1 or target_columns >= train_data.shape[1]:
        raise ValueError(f"Invalid number of target columns: {target_columns}. "
                        f"Must be between 1 and {train_data.shape[1]-1}")
    
    # Split features and targets
    feature_end_idx = train_data.shape[1] - target_columns
    features = train_data.iloc[:, :feature_end_idx]
    targets = train_data.iloc[:, feature_end_idx:]
    
    # Fit scaler on full training data (for inverse transform)
    full_scaler = preprocessing.StandardScaler().fit(train_data)
    train_transformed = full_scaler.transform(train_data)
    train_df = pd.DataFrame(train_transformed, columns=train_data.columns)
    
    # Fit feature scaler on training features only (for prediction)
    feature_scaler = preprocessing.StandardScaler().fit(features)
    
    test_df = None
    if test_path:
        test_data = await read_data_file(test_path)
        
        # Validate test data has correct number of features
        if test_data.shape[1] != feature_end_idx:
            raise ValueError(f"Test data has {test_data.shape[1]} columns, "
                           f"but expected {feature_end_idx} features")
        
        test_transformed = feature_scaler.transform(test_data)
        test_df = pd.DataFrame(test_transformed, columns=test_data.columns)
        
        if save_transformed:
            if _is_url(test_path):
                # Extract filename from URL and create local save path
                parsed_url = urlparse(test_path)
                filename = Path(parsed_url.path).name
                test_save_path = filename.rsplit('.', 1)[0] + "_transformed.csv"
            else:
                # Local file path
                test_save_path = test_path.rsplit('.', 1)[0] + "_transformed.csv"
            test_df.to_csv(test_save_path, index=False)

    if save_transformed:
        if _is_url(train_path):
            # Extract filename from URL and create local save path
            parsed_url = urlparse(train_path)
            filename = Path(parsed_url.path).name
            train_save_path = filename.rsplit('.', 1)[0] + "_transformed.csv"
        else:
            # Local file path
            train_save_path = train_path.rsplit('.', 1)[0] + "_transformed.csv"
        train_df.to_csv(train_save_path, index=False)
    
    return train_df, test_df, full_scaler, feature_scaler


async def preprocess_prediction_input(
    input_data: Union[List[float], np.ndarray, pd.DataFrame],
    feature_scaler: preprocessing.StandardScaler,
    feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """Preprocess input data for prediction.
    
    Args:
        input_data: Input data as list, array, or DataFrame
        feature_scaler: Fitted scaler for features
        feature_names: Optional list of feature names for validation
        
    Returns:
        Preprocessed data as numpy array
        
    Raises:
        ValueError: If input data format is invalid
    """
    if isinstance(input_data, list):
        input_array = np.array(input_data).reshape(1, -1)
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            input_array = input_data.reshape(1, -1)
        else:
            input_array = input_data
    elif isinstance(input_data, pd.DataFrame):
        if feature_names:
            # Ensure correct column order
            input_array = input_data[feature_names].values
        else:
            input_array = input_data.values
    else:
        raise ValueError(f"Unsupported input data type: {type(input_data)}")
    
    # Apply feature scaling
    scaled_data = feature_scaler.transform(input_array)
    return scaled_data


async def inverse_transform_predictions(
    predictions: np.ndarray,
    features: np.ndarray,
    full_scaler: preprocessing.StandardScaler,
    target_names: List[str]
) -> pd.DataFrame:
    """Apply inverse transformation to predictions and features.
    
    Args:
        predictions: Model predictions (can be multi-dimensional for multiple targets)
        features: Original feature values (scaled)
        full_scaler: Scaler fitted on full training data
        target_names: Names of the target columns
        
    Returns:
        DataFrame with inverse-transformed features and predictions
    """
    # Ensure predictions are 2D
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Combine features and predictions
    combined = np.column_stack([features, predictions])
    
    # Apply inverse transformation
    inverse_transformed = full_scaler.inverse_transform(combined)
    
    # Create DataFrame with proper column names
    feature_names = [f"feature_{i}" for i in range(features.shape[1])]
    prediction_names = [f"prediction_{name}" for name in target_names]
    columns = feature_names + prediction_names
    
    result_df = pd.DataFrame(inverse_transformed, columns=columns)
    return result_df


def validate_data_format(data: pd.DataFrame, target_columns: int = 1, min_features: int = 1) -> bool:
    """Validate data format for training/prediction.
    
    Args:
        data: DataFrame to validate
        target_columns: Number of target columns expected
        min_features: Minimum number of features required
        
    Returns:
        True if data format is valid
        
    Raises:
        ValueError: If data format is invalid
    """
    if data.empty:
        raise ValueError("Data is empty")
    
    total_required_columns = min_features + target_columns
    if data.shape[1] < total_required_columns:
        raise ValueError(f"Data must have at least {total_required_columns} columns "
                        f"({min_features} features + {target_columns} targets)")
    
    if data.isnull().any().any():
        raise ValueError("Data contains null values")
    
    return True


def extract_feature_target_info(data: pd.DataFrame, target_columns: int = 1) -> Tuple[List[str], List[str], int, int]:
    """Extract feature and target information from data.
    
    Args:
        data: DataFrame containing features and targets
        target_columns: Number of target columns at the end
        
    Returns:
        Tuple containing:
            - List of feature column names
            - List of target column names  
            - Number of features
            - Number of targets
    """
    feature_end_idx = data.shape[1] - target_columns
    
    feature_names = data.columns[:feature_end_idx].tolist()
    target_names = data.columns[feature_end_idx:].tolist()
    
    return feature_names, target_names, len(feature_names), len(target_names)


def encode_classification_labels(targets: pd.Series) -> Tuple[np.ndarray, Dict[str, Any], int]:
    """Automatically encode classification labels and detect task type.
    
    Handles both string and numeric labels, automatically detects number of classes.
    
    Args:
        targets: Target column (pandas Series)
        
    Returns:
        Tuple of (encoded_targets, label_info, num_classes)
        where label_info contains:
        - label_encoder: sklearn LabelEncoder object
        - classes: list of original class names
        - class_to_idx: mapping from class name to index
        - idx_to_class: mapping from index to class name
        - is_string_labels: whether original labels were strings
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Check if targets contain any non-numeric values
    is_string_labels = not pd.api.types.is_numeric_dtype(targets)
    
    if is_string_labels:
        # Use LabelEncoder for string labels
        label_encoder = LabelEncoder()
        encoded_targets = label_encoder.fit_transform(targets)
        classes = label_encoder.classes_.tolist()
        
        print(f"Detected string labels: {classes}")
        print(f"Encoded as: {list(range(len(classes)))}")
        
    else:
        # For numeric labels, ensure they start from 0 and are consecutive
        unique_values = sorted(targets.unique())
        
        # Check if values are already 0-indexed consecutive integers
        expected_values = list(range(len(unique_values)))
        
        if unique_values == expected_values:
            # Already properly encoded
            label_encoder = None
            encoded_targets = targets.values.astype(int)
            classes = [str(val) for val in unique_values]
        else:
            # Need to re-encode to ensure 0-indexing
            label_encoder = LabelEncoder()
            encoded_targets = label_encoder.fit_transform(targets.astype(str))
            classes = [str(val) for val in unique_values]
            print(f"Re-encoded numeric labels from {unique_values} to {list(range(len(unique_values)))}")
    
    num_classes = len(classes)
    
    # Create mappings
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
    
    label_info = {
        'label_encoder': label_encoder,
        'classes': classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'is_string_labels': is_string_labels,
        'num_classes': num_classes
    }
    
    print(f"Classification task detected: {num_classes} classes")
    print(f"Class mapping: {class_to_idx}")
    
    return encoded_targets, label_info, num_classes


async def preprocess_classification_data(
    file_path: str,
    target_column: int = -1
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """Preprocess data specifically for classification tasks.
    
    Args:
        file_path: Path to the data file
        target_column: Index of target column (-1 for last column)
        
    Returns:
        Tuple of (processed_dataframe, feature_scaler, label_info)
    """
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    data_df = await read_data_file(file_path)
    
    print(f"Loaded classification data: {data_df.shape}")
    print(f"Columns: {list(data_df.columns)}")
    
    # Validate minimum columns
    if data_df.shape[1] < 2:
        raise ValueError("Classification data must have at least 2 columns (features + target)")
    
    # Extract features and targets
    if target_column == -1:
        features = data_df.iloc[:, :-1]
        targets = data_df.iloc[:, -1]
        target_col_name = data_df.columns[-1]
    else:
        features = data_df.drop(data_df.columns[target_column], axis=1)
        targets = data_df.iloc[:, target_column]
        target_col_name = data_df.columns[target_column]
    
    print(f"Target column: {target_col_name}")
    print(f"Original target values (first 10): {targets.head(10).tolist()}")
    
    # Encode labels automatically
    encoded_targets, label_info, num_classes = encode_classification_labels(targets)
    
    # Scale features
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    # Create processed dataframe with scaled features and encoded targets
    feature_columns = features.columns.tolist()
    processed_df = pd.DataFrame(features_scaled, columns=feature_columns)
    processed_df[target_col_name] = encoded_targets
    
    # Add metadata to label_info
    label_info.update({
        'target_column_name': target_col_name,
        'feature_names': feature_columns,
        'feature_number': len(feature_columns)
    })
    
    print(f"Preprocessing completed:")
    print(f"  - Features: {len(feature_columns)} columns scaled")
    print(f"  - Target: {num_classes} classes encoded")
    print(f"  - Final data shape: {processed_df.shape}")
    
    return processed_df, feature_scaler, label_info 