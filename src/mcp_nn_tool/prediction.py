"""Prediction utilities for the MCP NN Tool.

This module contains functions for making predictions using trained models,
supporting both single and multi-target predictions.
"""

import numpy as np
import pandas as pd
import torch
import os
import json
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from .neural_network import MLPregression, MLPClassification
from .data_utils import (
    read_data_file, 
    preprocess_prediction_input, 
    inverse_transform_predictions
)


async def predict_with_ensemble(
    model_components: Dict[str, Any],
    preprocessed_features: np.ndarray,
    return_individual_predictions: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """Make predictions using ensemble of cross-validation models.
    
    Args:
        model_components: Dictionary containing model states and metadata
        preprocessed_features: Preprocessed feature data
        return_individual_predictions: Whether to return individual model predictions
        
    Returns:
        Ensemble predictions (optionally with individual predictions)
    """
    metadata = model_components['metadata']
    model_states = model_components['model_states']
    best_params = model_components['best_params']
    
    feature_number = metadata['feature_number']
    target_number = metadata['target_number']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(preprocessed_features).to(device)
    
    predictions_list = []
    
    # Make predictions with each model from cross-validation
    for model_state in model_states:
        model = MLPregression(best_params, feature_number, target_number).to(device)
        model.load_state_dict(model_state)
        model.eval()
        
        with torch.no_grad():
            pred = model(features_tensor).cpu().numpy()
            
            # Ensure prediction has correct shape
            if target_number == 1 and pred.ndim > 1:
                pred = pred.squeeze()
            elif target_number > 1 and pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            
            predictions_list.append(pred)
    
    # Average predictions from all models (ensemble)
    ensemble_pred = np.mean(predictions_list, axis=0)
    
    if return_individual_predictions:
        return ensemble_pred, predictions_list
    else:
        return ensemble_pred


async def predict_from_file(
    model_components: Dict[str, Any],
    prediction_file: str,
    generate_experiment_report: bool = False
) -> Tuple[pd.DataFrame, np.ndarray, Optional[str], str]:
    """Make predictions on data from a file with optional experiment reporting.
    
    Args:
        model_components: Dictionary containing model and preprocessing components
        prediction_file: Path to file containing prediction data
        generate_experiment_report: Whether to generate detailed experiment report (default: False)
        
    Returns:
        Tuple of (results_dataframe, raw_predictions, experiment_report_path, basic_csv_path)
        - experiment_report_path is None if generate_experiment_report=False
    """
    # Load prediction data
    pred_data = await read_data_file(prediction_file)
    
    # Preprocess data
    metadata = model_components['metadata']
    feature_scaler = model_components['feature_scaler']
    full_scaler = model_components['full_scaler']
    
    # Validate feature count
    expected_features = metadata['feature_number']
    if pred_data.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {pred_data.shape[1]}")
    
    preprocessed_data = await preprocess_prediction_input(
        pred_data, feature_scaler, metadata['feature_names']
    )
    
    # Make predictions
    predictions = await predict_with_ensemble(model_components, preprocessed_data)
    
    # Apply inverse transformation
    result_df = await inverse_transform_predictions(
        predictions, preprocessed_data, full_scaler, metadata['target_names']
    )
    
    # Always save basic prediction results to model folder (even without experiment report)
    model_id = metadata['model_id']
    model_base_path = os.path.join("trained_model", model_id)
    predictions_dir = os.path.join(model_base_path, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(os.path.basename(prediction_file))[0]
    
    # Get original scale data for basic saving
    original_scale_df = await inverse_transform_predictions(
        predictions, preprocessed_data, full_scaler, metadata['target_names']
    )
    original_features = original_scale_df.iloc[:, :expected_features].values
    original_predictions = original_scale_df.iloc[:, expected_features:].values
    
    # Save basic original scale CSV
    basic_results_data = {}
    for i, feature_name in enumerate(metadata['feature_names']):
        basic_results_data[f'{feature_name}'] = original_features[:, i]
    
    if metadata['target_number'] == 1:
        target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
        if original_predictions.ndim > 1:
            basic_results_data[f'{target_name}_prediction'] = original_predictions.flatten()
        else:
            basic_results_data[f'{target_name}_prediction'] = original_predictions
    else:
        for i, target_name in enumerate(metadata['target_names']):
            basic_results_data[f'{target_name}_prediction'] = original_predictions[:, i]
    
    basic_results_df = pd.DataFrame(basic_results_data)
    basic_csv_path = os.path.join(predictions_dir, f'{base_name}_{timestamp}.csv')
    basic_results_df.to_csv(basic_csv_path, index=False)
    
    # Update result_df to include original scale data
    result_df = basic_results_df
    
    # Generate experiment report if requested
    experiment_report_path = None
    if generate_experiment_report:
        # Get model folder path from model_id
        model_id = metadata['model_id']
        model_base_path = os.path.join("trained_model", model_id)
        predictions_dir = os.path.join(model_base_path, "predictions")
        
        # Create predictions directory if it doesn't exist
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Generate timestamp for unique file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(prediction_file))[0]
        experiment_name = f"Prediction_{base_name}_{timestamp}"
        
        # Get individual predictions for statistics
        ensemble_predictions, individual_predictions = await predict_with_ensemble(
            model_components, preprocessed_data, return_individual_predictions=True
        )
        
        # Get prediction statistics
        prediction_stats = await get_prediction_statistics(model_components, preprocessed_data)
        
        # Apply inverse transformation to get original scale features and predictions
        original_scale_df = await inverse_transform_predictions(
            ensemble_predictions, preprocessed_data, full_scaler, metadata['target_names']
        )
        
        # Extract original scale features and predictions from DataFrame
        target_number = metadata['target_number']
        original_features = original_scale_df.iloc[:, :expected_features].values
        original_predictions = original_scale_df.iloc[:, expected_features:].values
        
        # Create comprehensive results DataFrame with original scale data
        results_data = {}
        
        # Add original scale features
        for i, feature_name in enumerate(metadata['feature_names']):
            results_data[f'{feature_name}_original'] = original_features[:, i]
        
        # Add scaled features for reference
        for i, feature_name in enumerate(metadata['feature_names']):
            results_data[f'{feature_name}_scaled'] = preprocessed_data[:, i]
        
        # Add original scale predictions
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            results_data[f'{target_name}_prediction_original'] = original_predictions.flatten()
            results_data[f'{target_name}_prediction_scaled'] = ensemble_predictions
        else:
            for i, target_name in enumerate(metadata['target_names']):
                results_data[f'{target_name}_prediction_original'] = original_predictions[:, i]
                results_data[f'{target_name}_prediction_scaled'] = ensemble_predictions
        
        # Add prediction statistics
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            results_data[f'{target_name}_prediction_std'] = [prediction_stats['prediction_std']] * len(original_predictions)
            results_data[f'{target_name}_prediction_ci_lower'] = [prediction_stats['confidence_interval_95'][0]] * len(original_predictions)
            results_data[f'{target_name}_prediction_ci_upper'] = [prediction_stats['confidence_interval_95'][1]] * len(original_predictions)
        else:
            for i, target_name in enumerate(metadata['target_names']):
                results_data[f'{target_name}_prediction_std'] = [prediction_stats['prediction_std'][i]] * len(original_predictions)
                results_data[f'{target_name}_prediction_ci_lower'] = [prediction_stats['confidence_intervals_95'][i][0]] * len(original_predictions)
                results_data[f'{target_name}_prediction_ci_upper'] = [prediction_stats['confidence_intervals_95'][i][1]] * len(original_predictions)
        
        detailed_results_df = pd.DataFrame(results_data)
        
        # Save CSV files for experiment mode
        # 1. Scaled data CSV (preprocessed features + scaled predictions)
        scaled_results_data = {}
        for i, feature_name in enumerate(metadata['feature_names']):
            scaled_results_data[f'{feature_name}'] = preprocessed_data[:, i]
        
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            scaled_results_data[f'{target_name}_prediction'] = ensemble_predictions.flatten() if ensemble_predictions.ndim > 1 else ensemble_predictions
        else:
            for i, target_name in enumerate(metadata['target_names']):
                scaled_results_data[f'{target_name}_prediction'] = ensemble_predictions[:, i]
        
        scaled_results_df = pd.DataFrame(scaled_results_data)
        scaled_csv_path = os.path.join(predictions_dir, f'{experiment_name}_scaled.csv')
        scaled_results_df.to_csv(scaled_csv_path, index=False)
        
        # 2. Original scale data CSV (original features + original predictions)
        original_results_data = {}
        for i, feature_name in enumerate(metadata['feature_names']):
            original_results_data[f'{feature_name}'] = original_features[:, i]
        
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            if original_predictions.ndim > 1:
                original_results_data[f'{target_name}_prediction'] = original_predictions.flatten()
            else:
                original_results_data[f'{target_name}_prediction'] = original_predictions
        else:
            for i, target_name in enumerate(metadata['target_names']):
                original_results_data[f'{target_name}_prediction'] = original_predictions[:, i]
        
        original_results_df = pd.DataFrame(original_results_data)
        original_csv_path = os.path.join(predictions_dir, f'{experiment_name}_original.csv')
        original_results_df.to_csv(original_csv_path, index=False)
        
        # 3. Detailed CSV with all information (for compatibility)
        detailed_csv_path = os.path.join(predictions_dir, f'{experiment_name}_detailed.csv')
        detailed_results_df.to_csv(detailed_csv_path, index=False)
        
        # Save experiment metadata
        experiment_metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model_id': metadata['model_id'],
            'prediction_file': prediction_file,
            'num_samples': len(pred_data),
            'feature_number': expected_features,
            'target_number': metadata['target_number'],
            'feature_names': metadata['feature_names'],
            'target_names': metadata['target_names'],
            'prediction_statistics': prediction_stats,
            'files_generated': {
                'scaled_data': f'{experiment_name}_scaled.csv',
                'original_data': f'{experiment_name}_original.csv',
                'detailed_results': f'{experiment_name}_detailed.csv',
                'experiment_report': f'{experiment_name}_report.md'
            }
        }
        
        metadata_path = os.path.join(predictions_dir, f'{experiment_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        # Generate comprehensive experiment report
        report_path = os.path.join(predictions_dir, f'{experiment_name}_report.md')
        experiment_report_path = await generate_prediction_experiment_report(
            predictions_dir, experiment_metadata, detailed_results_df, model_components, prediction_stats
        )
        
        # Update the main result_df to be the detailed one with original scale data
        result_df = detailed_results_df
    
    return result_df, predictions, experiment_report_path, basic_csv_path # type: ignore


async def predict_from_values(
    model_components: Dict[str, Any],
    feature_values: Union[List[float], List[List[float]]],
    generate_experiment_report: bool = False
) -> Tuple[Dict[str, Any], Optional[str], str]:
    """Make predictions from feature values with optional experiment reporting.
    
    Supports both single prediction and batch prediction:
    - Single: feature_values = [1.0, 2.0, 3.0, 4.0]
    - Batch: feature_values = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    
    Args:
        model_components: Dictionary containing model and preprocessing components
        feature_values: List of feature values (1D) or list of feature value lists (2D) for batch prediction
        generate_experiment_report: Whether to generate detailed experiment report (default: False)
        
    Returns:
        Tuple of (prediction_results_dict, experiment_report_path, basic_csv_path)
        - experiment_report_path is None if generate_experiment_report=False
    """
    # Validate input and determine if it's single or batch prediction
    metadata = model_components['metadata']
    expected_features = metadata['feature_number']
    
    # Check if it's batch prediction (2D list) or single prediction (1D list)
    is_batch = isinstance(feature_values[0], (list, tuple)) if feature_values else False
    
    if is_batch:
        # Batch prediction - validate all samples
        num_samples = len(feature_values)
        for i, sample in enumerate(feature_values):
            if len(sample) != expected_features:
                raise ValueError(f"Sample {i}: Expected {expected_features} features, got {len(sample)}")
        
        # Convert to numpy array for processing
        feature_array = np.array(feature_values)
        filename_prefix = f"BatchValues_{num_samples}samples"
    else:
        # Single prediction - validate single sample
        if len(feature_values) != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {len(feature_values)}")
        
        # Convert to 2D array for consistent processing
        feature_array = np.array([feature_values])
        num_samples = 1
        filename_prefix = "SingleValue"
    
    # Convert to numpy array and preprocess
    feature_scaler = model_components['feature_scaler']
    full_scaler = model_components['full_scaler']
    
    # Convert to DataFrame for preprocessing
    feature_names = metadata['feature_names']
    input_df = pd.DataFrame(feature_array, columns=feature_names)
    
    # Preprocess
    preprocessed_data = await preprocess_prediction_input(
        input_df, feature_scaler, feature_names
    )
    
    # Make prediction
    predictions = await predict_with_ensemble(model_components, preprocessed_data)
    print(predictions)
    
    # Apply inverse transformation for final results
    result_df = await inverse_transform_predictions(
        predictions, preprocessed_data, full_scaler, metadata['target_names']
    )
    
    # Format results
    target_names = metadata['target_names']
    
    if is_batch:
        # Batch results - return list of predictions
        batch_results = []
        for i in range(num_samples):
            sample_features = feature_array[i]
            sample_preprocessed = preprocessed_data[i]
            
            if metadata['target_number'] == 1:
                # Single target
                prediction_value = float(predictions[i])
                sample_result = {
                    'input_features': dict(zip(feature_names, sample_features)),
                    'prediction': {target_names[0]: prediction_value},
                    'processed_input': dict(zip(feature_names, sample_preprocessed.tolist()))
                }
            else:
                # Multiple targets
                prediction_values = predictions[i].tolist()
                sample_result = {
                    'input_features': dict(zip(feature_names, sample_features)),
                    'predictions': dict(zip(target_names, prediction_values)),
                    'processed_input': dict(zip(feature_names, sample_preprocessed.tolist()))
                }
            batch_results.append(sample_result)
        
        results = {
            'batch_results': batch_results,
            'num_samples': num_samples,
            'prediction_type': 'batch'
        }
    else:
        # Single result - return single prediction
        if metadata['target_number'] == 1:
            # Single target
            prediction_value = float(predictions[0])
            results = {
                'input_features': dict(zip(feature_names, feature_values)),
                'prediction': {target_names[0]: prediction_value},
                'processed_input': dict(zip(feature_names, preprocessed_data[0].tolist())),
                'prediction_type': 'single'
            }
        else:
            # Multiple targets
            prediction_values = predictions[0].tolist()
            results = {
                'input_features': dict(zip(feature_names, feature_values)),
                'predictions': dict(zip(target_names, prediction_values)),
                'processed_input': dict(zip(feature_names, preprocessed_data[0].tolist())),
                'prediction_type': 'single'
            }
    
    # Always save basic prediction results to model folder (even without experiment report)
    model_id = metadata['model_id']
    model_base_path = os.path.join("trained_model", model_id)
    predictions_dir = os.path.join(model_base_path, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    print(predictions_dir)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get original scale data for basic saving
    original_scale_df = await inverse_transform_predictions(
        predictions, preprocessed_data, full_scaler, metadata['target_names']
    )
    original_features = original_scale_df.iloc[:, :expected_features].values
    original_predictions = original_scale_df.iloc[:, expected_features:].values
    
    # Save basic original scale CSV
    basic_results_data = {}
    for i, feature_name in enumerate(metadata['feature_names']):
        basic_results_data[f'{feature_name}'] = original_features[:, i]
    
    if metadata['target_number'] == 1:
        target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
        basic_results_data[f'{target_name}_prediction'] = original_predictions.flatten() if original_predictions.ndim > 1 else original_predictions
    else:
        for i, target_name in enumerate(metadata['target_names']):
            basic_results_data[f'{target_name}_prediction'] = original_predictions[:, i]
    
    basic_results_df = pd.DataFrame(basic_results_data)
    basic_csv_path = os.path.join(predictions_dir, f'{filename_prefix}_{timestamp}.csv')
    basic_results_df.to_csv(basic_csv_path, index=False)
    
    # Add basic file info to results
    results['basic_prediction_file'] = basic_csv_path
    results['num_samples'] = num_samples
    
    # Generate experiment report if requested
    experiment_report_path = None
    if generate_experiment_report:
        # Generate timestamp for unique file naming
        experiment_name = f"{filename_prefix}_{timestamp}"
        
        # Get individual predictions for statistics
        ensemble_predictions, individual_predictions = await predict_with_ensemble(
            model_components, preprocessed_data, return_individual_predictions=True
        )
        
        # Get prediction statistics
        prediction_stats = await get_prediction_statistics(model_components, preprocessed_data)
        
        # Apply inverse transformation to get original scale features and predictions
        original_scale_df = await inverse_transform_predictions(
            ensemble_predictions, preprocessed_data, full_scaler, metadata['target_names']
        )
        
        # Extract original scale features and predictions from DataFrame
        original_features = original_scale_df.iloc[:, :expected_features].values
        original_predictions = original_scale_df.iloc[:, expected_features:].values
        
        # Save prediction results as CSV files
        # 1. Scaled data CSV (preprocessed features + scaled predictions)
        scaled_results_data = {}
        for i, feature_name in enumerate(metadata['feature_names']):
            scaled_results_data[f'{feature_name}'] = preprocessed_data[:, i]
        
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            scaled_results_data[f'{target_name}_prediction'] = ensemble_predictions.flatten() if ensemble_predictions.ndim > 1 else ensemble_predictions
        else:
            for i, target_name in enumerate(metadata['target_names']):
                scaled_results_data[f'{target_name}_prediction'] = ensemble_predictions[:, i]
        
        scaled_results_df = pd.DataFrame(scaled_results_data)
        scaled_csv_path = os.path.join(predictions_dir, f'{experiment_name}_scaled.csv')
        scaled_results_df.to_csv(scaled_csv_path, index=False)
        
        # 2. Original scale data CSV (original features + original predictions)
        original_results_data = {}
        for i, feature_name in enumerate(metadata['feature_names']):
            original_results_data[f'{feature_name}'] = original_features[:, i]
        
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            original_results_data[f'{target_name}_prediction'] = original_predictions.flatten() if original_predictions.ndim > 1 else original_predictions
        else:
            for i, target_name in enumerate(metadata['target_names']):
                original_results_data[f'{target_name}_prediction'] = original_predictions[:, i]
        
        original_results_df = pd.DataFrame(original_results_data)
        original_csv_path = os.path.join(predictions_dir, f'{experiment_name}_original.csv')
        original_results_df.to_csv(original_csv_path, index=False)
        
        # Create comprehensive results DataFrame with original scale data
        results_data = {}
        
        # Add original scale features
        for i, feature_name in enumerate(metadata['feature_names']):
            results_data[f'{feature_name}_original'] = original_features[:, i]
        
        # Add scaled features for reference
        for i, feature_name in enumerate(metadata['feature_names']):
            results_data[f'{feature_name}_scaled'] = preprocessed_data[:, i]
        
        # Add original scale predictions
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            results_data[f'{target_name}_prediction_original'] = original_predictions.flatten() if original_predictions.ndim > 1 else original_predictions
            results_data[f'{target_name}_prediction_scaled'] = ensemble_predictions.flatten() if ensemble_predictions.ndim > 1 else ensemble_predictions
        else:
            for i, target_name in enumerate(metadata['target_names']):
                results_data[f'{target_name}_prediction_original'] = original_predictions[:, i]
                results_data[f'{target_name}_prediction_scaled'] = ensemble_predictions[:, i]
        
        # Add prediction statistics (for batch, these are aggregated statistics)
        if metadata['target_number'] == 1:
            target_name = metadata['target_names'][0] if metadata['target_names'] else 'target'
            # For batch predictions, prediction_stats contains statistics for all samples
            if num_samples == 1:
                results_data[f'{target_name}_prediction_std'] = [prediction_stats['prediction_std']]
                results_data[f'{target_name}_prediction_ci_lower'] = [prediction_stats['confidence_interval_95'][0]]
                results_data[f'{target_name}_prediction_ci_upper'] = [prediction_stats['confidence_interval_95'][1]]
            else:
                # For batch, add stats for each sample
                results_data[f'{target_name}_prediction_std'] = [prediction_stats['prediction_std']] * num_samples
                results_data[f'{target_name}_prediction_ci_lower'] = [prediction_stats['confidence_interval_95'][0]] * num_samples
                results_data[f'{target_name}_prediction_ci_upper'] = [prediction_stats['confidence_interval_95'][1]] * num_samples
        else:
            for i, target_name in enumerate(metadata['target_names']):
                if num_samples == 1:
                    results_data[f'{target_name}_prediction_std'] = [prediction_stats['prediction_std'][i]]
                    results_data[f'{target_name}_prediction_ci_lower'] = [prediction_stats['confidence_intervals_95'][i][0]]
                    results_data[f'{target_name}_prediction_ci_upper'] = [prediction_stats['confidence_intervals_95'][i][1]]
                else:
                    # For batch, add stats for each sample
                    results_data[f'{target_name}_prediction_std'] = [prediction_stats['prediction_std'][i]] * num_samples
                    results_data[f'{target_name}_prediction_ci_lower'] = [prediction_stats['confidence_intervals_95'][i][0]] * num_samples
                    results_data[f'{target_name}_prediction_ci_upper'] = [prediction_stats['confidence_intervals_95'][i][1]] * num_samples
        
        detailed_results_df = pd.DataFrame(results_data)
        
        # 3. Detailed CSV with all information
        detailed_csv_path = os.path.join(predictions_dir, f'{experiment_name}_detailed.csv')
        detailed_results_df.to_csv(detailed_csv_path, index=False)
        
        # Save experiment metadata
        experiment_metadata = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model_id': metadata['model_id'],
            'input_type': 'batch_feature_values' if is_batch else 'feature_values',
            'prediction_file': f'Feature Values (Manual Input - {"Batch" if is_batch else "Single"})',
            'input_values': feature_values,
            'feature_names': metadata['feature_names'],
            'target_names': metadata['target_names'],
            'num_samples': num_samples,
            'feature_number': expected_features,
            'target_number': metadata['target_number'],
            'prediction_statistics': prediction_stats,
            'files_generated': {
                'scaled_data': f'{experiment_name}_scaled.csv',
                'original_data': f'{experiment_name}_original.csv',
                'detailed_results': f'{experiment_name}_detailed.csv',
                'experiment_report': f'{experiment_name}_report.md'
            }
        }
        
        metadata_path = os.path.join(predictions_dir, f'{experiment_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        # Generate comprehensive experiment report
        experiment_report_path = await generate_prediction_experiment_report(
            predictions_dir, experiment_metadata, detailed_results_df, model_components, prediction_stats
        )
        
        # Update results to include detailed information
        results['experiment_details'] = {
            'output_directory': predictions_dir,
            'scaled_data_file': scaled_csv_path,
            'original_data_file': original_csv_path,
            'detailed_results_file': detailed_csv_path,
            'experiment_report': experiment_report_path,
            'prediction_statistics': prediction_stats
        }
    
    return results, experiment_report_path, basic_csv_path


async def batch_predict_from_values(
    model_components: Dict[str, Any],
    feature_value_list: List[List[float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Make batch predictions from multiple feature value sets.
    
    Args:
        model_components: Dictionary containing model and preprocessing components
        feature_value_list: List of feature value lists
        
    Returns:
        Tuple of (scaled_predictions, original_scale_predictions)
    """
    if not feature_value_list:
        raise ValueError("Feature value list cannot be empty")
    
    # Validate input
    metadata = model_components['metadata']
    expected_features = metadata['feature_number']
    
    for i, feature_values in enumerate(feature_value_list):
        if len(feature_values) != expected_features:
            raise ValueError(f"Sample {i}: Expected {expected_features} features, got {len(feature_values)}")
    
    # Convert to numpy array and preprocess
    feature_array = np.array(feature_value_list)
    feature_scaler = model_components['feature_scaler']
    preprocessed_data = await preprocess_prediction_input(
        feature_array, feature_scaler, metadata['feature_names']
    )
    
    # Make predictions
    scaled_predictions = await predict_with_ensemble(model_components, preprocessed_data)
    
    # Apply inverse transformation
    full_scaler = model_components['full_scaler']
    result_df = await inverse_transform_predictions(
        scaled_predictions, preprocessed_data, full_scaler, metadata['target_names']
    )
    
    # Extract prediction columns
    target_number = metadata['target_number']
    if target_number == 1:
        original_predictions = result_df.iloc[:, -1].values  # Last column
    else:
        # For multi-target, get all prediction columns
        original_predictions = result_df.iloc[:, -target_number:].values
    
    return scaled_predictions, original_predictions


async def get_prediction_statistics(
    model_components: Dict[str, Any],
    preprocessed_features: np.ndarray
) -> Dict[str, Any]:
    """Get prediction statistics including confidence intervals.
    
    Args:
        model_components: Dictionary containing model components
        preprocessed_features: Preprocessed feature data
        
    Returns:
        Dictionary containing prediction statistics
    """
    # Get individual predictions from all CV models
    ensemble_pred, individual_preds = await predict_with_ensemble(
        model_components, preprocessed_features, return_individual_predictions=True
    )
    
    individual_preds = np.array(individual_preds)
    metadata = model_components['metadata']
    target_number = metadata['target_number']
    
    stats = {
        "ensemble_prediction": ensemble_pred.tolist() if ensemble_pred.ndim > 0 else float(ensemble_pred),
        "individual_predictions": [pred.tolist() for pred in individual_preds],
        "num_models": len(individual_preds),
        "target_names": metadata['target_names']
    }
    
    # Calculate statistics for each target
    if target_number == 1:
        # Single target statistics
        std = np.std(individual_preds, axis=0)
        # Handle both scalar and array cases for std
        if np.isscalar(std) or std.size == 1:
            std_value = float(std)
        else:
            # If somehow we get an array, take the first element
            std_value = float(std.flatten()[0])
            
        # Ensure ensemble_pred is also scalar for single target
        if hasattr(ensemble_pred, 'size') and ensemble_pred.size == 1:
            ensemble_value = float(ensemble_pred)
        elif np.isscalar(ensemble_pred):
            ensemble_value = float(ensemble_pred)
        else:
            ensemble_value = float(ensemble_pred.flatten()[0])
            
        stats.update({
            "prediction_std": std_value,
            "confidence_interval_95": [
                float(ensemble_value - 1.96 * std_value),
                float(ensemble_value + 1.96 * std_value)
            ],
            "min_prediction": float(np.min(individual_preds)),
            "max_prediction": float(np.max(individual_preds))
        })
    else:
        # Multi-target statistics
        std = np.std(individual_preds, axis=0)
        stats.update({
            "prediction_std": std.tolist(),
            "confidence_intervals_95": [
                [float(ensemble_pred[i] - 1.96 * std[i]), 
                 float(ensemble_pred[i] + 1.96 * std[i])]
                for i in range(target_number)
            ],
            "min_predictions": np.min(individual_preds, axis=0).tolist(),
            "max_predictions": np.max(individual_preds, axis=0).tolist()
        })
    
    return stats


async def predict_and_save(
    model_components: Dict[str, Any],
    prediction_file: str,
    output_file: str = None
) -> str:
    """Make predictions and save results to file.
    
    Args:
        model_components: Dictionary containing model components
        prediction_file: Path to input prediction file
        output_file: Optional output file path
        
    Returns:
        Path to saved results file
    """
    # Make predictions
    result_df, _ = await predict_from_file(model_components, prediction_file)
    
    # Determine output file path
    if output_file is None:
        base_name = prediction_file.rsplit('.', 1)[0]
        output_file = f"{base_name}_predictions.csv"
    
    # Save results
    result_df.to_csv(output_file, index=False)
    
    return output_file


def create_prediction_summary(
    model_components: Dict[str, Any],
    predictions: np.ndarray,
    input_features: np.ndarray = None
) -> Dict[str, Any]:
    """Create a summary of predictions.
    
    Args:
        model_components: Dictionary containing model components
        predictions: Prediction results
        input_features: Optional input features for context
        
    Returns:
        Dictionary containing prediction summary
    """
    metadata = model_components['metadata']
    target_number = metadata['target_number']
    
    summary = {
        "model_id": metadata['model_id'],
        "num_predictions": len(predictions) if predictions.ndim > 0 else 1,
        "target_names": metadata['target_names'],
        "feature_names": metadata['feature_names']
    }
    
    if predictions.ndim == 0:
        predictions = np.array([predictions])
    
    if target_number == 1:
        # Single target summary
        summary.update({
            "prediction_stats": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            }
        })
    else:
        # Multi-target summary
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, target_number)
        
        target_stats = {}
        for i, target_name in enumerate(metadata['target_names']):
            target_preds = predictions[:, i]
            target_stats[target_name] = {
                "mean": float(np.mean(target_preds)),
                "std": float(np.std(target_preds)),
                "min": float(np.min(target_preds)),
                "max": float(np.max(target_preds))
            }
        
        summary["prediction_stats_by_target"] = target_stats
    
    return summary


async def generate_prediction_experiment_report(
    output_dir: str,
    experiment_metadata: Dict[str, Any],
    results_df: pd.DataFrame,
    model_components: Dict[str, Any],
    prediction_stats: Dict[str, Any]
) -> str:
    """Generate a comprehensive prediction experiment report.
    
    Args:
        output_dir: Directory to save the report
        experiment_metadata: Experiment metadata dictionary
        results_df: Results DataFrame with predictions
        model_components: Model components dictionary
        prediction_stats: Prediction statistics
        
    Returns:
        Path to the generated report
    """
    metadata = model_components['metadata']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Prediction Experiment Report

**Generated on:** {timestamp}  
**Experiment Name:** {experiment_metadata['experiment_name']}  
**Model ID:** `{metadata['model_id']}`  
**Output Directory:** `{output_dir}`

## Executive Summary

This report documents a comprehensive neural network prediction experiment conducted using a pre-trained ensemble model. The experiment involved preprocessing input data, making predictions with multiple cross-validation models, and providing detailed statistical analysis of the prediction results.

### Key Results
- **Number of Predictions:** {experiment_metadata['num_samples']}
- **Feature Count:** {experiment_metadata['feature_number']}
- **Target Count:** {experiment_metadata['target_number']}
- **Ensemble Models Used:** {prediction_stats['num_models']} (from cross-validation)

---

## 1. Experiment Setup

### 1.1 Input Data Information

| Parameter | Value |
|-----------|-------|
| Input File | `{experiment_metadata['prediction_file']}` |
| Number of Samples | {experiment_metadata['num_samples']} |
| Number of Features | {experiment_metadata['feature_number']} |
| Number of Targets | {experiment_metadata['target_number']} |
| Data Type | Numerical (floating-point) |

### 1.2 Feature Information

**Input Features ({experiment_metadata['feature_number']} columns):**
{', '.join(f'`{name}`' for name in experiment_metadata['feature_names'])}

**Target Variables ({experiment_metadata['target_number']} column{'s' if experiment_metadata['target_number'] > 1 else ''}):**
{', '.join(f'`{name}`' for name in experiment_metadata['target_names'])}

### 1.3 Model Information

| Component | Details |
|-----------|---------|
| **Model Type** | Multi-Layer Perceptron (MLP) Ensemble |
| **Ensemble Size** | {prediction_stats['num_models']} models (from cross-validation) |
| **Model ID** | `{metadata['model_id']}` |
| **Training Framework** | PyTorch |
| **Prediction Method** | Ensemble averaging |

---

## 2. Prediction Results

### 2.1 Prediction Statistics

"""

    # Add prediction statistics based on target type
    if experiment_metadata['target_number'] == 1:
        target_name = experiment_metadata['target_names'][0]
        original_preds = results_df[f'{target_name}_prediction_original'].values
        
        report += f"""
#### Single Target Prediction Statistics

**Target: {target_name}**

| Statistic | Value |
|-----------|-------|
| Mean Prediction | {np.mean(original_preds):.6f} |
| Standard Deviation | {np.std(original_preds):.6f} |
| Minimum Prediction | {np.min(original_preds):.6f} |
| Maximum Prediction | {np.max(original_preds):.6f} |
| Prediction Range | {np.max(original_preds) - np.min(original_preds):.6f} |

**Ensemble Uncertainty:**
- **Model Agreement (Std)**: {prediction_stats['prediction_std']:.6f}
- **95% Confidence Interval**: [{prediction_stats['confidence_interval_95'][0]:.6f}, {prediction_stats['confidence_interval_95'][1]:.6f}]
"""
    else:
        report += f"""
#### Multi-Target Prediction Statistics

| Target | Mean | Std Dev | Min | Max | Model Agreement (Std) |
|--------|------|---------|-----|-----|----------------------|"""
        
        for i, target_name in enumerate(experiment_metadata['target_names']):
            original_preds = results_df[f'{target_name}_prediction_original'].values
            model_agreement_std = prediction_stats['prediction_std'][i]
            
            report += f"""
| {target_name} | {np.mean(original_preds):.6f} | {np.std(original_preds):.6f} | {np.min(original_preds):.6f} | {np.max(original_preds):.6f} | {model_agreement_std:.6f} |"""
        
        report += f"""

**95% Confidence Intervals by Target:**
"""
        for i, target_name in enumerate(experiment_metadata['target_names']):
            ci_lower, ci_upper = prediction_stats['confidence_intervals_95'][i]
            report += f"""
- **{target_name}**: [{ci_lower:.6f}, {ci_upper:.6f}]"""

    report += f"""

---

## 3. Generated Files

| File | Description |
|------|-------------|
| `{experiment_metadata['files_generated']['scaled_data']}` | Preprocessed features + scaled predictions |
| `{experiment_metadata['files_generated']['original_data']}` | Original scale features + predictions |
| `{experiment_metadata['files_generated']['detailed_results']}` | Comprehensive results with statistics |
| `{experiment_metadata['files_generated']['experiment_report']}` | This report |

---

*Report generated on {timestamp}*
"""

    # Save the report
    report_path = os.path.join(output_dir, f"{experiment_metadata['experiment_name']}_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


# ================== CLASSIFICATION PREDICTION FUNCTIONS ==================

def load_classification_model(model_folder: str, feature_number: int, num_classes: int) -> Tuple[List[MLPClassification], Dict[str, Any], Any, Any]:
    """Load trained classification models and metadata."""
    import os
    import json
    import pickle
    import torch
    
    # Load metadata
    metadata_path = os.path.join(model_folder, "model_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load scalers (updated to use unified scalers.pkl)
    scalers_path = os.path.join(model_folder, "scalers.pkl")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    scaler = scalers['feature_scaler']
    
    # Load label encoder if it exists
    label_encoder = None
    label_encoder_path = os.path.join(model_folder, "label_encoder.pkl")
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    
    # Load model states (updated to use unified model_states.pth)
    models_path = os.path.join(model_folder, "model_states.pth")
    model_states = torch.load(models_path, map_location=torch.device('cpu'))
    
    # Create model instances from states
    models = []
    for model_state in model_states:
        model = MLPClassification(metadata['best_params'], feature_number, num_classes)
        model.load_state_dict(model_state)
        model.eval()
        models.append(model)
    
    return models, metadata, scaler, label_encoder


def classify_from_values(
    feature_values,  # Union[List[float], List[List[float]]]
    model_folder: str,
    feature_names: Optional[List[str]] = None,
    generate_experiment_report: bool = False
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """Classify using trained models from feature values.
    
    Supports both single sample and batch classification.
    
    Args:
        feature_values: Single sample [val1, val2, ...] or batch [[val1, val2, ...], [...]]
        model_folder: Path to trained model folder
        feature_names: Optional list of feature names
        generate_experiment_report: Whether to generate detailed report
        
    Returns:
        Tuple of (results_dict, experiment_report_path, basic_csv_path)
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    import torch
    from datetime import datetime
    
    try:
        # Load metadata first
        metadata_path = os.path.join(model_folder, "model_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        feature_number = metadata['data_info']['feature_number']
        num_classes = metadata['data_info'].get('num_classes', 2)  # Default to binary
        
        # Detect input type
        is_batch = isinstance(feature_values[0], (list, tuple))
        
        if is_batch:
            # Batch prediction
            feature_array = np.array(feature_values)
            if feature_array.shape[1] != feature_number:
                raise ValueError(f"Expected {feature_number} features, got {feature_array.shape[1]}")
            
            prediction_type = 'batch'
            num_samples = len(feature_values)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        else:
            # Single prediction
            if len(feature_values) != feature_number:
                raise ValueError(f"Expected {feature_number} features, got {len(feature_values)}")
            
            feature_array = np.array([feature_values])
            prediction_type = 'single'
            num_samples = 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Classification - Processing {prediction_type} prediction with {num_samples} sample(s)")
        
        # Load models
        models, metadata, scaler, label_encoder = load_classification_model(model_folder, feature_number, num_classes)
        
        # Scale features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Make predictions with ensemble
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_probs = []
        all_predictions = []
        
        for model in models:
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                features_tensor = torch.FloatTensor(feature_array_scaled).to(device)
                
                # Get probabilities
                probs = model.predict_proba(features_tensor).cpu().numpy()
                
                # Get class predictions
                if num_classes == 2:
                    predictions = (probs > 0.5).astype(int).squeeze()
                else:
                    predictions = model.predict_classes(features_tensor).cpu().numpy()
                
                all_probs.append(probs)
                all_predictions.append(predictions)
        
        # Ensemble predictions
        ensemble_probs = np.mean(all_probs, axis=0)
        
        if num_classes == 2:
            # Binary classification
            ensemble_predictions = (ensemble_probs > 0.5).astype(int)
            if ensemble_predictions.ndim > 1:
                ensemble_predictions = ensemble_predictions.squeeze()
        else:
            # Multi-class classification
            ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        
        # Prepare class names and handle label conversion
        class_names = metadata['data_info'].get('class_names', [f'Class_{i}' for i in range(num_classes)])
        
        # Function to convert predictions back to original labels
        def convert_prediction_to_original(pred_idx):
            if label_encoder is not None:
                # Convert back to original string labels
                return label_encoder.inverse_transform([pred_idx])[0]
            else:
                # Use stored class names or default
                return class_names[pred_idx]
        
        # Format results
        if prediction_type == 'single':
            prediction = int(ensemble_predictions[0]) if hasattr(ensemble_predictions, '__len__') and len(ensemble_predictions) > 0 else int(ensemble_predictions)
            probabilities = ensemble_probs[0] if ensemble_probs.ndim > 1 else ensemble_probs
            
            # Convert prediction back to original label
            original_class_name = convert_prediction_to_original(prediction)
            
            # Handle probabilities properly for both binary and multi-class
            if num_classes == 2:
                # For binary classification, probabilities might be a scalar (positive class probability)
                if np.isscalar(probabilities):
                    prob_dict = {
                        class_names[0]: float(1.0 - probabilities),  # negative class
                        class_names[1]: float(probabilities)         # positive class
                    }
                else:
                    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(num_classes)}
            else:
                # Multi-class classification
                prob_dict = {class_names[i]: float(probabilities[i]) for i in range(num_classes)}
            
            results = {
                'prediction_type': 'single',
                'predicted_class_index': prediction,
                'predicted_class': original_class_name,
                'predicted_class_name': original_class_name,  # For backward compatibility
                'probabilities': prob_dict,
                'confidence': float(np.max(probabilities) if not np.isscalar(probabilities) else probabilities),
                'feature_values': feature_values,
                'model_folder': model_folder,
                'num_models_used': len(models)
            }
            
            print(f"Classification result: {original_class_name} (confidence: {results['confidence']:.3f})")
            
        else:
            # Batch results
            batch_results = []
            for i in range(num_samples):
                prediction = int(ensemble_predictions[i])
                probabilities = ensemble_probs[i] if ensemble_probs.ndim > 1 else ensemble_probs
                
                # Convert prediction back to original label
                original_class_name = convert_prediction_to_original(prediction)
                
                # Handle probabilities properly for both binary and multi-class
                if num_classes == 2:
                    # For binary classification, probabilities might be a scalar (positive class probability)
                    if np.isscalar(probabilities):
                        prob_dict = {
                            class_names[0]: float(1.0 - probabilities),  # negative class
                            class_names[1]: float(probabilities)         # positive class
                        }
                        confidence = float(probabilities)
                    else:
                        prob_dict = {class_names[j]: float(probabilities[j]) for j in range(num_classes)}
                        confidence = float(np.max(probabilities))
                else:
                    # Multi-class classification
                    prob_dict = {class_names[j]: float(probabilities[j]) for j in range(num_classes)}
                    confidence = float(np.max(probabilities))
                
                sample_result = {
                    'predicted_class_index': prediction,
                    'predicted_class': original_class_name,
                    'predicted_class_name': original_class_name,  # For backward compatibility
                    'probabilities': prob_dict,
                    'confidence': confidence,
                    'feature_values': feature_values[i]
                }
                batch_results.append(sample_result)
            
            results = {
                'prediction_type': 'batch',
                'batch_results': batch_results,
                'batch_size': num_samples,
                'model_folder': model_folder,
                'num_models_used': len(models)
            }
            
            print(f"Batch classification completed: {num_samples} samples processed")
        
        # Generate CSV files
        experiment_report_path = None
        basic_csv_path = None
        
        if prediction_type == 'single':
            csv_filename = f"SingleClassification_{timestamp}.csv"
        else:
            csv_filename = f"BatchClassification_{num_samples}samples_{timestamp}.csv"
        
        basic_csv_path = os.path.join(model_folder, "predictions", csv_filename)
        os.makedirs(os.path.dirname(basic_csv_path), exist_ok=True)
        
        # Create basic CSV
        if prediction_type == 'single':
            csv_data = {
                'predicted_class_index': [results['predicted_class_index']],
                'predicted_class': [results['predicted_class']],
                'confidence': [results['confidence']]
            }
            
            # Add feature columns
            feature_names_used = feature_names or [f'Feature_{i}' for i in range(feature_number)]
            for i, name in enumerate(feature_names_used):
                csv_data[name] = [feature_values[i]]
            
            # Add probability columns
            for class_name, prob in results['probabilities'].items():
                csv_data[f'prob_{class_name}'] = [prob]
                
        else:
            csv_data = {
                'sample_index': list(range(num_samples)),
                'predicted_class_index': [r['predicted_class_index'] for r in batch_results],
                'predicted_class': [r['predicted_class'] for r in batch_results],
                'confidence': [r['confidence'] for r in batch_results]
            }
            
            # Add feature columns
            feature_names_used = feature_names or [f'Feature_{i}' for i in range(feature_number)]
            for i, name in enumerate(feature_names_used):
                csv_data[name] = [sample[i] for sample in feature_values]
            
            # Add probability columns
            for i, class_name in enumerate(class_names):
                csv_data[f'prob_{class_name}'] = [r['probabilities'][class_name] for r in batch_results]
        
        df = pd.DataFrame(csv_data)
        df.to_csv(basic_csv_path, index=False)
        print(f"Basic CSV saved: {basic_csv_path}")
        
        # Generate experiment report if requested
        if generate_experiment_report:
            experiment_report_path = generate_classification_experiment_report(
                results, metadata, model_folder, timestamp, csv_filename, feature_names_used
            )
        
        return results, experiment_report_path, basic_csv_path
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        raise


def generate_classification_experiment_report(
    results: Dict[str, Any],
    metadata: Dict[str, Any],
    model_folder: str,
    timestamp: str,
    csv_filename: str,
    feature_names: List[str]
) -> str:
    """Generate detailed experiment report for classification predictions."""
    import os
    import json
    from datetime import datetime
    
    prediction_type = results['prediction_type']
    
    # Prepare experiment metadata
    experiment_data = {
        'experiment_id': f"classification_experiment_{timestamp}",
        'timestamp': timestamp,
        'prediction_type': prediction_type,
        'model_info': {
            'model_folder': model_folder,
            'num_models_used': results['num_models_used'],
            'hyperparameters': metadata.get('hyperparameters', {}),
            'training_score': metadata.get('best_score', 'N/A')
        },
        'data_info': metadata.get('data_info', {}),
        'prediction_file': csv_filename,
        'feature_names': feature_names
    }
    
    if prediction_type == 'single':
        experiment_data['prediction_results'] = {
            'predicted_class_index': results['predicted_class_index'],
            'predicted_class': results['predicted_class'],
            'confidence': results['confidence'],
            'probabilities': results['probabilities'],
            'feature_values': results['feature_values']
        }
    else:
        experiment_data['prediction_results'] = {
            'batch_size': results['batch_size'],
            'summary': {
                'total_samples': len(results['batch_results']),
                'average_confidence': sum(r['confidence'] for r in results['batch_results']) / len(results['batch_results']),
                'class_distribution': {}
            }
        }
        
        # Calculate class distribution
        class_counts = {}
        for result in results['batch_results']:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        experiment_data['prediction_results']['summary']['class_distribution'] = class_counts
    
    # Save experiment metadata
    experiment_path = os.path.join(model_folder, "predictions", f"experiment_{timestamp}.json")
    with open(experiment_path, 'w') as f:
        json.dump(experiment_data, f, indent=2)
    
    print(f"Experiment report saved: {experiment_path}")
    return experiment_path


def classify_from_file(
    csv_file_path: str,
    model_folder: str,
    feature_names: Optional[List[str]] = None,
    generate_experiment_report: bool = False
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """Classify using trained models from CSV file.
    
    Args:
        csv_file_path: Path to CSV file with features
        model_folder: Path to trained model folder
        feature_names: Optional list of feature names
        generate_experiment_report: Whether to generate detailed report
        
    Returns:
        Tuple of (results_dict, experiment_report_path, basic_csv_path)
    """
    import pandas as pd
    
    # Load CSV data
    df = pd.read_csv(csv_file_path)
    
    # Convert to list of lists (batch format)
    feature_values = df.values.tolist()
    
    return classify_from_values(
        feature_values,
        model_folder,
        feature_names,
        generate_experiment_report
    )