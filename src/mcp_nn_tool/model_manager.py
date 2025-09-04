"""Model management utilities for the MCP NN Tool.

This module handles saving, loading, and managing trained models with their configurations
using a folder-based structure for better organization.
"""

import torch
import pickle
import json
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .neural_network import MLPregression, MLPClassification


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    created_at: str
    feature_number: int
    target_number: int
    feature_names: List[str]
    target_names: List[str]
    best_params: Dict[str, Any]
    best_mae: float
    cv_folds: int
    training_epochs: int
    

class ModelManager:
    """Manager for neural network models and their configurations using folder structure."""
    
    def __init__(self, models_dir: str = "saved_models"):
        """Initialize the model manager.
        
        Args:
            models_dir: Base directory to store saved model folders
        """
        self.models_dir = models_dir
        self.models_index_path = os.path.join(models_dir, "models_index.json")
        self._ensure_models_directory()
        self._load_models_index()
    
    def _ensure_models_directory(self) -> None:
        """Ensure the models directory exists."""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def _load_models_index(self) -> None:
        """Load the models index from disk."""
        if os.path.exists(self.models_index_path):
            with open(self.models_index_path, 'r') as f:
                self.models_index = json.load(f)
        else:
            self.models_index = {}
    
    def _save_models_index(self) -> None:
        """Save the models index to disk."""
        with open(self.models_index_path, 'w') as f:
            json.dump(self.models_index, f, indent=2)
    
    def _get_model_folder(self, model_id: str) -> str:
        """Get the folder path for a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Path to the model folder
        """
        return os.path.join(self.models_dir, model_id)
    
    async def save_model(
        self,
        model_states: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        best_mae: float,
        feature_names: List[str],
        target_names: List[str],
        full_scaler: StandardScaler,
        feature_scaler: StandardScaler,
        cv_folds: int = 5,
        training_epochs: int = 100,
        model_id: Optional[str] = None
    ) -> str:
        """Save a trained model with all its components in a dedicated folder.
        
        Args:
            model_states: List of model state dictionaries (one per CV fold)
            best_params: Best hyperparameters found
            best_mae: Best MAE achieved
            feature_names: Names of input features
            target_names: Names of target variables
            full_scaler: Scaler fitted on full training data
            feature_scaler: Scaler fitted on features only
            cv_folds: Number of cross-validation folds used
            training_epochs: Number of training epochs per fold
            model_id: Optional custom model ID
            
        Returns:
            Generated model ID
        """
        if model_id is None:
            model_id = str(uuid.uuid4())
        
        # Create model folder
        model_folder = self._get_model_folder(model_id)
        os.makedirs(model_folder, exist_ok=True)
        
        # Save model metadata
        metadata = {
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
            'task_type': 'regression',  # Add task_type for consistency
            'feature_number': len(feature_names),
            'target_number': len(target_names),
            'feature_names': feature_names,
            'target_names': target_names,
            'best_params': best_params,
            'best_mae': best_mae,
            'cv_folds': cv_folds,
            'training_epochs': training_epochs
        }
        
        # Save metadata as JSON (unified filename)
        metadata_path = os.path.join(model_folder, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model states (one per fold)
        models_path = os.path.join(model_folder, "model_states.pth")
        torch.save(model_states, models_path)
        
        # Save best parameters separately for easy access
        params_path = os.path.join(model_folder, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save scalers in the model folder
        scalers_path = os.path.join(model_folder, "scalers.pkl")
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'full_scaler': full_scaler,
                'feature_scaler': feature_scaler
            }, f)
        
        # Save feature and target names for easy reference
        names_path = os.path.join(model_folder, "column_names.json")
        with open(names_path, 'w') as f:
            json.dump({
                'feature_names': feature_names,
                'target_names': target_names
            }, f, indent=2)
        
        # Update models index
        self.models_index[model_id] = {
            'model_id': model_id,
            'created_at': metadata['created_at'],
            'feature_number': metadata['feature_number'],
            'target_number': metadata['target_number'],
            'target_names': target_names,
            'best_mae': best_mae,
            'folder_path': model_folder
        }
        self._save_models_index()
        
        return model_id
    
    async def save_classification_model(
        self,
        model_states: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        best_accuracy: float,
        feature_names: List[str],
        class_names: List[str],
        feature_scaler: StandardScaler,
        num_classes: int,
        cv_folds: int = 5,
        training_epochs: int = 100,
        model_id: Optional[str] = None,
        task_type: str = "classification",
        label_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a trained classification model with all its components in a dedicated folder.
        
        Args:
            model_states: List of model state dictionaries (one per CV fold)
            best_params: Best hyperparameters found
            best_accuracy: Best accuracy achieved
            feature_names: Names of input features
            class_names: Names of the classes
            feature_scaler: Scaler fitted on features only
            num_classes: Number of classes
            cv_folds: Number of cross-validation folds used
            training_epochs: Number of training epochs per fold
            model_id: Optional custom model ID
            task_type: Type of task ("classification")
            
        Returns:
            Generated model ID
        """
        if model_id is None:
            model_id = str(uuid.uuid4())
        
        # Create model folder
        model_folder = self._get_model_folder(model_id)
        os.makedirs(model_folder, exist_ok=True)
        
        # Save model metadata with classification-specific fields
        metadata = {
            'model_id': model_id,
            'created_at': datetime.now().isoformat(),
            'task_type': task_type,
            'feature_number': len(feature_names),
            'num_classes': num_classes,
            'feature_names': feature_names,
            'class_names': class_names,
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'cv_folds': cv_folds,
            'training_epochs': training_epochs,
            'data_info': {
                'feature_number': len(feature_names),
                'num_classes': num_classes,
                'class_names': class_names,
                'feature_names': feature_names
            }
        }
        
        # Add label encoding info if provided
        if label_info is not None:
            metadata['label_info'] = {
                'is_string_labels': label_info.get('is_string_labels', False),
                'class_to_idx': label_info.get('class_to_idx', {}),
                'idx_to_class': label_info.get('idx_to_class', {}),
                'target_column_name': label_info.get('target_column_name', 'target')
            }
        
        # Save metadata as JSON
        metadata_path = os.path.join(model_folder, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model states (one per fold) - consistent with regression model approach
        models_path = os.path.join(model_folder, "model_states.pth")
        torch.save(model_states, models_path)
        
        # Save best parameters separately for easy access
        params_path = os.path.join(model_folder, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save scalers in the model folder (consistent with regression model approach)
        scalers_path = os.path.join(model_folder, "scalers.pkl")
        with open(scalers_path, 'wb') as f:
            pickle.dump({
                'feature_scaler': feature_scaler
            }, f)
        
        # Save label encoder if present
        if label_info is not None and label_info.get('label_encoder') is not None:
            label_encoder_path = os.path.join(model_folder, "label_encoder.pkl")
            with open(label_encoder_path, 'wb') as f:
                pickle.dump(label_info['label_encoder'], f)
        
        # Save feature and class names for easy reference
        names_path = os.path.join(model_folder, "column_names.json")
        with open(names_path, 'w') as f:
            json.dump({
                'feature_names': feature_names,
                'class_names': class_names
            }, f, indent=2)
        
        # Update models index with classification-specific info
        self.models_index[model_id] = {
            'model_id': model_id,
            'created_at': metadata['created_at'],
            'task_type': task_type,
            'feature_number': metadata['feature_number'],
            'num_classes': num_classes,
            'class_names': class_names,
            'best_accuracy': best_accuracy,
            'folder_path': model_folder
        }
        self._save_models_index()
        
        return model_id
    
    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a trained model and its components from folder.
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Dictionary containing all model components
            
        Raises:
            ValueError: If model ID not found
            FileNotFoundError: If model files are missing
        """
        if model_id not in self.models_index:
            raise ValueError(f"Model ID '{model_id}' not found")
        
        model_folder = self._get_model_folder(model_id)
        
        # Load metadata (try both possible filenames)
        metadata_path = os.path.join(model_folder, "model_metadata.json")
        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(model_folder, "metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found for model {model_id}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model states
        models_path = os.path.join(model_folder, "model_states.pth")
        if not os.path.exists(models_path):
            raise FileNotFoundError(f"Model states file not found for model {model_id}")
        
        model_states = torch.load(models_path, map_location='cpu')
        
        # Load scalers
        scalers_path = os.path.join(model_folder, "scalers.pkl")
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(f"Scalers file not found for model {model_id}")
        
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        # Load best parameters
        params_path = os.path.join(model_folder, "best_params.json")
        best_params = metadata['best_params']  # Fallback to metadata
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                best_params = json.load(f)
        
        # Determine task type and prepare return data accordingly
        task_type = metadata.get('task_type', 'regression')  # Default to regression for old models
        
        if task_type == 'classification':
            # Classification models only have feature_scaler
            return {
                'metadata': metadata,
                'model_states': model_states,
                'best_params': best_params,
                'feature_scaler': scalers['feature_scaler'],
                'model_folder': model_folder,
                'task_type': task_type
            }
        else:
            # Regression models have both full_scaler and feature_scaler
            return {
                'metadata': metadata,
                'model_states': model_states,
                'best_params': best_params,
                'full_scaler': scalers['full_scaler'],
                'feature_scaler': scalers['feature_scaler'],
                'model_folder': model_folder,
                'task_type': task_type
            }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all saved models.
        
        Returns:
            List of model information dictionaries
        """
        return list(self.models_index.values())
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a saved model and its folder.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            ValueError: If model ID not found
        """
        if model_id not in self.models_index:
            raise ValueError(f"Model ID '{model_id}' not found")
        
        model_folder = self._get_model_folder(model_id)
        
        # Remove model folder and all contents
        try:
            import shutil
            if os.path.exists(model_folder):
                shutil.rmtree(model_folder)
            
            # Remove from index
            del self.models_index[model_id]
            self._save_models_index()
            
            return True
        except Exception as e:
            print(f"Error deleting model {model_id}: {e}")
            return False
    
    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists.
        
        Args:
            model_id: ID of the model to check
            
        Returns:
            True if model exists
        """
        return model_id in self.models_index
    
    def get_model_folder_path(self, model_id: str) -> Optional[str]:
        """Get the folder path for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Path to model folder or None if model doesn't exist
        """
        if model_id in self.models_index:
            return self._get_model_folder(model_id)
        return None
    
    async def save_prediction_data(
        self, 
        model_id: str, 
        input_data: pd.DataFrame, 
        predictions: pd.DataFrame,
        filename: str = None
    ) -> str:
        """Save prediction input and results to the model folder.
        
        Args:
            model_id: ID of the model
            input_data: Input data used for prediction
            predictions: Prediction results
            filename: Optional custom filename
            
        Returns:
            Path to saved prediction file
            
        Raises:
            ValueError: If model ID not found
        """
        if not self.model_exists(model_id):
            raise ValueError(f"Model ID '{model_id}' not found")
        
        model_folder = self._get_model_folder(model_id)
        predictions_folder = os.path.join(model_folder, "predictions")
        os.makedirs(predictions_folder, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_{timestamp}.csv"
        
        # Combine input and predictions
        combined_data = pd.concat([input_data.reset_index(drop=True), 
                                 predictions.reset_index(drop=True)], axis=1)
        
        prediction_path = os.path.join(predictions_folder, filename)
        combined_data.to_csv(prediction_path, index=False)
        
        return prediction_path 