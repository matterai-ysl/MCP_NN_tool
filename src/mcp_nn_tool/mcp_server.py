"""MCP Server for Neural Network Tool.

This module implements an MCP server that provides neural network training and prediction
capabilities through a standardized interface using FastMCP, supporting both single and
multi-target predictions.
"""
import sys
import os
from pathlib import Path
print(str(Path(__file__).parent.parent.parent))
print(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from .config import BASE_URL,get_download_url,get_static_url
import asyncio
import json
from typing import Any, Dict, List, Optional, Union
import logging
import uuid
import numpy as np
# FastMCP imports
from fastmcp import FastMCP
import time

from mcp_nn_tool.data_utils import (
    preprocess_data, 
    read_data_file, 
    validate_data_format,
    extract_feature_target_info,
    preprocess_classification_data
)
from mcp_nn_tool import prediction
from mcp_nn_tool.training import (optimize_hyperparameters, 
                                  train_final_model,
                                  optimize_classification_hyperparameters,
                                  train_final_classification_model,
                                  generate_academic_report
                                  )
from mcp_nn_tool.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server instance
mcp = FastMCP("Neural-Network-MCP-Tool",
              instructions="Neural Network MCP Tool",
              version="0.1.0"
              )

# Initialize model manager with './trained_model' directory
model_manager = ModelManager("./trained_model")


@mcp.tool()
async def train_neural_network_regression(
    training_file: str,
    target_columns: int = 1,
    n_trials: int = 100,
    cv_folds: int = 5,
    num_epochs: int = 500,
    algorithm: str = "TPE",
    loss_function: str = "MAE",
) -> Dict[str, Any]:
    """Train a neural network model with hyperparameter optimization.
    
    Args:
        training_file: Path to training data file (CSV/Excel)
        target_columns: Number of target columns at the end of the dataset (default: 1)
        n_trials: Number of hyperparameter optimization trials (default: 100)
        cv_folds: Number of cross-validation folds (default: 5)
        num_epochs: Number of training epochs per fold (default: 500)
        algorithm: Optimization algorithm - "TPE" or "GP" (default: "TPE")
        loss_function: Loss function to use - "MAE" or "MSE" (default: "MAE")
        
    Returns:
        Dictionary containing training results with model ID, best parameters, and performance metrics
    """
    logger.info(f"Starting neural network training with file: {training_file}, targets: {target_columns}")
    
    try:
        import time
        print("Starting neural network training")
        
        # Record training start time
        training_start_time = time.time()
        
        model_id = str(uuid.uuid4())
        # Load and preprocess data
        train_data, _, full_scaler, feature_scaler = await preprocess_data(
            training_file, target_columns=target_columns
        )
        validate_data_format(train_data, target_columns=target_columns)
        
        # Extract feature and target information
        feature_names, target_names, feature_number, target_number = extract_feature_target_info(
            train_data, target_columns
        )
        data_array = train_data.values
        
        # Progress callback for monitoring
        progress_updates = []
        async def progress_callback(info):
            progress_updates.append(info)
            logger.info(f"Training progress: {info}")
        
        # Create temporary model folder for saving training artifacts
        temp_model_folder = f"trained_model/{model_id}"
        os.makedirs(temp_model_folder, exist_ok=True)
        
        # Optimize hyperparameters with saving
        optimization_start_time = time.time()
        best_params, best_loss, trials_df = await optimize_hyperparameters(
            data_array,
            feature_number,
            target_number,
            n_trials=n_trials,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            algorithm=algorithm,
            progress_callback=progress_callback,
            save_dir=temp_model_folder,
            loss_function=loss_function
        )
        optimization_time = time.time() - optimization_start_time
        
        # Train final model with best parameters and detailed recording
        final_training_start_time = time.time()
        model_states, final_loss, cv_results = await train_final_model(
            best_params,
            data_array,
            feature_number,
            target_number,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            progress_callback=progress_callback,
            save_dir=temp_model_folder,
            full_scaler=full_scaler,
            target_names=target_names,
            feature_names=feature_names,
            loss_function=loss_function
        )
        final_training_time = time.time() - final_training_start_time
        
        # Save model (auto-generate model ID)
        saved_model_id = await model_manager.save_model(
            model_states=model_states,
            best_params=best_params,
            best_mae=best_loss,
            feature_names=feature_names,
            target_names=target_names,
            full_scaler=full_scaler,
            feature_scaler=feature_scaler, # type: ignore
            cv_folds=cv_folds,
            training_epochs=num_epochs,
            model_id=model_id
        )
        
        # Get actual model folder path
        model_folder = model_manager.get_model_folder_path(saved_model_id)
        
        # Generate academic report
        total_training_time = time.time() - training_start_time
        data_info = {
            "data_shape": list(train_data.shape),
            "feature_number": feature_number,
            "target_number": target_number,
            "feature_names": feature_names,
            "target_names": target_names
        }
        
        try:
            report_path = generate_academic_report(
                model_folder=model_folder, # type: ignore
                model_id=saved_model_id,
                training_file=training_file,
                data_info=data_info,
                best_params=best_params,
                best_score=best_loss,
                trials_df=trials_df,
                cv_results=cv_results, # type: ignore
                average_mae=final_loss,
                optimization_time=optimization_time,
                training_time=final_training_time,
                n_trials=n_trials,
                cv_folds=cv_folds,
                num_epochs=num_epochs,
                algorithm=algorithm,
                loss_function=loss_function
            )
            logger.info(f"Academic report generated: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to generate academic report: {str(e)}")
            report_path = None
        
        report_url_path = get_download_url(report_path) # type: ignore
        model_folder_url_path = get_download_url(model_folder) # type: ignore
        result = {
            "status": "success",
            "model_id": saved_model_id,
            "model_folder": model_folder,
            "best_parameters": best_params,
            "best_mae": float(best_loss),
            "feature_names": feature_names,
            "target_names": target_names,
            "training_summary": {
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "num_epochs": num_epochs,
                "algorithm": algorithm,
                "loss_function": loss_function,
                "target_columns": target_columns,
                "data_shape": list(train_data.shape),
                "optimization_time": optimization_time,
                "training_time": final_training_time,
                "total_time": total_training_time
            },
            "experiment_report": report_url_path,
            "training_artifacts_download_url": f"More training details, such as hyperparameter optimization, cross-validation scatter plots, and training logs, are available in the {model_folder_url_path}. The **experiment_report** at {report_url_path} provides comprehensive records and analysis for reproducibility and academic reference."
        }
        
        logger.info(f"Training completed successfully. Model ID: {saved_model_id}")
        logger.info(f"Model saved to: {model_folder}")
        logger.info(f"Academic report: {report_path}")
        return result
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise


@mcp.tool()
async def predict_from_file_neural_network(
    model_id: str,
    prediction_file: str,
    generate_experiment_report: bool = True
) -> dict:
    """Make predictions on data from a file (supports both regression and classification).
    
    Automatically detects model type from model metadata and calls the appropriate prediction method.
    
    Args:
        model_id: ID of the trained model to use for prediction
        prediction_file: Path to CSV/Excel file containing prediction data
        generate_experiment_report: Whether to generate detailed experiment report (default: True)
        
    Returns:
        Dictionary containing prediction results and optional experiment report path
    """
    try:
        # Load model components to determine task type
        model_components = await model_manager.load_model(model_id)
        task_type = model_components['metadata'].get('task_type', 'regression')
        
        logger.info(f"Model {model_id} detected as {task_type} task")
        
        if task_type == 'classification':
            # Use classification prediction
            results, experiment_report_path, basic_csv_path = await prediction.classify_from_file(
                prediction_file,
                model_components["model_folder"], # type: ignore
                generate_experiment_report=generate_experiment_report
            )
            
            # Format response for classification
            response = {
                'task_type': 'classification',
                'model_id': model_id,
                'results': results,
                'experiment_report_generated': generate_experiment_report,
                'basic_csv_path': basic_csv_path,
                "experiment_report_path": experiment_report_path,
                "prediction_details": f"Prediction task completed.More prediction details, such as the prediction results, can be found in the {basic_csv_path}. The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference."
            }
            
        else:
            # Use regression prediction (default)
            result_df, raw_predictions, experiment_report_path, basic_csv_path = await prediction.predict_from_file(
                model_components, 
                prediction_file,
                generate_experiment_report=generate_experiment_report
            )
            experiment_realve_path = get_download_url(experiment_report_path) # type: ignore
            # Format response for regression
            response = {
                'task_type': 'regression',
                'model_id': model_id,
                'num_predictions': len(result_df),
                'predictions': result_df.to_dict('records'),
                'experiment_report_generated': generate_experiment_report,
                'basic_prediction_file': basic_csv_path,
                'experiment_report_path': experiment_realve_path,
                           }
            
            if experiment_report_path:
                
                # Get model folder path for regression
                model_base_path = os.path.join("trained_model", model_id)
                predictions_dir = os.path.join(model_base_path, "predictions")
                predictions_dir_realve_path = get_download_url(predictions_dir) # type: ignore
                response['output_directory'] = predictions_dir_realve_path
                response['prediction_details'] = f"Prediction task completed.More prediction details, such as the prediction results, can be found in the {predictions_dir}. The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference."
        
        # Add experiment report path if available
        if experiment_report_path:
            experiment_report_realve_path = get_download_url(experiment_report_path) # type: ignore
            response['experiment_report_path'] = experiment_report_realve_path
        
        logger.info(f"Prediction completed for {task_type} model {model_id}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'error': f"Prediction failed: {str(e)}"}


@mcp.tool()
async def predict_from_values_neural_network(
    model_id: str, 
    feature_values: Union[list, List[list]],
    generate_experiment_report: bool = True
) -> dict:
    """Make prediction from feature values (supports both regression and classification).
    
    Automatically detects model type from model metadata and calls the appropriate prediction method.
    
    Supports both single prediction and batch prediction:
    - Single: feature_values = [1.0, 2.0, 3.0, 4.0]
    - Batch: feature_values = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    
    Args:
        model_id: ID of the trained model to use for prediction
        feature_values: List of numerical feature values (1D) or list of feature value lists (2D) for batch prediction
        generate_experiment_report: Whether to generate detailed experiment report (default: True)
        
    Returns:
        Dictionary containing prediction results and optional experiment report path
    """
    try:
        # Load model components to determine task type
        model_components = await model_manager.load_model(model_id)
        task_type = model_components['metadata'].get('task_type', 'regression')
        
        logger.info(f"Model {model_id} detected as {task_type} task")

        if task_type == 'classification':
            # Use classification prediction
            results, experiment_report_path, basic_csv_path = prediction.classify_from_values(
                feature_values,
                model_components["model_folder"],
                generate_experiment_report=generate_experiment_report
            )
            
            # Format response for classification
            response = {
                'task_type': 'classification',
                'model_id': model_id,
                'results': results,
                'experiment_report':f"The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference.",
                'basic_csv_path': basic_csv_path
            }
            
        else:
            # Use regression prediction (default)
            results, experiment_report_path, basic_csv_path = await prediction.predict_from_values(
                model_components,
                feature_values,
                generate_experiment_report=generate_experiment_report
            )
            
            # Format response for regression
            response = {
                'task_type': 'regression',
                'model_id': model_id,
                'results': results,
                'experiment_report': f"The **experiment_report** at {experiment_report_path} provides comprehensive records and analysis for reproducibility and academic reference.",
                'basic_csv_path': basic_csv_path
            }
            
            if 'experiment_details' in results:
                response['output_directory'] = results['experiment_details']['output_directory']
        
        # Add experiment report path if available
        if experiment_report_path:
            experiment_report_realve_path = get_download_url(experiment_report_path) # type: ignore
            response['experiment_report_path'] = experiment_report_realve_path
        
        logger.info(f"Prediction completed for {task_type} model {model_id}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'error': f"Prediction failed: {str(e)}"}


@mcp.tool()
async def list_neural_network_models() -> Dict[str, Any]:
    """List all saved neural network models.
    
    Returns:
        Dictionary containing list of model information
    """
    try:
        models = await model_manager.list_models()
        
        # Add folder paths to model info
        for model in models:
            model["folder_path"] = model_manager.get_model_folder_path(model["model_id"])
        
        result = {
            "status": "success",
            "models": models,
            "total_models": len(models),
            "models_directory": "./trained_model"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise


@mcp.tool()
async def get_neural_network_model_info(model_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        Dictionary containing detailed model metadata
    """
    try:
        model_components = await model_manager.load_model(model_id)
        metadata = model_components['metadata']
        
        # Add folder information
        metadata["model_folder"] = model_components["model_folder"]
        metadata["folder_exists"] = os.path.exists(model_components["model_folder"])
        
        # Add file information
        folder_contents = []
        if metadata["folder_exists"]:
            for item in os.listdir(model_components["model_folder"]):
                item_path = os.path.join(model_components["model_folder"], item)
                folder_contents.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
        
        result = {
            "status": "success",
            "model_info": metadata,
            "folder_contents": folder_contents
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise


@mcp.tool()
async def delete_neural_network_model(model_id: str) -> Dict[str, Any]:
    """Delete a saved neural network model and its folder.
    
    Args:
        model_id: ID of the model to delete
        
    Returns:
        Dictionary containing deletion status
    """
    try:
        # Get folder path before deletion
        folder_path = model_manager.get_model_folder_path(model_id)
        
        success = await model_manager.delete_model(model_id)
        
        result = {
            "status": "success" if success else "failed",
            "model_id": model_id,
            "deleted": success,
            "folder_path": folder_path
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        raise


# ================== CLASSIFICATION TOOLS ==================

@mcp.tool()
async def train_classification_model_neural_network(
    training_file: str,
    n_trials: int = 50,
    cv_folds: int = 5,
    num_epochs: int = 300,
    algorithm: str = "TPE",
) -> Dict[str, Any]:
    """Train a classification neural network model with hyperparameter optimization.
    
    Automatically detects class labels and converts them to numerical format.
    Supports both string labels ("cat", "dog") and numeric labels (0, 1, 2).
    
    Args:
        training_file: Path to training data file (CSV/Excel) 
        n_trials: Number of hyperparameter optimization trials (default: 50)
        cv_folds: Number of cross-validation folds (default: 5)
        num_epochs: Number of training epochs per fold (default: 300)
        algorithm: Optimization algorithm - "TPE" or "GP" (default: "TPE")
        
    Returns:
        Dictionary containing training results with model ID, best parameters, and performance metrics
    """
    logger.info(f"Starting classification model training with file: {training_file}")
    
    try:

        # Record training start time
        training_start_time = time.time()
        
        model_id = str(uuid.uuid4())
        
        # Load and preprocess classification data with automatic label encoding
        processed_df, feature_scaler, label_info = await preprocess_classification_data(training_file)
        
        # Extract information from label_info
        num_classes = label_info['num_classes']
        class_names = label_info['classes']
        feature_names = label_info['feature_names']
        feature_number = label_info['feature_number']
        
        logger.info(f"Detected {num_classes} classes: {class_names}")
        logger.info(f"Features: {feature_number} columns")
        
        # Convert to numpy array for training
        data_array = processed_df.values
        
        # Progress callback for monitoring
        progress_updates = []
        async def progress_callback(info):
            progress_updates.append(info)
            logger.info(f"Training progress: {info}")
        
        # Create temporary model folder for saving training artifacts
        temp_model_folder = f"trained_model/{model_id}"
        os.makedirs(temp_model_folder, exist_ok=True)
        
        # Optimize hyperparameters with saving
        optimization_start_time = time.time()
        best_params, best_score, trials_df = await optimize_classification_hyperparameters(
            data_array,
            feature_number,
            num_classes,
            n_trials=n_trials,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            algorithm=algorithm,
            progress_callback=progress_callback,
            save_dir=temp_model_folder
        )
        optimization_time = time.time() - optimization_start_time
        
        # Train final model with best parameters
        final_training_start_time = time.time()
        model_states, final_accuracy, cv_results = await train_final_classification_model(
            best_params,
            data_array,
            feature_number,
            num_classes,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            progress_callback=progress_callback,
            save_dir=temp_model_folder,
            feature_scaler=feature_scaler,
            class_names=class_names,
            feature_names=feature_names
        )
        final_training_time = time.time() - final_training_start_time
        
        # Save classification model with extended metadata including label info
        saved_model_id = await model_manager.save_classification_model(
            model_states=model_states,
            best_params=best_params,
            best_accuracy=final_accuracy,
            feature_names=feature_names,
            class_names=class_names,
            feature_scaler=feature_scaler,
            num_classes=num_classes,
            cv_folds=cv_folds,
            training_epochs=num_epochs,
            model_id=model_id,
            task_type="classification",
            label_info=label_info
        )
        
        # Get actual model folder path
        model_folder = model_manager.get_model_folder_path(saved_model_id)
        
        total_training_time = time.time() - training_start_time
        
        # Generate detailed academic report for classification
        from mcp_nn_tool.training import generate_classification_report
        
        data_info = {
            'data_shape': list(processed_df.shape),
            'feature_number': feature_number,
            'feature_names': feature_names
        }
        
        # Create label encoding information for the report using existing label_info
        # label_info already contains encoding information from preprocess_classification_data
        
        classification_report = generate_classification_report(
            model_folder=model_folder, # type: ignore
            model_id=saved_model_id,
            training_file=training_file,
            data_info=data_info,
            best_params=best_params,
            best_score=best_score,
            trials_df=trials_df,
            cv_results=cv_results, # type: ignore
            average_accuracy=final_accuracy,
            optimization_time=optimization_time,
            training_time=final_training_time,
            n_trials=n_trials,
            cv_folds=cv_folds,
            num_epochs=num_epochs,
            algorithm=algorithm,
            class_names=class_names,
            num_classes=num_classes,
            label_info=label_info
        )
        
        # Save the classification report
        report_path = os.path.join(model_folder, "CLASSIFICATION_TRAINING_REPORT.md") # type: ignore
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(classification_report)
        
        logger.info(f"Classification training report saved to: {report_path}")
        report_realve_path = get_download_url(report_path) # type: ignore
        result = {
            "status": "success",
            "task_type": "classification",
            "model_id": saved_model_id,
            "model_folder": model_folder,
            "best_parameters": best_params,
            "best_accuracy": float(final_accuracy),
            "best_score": float(1.0 - best_score),  # Convert back to accuracy
            "feature_names": feature_names,
            "class_names": class_names,
            "num_classes": num_classes,
            "classification_report_path": report_realve_path,
            "training_summary": {
                "n_trials": n_trials,
                "cv_folds": cv_folds,
                "num_epochs": num_epochs,
                "algorithm": algorithm,
                "data_shape": list(processed_df.shape),
                "feature_number": feature_number,
                "optimization_time": optimization_time,
                "final_training_time": final_training_time,
                "total_training_time": total_training_time,
            },
            "cv_results": {
                "fold_accuracies": cv_results["fold_accuracy_scores"], # type: ignore
                "average_accuracy": float(final_accuracy),
                "classification_report": cv_results.get("classification_report", {}), # type: ignore
                "auc_scores": cv_results.get("auc_scores", {}), # type: ignore
                "roc_data_path": cv_results.get("roc_data_path", ""), # type: ignore
                "roc_plot_path": cv_results.get("roc_plot_path", "") # type: ignore
            }
        }
        
        logger.info(f"Classification training completed. Model ID: {saved_model_id}, Accuracy: {final_accuracy:.4f}")
        return result
        
    except Exception as e:
        logger.error(f"Classification training failed: {str(e)}")
        raise



class NeuralNetworkMCPServer:
    """MCP Server for Neural Network operations using FastMCP."""
    
    def __init__(self, models_dir: str = "./trained_model"):
        """Initialize the Neural Network MCP Server.
        
        Args:
            models_dir: Directory to store trained models (default: './trained_model')
        """
        global model_manager
        model_manager = ModelManager(models_dir)
        self.mcp = mcp
        
    async def run(self, host: str = "localhost", port: int = 8000):
        """Run the MCP server.
        
        Args:
            host: Host to bind the server (not used in FastMCP STDIO mode)
            port: Port to bind the server (not used in FastMCP STDIO mode)
        """
        logger.info(f"Starting Neural Network MCP Server")
        logger.info(f"Models will be saved to: ./trained_model")
        # FastMCP typically runs in STDIO mode for MCP protocol
        await self.mcp.run() # type: ignore


# Factory function for creating the server
def create_neural_network_server(models_dir: str = "./trained_model") -> NeuralNetworkMCPServer:
    """Create a Neural Network MCP Server instance.
    
    Args:
        models_dir: Directory to store trained models (default: './trained_model')
        
    Returns:
        Configured NeuralNetworkMCPServer instance
    """
    return NeuralNetworkMCPServer(models_dir)