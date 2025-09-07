"""
Neural Network HTML Report Generator

This module provides comprehensive HTML report generation for
neural network training processes and prediction results.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import os
import base64

logger = logging.getLogger(__name__)

class NeuralNetworkHTMLReportGenerator:
    """
    Generates detailed HTML reports for neural network training and prediction processes.
    
    Features:
    - Training process reports with hyperparameter optimization details
    - Cross-validation visualization
    - Training curves and learning progress
    - Prediction result reports with confidence analysis
    - Performance metrics and model architecture details
    - Interactive visualizations
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize NeuralNetworkHTMLReportGenerator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized NeuralNetworkHTMLReportGenerator with directory: {output_dir}")
        
    def generate_training_report(
        self,
        model_directory: str,
        training_results: Dict[str, Any],
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a comprehensive neural network training report.
        
        Args:
            model_directory: Path to model directory with training artifacts
            training_results: Training process results
            include_visualizations: Whether to include visualization references
            
        Returns:
            Path to generated HTML report file
        """
        try:
            logger.info("Generating HTML neural network training report...")
            
            # Create HTML content
            html_content = self._create_html_training_report(
                model_directory, training_results, include_visualizations
            )
            
            # Save to output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_id = training_results.get('model_id', 'unknown_model')
            filename = f"nn_training_report_{model_id}_{timestamp}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML training report saved to: {filepath}")
            return str(filepath)
                
        except Exception as e:
            logger.error(f"Error generating training report: {str(e)}")
            raise
    
    def generate_prediction_report(
        self,
        prediction_results: Dict[str, Any],
        model_info: Dict[str, Any],
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a comprehensive neural network prediction report.
        
        Args:
            prediction_results: Prediction process results
            model_info: Model metadata and information
            include_visualizations: Whether to include visualization references
            
        Returns:
            Path to generated HTML report file
        """
        try:
            logger.info("Generating HTML neural network prediction report...")
            
            # Create HTML content
            html_content = self._create_html_prediction_report(
                prediction_results, model_info, include_visualizations
            )
            
            # Save to output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_id = prediction_results.get('model_id', 'unknown_model')
            filename = f"nn_prediction_report_{model_id}_{timestamp}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML prediction report saved to: {filepath}")
            return str(filepath)
                
        except Exception as e:
            logger.error(f"Error generating prediction report: {str(e)}")
            raise

    def _create_html_training_report(
        self, 
        model_directory: str, 
        training_results: Dict[str, Any],
        include_visualizations: bool = True
    ) -> str:
        """Create enhanced HTML training report with neural network specifics."""
        
        model_dir_path = Path(model_directory)
        task_type = training_results.get('task_type', 'regression')
        model_id = training_results.get('model_id', 'unknown_model')
        
        # Generate visualization sections
        training_curves_section = self._generate_training_curves_section(model_dir_path) if include_visualizations else ""
        cv_visualization_section = self._generate_cv_visualization_section(model_dir_path, task_type) if include_visualizations else ""
        hyperparameter_section = self._generate_hyperparameter_optimization_section(model_dir_path)
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Training Report - {model_id}</title>
    {self._get_training_report_styles()}
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>🧠 Neural Network Training Report</h1>
            <p class="subtitle">Comprehensive Analysis of Model Training Process</p>
            <div class="header-meta">
                <span class="task-type {task_type}">{task_type.title()}</span>
                <span class="model-id">Model ID: {model_id}</span>
            </div>
        </header>
        
        {self._generate_training_summary_section(training_results)}
        {self._generate_architecture_section(training_results)}
        {hyperparameter_section}
        {training_curves_section}
        {cv_visualization_section}
        {self._generate_performance_metrics_section(training_results, task_type)}
        {self._generate_training_recommendations_section(training_results)}
        
        <footer class="report-footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>MCP Neural Network Tool - Training Analysis Report v1.0.0</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html_template

    def _create_html_prediction_report(
        self, 
        prediction_results: Dict[str, Any],
        model_info: Dict[str, Any],
        include_visualizations: bool = True
    ) -> str:
        """Create enhanced HTML prediction report with neural network specifics."""
        
        task_type = model_info.get('task_type', 'regression')
        model_id = prediction_results.get('model_id', 'unknown_model')
        
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Prediction Report - {model_id}</title>
    {self._get_prediction_report_styles()}
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>🔮 Neural Network Prediction Report</h1>
            <p class="subtitle">Comprehensive Analysis of Model Predictions</p>
            <div class="header-meta">
                <span class="task-type {task_type}">{task_type.title()}</span>
                <span class="model-id">Model ID: {model_id}</span>
            </div>
        </header>
        
        {self._generate_prediction_summary_section(prediction_results, model_info)}
        {self._generate_model_info_section(model_info)}
        {self._generate_prediction_statistics_section(prediction_results, task_type)}
        {self._generate_prediction_distribution_section(prediction_results, task_type)}
        {self._generate_prediction_details_section(prediction_results)}
        {self._generate_prediction_recommendations_section(prediction_results, model_info)}
        
        <footer class="report-footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
            <p>MCP Neural Network Tool - Prediction Analysis Report v1.0.0</p>
        </footer>
    </div>
</body>
</html>"""
        
        return html_template

    def _get_training_report_styles(self) -> str:
        """Get CSS styles for neural network training report."""
        return """
    <style>
        /* Base Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        /* Header Styles */
        .report-header {
            text-align: center;
            padding: 30px 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        
        .report-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-bottom: 20px;
        }
        
        .header-meta {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .task-type {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .task-type.regression {
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
        }
        
        .task-type.classification {
            background: rgba(156, 39, 176, 0.2);
            border: 1px solid #9C27B0;
        }
        
        .model-id {
            padding: 5px 15px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            font-size: 0.9em;
        }
        
        /* Section Styles */
        .report-section {
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
        }
        
        .section-title {
            font-size: 1.5em;
            color: #1e3c72;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2a5298;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Table Styles */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .data-table th,
        .data-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .data-table th {
            background: #2a5298;
            color: white;
            font-weight: 600;
        }
        
        .data-table tr:hover {
            background: #f5f5f5;
        }
        
        /* Visualization Styles */
        .visualization-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .visualization-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .visualization-caption {
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        
        /* Progress Bar */
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        /* Architecture Display */
        .architecture-diagram {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .layer-info {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .layer-info:last-child {
            border-bottom: none;
        }
        
        /* Recommendations */
        .recommendation-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #FF9800;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .recommendation-title {
            font-weight: bold;
            color: #FF9800;
            margin-bottom: 5px;
        }
        
        /* Footer */
        .report-footer {
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #eee;
            margin-top: 40px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 15px;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
            
            .header-meta {
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>"""

    def _get_prediction_report_styles(self) -> str:
        """Get CSS styles for neural network prediction report."""
        # Similar styling but with prediction-specific colors
        return self._get_training_report_styles().replace(
            "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)",
            "linear-gradient(135deg, #2E7D32 0%, #388E3C 100%)"
        ).replace("#2a5298", "#388E3C").replace("#1e3c72", "#2E7D32")

    def _generate_training_summary_section(self, training_results: Dict[str, Any]) -> str:
        """Generate training summary section."""
        summary = training_results.get('training_summary', {})
        
        # Extract key metrics
        total_time = summary.get('total_time', 0)
        n_trials = summary.get('n_trials', 0)
        cv_folds = summary.get('cv_folds', 0)
        num_epochs = summary.get('num_epochs', 0)
        algorithm = summary.get('algorithm', 'Unknown')
        
        # Performance metrics
        if training_results.get('task_type') == 'classification':
            best_metric = training_results.get('best_accuracy', 0)
            metric_name = "Best Accuracy"
            metric_format = f"{best_metric:.3f}"
        else:
            best_metric = training_results.get('best_mae', float('inf'))
            metric_name = "Best MAE"
            metric_format = f"{best_metric:.6f}"
        
        return f"""
        <div class="report-section">
            <h2 class="section-title">📊 Training Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{metric_format}</div>
                    <div class="metric-label">{metric_name}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_time:.1f}s</div>
                    <div class="metric-label">Total Training Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{n_trials}</div>
                    <div class="metric-label">Optimization Trials</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{cv_folds}</div>
                    <div class="metric-label">CV Folds</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{num_epochs}</div>
                    <div class="metric-label">Epochs per Fold</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{algorithm}</div>
                    <div class="metric-label">Optimization Algorithm</div>
                </div>
            </div>
        </div>"""

    def _generate_architecture_section(self, training_results: Dict[str, Any]) -> str:
        """Generate neural network architecture section."""
        best_params = training_results.get('best_parameters', {})
        feature_names = training_results.get('feature_names', [])
        target_names = training_results.get('target_names', [])
        
        # Extract architecture details
        hidden_layers = best_params.get('hidden_layers', [])
        dropout_rate = best_params.get('dropout_rate', 0.0)
        learning_rate = best_params.get('learning_rate', 0.001)
        batch_size = best_params.get('batch_size', 32)
        activation = best_params.get('activation', 'ReLU')
        
        # Create architecture visualization
        architecture_html = f"""
        <div class="architecture-diagram">
            <div class="layer-info">
                <span><strong>Input Layer</strong></span>
                <span>{len(feature_names)} features</span>
            </div>"""
        
        for i, layer_size in enumerate(hidden_layers):
            architecture_html += f"""
            <div class="layer-info">
                <span><strong>Hidden Layer {i+1}</strong></span>
                <span>{layer_size} neurons ({activation})</span>
            </div>"""
        
        architecture_html += f"""
            <div class="layer-info">
                <span><strong>Output Layer</strong></span>
                <span>{len(target_names)} outputs</span>
            </div>
        </div>"""
        
        return f"""
        <div class="report-section">
            <h2 class="section-title">🏗️ Model Architecture</h2>
            {architecture_html}
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{dropout_rate:.3f}</div>
                    <div class="metric-label">Dropout Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{learning_rate:.6f}</div>
                    <div class="metric-label">Learning Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{batch_size}</div>
                    <div class="metric-label">Batch Size</div>
                </div>
            </div>
        </div>"""

    def _generate_hyperparameter_optimization_section(self, model_dir_path: Path) -> str:
        """Generate hyperparameter optimization section."""
        trials_file = model_dir_path / "hyperparameter_optimization_trials.csv"
        
        if not trials_file.exists():
            return """
            <div class="report-section">
                <h2 class="section-title">🔍 Hyperparameter Optimization</h2>
                <p>No hyperparameter optimization data available.</p>
            </div>"""
        
        try:
            import pandas as pd
            trials_df = pd.read_csv(trials_file)
            
            # Get best trials
            best_trials = trials_df.nsmallest(5, 'value')[['number', 'value', 'params_hidden_layers', 'params_learning_rate', 'params_dropout_rate']]
            
            table_html = """
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Trial</th>
                        <th>Score</th>
                        <th>Hidden Layers</th>
                        <th>Learning Rate</th>
                        <th>Dropout Rate</th>
                    </tr>
                </thead>
                <tbody>"""
            
            for _, row in best_trials.iterrows():
                table_html += f"""
                    <tr>
                        <td>#{int(row['number'])}</td>
                        <td>{row['value']:.6f}</td>
                        <td>{row.get('params_hidden_layers', 'N/A')}</td>
                        <td>{row.get('params_learning_rate', 'N/A')}</td>
                        <td>{row.get('params_dropout_rate', 'N/A')}</td>
                    </tr>"""
            
            table_html += """
                </tbody>
            </table>"""
            
            return f"""
            <div class="report-section">
                <h2 class="section-title">🔍 Hyperparameter Optimization</h2>
                <p>Top 5 performing hyperparameter combinations from {len(trials_df)} trials:</p>
                {table_html}
            </div>"""
            
        except Exception as e:
            logger.warning(f"Could not load hyperparameter optimization data: {e}")
            return """
            <div class="report-section">
                <h2 class="section-title">🔍 Hyperparameter Optimization</h2>
                <p>Could not load hyperparameter optimization data.</p>
            </div>"""

    def _generate_training_curves_section(self, model_dir_path: Path) -> str:
        """Generate training curves visualization section."""
        training_curves_file = model_dir_path / "training_curves.png"
        
        if not training_curves_file.exists():
            return """
            <div class="report-section">
                <h2 class="section-title">📈 Training Curves</h2>
                <p>No training curves visualization available.</p>
            </div>"""
        
        try:
            # Convert image to base64 for embedding
            with open(training_curves_file, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode()
            
            return f"""
            <div class="report-section">
                <h2 class="section-title">📈 Training Curves</h2>
                <div class="visualization-container">
                    <img src="data:image/png;base64,{img_data}" alt="Training Curves">
                    <p class="visualization-caption">Training and validation loss curves across epochs</p>
                </div>
            </div>"""
            
        except Exception as e:
            logger.warning(f"Could not load training curves: {e}")
            return """
            <div class="report-section">
                <h2 class="section-title">📈 Training Curves</h2>
                <p>Could not load training curves visualization.</p>
            </div>"""

    def _generate_cv_visualization_section(self, model_dir_path: Path, task_type: str) -> str:
        """Generate cross-validation visualization section."""
        if task_type == 'classification':
            # Look for ROC curves or confusion matrices
            roc_file = model_dir_path / "roc_curves.png"
            if roc_file.exists():
                try:
                    with open(roc_file, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    
                    return f"""
                    <div class="report-section">
                        <h2 class="section-title">🎯 Cross-Validation Results</h2>
                        <div class="visualization-container">
                            <img src="data:image/png;base64,{img_data}" alt="ROC Curves">
                            <p class="visualization-caption">ROC curves for cross-validation folds</p>
                        </div>
                    </div>"""
                except Exception as e:
                    logger.warning(f"Could not load ROC curves: {e}")
        else:
            # Look for scatter plots for regression
            scatter_files = list(model_dir_path.glob("cv_predictions_scatter*.png"))
            if scatter_files:
                try:
                    with open(scatter_files[0], 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    
                    return f"""
                    <div class="report-section">
                        <h2 class="section-title">🎯 Cross-Validation Results</h2>
                        <div class="visualization-container">
                            <img src="data:image/png;base64,{img_data}" alt="CV Predictions Scatter">
                            <p class="visualization-caption">Cross-validation predictions vs actual values</p>
                        </div>
                    </div>"""
                except Exception as e:
                    logger.warning(f"Could not load CV scatter plot: {e}")
        
        return """
        <div class="report-section">
            <h2 class="section-title">🎯 Cross-Validation Results</h2>
            <p>No cross-validation visualizations available.</p>
        </div>"""

    def _generate_performance_metrics_section(self, training_results: Dict[str, Any], task_type: str) -> str:
        """Generate performance metrics section."""
        if task_type == 'classification':
            cv_results = training_results.get('cv_results', {})
            fold_accuracies = cv_results.get('fold_accuracies', [])
            avg_accuracy = training_results.get('best_accuracy', 0)
            
            if fold_accuracies:
                min_acc = min(fold_accuracies)
                max_acc = max(fold_accuracies)
                std_acc = np.std(fold_accuracies) if len(fold_accuracies) > 1 else 0
                
                return f"""
                <div class="report-section">
                    <h2 class="section-title">📊 Performance Metrics</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{avg_accuracy:.3f}</div>
                            <div class="metric-label">Average Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{min_acc:.3f}</div>
                            <div class="metric-label">Min Fold Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{max_acc:.3f}</div>
                            <div class="metric-label">Max Fold Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{std_acc:.3f}</div>
                            <div class="metric-label">Std Deviation</div>
                        </div>
                    </div>
                </div>"""
        else:
            # Regression metrics
            best_mae = training_results.get('best_mae', float('inf'))
            
            return f"""
            <div class="report-section">
                <h2 class="section-title">📊 Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{best_mae:.6f}</div>
                        <div class="metric-label">Best MAE</div>
                    </div>
                </div>
            </div>"""

    def _generate_training_recommendations_section(self, training_results: Dict[str, Any]) -> str:
        """Generate training recommendations section."""
        recommendations = []
        
        # Analyze training results and generate recommendations
        best_mae = training_results.get('best_mae', float('inf'))
        total_time = training_results.get('training_summary', {}).get('total_time', 0)
        
        if best_mae > 0.1:
            recommendations.append({
                "title": "High Error Rate Detected",
                "content": "Consider increasing the number of trials or adjusting the hyperparameter search space."
            })
        
        if total_time < 60:
            recommendations.append({
                "title": "Quick Training Detected", 
                "content": "Training completed quickly. Consider increasing epochs or model complexity for potentially better performance."
            })
        
        if not recommendations:
            recommendations.append({
                "title": "Training Completed Successfully",
                "content": "The model training completed with good performance metrics."
            })
        
        recommendations_html = ""
        for rec in recommendations:
            recommendations_html += f"""
            <div class="recommendation-item">
                <div class="recommendation-title">{rec['title']}</div>
                <p>{rec['content']}</p>
            </div>"""
        
        return f"""
        <div class="report-section">
            <h2 class="section-title">💡 Recommendations</h2>
            {recommendations_html}
        </div>"""

    # Prediction report methods
    def _generate_prediction_summary_section(self, prediction_results: Dict[str, Any], model_info: Dict[str, Any]) -> str:
        """Generate prediction summary section."""
        num_predictions = prediction_results.get('num_predictions', 0)
        task_type = model_info.get('task_type', 'regression')
        
        return f"""
        <div class="report-section">
            <h2 class="section-title">📊 Prediction Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{num_predictions}</div>
                    <div class="metric-label">Total Predictions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{task_type.title()}</div>
                    <div class="metric-label">Task Type</div>
                </div>
            </div>
        </div>"""

    def _generate_model_info_section(self, model_info: Dict[str, Any]) -> str:
        """Generate model information section."""
        model_id = model_info.get('model_id', 'Unknown')
        feature_names = model_info.get('feature_names', [])
        target_names = model_info.get('target_names', [])
        
        return f"""
        <div class="report-section">
            <h2 class="section-title">🏗️ Model Information</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(feature_names)}</div>
                    <div class="metric-label">Input Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(target_names)}</div>
                    <div class="metric-label">Output Targets</div>
                </div>
            </div>
        </div>"""

    def _generate_prediction_statistics_section(self, prediction_results: Dict[str, Any], task_type: str) -> str:
        """Generate prediction statistics section."""
        predictions = prediction_results.get('predictions', [])
        
        if not predictions:
            return """
            <div class="report-section">
                <h2 class="section-title">📈 Prediction Statistics</h2>
                <p>No prediction statistics available.</p>
            </div>"""
        
        # Extract prediction values
        if task_type == 'regression':
            if isinstance(predictions[0], dict):
                pred_values = [list(pred.values())[0] if pred else 0 for pred in predictions]
            else:
                pred_values = predictions
            
            pred_array = np.array(pred_values)
            mean_pred = np.mean(pred_array)
            std_pred = np.std(pred_array)
            min_pred = np.min(pred_array)
            max_pred = np.max(pred_array)
            
            return f"""
            <div class="report-section">
                <h2 class="section-title">📈 Prediction Statistics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{mean_pred:.4f}</div>
                        <div class="metric-label">Mean Prediction</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{std_pred:.4f}</div>
                        <div class="metric-label">Std Deviation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{min_pred:.4f}</div>
                        <div class="metric-label">Min Prediction</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{max_pred:.4f}</div>
                        <div class="metric-label">Max Prediction</div>
                    </div>
                </div>
            </div>"""
        else:
            # Classification statistics
            return """
            <div class="report-section">
                <h2 class="section-title">📈 Prediction Statistics</h2>
                <p>Classification prediction statistics would be displayed here.</p>
            </div>"""

    def _generate_prediction_distribution_section(self, prediction_results: Dict[str, Any], task_type: str) -> str:
        """Generate prediction distribution section."""
        return """
        <div class="report-section">
            <h2 class="section-title">📊 Prediction Distribution</h2>
            <p>Prediction distribution visualization would be displayed here.</p>
        </div>"""

    def _generate_prediction_details_section(self, prediction_results: Dict[str, Any]) -> str:
        """Generate detailed prediction results section."""
        predictions = prediction_results.get('predictions', [])
        
        if not predictions:
            return """
            <div class="report-section">
                <h2 class="section-title">📋 Prediction Details</h2>
                <p>No prediction details available.</p>
            </div>"""
        
        # Show first few predictions
        table_html = """
        <table class="data-table">
            <thead>
                <tr>
                    <th>Index</th>
                    <th>Prediction</th>
                </tr>
            </thead>
            <tbody>"""
        
        for i, pred in enumerate(predictions[:10]):  # Show first 10
            if isinstance(pred, dict):
                pred_str = ", ".join([f"{k}: {v:.4f}" for k, v in pred.items()])
            else:
                pred_str = f"{pred:.4f}" if isinstance(pred, (int, float)) else str(pred)
            
            table_html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{pred_str}</td>
                </tr>"""
        
        table_html += """
            </tbody>
        </table>"""
        
        more_info = f"<p><em>Showing first 10 of {len(predictions)} predictions</em></p>" if len(predictions) > 10 else ""
        
        return f"""
        <div class="report-section">
            <h2 class="section-title">📋 Prediction Details</h2>
            {table_html}
            {more_info}
        </div>"""

    def _generate_prediction_recommendations_section(self, prediction_results: Dict[str, Any], model_info: Dict[str, Any]) -> str:
        """Generate prediction recommendations section."""
        return """
        <div class="report-section">
            <h2 class="section-title">💡 Recommendations</h2>
            <div class="recommendation-item">
                <div class="recommendation-title">Prediction Completed Successfully</div>
                <p>The model has generated predictions for all input samples. Review the prediction statistics and details above.</p>
            </div>
        </div>"""