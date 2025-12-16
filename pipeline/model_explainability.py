"""
Model Explainability using SHAP (SHapley Additive exPlanations).

Provides feature importance and prediction explanations for ML models
to improve interpretability and trust in model decisions.
"""

from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import json

try:
    import shap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Model explainability will be disabled.")

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Provides model explanations using SHAP values.
    
    Usage:
        explainer = ModelExplainer(model, X_train)
        
        # Get feature importance
        importance = explainer.get_feature_importance()
        
        # Explain single prediction
        explanation = explainer.explain_prediction(X_sample)
    """
    
    def __init__(
        self,
        model: Any,
        background_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        model_type: str = "tree"
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (sklearn, xgboost, etc.)
            background_data: Background dataset for SHAP (sample of training data)
            feature_names: List of feature names
            model_type: Type of model ("tree", "linear", "kernel")
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.enabled = SHAP_AVAILABLE
        
        if self.enabled:
            self._initialize_explainer()
        else:
            logger.warning("SHAP explainer disabled (SHAP not installed)")
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        try:
            if self.model_type == "tree":
                # For tree-based models (RandomForest, GradientBoosting, XGBoost)
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "linear":
                # For linear models (LogisticRegression, LinearRegression)
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
            elif self.model_type == "kernel":
                # Model-agnostic explainer (slower but works for any model)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(self.background_data, 100)
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            logger.info(f"Initialized {self.model_type} SHAP explainer")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.enabled = False
    
    def get_feature_importance(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        top_n: int = 20
    ) -> Dict[str, float]:
        """
        Get global feature importance using SHAP values.
        
        Args:
            X: Data to compute SHAP values on (uses background_data if None)
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature_name -> importance_score
        """
        if not self.enabled or self.explainer is None:
            return {}
        
        try:
            if X is None:
                X = self.background_data
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class output (take class 1 for binary classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dictionary
            if self.feature_names:
                importance = dict(zip(self.feature_names, mean_abs_shap))
            else:
                importance = {f"feature_{i}": val for i, val in enumerate(mean_abs_shap)}
            
            # Sort and return top N
            sorted_importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
            )
            
            return sorted_importance
        except Exception as e:
            logger.error(f"Failed to compute feature importance: {e}")
            return {}
    
    def explain_prediction(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X: Single sample to explain (1D array or single-row DataFrame)
            top_n: Number of top contributing features to return
            
        Returns:
            Dictionary with prediction explanation
        """
        if not self.enabled or self.explainer is None:
            return {"error": "SHAP explainer not available"}
        
        try:
            # Ensure X is 2D
            if isinstance(X, pd.Series):
                X = X.values.reshape(1, -1)
            elif isinstance(X, np.ndarray) and X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 for binary classification
            
            # Get SHAP values for this prediction
            sample_shap = shap_values[0] if shap_values.ndim > 1 else shap_values
            
            # Create feature contributions dictionary
            if self.feature_names:
                contributions = dict(zip(self.feature_names, sample_shap))
            else:
                contributions = {f"feature_{i}": val for i, val in enumerate(sample_shap)}
            
            # Sort by absolute contribution
            sorted_contributions = dict(
                sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
            )
            
            # Get base value (expected value)
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]  # Class 1 for binary
            
            return {
                "base_value": float(base_value),
                "prediction_value": float(base_value + sample_shap.sum()),
                "top_contributions": sorted_contributions,
                "total_effect": float(sample_shap.sum())
            }
        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return {"error": str(e)}
    
    def save_summary_plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        output_path: Path,
        max_display: int = 20
    ):
        """
        Save SHAP summary plot showing feature importance.
        
        Args:
            X: Data to compute SHAP values on
            output_path: Path to save plot
            max_display: Maximum number of features to display
        """
        if not self.enabled or self.explainer is None:
            logger.warning("Cannot create summary plot: SHAP not available")
            return
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved SHAP summary plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save summary plot: {e}")
    
    def save_force_plot(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        output_path: Path,
        sample_idx: int = 0
    ):
        """
        Save SHAP force plot for a single prediction.
        
        Args:
            X: Data containing the sample
            output_path: Path to save plot
            sample_idx: Index of sample to explain
        """
        if not self.enabled or self.explainer is None:
            logger.warning("Cannot create force plot: SHAP not available")
            return
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]
            
            # Create force plot
            shap.force_plot(
                base_value,
                shap_values[sample_idx],
                X[sample_idx] if isinstance(X, np.ndarray) else X.iloc[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved SHAP force plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save force plot: {e}")
    
    def export_feature_importance(self, output_path: Path, top_n: int = 50):
        """Export feature importance to JSON file."""
        importance = self.get_feature_importance(top_n=top_n)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(importance, f, indent=2)
            logger.info(f"Exported feature importance to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export feature importance: {e}")


def explain_model_predictions(
    model: Any,
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    model_type: str = "tree",
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Comprehensive model explanation function.
    
    Args:
        model: Trained model
        X_train: Training data (for background)
        X_test: Test data (for explanations)
        feature_names: Feature names
        model_type: Model type for SHAP
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with feature importance and sample explanations
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not installed"}
    
    # Initialize explainer
    explainer = ModelExplainer(
        model=model,
        background_data=X_train[:100],  # Sample for efficiency
        feature_names=feature_names,
        model_type=model_type
    )
    
    # Get feature importance
    importance = explainer.get_feature_importance(X_test, top_n=20)
    
    # Explain a few sample predictions
    sample_explanations = []
    for i in range(min(5, len(X_test))):
        explanation = explainer.explain_prediction(X_test[i:i+1])
        sample_explanations.append(explanation)
    
    # Save plots if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        explainer.save_summary_plot(X_test, output_dir / "shap_summary.png")
        explainer.save_force_plot(X_test, output_dir / "shap_force_plot.png")
        explainer.export_feature_importance(output_dir / "feature_importance.json")
    
    return {
        "feature_importance": importance,
        "sample_explanations": sample_explanations
    }


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Explain model
    feature_names = [f"feature_{i}" for i in range(20)]
    results = explain_model_predictions(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        model_type="tree",
        output_dir=Path("./shap_output")
    )
    
    print("Top 10 Important Features:")
    for feature, importance in list(results["feature_importance"].items())[:10]:
        print(f"  {feature}: {importance:.4f}")
