"""
MLflow Integration for Experiment Tracking.

Provides utilities for tracking experiments, logging metrics, parameters,
and models using MLflow.
"""

from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
from contextlib import contextmanager
import json

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Experiment tracking will be disabled.")

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.
    
    Usage:
        tracker = MLflowTracker(experiment_name="secom_training")
        
        with tracker.start_run(run_name="random_forest_v1"):
            tracker.log_params({"n_estimators": 100, "max_depth": 10})
            tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
            tracker.log_model(model, "model")
    """
    
    def __init__(
        self,
        experiment_name: str = "secom_pipeline",
        tracking_uri: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            enabled: Whether tracking is enabled
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.active_run = None
        
        if self.enabled:
            self._initialize_mlflow()
        else:
            logger.info("MLflow tracking disabled")
    
    def _initialize_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created MLflow experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized: {self.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            self.enabled = False
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start an MLflow run (context manager).
        
        Args:
            run_name: Name for this run
            tags: Dictionary of tags to add to the run
            
        Yields:
            MLflow run object
        """
        if not self.enabled:
            yield None
            return
        
        try:
            with mlflow.start_run(run_name=run_name) as run:
                self.active_run = run
                
                # Set tags
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                yield run
        except Exception as e:
            logger.error(f"MLflow run failed: {e}")
            yield None
        finally:
            self.active_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters
        """
        if not self.enabled:
            return
        
        try:
            # Flatten nested dicts
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
        except Exception as e:
            logger.error(f"Failed to log params: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ):
        """
        Log model to MLflow.
        
        Args:
            model: Model object (sklearn, pytorch, etc.)
            artifact_path: Path within run artifacts
            registered_model_name: Name for model registry
        """
        if not self.enabled:
            return
        
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact file to MLflow.
        
        Args:
            local_path: Local file path
            artifact_path: Path within run artifacts
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_dict(self, dictionary: Dict[str, Any], filename: str):
        """
        Log dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            filename: Filename for the artifact
        """
        if not self.enabled:
            return
        
        try:
            mlflow.log_dict(dictionary, filename)
        except Exception as e:
            logger.error(f"Failed to log dict: {e}")
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for current run.
        
        Args:
            tags: Dictionary of tags
        """
        if not self.enabled:
            return
        
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")
    
    def log_training_run(
        self,
        model_type: str,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        cv_results: Optional[Dict[str, Any]] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ):
        """
        Log complete training run with all artifacts.
        
        Args:
            model_type: Type of model (e.g., "random_forest")
            model: Trained model object
            params: Model hyperparameters
            metrics: Evaluation metrics
            cv_results: Cross-validation results
            feature_importance: Feature importance scores
        """
        if not self.enabled:
            return
        
        run_name = f"{model_type}_{metrics.get('test_f1', 0):.4f}"
        
        with self.start_run(run_name=run_name, tags={"model_type": model_type}):
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log CV results if available
            if cv_results:
                cv_metrics = {
                    "cv_mean_score": cv_results.get("mean_test_score", [0])[0],
                    "cv_std_score": cv_results.get("std_test_score", [0])[0]
                }
                self.log_metrics(cv_metrics)
                self.log_dict(cv_results, "cv_results.json")
            
            # Log feature importance
            if feature_importance:
                self.log_dict(feature_importance, "feature_importance.json")
            
            # Log model
            self.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"secom_{model_type}"
            )
    
    def compare_runs(self, metric: str = "test_f1", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Compare runs by metric.
        
        Args:
            metric: Metric to compare by
            limit: Number of top runs to return
            
        Returns:
            List of run information dictionaries
        """
        if not self.enabled:
            return []
        
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                return []
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} DESC"],
                max_results=limit
            )
            
            return runs.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return []
    
    def get_best_run(self, metric: str = "test_f1") -> Optional[Dict[str, Any]]:
        """
        Get best run by metric.
        
        Args:
            metric: Metric to optimize
            
        Returns:
            Best run information
        """
        runs = self.compare_runs(metric=metric, limit=1)
        return runs[0] if runs else None
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to string if not a simple type
                if not isinstance(v, (str, int, float, bool)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)


# Convenience function for quick tracking
def track_experiment(
    experiment_name: str = "secom_pipeline",
    run_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    model: Optional[Any] = None,
    enabled: bool = True
):
    """
    Quick experiment tracking function.
    
    Args:
        experiment_name: Experiment name
        run_name: Run name
        params: Parameters to log
        metrics: Metrics to log
        model: Model to log
        enabled: Whether tracking is enabled
    """
    if not enabled or not MLFLOW_AVAILABLE:
        return
    
    tracker = MLflowTracker(experiment_name=experiment_name, enabled=enabled)
    
    with tracker.start_run(run_name=run_name):
        if params:
            tracker.log_params(params)
        if metrics:
            tracker.log_metrics(metrics)
        if model:
            tracker.log_model(model, "model")


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred)
    }
    
    # Track with MLflow
    tracker = MLflowTracker(experiment_name="example_experiment")
    tracker.log_training_run(
        model_type="random_forest",
        model=model,
        params={"n_estimators": 100, "max_depth": 10},
        metrics=metrics
    )
    
    print("Experiment tracked successfully!")
