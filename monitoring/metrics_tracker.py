"""Comprehensive metrics tracking for distributed ML training."""
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import time
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track training metrics with support for MLflow, Prometheus, and W&B.
    
    Provides unified interface for logging metrics, artifacts, and model checkpoints
    across multiple monitoring backends.
    """
    
    def __init__(self, 
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 enable_mlflow: bool = False,
                 enable_prometheus: bool = False,
                 enable_wandb: bool = False,
                 log_dir: str = "logs",
                 tags: Optional[Dict[str, str]] = None):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of this specific run
            enable_mlflow: Enable MLflow tracking
            enable_prometheus: Enable Prometheus metrics
            enable_wandb: Enable Weights & Biases tracking
            log_dir: Directory for local logs
            tags: Additional tags for the run
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_mlflow = enable_mlflow
        self.enable_prometheus = enable_prometheus
        self.enable_wandb = enable_wandb
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tags = tags or {}
        
        # Initialize backends
        self.mlflow_run = None
        self.prom_metrics = {}
        self.wandb_run = None
        
        # Local metrics storage
        self.metrics_history = []
        self.start_time = time.time()
        
        self._initialize_backends()
        logger.info(f"MetricsTracker initialized: {experiment_name}/{self.run_name}")
    
    def _initialize_backends(self):
        """Initialize enabled monitoring backends."""
        # MLflow initialization
        if self.enable_mlflow:
            try:
                import mlflow
                mlflow.set_experiment(self.experiment_name)
                self.mlflow_run = mlflow.start_run(run_name=self.run_name)
                
                # Log tags
                for key, value in self.tags.items():
                    mlflow.set_tag(key, value)
                
                logger.info("MLflow tracking enabled")
            except ImportError:
                logger.warning("MLflow not installed. Install with: pip install mlflow")
                self.enable_mlflow = False
            except Exception as e:
                logger.error(f"Failed to initialize MLflow: {e}")
                self.enable_mlflow = False
        
        # Prometheus initialization
        if self.enable_prometheus:
            try:
                import prometheus_client as prom
                
                # Define Prometheus metrics
                self.prom_metrics = {
                    'training_loss': prom.Gauge('training_loss', 'Training loss'),
                    'validation_loss': prom.Gauge('validation_loss', 'Validation loss'),
                    'training_accuracy': prom.Gauge('training_accuracy', 'Training accuracy'),
                    'validation_accuracy': prom.Gauge('validation_accuracy', 'Validation accuracy'),
                    'learning_rate': prom.Gauge('learning_rate', 'Current learning rate'),
                    'epoch_time': prom.Histogram('epoch_time_seconds', 'Epoch time in seconds'),
                    'gpu_memory_allocated': prom.Gauge('gpu_memory_allocated_mb', 'GPU memory allocated in MB'),
                    'gpu_memory_cached': prom.Gauge('gpu_memory_cached_mb', 'GPU memory cached in MB'),
                    'batch_processing_time': prom.Histogram('batch_processing_time_seconds', 'Batch processing time'),
                }
                
                logger.info("Prometheus metrics enabled")
            except ImportError:
                logger.warning("Prometheus client not installed. Install with: pip install prometheus-client")
                self.enable_prometheus = False
            except Exception as e:
                logger.error(f"Failed to initialize Prometheus: {e}")
                self.enable_prometheus = False
        
        # Weights & Biases initialization
        if self.enable_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.experiment_name,
                    name=self.run_name,
                    tags=list(self.tags.values()),
                    config=self.tags
                )
                logger.info("W&B tracking enabled")
            except ImportError:
                logger.warning("Weights & Biases not installed. Install with: pip install wandb")
                self.enable_wandb = False
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                self.enable_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch number
            prefix: Optional prefix for metric names
        """
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Add timestamp
        metrics['timestamp'] = time.time() - self.start_time
        metrics['step'] = step
        
        # Store locally
        self.metrics_history.append(metrics.copy())
        
        # Log to MLflow
        if self.enable_mlflow:
            try:
                import mlflow
                mlflow.log_metrics({k: v for k, v in metrics.items() if k not in ['timestamp', 'step']}, step=step)
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {e}")
        
        # Log to Prometheus
        if self.enable_prometheus:
            try:
                for key, value in metrics.items():
                    if key in self.prom_metrics and isinstance(value, (int, float)):
                        self.prom_metrics[key].set(value)
            except Exception as e:
                logger.error(f"Failed to log to Prometheus: {e}")
        
        # Log to W&B
        if self.enable_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log to W&B: {e}")
        
        logger.debug(f"Logged metrics at step {step}: {metrics}")
    
    def log_model(self, model, model_path: Path, artifacts: Optional[Dict[str, Path]] = None):
        """
        Log model and associated artifacts.
        
        Args:
            model: The model object (PyTorch model, sklearn model, etc.)
            model_path: Path where model is saved
            artifacts: Dictionary of artifact names and paths
        """
        # Log to MLflow
        if self.enable_mlflow:
            try:
                import mlflow
                import torch
                
                if isinstance(model, torch.nn.Module):
                    mlflow.pytorch.log_model(model, "model")
                else:
                    # For sklearn models or others
                    mlflow.sklearn.log_model(model, "model")
                
                # Log artifacts
                if artifacts:
                    for name, path in artifacts.items():
                        if path.exists():
                            mlflow.log_artifact(str(path), artifact_path=name)
                
                logger.info("Model logged to MLflow")
            except Exception as e:
                logger.error(f"Failed to log model to MLflow: {e}")
        
        # Log to W&B
        if self.enable_wandb:
            try:
                import wandb
                
                # Save model as artifact
                artifact = wandb.Artifact(
                    name=f"model_{self.run_name}",
                    type="model",
                    description=f"Model from {self.experiment_name}"
                )
                artifact.add_file(str(model_path))
                
                # Add additional artifacts
                if artifacts:
                    for name, path in artifacts.items():
                        if path.exists():
                            artifact.add_file(str(path), name=name)
                
                self.wandb_run.log_artifact(artifact)
                logger.info("Model logged to W&B")
            except Exception as e:
                logger.error(f"Failed to log model to W&B: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log hyperparameters and configuration.
        
        Args:
            params: Dictionary of parameters
        """
        # Log to MLflow
        if self.enable_mlflow:
            try:
                import mlflow
                for key, value in params.items():
                    mlflow.log_param(key, value)
            except Exception as e:
                logger.error(f"Failed to log parameters to MLflow: {e}")
        
        # Log to W&B
        if self.enable_wandb:
            try:
                import wandb
                wandb.config.update(params)
            except Exception as e:
                logger.error(f"Failed to log parameters to W&B: {e}")
        
        # Save locally
        param_file = self.log_dir / f"{self.run_name}_parameters.json"
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Parameters logged: {list(params.keys())}")
    
    def log_system_metrics(self, gpu_id: Optional[int] = 0):
        """
        Log system metrics (GPU usage, memory, etc.).
        
        Args:
            gpu_id: GPU device ID to monitor
        """
        try:
            import torch
            
            if torch.cuda.is_available():
                metrics = {
                    'gpu_memory_allocated_mb': torch.cuda.memory_allocated(gpu_id) / 1024**2,
                    'gpu_memory_cached_mb': torch.cuda.memory_reserved(gpu_id) / 1024**2,
                    'gpu_utilization': torch.cuda.utilization(gpu_id) if hasattr(torch.cuda, 'utilization') else 0,
                }
                
                # Log to Prometheus
                if self.enable_prometheus:
                    for key, value in metrics.items():
                        if key in self.prom_metrics:
                            self.prom_metrics[key].set(value)
                
                return metrics
        except Exception as e:
            logger.debug(f"Failed to log system metrics: {e}")
        
        return {}
    
    def save_metrics_history(self):
        """Save metrics history to local file."""
        history_file = self.log_dir / f"{self.run_name}_metrics_history.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Metrics history saved to {history_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics history: {e}")
    
    def finish(self):
        """Finalize tracking and close all backends."""
        # Save local metrics
        self.save_metrics_history()
        
        # End MLflow run
        if self.enable_mlflow and self.mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
                logger.info("MLflow run ended")
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")
        
        # Finish W&B run
        if self.enable_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error(f"Failed to finish W&B run: {e}")
        
        logger.info(f"MetricsTracker finished: {self.experiment_name}/{self.run_name}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
