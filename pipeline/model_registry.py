"""Model registry and metadata tracking.

Provides a unified interface for logging models with rich metadata including:
- Config hash (from ConfigManager)
- Data version hash (from DataVersionStore)
- Git commit SHA (if available)
- Training metrics
- Model artifacts

Integrates with MLflow when available, otherwise stores metadata locally.
"""
import logging
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Container for model metadata."""
    
    def __init__(
        self,
        model_name: str,
        version: str,
        config_hash: Optional[str] = None,
        data_hash: Optional[str] = None,
        git_commit: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ):
        self.model_name = model_name
        self.version = version
        self.config_hash = config_hash
        self.data_hash = data_hash
        self.git_commit = git_commit or self._get_git_commit()
        self.metrics = metrics or {}
        self.tags = tags or {}
        self.description = description
        self.timestamp = int(time.time())
        self.metadata_hash = self._compute_metadata_hash()
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit SHA."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _compute_metadata_hash(self) -> str:
        """Compute hash of core metadata for versioning."""
        data = {
            'model_name': self.model_name,
            'config_hash': self.config_hash,
            'data_hash': self.data_hash,
            'git_commit': self.git_commit,
        }
        serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'version': self.version,
            'config_hash': self.config_hash,
            'data_hash': self.data_hash,
            'git_commit': self.git_commit,
            'metrics': self.metrics,
            'tags': self.tags,
            'description': self.description,
            'timestamp': self.timestamp,
            'metadata_hash': self.metadata_hash,
        }


class ModelRegistry:
    """Model registry with local and optional MLflow backend.
    
    This registry stores model metadata and artifacts. If MLflow is available
    and configured, it will also log to MLflow model registry.
    """
    
    def __init__(
        self,
        registry_dir: Path = Path('model_registry'),
        use_mlflow: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        self._mlflow = None
        
        if use_mlflow:
            try:
                import mlflow
                self._mlflow = mlflow
                if mlflow_tracking_uri:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info("MLflow integration enabled")
            except ImportError:
                logger.warning("MLflow not available, using local registry only")
                self.use_mlflow = False
    
    def register_model(
        self,
        model,
        metadata: ModelMetadata,
        model_path: Optional[Path] = None,
        artifacts: Optional[Dict[str, Path]] = None,
    ) -> str:
        """Register a model with metadata.
        
        Args:
            model: Model object (sklearn, pytorch, etc.)
            metadata: ModelMetadata instance
            model_path: Optional path where model is saved
            artifacts: Optional dict of artifact name -> path
            
        Returns:
            Model registry ID (metadata hash)
        """
        registry_id = metadata.metadata_hash
        model_dir = self.registry_dir / registry_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata locally
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save artifacts locally
        if artifacts:
            artifacts_dir = model_dir / 'artifacts'
            artifacts_dir.mkdir(exist_ok=True)
            for name, path in artifacts.items():
                if path.exists():
                    import shutil
                    dest = artifacts_dir / path.name
                    shutil.copy2(path, dest)
        
        logger.info(f"Model registered locally: {registry_id} at {model_dir}")
        
        # Log to MLflow if available
        if self.use_mlflow and self._mlflow:
            try:
                self._log_to_mlflow(model, metadata, model_path, artifacts)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        return registry_id
    
    def _log_to_mlflow(
        self,
        model,
        metadata: ModelMetadata,
        model_path: Optional[Path],
        artifacts: Optional[Dict[str, Path]],
    ):
        """Log model and metadata to MLflow."""
        with self._mlflow.start_run(run_name=f"{metadata.model_name}_{metadata.version}"):
            # Log metadata as tags
            self._mlflow.set_tag('model_name', metadata.model_name)
            self._mlflow.set_tag('version', metadata.version)
            if metadata.config_hash:
                self._mlflow.set_tag('config_hash', metadata.config_hash)
            if metadata.data_hash:
                self._mlflow.set_tag('data_hash', metadata.data_hash)
            if metadata.git_commit:
                self._mlflow.set_tag('git_commit', metadata.git_commit)
            if metadata.description:
                self._mlflow.set_tag('description', metadata.description)
            
            for key, value in metadata.tags.items():
                self._mlflow.set_tag(key, value)
            
            # Log metrics
            for key, value in metadata.metrics.items():
                self._mlflow.log_metric(key, value)
            
            # Log model
            try:
                import torch
                if isinstance(model, torch.nn.Module):
                    self._mlflow.pytorch.log_model(model, 'model')
                else:
                    self._mlflow.sklearn.log_model(model, 'model')
            except Exception:
                if model_path and model_path.exists():
                    self._mlflow.log_artifact(str(model_path), 'model')
            
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if path.exists():
                        self._mlflow.log_artifact(str(path), f'artifacts/{name}')
            
            logger.info(f"Model logged to MLflow: {metadata.model_name}")
    
    def get_model_metadata(self, registry_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by registry ID."""
        metadata_path = self.registry_dir / registry_id / 'metadata.json'
        if not metadata_path.exists():
            return None
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        return ModelMetadata(
            model_name=data['model_name'],
            version=data['version'],
            config_hash=data.get('config_hash'),
            data_hash=data.get('data_hash'),
            git_commit=data.get('git_commit'),
            metrics=data.get('metrics', {}),
            tags=data.get('tags', {}),
            description=data.get('description'),
        )
    
    def list_models(self) -> list:
        """List all registered models."""
        models = []
        for model_dir in self.registry_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        data = json.load(f)
                    models.append(data)
        return models
    
    def get_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of a model by name."""
        models = [m for m in self.list_models() if m['model_name'] == model_name]
        if not models:
            return None
        return max(models, key=lambda m: m['timestamp'])


def create_model_metadata_from_training(
    model_name: str,
    version: str,
    config_manager,
    data_version_record: Dict[str, Any],
    metrics: Dict[str, float],
    description: Optional[str] = None,
) -> ModelMetadata:
    """Helper to create ModelMetadata from training context.
    
    Args:
        model_name: Name of the model
        version: Model version
        config_manager: ConfigManager instance
        data_version_record: Record from DataLineage.build_lineage_record
        metrics: Training metrics dict
        description: Optional model description
        
    Returns:
        ModelMetadata instance
    """
    # Get config hash
    config_hash = None
    try:
        import yaml
        config_dict = config_manager.get_all_configs()
        serialized = yaml.dump(config_dict, sort_keys=True).encode('utf-8')
        config_hash = hashlib.sha256(serialized).hexdigest()[:16]
    except Exception:
        pass
    
    return ModelMetadata(
        model_name=model_name,
        version=version,
        config_hash=config_hash,
        data_hash=data_version_record.get('data_hash'),
        metrics=metrics,
        description=description,
    )
