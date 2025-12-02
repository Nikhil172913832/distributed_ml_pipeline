"""Tests for model registry and metadata tracking."""
import pytest
from pathlib import Path
import json
from pipeline.model_registry import ModelMetadata, ModelRegistry, create_model_metadata_from_training


def test_model_metadata_creation():
    """Test ModelMetadata creation and hash computation."""
    metadata = ModelMetadata(
        model_name='test_model',
        version='1.0',
        config_hash='abc123',
        data_hash='def456',
        metrics={'accuracy': 0.95},
        tags={'experiment': 'test'},
        description='Test model'
    )
    
    assert metadata.model_name == 'test_model'
    assert metadata.version == '1.0'
    assert metadata.metadata_hash is not None
    assert len(metadata.metadata_hash) == 16
    
    metadata_dict = metadata.to_dict()
    assert 'model_name' in metadata_dict
    assert 'timestamp' in metadata_dict


def test_model_registry_local(tmp_path):
    """Test local model registry without MLflow."""
    registry = ModelRegistry(registry_dir=tmp_path / 'registry', use_mlflow=False)
    
    # Create dummy model
    class DummyModel:
        pass
    
    model = DummyModel()
    metadata = ModelMetadata(
        model_name='dummy',
        version='1.0',
        config_hash='test123',
        metrics={'loss': 0.1}
    )
    
    # Register model
    registry_id = registry.register_model(model, metadata)
    
    assert registry_id is not None
    assert (tmp_path / 'registry' / registry_id / 'metadata.json').exists()
    
    # Retrieve metadata
    retrieved = registry.get_model_metadata(registry_id)
    assert retrieved is not None
    assert retrieved.model_name == 'dummy'
    assert retrieved.config_hash == 'test123'


def test_model_registry_list_models(tmp_path):
    """Test listing registered models."""
    registry = ModelRegistry(registry_dir=tmp_path / 'registry')
    
    # Register multiple models
    for i in range(3):
        model = None
        metadata = ModelMetadata(
            model_name=f'model_{i}',
            version='1.0',
            metrics={'score': float(i)}
        )
        registry.register_model(model, metadata)
    
    models = registry.list_models()
    assert len(models) == 3


def test_get_latest_model(tmp_path):
    """Test getting latest version of a model."""
    registry = ModelRegistry(registry_dir=tmp_path / 'registry')
    
    # Register multiple versions
    import time
    for version in ['1.0', '1.1', '1.2']:
        metadata = ModelMetadata(
            model_name='test_model',
            version=version,
            metrics={'score': 0.9}
        )
        registry.register_model(None, metadata)
        time.sleep(0.01)  # Ensure different timestamps
    
    latest = registry.get_latest_model('test_model')
    assert latest is not None
    assert latest['version'] == '1.2'


def test_model_metadata_with_artifacts(tmp_path):
    """Test registering model with artifacts."""
    registry = ModelRegistry(registry_dir=tmp_path / 'registry')
    
    # Create dummy artifact
    artifact_path = tmp_path / 'test_artifact.txt'
    artifact_path.write_text('test data')
    
    metadata = ModelMetadata(
        model_name='model_with_artifacts',
        version='1.0'
    )
    
    registry_id = registry.register_model(
        None,
        metadata,
        artifacts={'test_file': artifact_path}
    )
    
    # Check artifact was copied
    artifacts_dir = tmp_path / 'registry' / registry_id / 'artifacts'
    assert artifacts_dir.exists()
    assert (artifacts_dir / 'test_artifact.txt').exists()


class MockConfigManager:
    """Mock config manager for testing."""
    def get_all_configs(self):
        return {'model': {'name': 'test'}, 'training': {'epochs': 10}}


def test_create_metadata_from_training():
    """Test helper function for creating metadata from training context."""
    config_manager = MockConfigManager()
    data_record = {'data_hash': 'abc123', 'rows': 1000}
    metrics = {'accuracy': 0.92, 'loss': 0.15}
    
    metadata = create_model_metadata_from_training(
        model_name='trained_model',
        version='1.0',
        config_manager=config_manager,
        data_version_record=data_record,
        metrics=metrics,
        description='Test training run'
    )
    
    assert metadata.model_name == 'trained_model'
    assert metadata.data_hash == 'abc123'
    assert metadata.metrics['accuracy'] == 0.92
    assert metadata.config_hash is not None
