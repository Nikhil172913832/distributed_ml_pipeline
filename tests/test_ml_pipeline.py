"""
Integration tests for the complete ML pipeline
"""

import pytest
import time
import uuid
from datetime import datetime
import pandas as pd
import numpy as np

from pipeline.database import (
    DatabaseManager, RawDataRepository, PreprocessedDataRepository,
    ModelRegistryRepository, PredictionRepository, 
    ModelPerformanceRepository, RetrainingTriggerRepository
)


@pytest.fixture(scope="module")
def db_manager():
    """Create database manager for tests"""
    manager = DatabaseManager()
    yield manager
    manager.close_all()


@pytest.fixture
def raw_repo(db_manager):
    return RawDataRepository(db_manager)


@pytest.fixture
def preprocessed_repo(db_manager):
    return PreprocessedDataRepository(db_manager)


@pytest.fixture
def model_registry_repo(db_manager):
    return ModelRegistryRepository(db_manager)


@pytest.fixture
def prediction_repo(db_manager):
    return PredictionRepository(db_manager)


@pytest.fixture
def performance_repo(db_manager):
    return ModelPerformanceRepository(db_manager)


@pytest.fixture
def retraining_repo(db_manager):
    return RetrainingTriggerRepository(db_manager)


class TestDataIngestionFlow:
    """Test data ingestion pipeline"""
    
    def test_raw_data_insertion(self, raw_repo):
        """Test inserting raw data"""
        batch_id = f"test_batch_{uuid.uuid4().hex[:8]}"
        
        samples = [
            {
                'batch_id': batch_id,
                'sample_index': i,
                'features': {f'feature_{j}': np.random.randn() for j in range(10)},
                'target': np.random.choice([-1, 1])
            }
            for i in range(5)
        ]
        
        count = raw_repo.insert_batch(samples)
        assert count == 5
        
        # Verify data
        batch_samples = raw_repo.get_batch_samples(batch_id)
        assert len(batch_samples) == 5
    
    def test_preprocessed_data_insertion(self, raw_repo, preprocessed_repo):
        """Test inserting preprocessed data"""
        batch_id = f"test_batch_{uuid.uuid4().hex[:8]}"
        
        # Insert raw data first
        raw_samples = [
            {
                'batch_id': batch_id,
                'sample_index': i,
                'features': {f'feature_{j}': np.random.randn() for j in range(10)},
                'target': -1
            }
            for i in range(3)
        ]
        raw_repo.insert_batch(raw_samples)
        
        # Get raw data IDs
        batch_samples = raw_repo.get_batch_samples(batch_id)
        
        # Create preprocessed samples
        preprocessed_samples = [
            {
                'raw_data_id': sample[0],  # ID from raw data
                'features': {f'feature_{j}': np.random.randn() for j in range(10)},
                'target': -1,
                'missing_count': 0,
                'imputation_applied': False,
                'feature_count': 10,
                'processing_duration_ms': 5.0
            }
            for sample in batch_samples
        ]
        
        count = preprocessed_repo.insert_batch(preprocessed_samples)
        assert count == 3


class TestModelRegistry:
    """Test model registry operations"""
    
    def test_register_model(self, model_registry_repo):
        """Test registering a new model"""
        model_name = "test_model"
        model_version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        model_id = model_registry_repo.register_model(
            model_name=model_name,
            model_version=model_version,
            model_type="logistic_regression",
            model_path=f"./models/test_model_{model_version}.joblib",
            hyperparameters={'C': 1.0, 'penalty': 'l2'},
            test_metrics={
                'accuracy': 0.89,
                'precision': 0.87,
                'recall': 0.85,
                'f1_score': 0.86,
                'roc_auc': 0.92
            },
            training_metadata={
                'dataset_size': 1000,
                'start_time': datetime.utcnow(),
                'end_time': datetime.utcnow(),
                'duration_ms': 5000
            },
            triggered_by='test'
        )
        
        assert model_id is not None
        
        # Verify model
        model_info = model_registry_repo.get_model_by_id(model_id)
        assert model_info is not None
        assert model_info[1] == model_name
    
    def test_activate_model(self, model_registry_repo):
        """Test activating a model"""
        # Register a model
        model_version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        model_id = model_registry_repo.register_model(
            model_name="test_model",
            model_version=model_version,
            model_type="random_forest",
            model_path=f"./models/test_{model_version}.joblib",
            test_metrics={'accuracy': 0.90, 'f1_score': 0.88}
        )
        
        # Activate it
        model_registry_repo.activate_model(model_id)
        
        # Verify it's active
        active_model = model_registry_repo.get_active_model()
        assert active_model is not None
        assert str(active_model[0]) == str(model_id)


class TestInferencePipeline:
    """Test inference pipeline"""
    
    def test_prediction_insertion(self, model_registry_repo, prediction_repo, preprocessed_repo, raw_repo):
        """Test inserting predictions"""
        # Register a model
        model_version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_id = model_registry_repo.register_model(
            model_name="test_model",
            model_version=model_version,
            model_type="logistic_regression",
            model_path=f"./models/test_{model_version}.joblib",
            test_metrics={'accuracy': 0.85, 'f1_score': 0.83}
        )
        
        # Create some preprocessed data
        batch_id = f"test_batch_{uuid.uuid4().hex[:8]}"
        raw_samples = [
            {
                'batch_id': batch_id,
                'sample_index': i,
                'features': {f'feature_{j}': np.random.randn() for j in range(10)},
                'target': -1
            }
            for i in range(3)
        ]
        raw_repo.insert_batch(raw_samples)
        
        batch_samples = raw_repo.get_batch_samples(batch_id)
        preprocessed_samples = [
            {
                'raw_data_id': sample[0],
                'features': {f'feature_{j}': np.random.randn() for j in range(10)},
                'target': -1,
                'missing_count': 0,
                'imputation_applied': False,
                'feature_count': 10,
                'processing_duration_ms': 5.0
            }
            for sample in batch_samples
        ]
        preprocessed_repo.insert_batch(preprocessed_samples)
        
        # Create predictions
        predictions = []
        for sample in batch_samples:
            pred = {
                'preprocessed_data_id': str(sample[0]),
                'model_id': str(model_id),
                'prediction': -1,
                'prediction_probability': 0.85,
                'prediction_proba_pass': 0.85,
                'prediction_proba_fail': 0.15,
                'actual_target': -1,
                'is_correct': True,
                'confidence_score': 0.85,
                'uncertainty_score': 0.15,
                'inference_duration_ms': 2.5,
                'batch_id': batch_id
            }
            predictions.append(pred)
        
        count = prediction_repo.insert_predictions(predictions)
        assert count == 3
        
        # Verify predictions
        batch_predictions = prediction_repo.get_batch_predictions(batch_id)
        assert len(batch_predictions) == 3


class TestPerformanceMonitoring:
    """Test performance monitoring"""
    
    def test_calculate_performance_window(self, model_registry_repo, performance_repo, 
                                          prediction_repo, preprocessed_repo, raw_repo):
        """Test calculating performance metrics"""
        # Register and activate model
        model_version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_id = model_registry_repo.register_model(
            model_name="test_model",
            model_version=model_version,
            model_type="logistic_regression",
            model_path=f"./models/test_{model_version}.joblib",
            test_metrics={'accuracy': 0.85, 'f1_score': 0.83}
        )
        
        # Create test data with predictions
        batch_id = f"test_batch_{uuid.uuid4().hex[:8]}"
        raw_samples = [
            {
                'batch_id': batch_id,
                'sample_index': i,
                'features': {f'feature_{j}': np.random.randn() for j in range(10)},
                'target': np.random.choice([-1, 1])
            }
            for i in range(10)
        ]
        raw_repo.insert_batch(raw_samples)
        
        batch_samples = raw_repo.get_batch_samples(batch_id)
        preprocessed_samples = [
            {
                'raw_data_id': sample[0],
                'features': {f'feature_{j}': np.random.randn() for j in range(10)},
                'target': sample[4],
                'missing_count': 0,
                'imputation_applied': False,
                'feature_count': 10,
                'processing_duration_ms': 5.0
            }
            for sample in batch_samples
        ]
        preprocessed_repo.insert_batch(preprocessed_samples)
        
        # Create predictions (80% accurate)
        predictions = []
        for idx, sample in enumerate(batch_samples):
            actual = sample[4]
            pred = actual if idx < 8 else (-actual)  # 80% correct
            
            predictions.append({
                'preprocessed_data_id': str(sample[0]),
                'model_id': str(model_id),
                'prediction': pred,
                'prediction_probability': 0.75,
                'prediction_proba_pass': 0.75 if pred == -1 else 0.25,
                'prediction_proba_fail': 0.25 if pred == -1 else 0.75,
                'actual_target': actual,
                'is_correct': (pred == actual),
                'confidence_score': 0.75,
                'uncertainty_score': 0.25,
                'inference_duration_ms': 2.5,
                'batch_id': batch_id
            })
        
        prediction_repo.insert_predictions(predictions)
        
        # Calculate performance
        from datetime import timedelta
        now = datetime.utcnow()
        metric_id = performance_repo.calculate_performance_window(
            model_id=model_id,
            window_start=now - timedelta(hours=1),
            window_end=now,
            window_type='hourly'
        )
        
        assert metric_id is not None
        
        # Verify performance
        perf = performance_repo.get_latest_performance(model_id, 'hourly')
        assert perf is not None
        assert perf[0] == 0.8  # 80% accuracy


class TestContinuousLearning:
    """Test continuous learning workflow"""
    
    def test_retraining_trigger_creation(self, retraining_repo, model_registry_repo):
        """Test creating retraining triggers"""
        # Register a model
        model_version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_id = model_registry_repo.register_model(
            model_name="test_model",
            model_version=model_version,
            model_type="logistic_regression",
            model_path=f"./models/test_{model_version}.joblib",
            test_metrics={'accuracy': 0.85, 'f1_score': 0.83}
        )
        
        # Create trigger
        trigger_id = retraining_repo.create_trigger(
            trigger_type='performance_degradation',
            trigger_reason='Accuracy dropped below threshold',
            model_id=model_id,
            performance_data={
                'threshold': 0.85,
                'current_value': 0.75
            }
        )
        
        assert trigger_id is not None
        
        # Verify pending triggers
        pending = retraining_repo.get_pending_triggers()
        assert len(pending) > 0
    
    def test_retraining_trigger_update(self, retraining_repo):
        """Test updating retraining trigger status"""
        # Create trigger
        trigger_id = retraining_repo.create_trigger(
            trigger_type='manual',
            trigger_reason='Test trigger'
        )
        
        # Update to in_progress
        retraining_repo.update_trigger_status(trigger_id, 'in_progress')
        
        # Update to completed
        retraining_repo.update_trigger_status(trigger_id, 'completed')


class TestEndToEndPipeline:
    """End-to-end pipeline tests"""
    
    def test_complete_ml_pipeline(self, raw_repo, preprocessed_repo, model_registry_repo,
                                  prediction_repo, performance_repo):
        """Test complete ML pipeline flow"""
        batch_id = f"e2e_test_{uuid.uuid4().hex[:8]}"
        
        # Step 1: Ingest raw data
        raw_samples = [
            {
                'batch_id': batch_id,
                'sample_index': i,
                'features': {f'feature_{j}': float(np.random.randn()) for j in range(20)},
                'target': int(np.random.choice([-1, 1]))
            }
            for i in range(50)
        ]
        
        raw_count = raw_repo.insert_batch(raw_samples)
        assert raw_count == 50
        
        # Step 2: Preprocess data
        batch_samples = raw_repo.get_batch_samples(batch_id)
        preprocessed_samples = [
            {
                'raw_data_id': sample[0],
                'features': {f'feature_{j}': float(np.random.randn()) for j in range(20)},
                'target': int(sample[4]),
                'missing_count': 0,
                'imputation_applied': False,
                'feature_count': 20,
                'processing_duration_ms': 3.5
            }
            for sample in batch_samples
        ]
        
        prep_count = preprocessed_repo.insert_batch(preprocessed_samples)
        assert prep_count == 50
        
        # Step 3: Register model
        model_version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        model_id = model_registry_repo.register_model(
            model_name="e2e_test_model",
            model_version=model_version,
            model_type="logistic_regression",
            model_path=f"./models/e2e_test_{model_version}.joblib",
            test_metrics={
                'accuracy': 0.88,
                'precision': 0.86,
                'recall': 0.84,
                'f1_score': 0.85,
                'roc_auc': 0.91
            }
        )
        
        model_registry_repo.activate_model(model_id)
        
        # Step 4: Make predictions
        predictions = []
        for sample in batch_samples:
            actual = int(sample[4])
            # Simulate 85% accuracy
            pred = actual if np.random.random() < 0.85 else -actual
            
            predictions.append({
                'preprocessed_data_id': str(sample[0]),
                'model_id': str(model_id),
                'prediction': int(pred),
                'prediction_probability': 0.80,
                'prediction_proba_pass': 0.80 if pred == -1 else 0.20,
                'prediction_proba_fail': 0.20 if pred == -1 else 0.80,
                'actual_target': actual,
                'is_correct': bool(pred == actual),
                'confidence_score': 0.80,
                'uncertainty_score': 0.20,
                'inference_duration_ms': 2.0,
                'batch_id': batch_id
            })
        
        pred_count = prediction_repo.insert_predictions(predictions)
        assert pred_count == 50
        
        # Step 5: Calculate performance
        from datetime import timedelta
        now = datetime.utcnow()
        metric_id = performance_repo.calculate_performance_window(
            model_id=model_id,
            window_start=now - timedelta(hours=1),
            window_end=now,
            window_type='hourly'
        )
        
        assert metric_id is not None
        
        # Verify we have metrics
        perf = performance_repo.get_latest_performance(model_id, 'hourly')
        assert perf is not None
        assert perf[4] == 50  # total_predictions
        
        print(f"\n✓ End-to-end test completed successfully!")
        print(f"  - Batch ID: {batch_id}")
        print(f"  - Raw samples: {raw_count}")
        print(f"  - Preprocessed samples: {prep_count}")
        print(f"  - Predictions: {pred_count}")
        print(f"  - Model accuracy: {perf[0]:.2%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
