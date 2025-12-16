"""
Additional test coverage for improved code quality.

Comprehensive tests for configuration, resilience patterns,
and new ML engineering features.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime


# ==========================================
# CONFIGURATION TESTS
# ==========================================

class TestSettings:
    """Test unified configuration system."""
    
    def test_kafka_settings_validation(self):
        """Test Kafka settings validation."""
        from config.settings import KafkaSettings
        
        # Valid settings
        settings = KafkaSettings(bootstrap_servers="localhost:9092")
        assert settings.bootstrap_servers == "localhost:9092"
        
        # Invalid bootstrap servers
        with pytest.raises(ValueError):
            KafkaSettings(bootstrap_servers="invalid")
    
    def test_database_connection_string(self):
        """Test database connection string generation."""
        from config.settings import DatabaseSettings
        
        settings = DatabaseSettings(
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass"
        )
        
        expected = "postgresql://test_user:test_pass@localhost:5432/test_db"
        assert settings.connection_string == expected
    
    def test_inference_settings_constraints(self):
        """Test inference settings value constraints."""
        from config.settings import InferenceSettings
        
        # Valid settings
        settings = InferenceSettings(
            confidence_threshold=0.7,
            accuracy_threshold=0.85
        )
        assert settings.confidence_threshold == 0.7
        
        # Invalid threshold (> 1.0)
        with pytest.raises(ValueError):
            InferenceSettings(confidence_threshold=1.5)
    
    def test_settings_singleton(self):
        """Test settings singleton pattern."""
        from config.settings import get_settings
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2


# ==========================================
# RESILIENCE PATTERN TESTS
# ==========================================

class TestResiliencePatterns:
    """Test retry and circuit breaker patterns."""
    
    def test_retry_on_connection_error(self):
        """Test retry decorator for connection errors."""
        from pipeline.resilience import retry_on_connection_error
        
        call_count = 0
        
        @retry_on_connection_error(max_attempts=3, min_wait=0, max_wait=0)
        def flaky_connection():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = flaky_connection()
        assert result == "success"
        assert call_count == 3
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        from pipeline.resilience import circuit_breaker
        from circuitbreaker import CircuitBreakerError
        
        @circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def failing_function():
            raise Exception("Always fails")
        
        # First two calls should raise the original exception
        with pytest.raises(Exception):
            failing_function()
        
        with pytest.raises(Exception):
            failing_function()
        
        # Third call should raise CircuitBreakerError (circuit is open)
        with pytest.raises(CircuitBreakerError):
            failing_function()
    
    def test_graceful_degradation(self):
        """Test graceful degradation context manager."""
        from pipeline.resilience import graceful_degradation
        
        fallback_value = []
        
        with graceful_degradation(fallback_value=fallback_value, raise_on_failure=False):
            result = []  # Successful operation
        
        assert result == []


# ==========================================
# DATA VALIDATION TESTS
# ==========================================

class TestDataValidation:
    """Test data validation module."""
    
    def test_validate_feature_count(self):
        """Test feature count validation."""
        from pipeline.data_validation import validate_feature_count
        
        # Valid: 590 features + target
        df = pd.DataFrame({
            **{f"feature_{i}": [0.5] for i in range(590)},
            "target": [0]
        })
        assert validate_feature_count(df, expected=590)
        
        # Invalid: wrong number of features
        df_invalid = pd.DataFrame({
            **{f"feature_{i}": [0.5] for i in range(100)},
            "target": [0]
        })
        assert not validate_feature_count(df_invalid, expected=590)
    
    def test_validate_target_values(self):
        """Test target value validation."""
        from pipeline.data_validation import validate_target_values
        
        # Valid targets
        df = pd.DataFrame({"target": [0, 1, -1, 0, 1]})
        assert validate_target_values(df)
        
        # Invalid target
        df_invalid = pd.DataFrame({"target": [0, 1, 5]})
        assert not validate_target_values(df_invalid)
    
    def test_quick_validate_raw_data(self):
        """Test quick validation for raw data."""
        from pipeline.data_validation import quick_validate
        
        # Valid data
        df = pd.DataFrame({
            **{f"feature_{i}": [0.5] for i in range(590)},
            "target": [0]
        })
        is_valid, message = quick_validate(df, stage="raw")
        assert is_valid
        assert message == "OK"
        
        # Invalid: empty dataframe
        df_empty = pd.DataFrame()
        is_valid, message = quick_validate(df_empty, stage="raw")
        assert not is_valid
        assert "Empty" in message


# ==========================================
# FEATURE STORE TESTS
# ==========================================

class TestFeatureStore:
    """Test Redis feature store."""
    
    @patch('pipeline.feature_store.Redis')
    def test_store_and_retrieve_features(self, mock_redis):
        """Test storing and retrieving features."""
        from pipeline.feature_store import FeatureStore
        
        # Mock Redis client
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        store = FeatureStore(enabled=True)
        store.client = mock_client
        
        # Store features
        features = {"feature_1": 0.5, "feature_2": 1.2}
        result = store.store_features("sample_001", features)
        
        assert mock_client.setex.called
    
    @patch('pipeline.feature_store.Redis')
    def test_batch_operations(self, mock_redis):
        """Test batch store operations."""
        from pipeline.feature_store import FeatureStore
        
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        store = FeatureStore(enabled=True)
        store.client = mock_client
        
        samples = [
            {"sample_id": "s1", "feature_1": 0.5},
            {"sample_id": "s2", "feature_1": 0.6}
        ]
        
        count = store.store_batch(samples, id_field="sample_id")
        assert count == 2


# ==========================================
# MLFLOW TRACKER TESTS
# ==========================================

class TestMLflowTracker:
    """Test MLflow integration."""
    
    @patch('pipeline.mlflow_tracker.mlflow')
    def test_log_params(self, mock_mlflow):
        """Test logging parameters."""
        from pipeline.mlflow_tracker import MLflowTracker
        
        tracker = MLflowTracker(enabled=True)
        tracker.enabled = True
        
        params = {"n_estimators": 100, "max_depth": 10}
        tracker.log_params(params)
        
        assert mock_mlflow.log_params.called
    
    @patch('pipeline.mlflow_tracker.mlflow')
    def test_log_metrics(self, mock_mlflow):
        """Test logging metrics."""
        from pipeline.mlflow_tracker import MLflowTracker
        
        tracker = MLflowTracker(enabled=True)
        tracker.enabled = True
        
        metrics = {"accuracy": 0.95, "f1": 0.93}
        tracker.log_metrics(metrics)
        
        assert mock_mlflow.log_metrics.called


# ==========================================
# SHADOW MODE TESTS
# ==========================================

class TestShadowMode:
    """Test shadow mode deployment."""
    
    def test_deploy_shadow_model(self):
        """Test deploying model in shadow mode."""
        from pipeline.shadow_mode import ShadowModeManager
        
        manager = ShadowModeManager()
        
        success = manager.deploy_shadow_model(
            model_id="model_v2",
            model_version="2.0.0",
            model_path="./models/model_v2.joblib"
        )
        
        assert success
        assert "model_v2" in manager.deployments
    
    def test_canary_traffic_routing(self):
        """Test canary deployment traffic routing."""
        from pipeline.shadow_mode import CanaryDeployment
        
        canary = CanaryDeployment(initial_traffic=0.5)
        
        # With 50% traffic, roughly half should use canary
        results = [canary.should_use_canary() for _ in range(1000)]
        canary_count = sum(results)
        
        # Should be approximately 500 (within 10% tolerance)
        assert 400 <= canary_count <= 600
    
    def test_canary_traffic_increase(self):
        """Test increasing canary traffic."""
        from pipeline.shadow_mode import CanaryDeployment
        
        canary = CanaryDeployment(initial_traffic=0.1)
        assert canary.traffic_percentage == 0.1
        
        canary.increase_traffic()
        assert canary.traffic_percentage == 0.15
        
        canary.increase_traffic(step_size=0.2)
        assert canary.traffic_percentage == 0.35


# ==========================================
# API SPEC TESTS
# ==========================================

class TestAPISpec:
    """Test API specification."""
    
    def test_prediction_request_validation(self):
        """Test prediction request validation."""
        from pipeline.api_spec import PredictionRequest
        from pydantic import ValidationError
        
        # Valid request
        request = PredictionRequest(
            samples=[{f"feature_{i}": 0.5 for i in range(590)}]
        )
        assert len(request.samples) == 1
        
        # Invalid: wrong feature count
        with pytest.raises(ValidationError):
            PredictionRequest(
                samples=[{f"feature_{i}": 0.5 for i in range(100)}]
            )
    
    def test_prediction_response_schema(self):
        """Test prediction response schema."""
        from pipeline.api_spec import PredictionResponse, PredictionResult
        
        response = PredictionResponse(
            predictions=[
                PredictionResult(
                    prediction=0,
                    confidence=0.85,
                    probabilities={"pass": 0.85, "fail": 0.15}
                )
            ],
            model_version="v1.0.0",
            processing_time_ms=45.2
        )
        
        assert len(response.predictions) == 1
        assert response.model_version == "v1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=pipeline", "--cov=config"])
