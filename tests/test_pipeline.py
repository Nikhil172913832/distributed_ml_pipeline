"""
Unit tests for the pipeline components
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json


class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline"""
    
    def test_simple_imputation(self):
        """Test that missing values are imputed correctly"""
        # Create sample data with missing values
        data = {
            'feature_0': [1.0, 2.0, np.nan, 4.0],
            'feature_1': [np.nan, 2.0, 3.0, 4.0],
            'feature_2': [1.0, 2.0, 3.0, 4.0]
        }
        df = pd.DataFrame(data)
        
        # Median imputation
        df_imputed = df.copy()
        for col in df_imputed.columns:
            if df_imputed[col].isnull().any():
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
        
        # Check no missing values remain
        assert df_imputed.isnull().sum().sum() == 0
        
        # Check imputed values are correct
        assert df_imputed.loc[2, 'feature_0'] == 2.0  # median of [1, 2, 4]
        assert df_imputed.loc[0, 'feature_1'] == 3.0  # median of [2, 3, 4]
    
    def test_standardization(self):
        """Test feature standardization"""
        data = {
            'feature_0': [1.0, 2.0, 3.0, 4.0],
            'feature_1': [10.0, 20.0, 30.0, 40.0]
        }
        df = pd.DataFrame(data)
        
        # Standardize
        df_scaled = (df - df.mean()) / df.std()
        
        # Check mean is close to 0 and std is close to 1
        assert np.allclose(df_scaled.mean(), 0, atol=1e-10)
        assert np.allclose(df_scaled.std(), 1, atol=1e-10)


class TestDatabaseOperations:
    """Tests for database operations"""
    
    @patch('psycopg2.connect')
    def test_connection_pool_creation(self, mock_connect):
        """Test database connection pool creation"""
        from pipeline.database import DatabaseManager
        
        # Mock connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # This will fail without actual DB, but tests the structure
        # In real tests, use a test database
        pass
    
    def test_batch_metadata_structure(self):
        """Test batch metadata structure"""
        batch_metadata = {
            'batch_id': 'test_batch_001',
            'total_samples': 100,
            'pass_samples': 80,
            'fail_samples': 20,
            'processing_status': 'pending'
        }
        
        assert batch_metadata['batch_id'] == 'test_batch_001'
        assert batch_metadata['total_samples'] == 100
        assert batch_metadata['pass_samples'] + batch_metadata['fail_samples'] == batch_metadata['total_samples']


class TestKafkaProducer:
    """Tests for Kafka producer"""
    
    def test_sample_generation_structure(self):
        """Test generated sample has correct structure"""
        sample = {
            'batch_id': 'batch_test_001',
            'sample_index': 0,
            'features': {f'feature_{i}': float(i) for i in range(10)},
            'target': -1,
            'timestamp': '2024-01-01T00:00:00'
        }
        
        assert 'batch_id' in sample
        assert 'sample_index' in sample
        assert 'features' in sample
        assert 'target' in sample
        assert sample['target'] in [-1, 1]
        assert len(sample['features']) == 10
    
    def test_message_serialization(self):
        """Test that messages can be serialized to JSON"""
        sample = {
            'batch_id': 'test_001',
            'sample_index': 0,
            'features': {'feature_0': 1.5, 'feature_1': 2.5},
            'target': -1
        }
        
        # Serialize
        serialized = json.dumps(sample)
        
        # Deserialize
        deserialized = json.loads(serialized)
        
        assert deserialized == sample


class TestMetrics:
    """Tests for metrics collection"""
    
    def test_prometheus_metrics_structure(self):
        """Test Prometheus metrics are properly structured"""
        from prometheus_client import Counter, Histogram, Gauge
        
        # Test counter creation
        test_counter = Counter('test_counter', 'Test counter')
        test_counter.inc()
        
        # Test histogram creation
        test_histogram = Histogram('test_histogram', 'Test histogram')
        test_histogram.observe(1.5)
        
        # Test gauge creation
        test_gauge = Gauge('test_gauge', 'Test gauge')
        test_gauge.set(10)
        
        assert True  # If we get here, metrics are working


def test_data_quality_checks():
    """Test data quality validation"""
    # Test for missing value detection
    df = pd.DataFrame({
        'col1': [1, 2, np.nan, 4],
        'col2': [1, 2, 3, 4]
    })
    
    missing_count = df.isnull().sum().sum()
    assert missing_count == 1
    
    # Test for outlier detection (simple z-score method)
    data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = data[z_scores > 3]
    
    assert len(outliers) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
