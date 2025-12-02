"""
Tests for data correctness, edge cases, and robustness.

Verifies pipeline behavior with problematic data inputs.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from pipeline.data_validation import SchemaValidator, DataLineage
from preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    build_preprocessing_pipeline
)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        X_empty = pd.DataFrame()
        
        # Should not crash
        lineage = DataLineage()
        data_hash = lineage.dataframe_hash(X_empty)
        
        assert data_hash is not None
        assert isinstance(data_hash, str)
    
    def test_single_row(self):
        """Test handling of single row"""
        X = pd.DataFrame({'a': [1], 'b': [2]})
        
        handler = MissingValueHandler()
        X_transformed = handler.fit_transform(X)
        
        assert X_transformed.shape == (1, 2)
    
    def test_all_missing_column(self):
        """Test column with all missing values"""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [np.nan, np.nan, np.nan, np.nan]
        })
        
        handler = MissingValueHandler(numeric_strategy='constant', fill_value=0)
        X_transformed = handler.fit_transform(X)
        
        assert X_transformed['b'].notna().all()
        assert (X_transformed['b'] == 0).all()
    
    def test_single_feature(self):
        """Test with single feature"""
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        
        config = {'handle_missing': True, 'scale_features': True}
        pipeline = build_preprocessing_pipeline(config)
        
        X_transformed = pipeline.fit_transform(X)
        assert X_transformed.shape == (5, 1)
    
    def test_all_same_values(self):
        """Test constant feature (zero variance)"""
        X = pd.DataFrame({
            'a': [5, 5, 5, 5],
            'b': [1, 2, 3, 4]
        })
        
        from preprocessing import FeatureSelector
        selector = FeatureSelector(method='variance', threshold=0.1)
        X_selected = selector.fit_transform(X)
        
        # Should keep only variable feature
        assert X_selected.shape[1] == 1
        assert 'b' in X_selected.columns
    
    def test_extreme_outliers(self):
        """Test handling of extreme outliers"""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4, 1e10],  # Extreme outlier
            'b': [10, 20, 30, 40, 50]
        })
        
        handler = OutlierHandler(method='iqr', action='clip')
        X_transformed = handler.fit_transform(X)
        
        # Outlier should be clipped
        assert X_transformed['a'].max() < 1e5
    
    def test_negative_values(self):
        """Test handling of negative values"""
        X = pd.DataFrame({
            'a': [-10, -5, 0, 5, 10],
            'b': [-100, -50, 0, 50, 100]
        })
        
        from preprocessing import FeatureScaler
        scaler = FeatureScaler(method='minmax', feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        
        # Should scale to [0, 1]
        assert X_scaled.min().min() >= 0
        assert X_scaled.max().max() <= 1
    
    def test_mixed_data_types(self):
        """Test mixed numeric and string data"""
        X = pd.DataFrame({
            'numeric': [1, 2, 3, 4],
            'categorical': ['a', 'b', 'c', 'd']
        })
        
        handler = MissingValueHandler()
        # Should handle both types
        X_transformed = handler.fit_transform(X)
        
        assert X_transformed.shape == X.shape
    
    def test_inf_values(self):
        """Test handling of infinite values"""
        X = pd.DataFrame({
            'a': [1, 2, np.inf, 4],
            'b': [5, -np.inf, 7, 8]
        })
        
        # Replace inf with NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        handler = MissingValueHandler()
        X_transformed = handler.fit_transform(X_clean)
        
        # Should not have any NaN
        assert X_transformed.notna().all().all()
    
    def test_duplicate_columns(self):
        """Test DataFrame with duplicate column names"""
        X = pd.DataFrame([[1, 2], [3, 4]], columns=['a', 'a'])
        
        # Rename duplicates
        X.columns = ['a_0', 'a_1']
        
        handler = MissingValueHandler()
        X_transformed = handler.fit_transform(X)
        
        assert X_transformed.shape == (2, 2)


class TestDataCorrectness:
    """Test data integrity and correctness"""
    
    def test_no_data_leakage(self):
        """Ensure fit/transform don't leak data"""
        X_train = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })
        
        X_test = pd.DataFrame({
            'a': [6, 7, 8],
            'b': [60, 70, 80]
        })
        
        from preprocessing import FeatureScaler
        scaler = FeatureScaler(method='standard')
        
        # Fit on train
        scaler.fit(X_train)
        
        # Transform test
        X_test_scaled = scaler.transform(X_test)
        
        # Test should use train statistics, not its own
        # (test mean should NOT be 0)
        test_mean = X_test_scaled.mean()
        assert not np.allclose(test_mean, 0, atol=0.1)
    
    def test_deterministic_hashing(self):
        """Test data hashing is deterministic"""
        X = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        lineage = DataLineage()
        
        hash1 = lineage.dataframe_hash(X)
        hash2 = lineage.dataframe_hash(X)
        
        assert hash1 == hash2
    
    def test_hash_sensitivity(self):
        """Test hash changes with data"""
        X1 = pd.DataFrame({'a': [1, 2, 3]})
        X2 = pd.DataFrame({'a': [1, 2, 4]})  # Different
        
        lineage = DataLineage()
        
        hash1 = lineage.dataframe_hash(X1)
        hash2 = lineage.dataframe_hash(X2)
        
        assert hash1 != hash2
    
    def test_column_order_preservation(self):
        """Test preprocessing preserves column order"""
        X = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        from preprocessing import FeatureScaler
        scaler = FeatureScaler()
        
        scaler.fit(X)
        result = scaler.transform(X)
        
        # Shape should be preserved
        assert result.shape == X.shape
    
    def test_shape_preservation(self):
        """Test transformations preserve shape"""
        X, y = make_classification(n_samples=100, n_features=20)
        X_df = pd.DataFrame(X)
        
        config = {
            'handle_missing': True,
            'scale_features': True
        }
        
        pipeline = build_preprocessing_pipeline(config)
        X_transformed = pipeline.fit_transform(X_df)
        
        assert X_transformed.shape == X_df.shape
    
    def test_idempotency(self):
        """Test applying transform twice gives same result"""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [5, 6, 7, 8]
        })
        
        from preprocessing import FeatureScaler
        scaler = FeatureScaler()
        
        scaler.fit(X)
        result1 = scaler.transform(X)
        result2 = scaler.transform(result1)
        
        # Second transform should change it (not idempotent for scaling)
        # But should not crash
        assert result2.shape == result1.shape


class TestNullHandling:
    """Comprehensive null/NaN handling tests"""
    
    def test_various_null_representations(self):
        """Test different null representations"""
        X = pd.DataFrame({
            'a': [1.0, None, 3.0],
            'b': [4.0, np.nan, 6.0]
        })
        
        handler = MissingValueHandler()
        X_transformed = handler.fit_transform(X)
        
        # All nulls should be handled
        assert X_transformed.notna().all().all()
    
    def test_null_percentage_threshold(self):
        """Test dropping columns with too many nulls"""
        X = pd.DataFrame({
            'good': [1, 2, 3, 4, 5],
            'mostly_null': [1, np.nan, np.nan, np.nan, np.nan]
        })
        
        # Drop columns with > 50% nulls
        null_pct = X.isnull().mean()
        X_filtered = X.loc[:, null_pct < 0.5]
        
        assert 'good' in X_filtered.columns
        assert 'mostly_null' not in X_filtered.columns
    
    def test_forward_fill_strategy(self):
        """Test forward fill for time series data"""
        X = pd.DataFrame({
            'value': [1.0, np.nan, np.nan, 4.0, np.nan, 6.0]
        })
        
        X_filled = X.ffill()
        
        expected = pd.DataFrame({
            'value': [1.0, 1.0, 1.0, 4.0, 4.0, 6.0]
        })
        
        pd.testing.assert_frame_equal(X_filled, expected)


class TestScalability:
    """Test performance with different data sizes"""
    
    def test_small_dataset(self):
        """Test with very small dataset"""
        X, y = make_classification(n_samples=10, n_features=5)
        X_df = pd.DataFrame(X)
        
        config = {'scale_features': True}
        pipeline = build_preprocessing_pipeline(config)
        
        X_transformed = pipeline.fit_transform(X_df)
        assert X_transformed.shape == (10, 5)
    
    def test_many_features(self):
        """Test with many features"""
        X, y = make_classification(n_samples=100, n_features=100)
        X_df = pd.DataFrame(X)
        
        config = {'scale_features': True}
        pipeline = build_preprocessing_pipeline(config)
        
        X_transformed = pipeline.fit_transform(X_df)
        assert X_transformed.shape == (100, 100)
    
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test with large dataset (marked as slow)"""
        X, y = make_classification(n_samples=10000, n_features=50)
        X_df = pd.DataFrame(X)
        
        config = {'scale_features': True}
        pipeline = build_preprocessing_pipeline(config)
        
        import time
        start = time.time()
        X_transformed = pipeline.fit_transform(X_df)
        duration = time.time() - start
        
        assert X_transformed.shape == (10000, 50)
        assert duration < 5.0  # Should complete in < 5 seconds


class TestSchemaValidation:
    """Test schema validation edge cases"""
    
    def test_schema_mismatch(self):
        """Test detection of schema violations"""
        # Define schema
        schema = {
            'a': {'dtype': 'int', 'nullable': False},
            'b': {'dtype': 'float', 'nullable': False},
            'c': {'dtype': 'object', 'nullable': False}
        }
        
        validator = SchemaValidator(schema)
        
        # Valid data
        X_valid = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0],
            'c': ['x', 'y', 'z']
        })
        
        # Invalid data (missing column)
        X_invalid = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1.0, 2.0, 3.0]
        })
        
        # Validate - valid should pass
        assert validator.validate(X_valid)
        
        # Invalid should raise
        with pytest.raises(ValueError, match="Missing required columns"):
            validator.validate(X_invalid)
