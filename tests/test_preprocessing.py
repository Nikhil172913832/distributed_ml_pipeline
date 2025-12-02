"""Tests for preprocessing module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from preprocessing import (
    MissingValueHandler,
    OutlierHandler,
    FeatureScaler,
    FeatureSelector,
    FeatureEngineer,
    build_preprocessing_pipeline
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some missing values
    X_df.iloc[0:5, 0] = np.nan
    X_df.iloc[10:15, 5] = np.nan
    
    return X_df, y


class TestMissingValueHandler:
    """Test missing value handler"""
    
    def test_fit_transform(self, sample_data):
        """Test fitting and transforming"""
        X, y = sample_data
        
        handler = MissingValueHandler(numeric_strategy='median')
        X_transformed = handler.fit_transform(X, y)
        
        # Check no missing values remain
        assert X_transformed.isnull().sum().sum() == 0
        assert X_transformed.shape == X.shape
    
    def test_mean_strategy(self, sample_data):
        """Test mean imputation"""
        X, y = sample_data
        
        handler = MissingValueHandler(numeric_strategy='mean')
        handler.fit(X, y)
        X_transformed = handler.transform(X)
        
        assert X_transformed.isnull().sum().sum() == 0
    
    def test_constant_strategy(self):
        """Test constant imputation"""
        X = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        
        handler = MissingValueHandler(numeric_strategy='constant', fill_value=0)
        X_transformed = handler.fit_transform(X)
        
        assert X_transformed.isnull().sum().sum() == 0
        assert X_transformed.iloc[2, 0] == 0


class TestOutlierHandler:
    """Test outlier handler"""
    
    def test_iqr_clip(self):
        """Test IQR method with clipping"""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 100],  # 100 is outlier
            'b': [10, 20, 30, 40, 50, 60]
        })
        
        handler = OutlierHandler(method='iqr', threshold=1.5, action='clip')
        X_transformed = handler.fit_transform(X)
        
        # Outlier should be clipped
        assert X_transformed['a'].max() < 100
    
    def test_zscore_method(self):
        """Test Z-score method"""
        X = np.array([[1, 2], [2, 3], [3, 4], [100, 5]])
        
        handler = OutlierHandler(method='zscore', threshold=2.0, action='clip')
        X_transformed = handler.fit_transform(X)
        
        # Check shape preserved
        assert X_transformed.shape == X.shape


class TestFeatureScaler:
    """Test feature scaler"""
    
    def test_standard_scaling(self, sample_data):
        """Test standard scaling"""
        X, y = sample_data
        
        # Remove NaN for scaler test
        X = X.fillna(0)
        
        scaler = FeatureScaler(method='standard')
        X_scaled = scaler.fit_transform(X)
        
        # Check mean ~ 0, std ~ 1
        assert np.abs(X_scaled.mean()).max() < 1e-10
        assert np.abs(X_scaled.std() - 1).max() < 1e-10
    
    def test_minmax_scaling(self, sample_data):
        """Test min-max scaling"""
        X, y = sample_data
        X = X.fillna(0)
        
        scaler = FeatureScaler(method='minmax', feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        
        # Check range [0, 1]
        assert X_scaled.min().min() >= 0
        assert X_scaled.max().max() <= 1
    
    def test_robust_scaling(self, sample_data):
        """Test robust scaling"""
        X, y = sample_data
        X = X.fillna(0)
        
        scaler = FeatureScaler(method='robust')
        X_scaled = scaler.fit_transform(X)
        
        # Check shape preserved
        assert X_scaled.shape == X.shape


class TestFeatureSelector:
    """Test feature selector"""
    
    def test_variance_selection(self):
        """Test variance-based selection"""
        X = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'low_var': [1, 1, 1, 2],
            'high_var': [1, 5, 10, 20]
        })
        
        selector = FeatureSelector(method='variance', threshold=0.5)
        X_selected = selector.fit_transform(X)
        
        # Only high_var should remain
        assert X_selected.shape[1] < X.shape[1]
    
    def test_manual_selection(self, sample_data):
        """Test manual feature selection"""
        X, y = sample_data
        
        selected = ['feature_0', 'feature_1', 'feature_2']
        selector = FeatureSelector(method='manual', feature_names=selected)
        X_selected = selector.fit_transform(X)
        
        assert X_selected.shape[1] == 3
        assert list(X_selected.columns) == selected


class TestFeatureEngineer:
    """Test feature engineer"""
    
    def test_interaction_features(self):
        """Test interaction feature creation"""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [2, 3, 4, 5]
        })
        
        engineer = FeatureEngineer(
            create_interactions=True,
            interaction_pairs=[('a', 'b')]
        )
        X_engineered = engineer.fit_transform(X)
        
        # Should have original + interaction
        assert X_engineered.shape[1] == 3
        assert 'a_x_b' in X_engineered.columns
        assert (X_engineered['a_x_b'] == X['a'] * X['b']).all()
    
    def test_polynomial_features(self):
        """Test polynomial feature creation"""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4]
        })
        
        engineer = FeatureEngineer(
            create_polynomials=True,
            polynomial_degree=3
        )
        X_engineered = engineer.fit_transform(X)
        
        # Should have a, a^2, a^3
        assert X_engineered.shape[1] == 3
        assert 'a_pow2' in X_engineered.columns
        assert 'a_pow3' in X_engineered.columns
    
    def test_custom_features(self):
        """Test custom feature creation"""
        X = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [2, 4, 6, 8]
        })
        
        def ratio_feature(df):
            return df['a'] / df['b']
        
        engineer = FeatureEngineer(
            custom_features={'a_b_ratio': ratio_feature}
        )
        X_engineered = engineer.fit_transform(X)
        
        assert 'a_b_ratio' in X_engineered.columns
        assert (X_engineered['a_b_ratio'] == 0.5).all()


class TestPreprocessingPipeline:
    """Test complete preprocessing pipeline"""
    
    def test_pipeline_creation(self):
        """Test pipeline building from config"""
        config = {
            'handle_missing': True,
            'missing_strategy': 'median',
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'scale_features': True,
            'scaling_method': 'standard'
        }
        
        pipeline = build_preprocessing_pipeline(config)
        
        # Check pipeline has expected steps
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0][0] == 'imputer'
        assert pipeline.steps[1][0] == 'outlier_handler'
        assert pipeline.steps[2][0] == 'scaler'
    
    def test_pipeline_fit_transform(self, sample_data):
        """Test full pipeline fit and transform"""
        X, y = sample_data
        
        config = {
            'handle_missing': True,
            'scale_features': True
        }
        
        pipeline = build_preprocessing_pipeline(config)
        X_transformed = pipeline.fit_transform(X)
        
        # Check no missing values and scaled
        assert not np.isnan(X_transformed).any()
        assert X_transformed.shape == X.shape
