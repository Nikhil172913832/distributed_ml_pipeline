"""
Modular preprocessing components for feature engineering.

Provides reusable preprocessors that can be composed into pipelines
and integrated with sklearn Pipeline or custom processing flows.
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import logging

logger = logging.getLogger(__name__)


class BasePreprocessor(BaseEstimator, TransformerMixin):
    """
    Base class for custom preprocessors.
    
    All preprocessors should inherit from this to ensure sklearn compatibility.
    """
    
    def fit(self, X, y=None):
        """Fit preprocessor (to be overridden)"""
        return self
    
    def transform(self, X):
        """Transform data (to be overridden)"""
        return X
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        return self.fit(X, y).transform(X)


class MissingValueHandler(BasePreprocessor):
    """
    Handle missing values with configurable strategies.
    
    Supports multiple imputation strategies per column type.
    """
    
    def __init__(
        self,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        fill_value: Optional[float] = None
    ):
        """
        Initialize missing value handler.
        
        Args:
            numeric_strategy: Strategy for numeric columns ('mean', 'median', 'constant')
            categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant')
            fill_value: Value for constant strategy
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.numeric_cols = None
        self.categorical_cols = None
    
    def fit(self, X, y=None):
        """Fit imputers on training data"""
        if isinstance(X, pd.DataFrame):
            self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            # Assume all numeric for numpy arrays
            self.numeric_cols = list(range(X.shape[1]))
            self.categorical_cols = []
        
        # Fit numeric imputer
        if self.numeric_cols:
            self.numeric_imputer = SimpleImputer(
                strategy=self.numeric_strategy,
                fill_value=self.fill_value
            )
            X_numeric = X[self.numeric_cols] if isinstance(X, pd.DataFrame) else X
            self.numeric_imputer.fit(X_numeric)
        
        # Fit categorical imputer
        if self.categorical_cols:
            self.categorical_imputer = SimpleImputer(
                strategy=self.categorical_strategy,
                fill_value=self.fill_value
            )
            self.categorical_imputer.fit(X[self.categorical_cols])
        
        logger.info(f"MissingValueHandler fitted: {len(self.numeric_cols)} numeric, "
                   f"{len(self.categorical_cols)} categorical columns")
        
        return self
    
    def transform(self, X):
        """Transform data by imputing missing values"""
        X_transformed = X.copy() if isinstance(X, pd.DataFrame) else X.copy()
        
        # Impute numeric columns
        if self.numeric_cols and self.numeric_imputer:
            if isinstance(X, pd.DataFrame):
                X_transformed[self.numeric_cols] = self.numeric_imputer.transform(
                    X[self.numeric_cols]
                )
            else:
                X_transformed = self.numeric_imputer.transform(X_transformed)
        
        # Impute categorical columns
        if self.categorical_cols and self.categorical_imputer:
            X_transformed[self.categorical_cols] = self.categorical_imputer.transform(
                X[self.categorical_cols]
            )
        
        return X_transformed


class OutlierHandler(BasePreprocessor):
    """
    Detect and handle outliers using IQR or Z-score methods.
    """
    
    def __init__(
        self,
        method: str = 'iqr',
        threshold: float = 3.0,
        action: str = 'clip'
    ):
        """
        Initialize outlier handler.
        
        Args:
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (IQR multiplier or Z-score)
            action: Action to take ('clip', 'remove', 'impute')
        """
        self.method = method
        self.threshold = threshold
        self.action = action
        
        self.lower_bounds = None
        self.upper_bounds = None
    
    def fit(self, X, y=None):
        """Calculate outlier bounds"""
        if isinstance(X, pd.DataFrame):
            X_array = X.select_dtypes(include=[np.number]).values
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            X_array = X
            self.columns = list(range(X.shape[1]))
        
        if self.method == 'iqr':
            q1 = np.percentile(X_array, 25, axis=0)
            q3 = np.percentile(X_array, 75, axis=0)
            iqr = q3 - q1
            
            self.lower_bounds = q1 - self.threshold * iqr
            self.upper_bounds = q3 + self.threshold * iqr
            
        elif self.method == 'zscore':
            mean = np.mean(X_array, axis=0)
            std = np.std(X_array, axis=0)
            
            self.lower_bounds = mean - self.threshold * std
            self.upper_bounds = mean + self.threshold * std
        
        logger.info(f"OutlierHandler fitted using {self.method} method")
        return self
    
    def transform(self, X):
        """Handle outliers"""
        X_transformed = X.copy()
        
        if isinstance(X, pd.DataFrame):
            X_numeric = X[self.columns].values
        else:
            X_numeric = X
        
        if self.action == 'clip':
            X_numeric = np.clip(X_numeric, self.lower_bounds, self.upper_bounds)
        
        elif self.action == 'remove':
            # Mark rows with outliers (would need separate handling)
            mask = ((X_numeric >= self.lower_bounds) & 
                   (X_numeric <= self.upper_bounds)).all(axis=1)
            X_transformed = X_transformed[mask] if isinstance(X, pd.DataFrame) else X_numeric[mask]
            logger.info(f"Removed {(~mask).sum()} outlier rows")
        
        elif self.action == 'impute':
            # Replace outliers with bounds
            X_numeric = np.where(X_numeric < self.lower_bounds, self.lower_bounds, X_numeric)
            X_numeric = np.where(X_numeric > self.upper_bounds, self.upper_bounds, X_numeric)
        
        if isinstance(X, pd.DataFrame):
            X_transformed[self.columns] = X_numeric
        else:
            X_transformed = X_numeric
        
        return X_transformed


class FeatureScaler(BasePreprocessor):
    """
    Scale features using various scaling strategies.
    """
    
    def __init__(
        self,
        method: str = 'standard',
        feature_range: Tuple[float, float] = (0, 1)
    ):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
            feature_range: Range for MinMax scaling
        """
        self.method = method
        self.feature_range = feature_range
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit(self, X, y=None):
        """Fit scaler"""
        self.scaler.fit(X)
        logger.info(f"FeatureScaler fitted using {self.method} method")
        return self
    
    def transform(self, X):
        """Scale features"""
        return self.scaler.transform(X)


class FeatureSelector(BasePreprocessor):
    """
    Select features based on variance, correlation, or custom criteria.
    """
    
    def __init__(
        self,
        method: str = 'variance',
        threshold: float = 0.01,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ('variance', 'correlation', 'manual')
            threshold: Threshold for variance/correlation
            feature_names: Manual list of features to keep
        """
        self.method = method
        self.threshold = threshold
        self.feature_names = feature_names
        
        self.selected_features = None
    
    def fit(self, X, y=None):
        """Fit selector"""
        if self.method == 'variance':
            # Select features with variance above threshold
            if isinstance(X, pd.DataFrame):
                variances = X.var()
                self.selected_features = variances[variances > self.threshold].index.tolist()
            else:
                variances = np.var(X, axis=0)
                self.selected_features = np.where(variances > self.threshold)[0].tolist()
        
        elif self.method == 'correlation':
            # Remove highly correlated features
            if isinstance(X, pd.DataFrame):
                corr_matrix = X.corr().abs()
                upper = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                to_drop = [col for col in upper.columns if any(upper[col] > self.threshold)]
                self.selected_features = [col for col in X.columns if col not in to_drop]
            else:
                # For numpy arrays, keep all
                self.selected_features = list(range(X.shape[1]))
        
        elif self.method == 'manual':
            self.selected_features = self.feature_names
        
        logger.info(f"FeatureSelector: selected {len(self.selected_features)} features "
                   f"using {self.method} method")
        return self
    
    def transform(self, X):
        """Select features"""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            return X[:, self.selected_features]


class FeatureEngineer(BasePreprocessor):
    """
    Create new features through engineering.
    
    Supports polynomial features, interaction terms, and domain-specific transformations.
    """
    
    def __init__(
        self,
        create_interactions: bool = False,
        interaction_pairs: Optional[List[Tuple[str, str]]] = None,
        create_polynomials: bool = False,
        polynomial_degree: int = 2,
        custom_features: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize feature engineer.
        
        Args:
            create_interactions: Create pairwise interaction features
            interaction_pairs: Specific pairs for interactions
            create_polynomials: Create polynomial features
            polynomial_degree: Degree for polynomial features
            custom_features: Dictionary of {name: function} for custom features
        """
        self.create_interactions = create_interactions
        self.interaction_pairs = interaction_pairs
        self.create_polynomials = create_polynomials
        self.polynomial_degree = polynomial_degree
        self.custom_features = custom_features or {}
        
        self.feature_names = None
    
    def fit(self, X, y=None):
        """Fit (mostly just store column names)"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return self
    
    def transform(self, X):
        """Engineer features"""
        X_transformed = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # Create interaction features
        if self.create_interactions:
            if self.interaction_pairs:
                for col1, col2 in self.interaction_pairs:
                    X_transformed[f"{col1}_x_{col2}"] = X_transformed[col1] * X_transformed[col2]
            else:
                # Create all pairwise interactions (be careful with many features)
                numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        X_transformed[f"{col1}_x_{col2}"] = X_transformed[col1] * X_transformed[col2]
        
        # Create polynomial features
        if self.create_polynomials:
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:10]:  # Limit to avoid explosion
                for degree in range(2, self.polynomial_degree + 1):
                    X_transformed[f"{col}_pow{degree}"] = X_transformed[col] ** degree
        
        # Create custom features
        for name, func in self.custom_features.items():
            X_transformed[name] = func(X_transformed)
        
        logger.info(f"FeatureEngineer: created {X_transformed.shape[1] - len(self.feature_names)} new features")
        
        return X_transformed


def build_preprocessing_pipeline(
    config: Dict[str, Any],
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Pipeline:
    """
    Build a complete preprocessing pipeline from configuration.
    
    Args:
        config: Configuration dictionary with preprocessing steps
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
    
    Returns:
        sklearn Pipeline object
    """
    steps = []
    
    # Missing value handling
    if config.get('handle_missing', True):
        steps.append(('imputer', MissingValueHandler(
            numeric_strategy=config.get('missing_strategy', 'median')
        )))
    
    # Outlier handling
    if config.get('handle_outliers', False):
        steps.append(('outlier_handler', OutlierHandler(
            method=config.get('outlier_method', 'iqr'),
            threshold=config.get('outlier_threshold', 3.0),
            action=config.get('outlier_action', 'clip')
        )))
    
    # Feature engineering
    if config.get('engineer_features', False):
        steps.append(('feature_engineer', FeatureEngineer(
            create_interactions=config.get('create_interactions', False),
            create_polynomials=config.get('create_polynomials', False),
            polynomial_degree=config.get('polynomial_degree', 2)
        )))
    
    # Feature selection
    if config.get('select_features', False):
        steps.append(('feature_selector', FeatureSelector(
            method=config.get('selection_method', 'variance'),
            threshold=config.get('selection_threshold', 0.01)
        )))
    
    # Scaling (typically last step)
    if config.get('scale_features', True):
        steps.append(('scaler', FeatureScaler(
            method=config.get('scaling_method', 'standard')
        )))
    
    pipeline = Pipeline(steps)
    
    logger.info(f"Built preprocessing pipeline with {len(steps)} steps: "
               f"{[name for name, _ in steps]}")
    
    return pipeline
