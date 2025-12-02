"""
Model abstraction layer for interchangeable model implementations.

Provides unified interface for sklearn, PyTorch, and TensorFlow models,
enabling easy model swapping and experimentation.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    All model implementations should inherit from this to ensure consistent interface.
    """
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize model wrapper.
        
        Args:
            model_params: Model hyperparameters
        """
        self.model_params = model_params or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.model_type = self.__class__.__name__
    
    @abstractmethod
    def fit(self, X, y, validation_data: Optional[Tuple] = None):
        """
        Fit model on training data.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
        """
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability array
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]):
        """Load model from disk"""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.model_params
    
    def set_params(self, **params):
        """Set model parameters"""
        self.model_params.update(params)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance (if supported)"""
        return None


class SklearnModelWrapper(BaseModelWrapper):
    """
    Wrapper for scikit-learn models.
    
    Supports any sklearn-compatible estimator.
    """
    
    def __init__(self, model_class, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize sklearn model wrapper.
        
        Args:
            model_class: sklearn model class (e.g., LogisticRegression)
            model_params: Model hyperparameters
        """
        super().__init__(model_params)
        self.model_class = model_class
        self.model = model_class(**self.model_params)
        logger.info(f"Initialized sklearn model: {model_class.__name__}")
    
    def fit(self, X, y, validation_data: Optional[Tuple] = None):
        """Fit sklearn model"""
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"Model fitted: {self.model_class.__name__}")
        return self
    
    def predict(self, X) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return predictions as one-hot
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(predictions), n_classes))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba
    
    def save(self, path: Union[str, Path]):
        """Save model using joblib"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load model from joblib"""
        path = Path(path)
        
        data = joblib.load(path)
        self.model = data['model']
        self.model_params = data['model_params']
        self.feature_names = data.get('feature_names')
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return self
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            return np.abs(self.model.coef_).flatten()
        return None


class PyTorchModelWrapper(BaseModelWrapper):
    """
    Wrapper for PyTorch models.
    
    Provides sklearn-like interface for PyTorch neural networks.
    """
    
    def __init__(
        self,
        model_class,
        model_params: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PyTorch model wrapper.
        
        Args:
            model_class: PyTorch model class
            model_params: Model architecture parameters
            training_params: Training hyperparameters (epochs, lr, etc.)
        """
        super().__init__(model_params)
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        self.torch = torch
        self.nn = nn
        self.optim = optim
        self.DataLoader = DataLoader
        self.TensorDataset = TensorDataset
        
        self.model_class = model_class
        self.training_params = training_params or {
            'epochs': 10,
            'lr': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'loss_fn': 'cross_entropy'
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized PyTorch model on {self.device}")
    
    def _create_model(self, input_dim: int, output_dim: int):
        """Create model instance"""
        self.model = self.model_class(
            input_dim=input_dim,
            output_dim=output_dim,
            **self.model_params
        )
        self.model.to(self.device)
    
    def _create_dataloader(self, X, y, shuffle: bool = True):
        """Create PyTorch DataLoader"""
        X_tensor = self.torch.FloatTensor(X)
        y_tensor = self.torch.LongTensor(y)
        
        dataset = self.TensorDataset(X_tensor, y_tensor)
        dataloader = self.DataLoader(
            dataset,
            batch_size=self.training_params['batch_size'],
            shuffle=shuffle
        )
        
        return dataloader
    
    def fit(self, X, y, validation_data: Optional[Tuple] = None):
        """Fit PyTorch model"""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Create model
        input_dim = X.shape[1]
        output_dim = len(np.unique(y))
        self._create_model(input_dim, output_dim)
        
        # Setup optimizer
        if self.training_params['optimizer'] == 'adam':
            optimizer = self.optim.Adam(self.model.parameters(), lr=self.training_params['lr'])
        else:
            optimizer = self.optim.SGD(self.model.parameters(), lr=self.training_params['lr'])
        
        # Setup loss function
        if self.training_params['loss_fn'] == 'cross_entropy':
            criterion = self.nn.CrossEntropyLoss()
        else:
            criterion = self.nn.BCEWithLogitsLoss()
        
        # Create dataloaders
        train_loader = self._create_dataloader(X, y, shuffle=True)
        
        if validation_data:
            X_val, y_val = validation_data
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        # Training loop
        self.model.train()
        for epoch in range(self.training_params['epochs']):
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if validation_data:
                val_loss = self._evaluate(val_loader, criterion)
                logger.info(f"Epoch {epoch+1}/{self.training_params['epochs']}: "
                          f"train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{self.training_params['epochs']}: "
                          f"train_loss={avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def _evaluate(self, dataloader, criterion):
        """Evaluate model on dataloader"""
        self.model.eval()
        total_loss = 0
        
        with self.torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(dataloader)
    
    def predict(self, X) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(X, 'values'):
            X = X.values
        
        self.model.eval()
        with self.torch.no_grad():
            X_tensor = self.torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = self.torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(X, 'values'):
            X = X.values
        
        self.model.eval()
        with self.torch.no_grad():
            X_tensor = self.torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probas = self.torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
    
    def save(self, path: Union[str, Path]):
        """Save PyTorch model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'training_params': self.training_params,
            'model_class': self.model_class.__name__
        }, path)
        
        logger.info(f"PyTorch model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Load PyTorch model"""
        path = Path(path)
        
        checkpoint = self.torch.load(path, map_location=self.device)
        
        # Reconstruct model (assumes model_class and params are available)
        self.model_params = checkpoint['model_params']
        self.training_params = checkpoint['training_params']
        
        # Note: This requires model_class to be passed during wrapper initialization
        # In production, you'd need to save/load the architecture definition
        if self.model is None:
            raise ValueError("Model architecture must be initialized before loading weights")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        
        logger.info(f"PyTorch model loaded from {path}")
        return self


class ModelFactory:
    """
    Factory for creating model wrappers from configuration.
    """
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModelWrapper:
        """
        Create model wrapper from configuration.
        
        Args:
            config: Model configuration with 'type' and 'params' keys
            
        Returns:
            Initialized model wrapper
        """
        model_type = config.get('type', 'sklearn')
        model_name = config.get('name', 'LogisticRegression')
        model_params = config.get('params', {})
        
        if model_type == 'sklearn':
            # Import sklearn model class
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier
            
            model_classes = {
                'LogisticRegression': LogisticRegression,
                'RandomForest': RandomForestClassifier,
                'GradientBoosting': GradientBoostingClassifier,
                'SVM': SVC,
                'DecisionTree': DecisionTreeClassifier
            }
            
            if model_name not in model_classes:
                raise ValueError(f"Unknown sklearn model: {model_name}")
            
            return SklearnModelWrapper(model_classes[model_name], model_params)
        
        elif model_type == 'pytorch':
            # For PyTorch, you'd pass a custom model class
            # This is a placeholder - in practice, define your architectures
            training_params = config.get('training_params', {})
            
            logger.warning("PyTorch models require custom model class definition")
            # return PyTorchModelWrapper(YourModelClass, model_params, training_params)
            raise NotImplementedError("Define PyTorch model class first")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example usage and utilities
def compare_models(
    models: List[BaseModelWrapper],
    X_train, y_train,
    X_test, y_test,
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on same dataset.
    
    Args:
        models: List of model wrappers
        X_train, y_train: Training data
        X_test, y_test: Test data
        metrics: List of metric names to compute
        
    Returns:
        Dictionary of {model_name: {metric: score}}
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    metrics = metrics or ['accuracy', 'f1', 'precision', 'recall']
    results = {}
    
    for model in models:
        # Train model
        logger.info(f"Training {model.model_type}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        model_results = {}
        if 'accuracy' in metrics:
            model_results['accuracy'] = accuracy_score(y_test, y_pred)
        if 'f1' in metrics:
            model_results['f1'] = f1_score(y_test, y_pred, average='weighted')
        if 'precision' in metrics:
            model_results['precision'] = precision_score(y_test, y_pred, average='weighted')
        if 'recall' in metrics:
            model_results['recall'] = recall_score(y_test, y_pred, average='weighted')
        
        results[model.model_type] = model_results
        logger.info(f"{model.model_type} results: {model_results}")
    
    return results
