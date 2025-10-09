"""
Model Training Pipeline

Automated model training with hyperparameter tuning, cross-validation,
and model comparison for continuous learning.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from loguru import logger
from dotenv import load_dotenv

from database import DatabaseManager, ModelRegistryRepository, AuditLogRepository

load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'training': {
        'test_size': float(os.getenv('TRAIN_TEST_SPLIT', 0.2)),
        'random_state': int(os.getenv('RANDOM_STATE', 42)),
        'cv_folds': int(os.getenv('CV_FOLDS', 5)),
        'n_jobs': int(os.getenv('N_JOBS', -1)),
    },
    'models': {
        'logistic_regression': {
            'enabled': True,
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000],
                'class_weight': ['balanced']
            }
        },
        'random_forest': {
            'enabled': True,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
        },
        'gradient_boosting': {
            'enabled': True,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
        }
    },
    'paths': {
        'models_dir': Path(os.getenv('MODELS_DIR', './models')),
        'artifacts_dir': Path(os.getenv('ARTIFACTS_DIR', './models/artifacts')),
    }
}

# ==========================================
# LOGGING SETUP
# ==========================================
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/training_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="90 days",
    compression="zip",
    level="DEBUG"
)


class ModelTrainer:
    """Handles model training with hyperparameter optimization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models_config = config['models']
        self.training_config = config['training']
        
        # Create directories
        self.models_dir = config['paths']['models_dir']
        self.artifacts_dir = config['paths']['artifacts_dir']
        self.models_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
    
    def _get_model_class(self, model_type: str):
        """Get model class and base parameters"""
        model_map = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier
        }
        return model_map.get(model_type)
    
    def train_with_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str
    ) -> Tuple[object, Dict, Dict]:
        """
        Train model with grid search hyperparameter optimization
        
        Returns:
            Tuple of (best_model, best_params, cv_results)
        """
        logger.info(f"Training {model_type} with GridSearchCV...")
        
        model_class = self._get_model_class(model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        param_grid = self.models_config[model_type]['params']
        
        # Setup cross-validation
        cv = StratifiedKFold(
            n_splits=self.training_config['cv_folds'],
            shuffle=True,
            random_state=self.training_config['random_state']
        )
        
        # Create base model
        base_model = model_class(random_state=self.training_config['random_state'])
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',  # Optimize for F1 score
            n_jobs=self.training_config['n_jobs'],
            verbose=1,
            return_train_score=True
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_duration = time.time() - start_time
        
        logger.info(f"✓ GridSearchCV completed in {training_duration:.2f}s")
        logger.info(f"  Best score: {grid_search.best_score_:.4f}")
        logger.info(f"  Best params: {grid_search.best_params_}")
        
        cv_results = {
            'best_score': float(grid_search.best_score_),
            'best_params': grid_search.best_params_,
            'cv_scores': grid_search.cv_results_['mean_test_score'].tolist(),
            'training_duration_seconds': training_duration
        }
        
        return grid_search.best_estimator_, grid_search.best_params_, cv_results
    
    def evaluate_model(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='binary', pos_label=1)),
            'recall': float(recall_score(y_test, y_pred, average='binary', pos_label=1)),
            'f1_score': float(f1_score(y_test, y_pred, average='binary', pos_label=1)),
        }
        
        if y_proba is not None:
            # Convert labels to binary (0, 1) for ROC AUC
            y_test_binary = (y_test == 1).astype(int)
            metrics['roc_auc'] = float(roc_auc_score(y_test_binary, y_proba))
        else:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_test, y_pred, target_names=['Pass (-1)', 'Fail (1)']
        )
        
        logger.info("Test set metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {metrics.get('roc_auc', 'N/A')}")
        
        return metrics
    
    def save_model(
        self,
        model: object,
        model_name: str,
        model_type: str,
        version: str
    ) -> str:
        """
        Save trained model to disk
        
        Returns:
            Path to saved model
        """
        filename = f"{model_name}_{model_type}_v{version}.joblib"
        model_path = self.models_dir / filename
        
        joblib.dump(model, model_path)
        logger.info(f"✓ Model saved to {model_path}")
        
        return str(model_path)


class TrainingOrchestrator:
    """Orchestrates the complete training pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.trainer = ModelTrainer(config)
        
        # Database components
        self.db_manager = DatabaseManager()
        self.model_registry_repo = ModelRegistryRepository(self.db_manager)
        self.audit_repo = AuditLogRepository(self.db_manager)
    
    def load_training_data(
        self,
        days: int = 7,
        max_samples: int = 50000
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from database
        
        Args:
            days: Number of days of data to use
            max_samples: Maximum number of samples to load
            
        Returns:
            Tuple of (features_df, target_series)
        """
        logger.info(f"Loading training data (last {days} days, max {max_samples} samples)...")
        
        query = f"""
        SELECT pd.features, pd.target
        FROM secom.preprocessed_data pd
        WHERE pd.created_at >= NOW() - INTERVAL '{days} days'
        ORDER BY pd.created_at DESC
        LIMIT {max_samples}
        """
        
        try:
            with self.db_manager.get_connection() as conn:
                df = pd.read_sql(query, conn)
                
                if len(df) == 0:
                    raise ValueError("No training data available")
                
                # Extract features
                features_df = pd.DataFrame(df['features'].tolist())
                target_series = df['target']
                
                logger.info(f"✓ Loaded {len(df)} samples with {len(features_df.columns)} features")
                logger.info(f"  Target distribution: {target_series.value_counts().to_dict()}")
                
                return features_df, target_series
                
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        triggered_by: str = 'manual'
    ) -> List[Dict]:
        """
        Train all enabled model types and compare them
        
        Returns:
            List of model results dictionaries
        """
        results = []
        
        for model_type, model_config in self.config['models'].items():
            if not model_config['enabled']:
                logger.info(f"Skipping {model_type} (disabled)")
                continue
            
            try:
                logger.info("=" * 80)
                logger.info(f"Training {model_type.upper()}")
                logger.info("=" * 80)
                
                training_start = datetime.utcnow()
                
                # Train with grid search
                best_model, best_params, cv_results = self.trainer.train_with_grid_search(
                    X_train, y_train, model_type
                )
                
                # Evaluate on test set
                test_metrics = self.trainer.evaluate_model(best_model, X_test, y_test)
                
                training_end = datetime.utcnow()
                training_duration_ms = (training_end - training_start).total_seconds() * 1000
                
                # Generate version
                version = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                
                # Save model
                model_path = self.trainer.save_model(
                    best_model,
                    model_name='secom_classifier',
                    model_type=model_type,
                    version=version
                )
                
                # Register in database
                model_id = self.model_registry_repo.register_model(
                    model_name='secom_classifier',
                    model_version=version,
                    model_type=model_type,
                    model_path=model_path,
                    hyperparameters=best_params,
                    test_metrics=test_metrics,
                    training_metadata={
                        'dataset_size': len(X_train) + len(X_test),
                        'start_time': training_start,
                        'end_time': training_end,
                        'duration_ms': training_duration_ms
                    },
                    triggered_by=triggered_by
                )
                
                result = {
                    'model_id': model_id,
                    'model_type': model_type,
                    'model_version': version,
                    'model_path': model_path,
                    'best_params': best_params,
                    'cv_results': cv_results,
                    'test_metrics': test_metrics,
                    'training_duration_ms': training_duration_ms
                }
                
                results.append(result)
                
                logger.info(f"✓ {model_type} training completed")
                logger.info(f"  Model ID: {model_id}")
                logger.info(f"  Version: {version}")
                
                # Log audit event
                self.audit_repo.log_event(
                    event_type='model_training',
                    event_status='success',
                    component='trainer',
                    message=f"Trained {model_type} model v{version}",
                    metadata={
                        'model_id': str(model_id),
                        'model_type': model_type,
                        'test_metrics': test_metrics,
                        'hyperparameters': best_params
                    },
                    duration_ms=training_duration_ms
                )
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                
                # Log failure
                self.audit_repo.log_event(
                    event_type='model_training',
                    event_status='failure',
                    component='trainer',
                    message=f"Failed to train {model_type}: {str(e)}",
                    metadata={'model_type': model_type, 'error': str(e)}
                )
        
        return results
    
    def select_best_model(self, results: List[Dict]) -> Dict:
        """
        Select best model based on F1 score
        
        Returns:
            Best model result dictionary
        """
        if not results:
            raise ValueError("No models to select from")
        
        # Sort by F1 score
        sorted_results = sorted(
            results,
            key=lambda x: x['test_metrics']['f1_score'],
            reverse=True
        )
        
        best_model = sorted_results[0]
        
        logger.info("=" * 80)
        logger.info("MODEL COMPARISON RESULTS")
        logger.info("=" * 80)
        
        for rank, result in enumerate(sorted_results, 1):
            logger.info(
                f"{rank}. {result['model_type']}: "
                f"F1={result['test_metrics']['f1_score']:.4f}, "
                f"Accuracy={result['test_metrics']['accuracy']:.4f}, "
                f"ROC-AUC={result['test_metrics'].get('roc_auc', 'N/A')}"
            )
        
        logger.info("=" * 80)
        logger.info(f"✓ Best model: {best_model['model_type']} (F1: {best_model['test_metrics']['f1_score']:.4f})")
        
        return best_model
    
    def deploy_model(self, model_id: uuid.UUID):
        """Deploy a model by activating it"""
        try:
            self.model_registry_repo.activate_model(model_id)
            logger.info(f"✓ Model {model_id} deployed and activated")
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='model_deployment',
                event_status='success',
                component='trainer',
                message=f"Deployed model {model_id}",
                metadata={'model_id': str(model_id)}
            )
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def run_training_pipeline(
        self,
        triggered_by: str = 'manual',
        auto_deploy: bool = False
    ) -> Dict:
        """
        Run complete training pipeline
        
        Args:
            triggered_by: What triggered the training
            auto_deploy: Whether to automatically deploy the best model
            
        Returns:
            Best model result dictionary
        """
        logger.info("=" * 80)
        logger.info("SECOM MODEL TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Triggered by: {triggered_by}")
        logger.info(f"Auto-deploy: {auto_deploy}")
        logger.info("=" * 80)
        
        try:
            # 1. Load data
            X, y = self.load_training_data()
            
            # 2. Split data
            logger.info("Splitting data into train/test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['training']['test_size'],
                random_state=self.config['training']['random_state'],
                stratify=y
            )
            
            logger.info(f"  Training samples: {len(X_train)}")
            logger.info(f"  Test samples: {len(X_test)}")
            
            # 3. Train all models
            results = self.train_all_models(
                X_train.values, X_test.values,
                y_train.values, y_test.values,
                triggered_by=triggered_by
            )
            
            if not results:
                raise RuntimeError("No models were successfully trained")
            
            # 4. Select best model
            best_model = self.select_best_model(results)
            
            # 5. Deploy if auto-deploy enabled
            if auto_deploy:
                logger.info("Auto-deploy enabled, deploying best model...")
                self.deploy_model(best_model['model_id'])
            else:
                logger.info("Auto-deploy disabled. Model ready for manual deployment.")
            
            logger.info("=" * 80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return best_model
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
        finally:
            self.db_manager.close_all()


def main():
    """Main entry point for manual training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SECOM classification models')
    parser.add_argument(
        '--triggered-by',
        type=str,
        default='manual',
        help='What triggered this training (manual, scheduled, performance_degradation, data_drift)'
    )
    parser.add_argument(
        '--auto-deploy',
        action='store_true',
        help='Automatically deploy the best model'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days of data to use for training'
    )
    
    args = parser.parse_args()
    
    try:
        orchestrator = TrainingOrchestrator(CONFIG)
        
        # Override data loading if specified
        if args.days != 7:
            logger.info(f"Using {args.days} days of training data")
        
        result = orchestrator.run_training_pipeline(
            triggered_by=args.triggered_by,
            auto_deploy=args.auto_deploy
        )
        
        logger.info(f"Training complete! Best model ID: {result['model_id']}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
