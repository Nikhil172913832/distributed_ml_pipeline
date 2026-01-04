"""
Advanced model trainer integrating deep learning with traditional ML.

Handles TabNet, deep ensembles, and hybrid approaches.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import joblib

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from pytorch_tabnet.tab_model import TabNetClassifier
from loguru import logger

from deep_models import (
    DeepClassifier, Autoencoder, DeepEnsemble,
    ModelTrainer, SECOMDataset
)


class AdvancedModelTrainer:
    """
    Trains advanced models including TabNet and deep learning.
    Handles class imbalance with SMOTE and focal loss.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_smote: bool = True,
        test_size: float = 0.2
    ) -> Tuple:
        """
        Prepare data with train/test split and optional SMOTE.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            stratify=y, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if use_smote:
            logger.info("Applying SMOTE for class balance")
            smote = SMOTETomek(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            logger.info(f"After SMOTE: {len(y_train)} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_tabnet(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> TabNetClassifier:
        """
        Train TabNet model.
        TabNet uses attention for feature selection.
        """
        logger.info("Training TabNet model")
        
        model = TabNetClassifier(
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params=dict(
                mode="min",
                patience=5,
                min_lr=1e-5,
                factor=0.5
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            mask_type='entmax',
            verbose=0
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc', 'accuracy'],
            max_epochs=200,
            patience=20,
            batch_size=256,
            virtual_batch_size=128
        )
        
        return model
    
    def train_deep_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        use_attention: bool = True
    ) -> DeepClassifier:
        """
        Train deep neural network with attention.
        """
        logger.info(f"Training deep classifier (attention={use_attention})")
        
        input_dim = X_train.shape[1]
        model = DeepClassifier(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128, 64],
            dropout=0.3,
            use_attention=use_attention
        )
        
        train_dataset = SECOMDataset(X_train, y_train)
        val_dataset = SECOMDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, batch_size=128,
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=256,
            shuffle=False, num_workers=0
        )
        
        trainer = ModelTrainer(model, device=self.device, use_focal_loss=True)
        model = trainer.fit(
            train_loader, val_loader,
            epochs=100, lr=0.001, patience=15
        )
        
        return model
    
    def train_autoencoder(
        self,
        X_train: np.ndarray,
        latent_dim: int = 64
    ) -> Autoencoder:
        """
        Train autoencoder for anomaly detection.
        Uses only normal samples for training.
        """
        logger.info("Training autoencoder for anomaly detection")
        
        input_dim = X_train.shape[1]
        model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
        model.to(self.device)
        
        dataset = SECOMDataset(X_train, np.zeros(len(X_train)))
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = model(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"Autoencoder epoch {epoch+1}/50 - Loss: {avg_loss:.4f}")
        
        return model
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_models: int = 5
    ) -> DeepEnsemble:
        """
        Train ensemble of deep models for uncertainty estimation.
        """
        logger.info(f"Training ensemble of {n_models} models")
        
        input_dim = X_train.shape[1]
        ensemble = DeepEnsemble(
            n_models=n_models,
            input_dim=input_dim,
            hidden_dims=[256, 128, 64],
            dropout=0.3
        )
        
        train_dataset = SECOMDataset(X_train, y_train)
        val_dataset = SECOMDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        for i, model in enumerate(ensemble.models):
            logger.info(f"Training ensemble model {i+1}/{n_models}")
            trainer = ModelTrainer(model, device=self.device)
            trainer.fit(train_loader, val_loader, epochs=50, patience=10)
        
        return ensemble
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str
    ) -> Dict:
        """
        Evaluate model and return metrics.
        """
        if model_type == 'tabnet':
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        elif model_type == 'deep':
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test).to(self.device)
                logits = model(X_tensor)
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        elif model_type == 'ensemble':
            probs, uncertainty = model.predict_proba(X_test)
            y_pred = np.argmax(probs, axis=1)
            y_proba = probs[:, 1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        logger.info(f"{model_type} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def train_all_models(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Train all advanced models and return results.
        """
        logger.info("Starting advanced model training")
        
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, use_smote=True)
        X_train_raw, X_val, y_train_raw, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        
        results = {}
        
        # TabNet
        try:
            tabnet = self.train_tabnet(X_train_raw, y_train_raw, X_val, y_val)
            self.models['tabnet'] = tabnet
            results['tabnet'] = self.evaluate_model(tabnet, X_test, y_test, 'tabnet')
        except Exception as e:
            logger.error(f"TabNet training failed: {e}")
        
        # Deep classifier with attention
        try:
            deep_model = self.train_deep_classifier(
                X_train_raw, y_train_raw, X_val, y_val, use_attention=True
            )
            self.models['deep_attention'] = deep_model
            results['deep_attention'] = self.evaluate_model(deep_model, X_test, y_test, 'deep')
        except Exception as e:
            logger.error(f"Deep classifier training failed: {e}")
        
        # Autoencoder
        try:
            normal_samples = X_train_raw[y_train_raw == 0]
            autoencoder = self.train_autoencoder(normal_samples)
            self.models['autoencoder'] = autoencoder
        except Exception as e:
            logger.error(f"Autoencoder training failed: {e}")
        
        # Deep ensemble
        try:
            ensemble = self.train_ensemble(X_train_raw, y_train_raw, X_val, y_val, n_models=3)
            self.models['ensemble'] = ensemble
            results['ensemble'] = self.evaluate_model(ensemble, X_test, y_test, 'ensemble')
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
        
        self.results = results
        return results
    
    def save_models(self, output_dir: Path):
        """Save trained models."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.models.items():
            model_path = output_dir / f"{name}_{timestamp}.pt"
            
            if name == 'tabnet':
                model.save_model(str(model_path))
            else:
                torch.save(model.state_dict(), model_path)
            
            logger.info(f"Saved {name} to {model_path}")
        
        scaler_path = output_dir / f"scaler_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        results_path = output_dir / f"results_{timestamp}.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
