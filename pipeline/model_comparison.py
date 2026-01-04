"""
Model comparison and benchmarking framework.

Compares traditional ML with deep learning models.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from loguru import logger


class ModelComparison:
    """
    Compare multiple models on same test set.
    Tracks performance and inference speed.
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = 'sklearn'
    ) -> Dict:
        """
        Evaluate a single model and record metrics.
        """
        logger.info(f"Evaluating {model_name}")
        
        start_time = time.time()
        
        if model_type == 'sklearn':
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        elif model_type == 'pytorch':
            import torch
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                if torch.cuda.is_available():
                    X_tensor = X_tensor.cuda()
                    model = model.cuda()
                
                logits = model(X_tensor)
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        elif model_type == 'tabnet':
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        inference_time = time.time() - start_time
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'inference_time': inference_time,
            'samples_per_second': len(X_test) / inference_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.results[model_name] = metrics
        
        logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, "
                   f"Speed: {metrics['samples_per_second']:.0f} samples/sec")
        
        return metrics
    
    def compare_all(self) -> pd.DataFrame:
        """
        Create comparison table of all evaluated models.
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC AUC': metrics['roc_auc'],
                'Inference Time (s)': metrics['inference_time'],
                'Samples/sec': metrics['samples_per_second']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 Score', ascending=False)
        
        return df
    
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """Get name of best performing model."""
        if not self.results:
            return None
        
        best_model = max(
            self.results.items(),
            key=lambda x: x[1][metric]
        )
        return best_model[0]
    
    def save_results(self, output_path: Path):
        """Save comparison results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print formatted comparison summary."""
        if not self.results:
            logger.warning("No results to display")
            return
        
        df = self.compare_all()
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        best_f1 = self.get_best_model('f1_score')
        best_speed = self.get_best_model('samples_per_second')
        
        print(f"\nBest F1 Score: {best_f1}")
        print(f"Fastest Inference: {best_speed}")
        print()


def run_model_comparison(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path = None
) -> ModelComparison:
    """
    Run comparison across all provided models.
    
    Args:
        models: Dict of {model_name: (model, model_type)}
        X_test: Test features
        y_test: Test labels
        output_dir: Optional directory to save results
    
    Returns:
        ModelComparison object with results
    """
    comparison = ModelComparison()
    
    for model_name, (model, model_type) in models.items():
        try:
            comparison.evaluate_model(
                model_name, model, X_test, y_test, model_type
            )
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
    
    comparison.print_summary()
    
    if output_dir:
        comparison.save_results(output_dir / 'model_comparison.json')
        
        df = comparison.compare_all()
        df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    return comparison
