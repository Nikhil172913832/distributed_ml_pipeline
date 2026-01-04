"""
Benchmark script to compare all model types.

Usage:
    python benchmark_models.py --data-days 7
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from database import DatabaseManager
from model_comparison import run_model_comparison


def load_test_data(db_manager: DatabaseManager, days: int = 7):
    """Load test data from database."""
    query = f"""
    SELECT features, target
    FROM secom.preprocessed_data
    WHERE created_at >= NOW() - INTERVAL '{days} days'
    ORDER BY created_at DESC
    LIMIT 1000
    """
    
    with db_manager.get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    if len(df) == 0:
        raise ValueError("No test data found")
    
    features = np.array([list(f.values()) for f in df['features']])
    targets = df['target'].values
    targets = (targets + 1) // 2
    
    return features, targets


def load_trained_models(models_dir: Path):
    """Load all available trained models."""
    import joblib
    import torch
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    models = {}
    
    # Load sklearn models if available
    for model_file in models_dir.glob("*.joblib"):
        if 'scaler' not in model_file.name and 'preprocessing' not in model_file.name:
            try:
                model = joblib.load(model_file)
                model_name = model_file.stem
                models[model_name] = (model, 'sklearn')
                logger.info(f"Loaded sklearn model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
    
    # Load PyTorch models if available
    for model_file in models_dir.glob("*.pt"):
        if 'tabnet' in model_file.name:
            try:
                from pytorch_tabnet.tab_model import TabNetClassifier
                model = TabNetClassifier()
                model.load_model(str(model_file))
                models['tabnet'] = (model, 'tabnet')
                logger.info("Loaded TabNet model")
            except Exception as e:
                logger.warning(f"Failed to load TabNet: {e}")
    
    return models


def main():
    parser = argparse.ArgumentParser(description='Benchmark all models')
    parser.add_argument('--data-days', type=int, default=7,
                       help='Days of data to use for testing')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='./benchmarks/results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    logger.info("Starting model benchmark")
    
    db_manager = DatabaseManager()
    
    try:
        X_test, y_test = load_test_data(db_manager, args.data_days)
        logger.info(f"Test data shape: {X_test.shape}")
        
        models_dir = Path(args.models_dir)
        models = load_trained_models(models_dir)
        
        if not models:
            logger.error("No models found to benchmark")
            sys.exit(1)
        
        logger.info(f"Found {len(models)} models to benchmark")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison = run_model_comparison(models, X_test, y_test, output_dir)
        
        logger.info("Benchmark complete")
        logger.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
    finally:
        db_manager.close_all()


if __name__ == '__main__':
    main()
