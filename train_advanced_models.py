"""
Train advanced models on SECOM data.

Usage:
    python train_advanced_models.py --data-days 7 --models tabnet,deep,ensemble
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from database import DatabaseManager
from advanced_trainer import AdvancedModelTrainer


def load_training_data(db_manager: DatabaseManager, days: int = 7):
    """Load preprocessed data from database."""
    query = f"""
    SELECT features, target
    FROM secom.preprocessed_data
    WHERE created_at >= NOW() - INTERVAL '{days} days'
    ORDER BY created_at DESC
    """
    
    with db_manager.get_connection() as conn:
        df = pd.read_sql(query, conn)
    
    if len(df) == 0:
        raise ValueError("No training data found")
    
    logger.info(f"Loaded {len(df)} samples from last {days} days")
    
    features = np.array([list(f.values()) for f in df['features']])
    targets = df['target'].values
    
    # Convert target from -1/1 to 0/1
    targets = (targets + 1) // 2
    
    return features, targets


def main():
    parser = argparse.ArgumentParser(description='Train advanced ML models')
    parser.add_argument('--data-days', type=int, default=7,
                       help='Days of data to use for training')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models to train')
    parser.add_argument('--output-dir', type=str, default='./models/advanced',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    logger.info("Starting advanced model training")
    logger.info(f"Data days: {args.data_days}")
    logger.info(f"Models: {args.models}")
    
    db_manager = DatabaseManager()
    
    try:
        X, y = load_training_data(db_manager, args.data_days)
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        config = {
            'models': args.models.split(','),
            'output_dir': args.output_dir
        }
        
        trainer = AdvancedModelTrainer(config)
        results = trainer.train_all_models(X, y)
        
        logger.info("Training complete")
        logger.info("Results:")
        for model_name, metrics in results.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"    F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"    ROC AUC: {metrics['roc_auc']:.4f}")
        
        output_dir = Path(args.output_dir)
        trainer.save_models(output_dir)
        
        logger.info(f"Models saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    finally:
        db_manager.close_all()


if __name__ == '__main__':
    main()
