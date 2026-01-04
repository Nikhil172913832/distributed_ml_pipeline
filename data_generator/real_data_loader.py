"""
Real SECOM data loader with time-based batching and drift simulation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger


class SECOMDataLoader:
    
    def __init__(self, data_path: str, labels_path: str):
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path)
        self.data = None
        self.labels = None
        self.current_index = 0
        self._load_data()
    
    def _load_data(self):
        """Load SECOM dataset from files."""
        logger.info(f"Loading SECOM data from {self.data_path}")
        
        # Load features (590 columns, space-separated)
        self.data = pd.read_csv(self.data_path, sep=' ', header=None)
        self.data.columns = [f'feature_{i}' for i in range(len(self.data.columns))]
        
        # Load labels (pass=-1, fail=1)
        labels_df = pd.read_csv(self.labels_path, sep=' ', header=None)
        self.labels = labels_df[0].values
        
        # Convert labels: -1 (pass) and 1 (fail)
        self.labels = np.where(self.labels == -1, -1, 1)
        
        # Add timestamp (simulate chronological order)
        base_time = datetime(2020, 1, 1)
        self.data['timestamp'] = [
            base_time + timedelta(seconds=i*300) for i in range(len(self.data))
        ]
        
        logger.info(f"Loaded {len(self.data)} samples")
        logger.info(f"Class distribution: Pass={np.sum(self.labels == -1)}, Fail={np.sum(self.labels == 1)}")
    
    def get_batch(self, batch_size: int = 100, batch_id: Optional[str] = None) -> List[Dict]:
        """Get next batch of samples in chronological order."""
        if self.current_index >= len(self.data):
            # Replay from beginning
            self.current_index = 0
            logger.info("Replaying dataset from beginning")
        
        end_index = min(self.current_index + batch_size, len(self.data))
        batch_data = self.data.iloc[self.current_index:end_index]
        batch_labels = self.labels[self.current_index:end_index]
        
        samples = []
        for idx, (_, row) in enumerate(batch_data.iterrows()):
            features = row.drop('timestamp').to_dict()
            samples.append({
                'batch_id': batch_id or f"batch_{self.current_index}",
                'sample_index': idx,
                'features': {k: float(v) if not pd.isna(v) else None for k, v in features.items()},
                'target': int(batch_labels[idx]),
                'timestamp': row['timestamp'].isoformat()
            })
        
        self.current_index = end_index
        return samples
    
    def get_train_test_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Get time-based train/test split."""
        split_idx = int(len(self.data) * (1 - test_size))
        
        X_train = self.data.iloc[:split_idx].drop('timestamp', axis=1)
        X_test = self.data.iloc[split_idx:].drop('timestamp', axis=1)
        y_train = self.labels[:split_idx]
        y_test = self.labels[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
    
    def reset(self):
        """Reset batch index to beginning."""
        self.current_index = 0


class DriftSimulator:
    
    def __init__(self, drift_start_ratio: float = 0.8, drift_magnitude: float = 0.3):
        self.drift_start_ratio = drift_start_ratio
        self.drift_magnitude = drift_magnitude
        self.total_samples_seen = 0
        self.drift_started = False
    
    def apply_drift(self, features: Dict[str, float], sample_index: int, total_samples: int) -> Dict[str, float]:
        """Apply gradual drift to features."""
        drift_start = int(total_samples * self.drift_start_ratio)
        
        if sample_index < drift_start:
            return features  # No drift yet
        
        if not self.drift_started:
            logger.warning(f"Drift simulation started at sample {sample_index}")
            self.drift_started = True
        
        # Calculate drift progress (0 to 1)
        drift_progress = min(1.0, (sample_index - drift_start) / (total_samples * 0.1))
        
        # Apply drift to subset of features
        drifted_features = features.copy()
        feature_keys = list(features.keys())
        
        # Drift 20% of features
        num_drift_features = int(len(feature_keys) * 0.2)
        drift_feature_indices = np.random.RandomState(42).choice(
            len(feature_keys), num_drift_features, replace=False
        )
        
        for idx in drift_feature_indices:
            key = feature_keys[idx]
            if features[key] is not None:
                # Gradual shift in mean
                shift = self.drift_magnitude * drift_progress * np.random.randn()
                drifted_features[key] = features[key] + shift
        
        return drifted_features
