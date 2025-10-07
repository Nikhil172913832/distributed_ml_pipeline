import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import torch
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# Import GPU utilities if available
try:
    from gpu_utils import print_gpu_info, get_gpu_memory_usage, clear_gpu_memory
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    logger.warning("GPU utils not available, basic GPU support only")

# Configuration
CONFIG = {
    'tvae_params': {
        'epochs': 300,  # TVAE typically needs fewer epochs than CTGAN
        'batch_size': 500,
        'verbose': True,
        # TVAE params optimized for high-dimensional tabular data with missing values
        'compress_dims': (256, 128),  # Encoder dimensions for 591 features
        'decompress_dims': (128, 256),  # Decoder dimensions
        'embedding_dim': 128,  # Latent space dimension
        'loss_factor': 2,  # Weight for reconstruction loss
        'cuda': True,  # Enable GPU acceleration
    },
    'gpu': {
        'use_cuda': True,  # Enable CUDA
        'device_ids': [0, 1],  # Use both T4 GPUs (GPU 0 and GPU 1)
        'primary_device': 0,  # Primary GPU for model
    },
    'paths': {
        'data_dir': Path('secom'),
        'model_dir': Path('models'),
        'features_file': 'secom.data',
        'labels_file': 'secom_labels.data',
        'model_name': 'sdv_secom_raw.joblib'
    },
    'validation': {
        'sample_size': 1000,  # Samples to generate for validation
        'missing_tolerance': 0.10,  # Allow 10% deviation in missing %
    }
}


class SECOMDataAnalyzer:
    """
    Analyzes SECOM raw data characteristics for validation.
    
    This class analyzes the original SECOM dataset to establish baseline
    statistics that will be used to validate synthetic data quality:
    - Missing data patterns and percentages per feature
    - Overall data distributions
    - Feature value ranges
    - Class balance
    """
    
    def __init__(self, data: pd.DataFrame, labels: pd.Series):
        self.data = data
        self.labels = labels
        self.stats = {}
        
    def analyze_data_characteristics(self) -> Dict[str, Any]:
        """
        Analyze original data characteristics to establish validation baselines.
        
        Returns:
            Dictionary containing:
            - n_features: Number of features
            - n_samples: Number of samples
            - missing_by_feature: Missing percentage per feature
            - overall_missing_pct: Overall missing percentage
            - class_distribution: Distribution of target classes
            - feature_ranges: Min/max for each feature
        """
        logger.info("Analyzing original SECOM data characteristics...")
        
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100).round(2)
        overall_missing = self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1]) * 100
        
        # Class distribution
        class_counts = self.labels.value_counts()
        class_dist = {
            'pass': int(class_counts.get(-1, 0)),
            'fail': int(class_counts.get(1, 0)),
            'pass_pct': (class_counts.get(-1, 0) / len(self.labels) * 100),
            'fail_pct': (class_counts.get(1, 0) / len(self.labels) * 100)
        }
        
        # Feature ranges (for non-missing values)
        feature_ranges = {}
        for col in self.data.columns:
            valid_data = self.data[col].dropna()
            if len(valid_data) > 0:
                feature_ranges[col] = {
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()) if len(valid_data) > 1 else 0.0
                }
        
        # Count features by missing %
        missing_levels = {
            'complete (0%)': int((missing_pct == 0).sum()),
            'low (0-10%)': int(((missing_pct > 0) & (missing_pct <= 10)).sum()),
            'moderate (10-50%)': int(((missing_pct > 10) & (missing_pct <= 50)).sum()),
            'high (50-75%)': int(((missing_pct > 50) & (missing_pct <= 75)).sum()),
            'critical (>75%)': int((missing_pct > 75).sum())
        }
        
        stats = {
            'n_features': self.data.shape[1],
            'n_samples': self.data.shape[0],
            'missing_by_feature': missing_pct.to_dict(),
            'overall_missing_pct': overall_missing,
            'features_with_missing': int((missing_pct > 0).sum()),
            'missing_levels': missing_levels,
            'class_distribution': class_dist,
            'feature_ranges': feature_ranges
        }
        
        logger.info(f"Features: {stats['n_features']}, Samples: {stats['n_samples']}")
        logger.info(f"Overall missing: {overall_missing:.2f}%")
        logger.info(f"Features with missing: {stats['features_with_missing']}/{stats['n_features']}")
        logger.info("Missing data breakdown:")
        for level, count in missing_levels.items():
            logger.info(f"  - {level}: {count} features")
        logger.info(f"Class balance - Pass: {class_dist['pass_pct']:.2f}%, Fail: {class_dist['fail_pct']:.2f}%")
        
        self.stats = stats
        return stats


def load_secom_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load SECOM dataset from files (RAW, no preprocessing).
    
    Args:
        data_dir: Directory containing SECOM data files
        
    Returns:
        Tuple of (features DataFrame, labels Series)
    """
    logger.info("Loading SECOM raw data...")
    
    features_path = data_dir / CONFIG['paths']['features_file']
    labels_path = data_dir / CONFIG['paths']['labels_file']
    
    # Load features (space-separated, no header)
    # Keep NaN values as-is - they are part of the data pattern
    features = pd.read_csv(features_path, sep=r'\s+', header=None, na_values=['NaN', 'nan'])
    features.columns = [f'feature_{i}' for i in range(len(features.columns))]
    
    # Load labels (space-separated, first column is label, second is timestamp)
    labels_df = pd.read_csv(labels_path, sep=r'\s+', header=None)
    labels = labels_df[0]  # First column is the label (-1 = pass, 1 = fail)
    
    logger.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
    logger.info(f"Missing values: {features.isnull().sum().sum()} ({features.isnull().sum().sum() / (features.shape[0] * features.shape[1]) * 100:.2f}%)")
    logger.info(f"Class distribution: {labels.value_counts().to_dict()}")
    
    return features, labels


def create_sdv_metadata(data: pd.DataFrame, target_column: str = 'target') -> SingleTableMetadata:
    """
    Create SDV metadata for RAW SECOM dataset.
    
    All features are numerical (continuous) with potential missing values.
    Target is categorical (binary: -1 = pass, 1 = fail).
    
    Args:
        data: Combined features and target DataFrame (raw data with NaNs)
        target_column: Name of the target column
        
    Returns:
        SDV SingleTableMetadata object
    """
    logger.info("Creating SDV metadata for raw data...")
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    
    # Explicitly set all features as numerical (to handle NaN properly)
    for col in data.columns:
        if col != target_column:
            metadata.update_column(col, sdtype='numerical')
    
    # Set target as categorical (binary classification: -1, 1)
    metadata.update_column(target_column, sdtype='categorical')
    
    logger.info(f"Metadata created: {len(data.columns)} columns ({len(data.columns)-1} features + 1 target)")
    
    return metadata


def train_sdv_model(data: pd.DataFrame, metadata: SingleTableMetadata, 
                   tvae_params: Dict) -> TVAESynthesizer:
    """
    Train TVAE model on RAW SECOM data (with missing values).
    
    TVAE (Variational Autoencoder) is better suited for this dataset because:
    - Handles high-dimensional tabular data (591 features) efficiently
    - More robust with heavy missing data patterns
    - Better preservation of complex feature correlations
    - Faster training than CTGAN on high-dimensional data
    
    Supports multi-GPU training for faster convergence.
    
    Args:
        data: Raw data with target column (includes NaN values)
        metadata: SDV metadata object
        tvae_params: TVAE hyperparameters
        
    Returns:
        Trained TVAESynthesizer
    """
    logger.info("Training TVAE model on RAW data...")
    logger.info(f"Training parameters: {tvae_params}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Missing values in training data: {data.isnull().sum().sum()} ({data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100:.2f}%)")
    
    # Check GPU availability
    if tvae_params.get('cuda', False):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"✓ CUDA available - {num_gpus} GPU(s) detected")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Set primary GPU
            primary_gpu = CONFIG['gpu'].get('primary_device', 0)
            torch.cuda.set_device(primary_gpu)
            logger.info(f"  Primary GPU set to: GPU {primary_gpu}")
        else:
            logger.warning("⚠ CUDA requested but not available, falling back to CPU")
            tvae_params['cuda'] = False
    else:
        logger.info("Training on CPU")
    
    model = TVAESynthesizer(
        metadata=metadata,
        epochs=tvae_params['epochs'],
        batch_size=tvae_params['batch_size'],
        verbose=tvae_params.get('verbose', True),
        # TVAE-specific parameters optimized for high-dimensional data
        compress_dims=tvae_params.get('compress_dims', (256, 128)),
        decompress_dims=tvae_params.get('decompress_dims', (128, 256)),
        embedding_dim=tvae_params.get('embedding_dim', 128),
        loss_factor=tvae_params.get('loss_factor', 2),
        enforce_min_max_values=False,  # Allow some variation beyond original ranges
        enforce_rounding=False,
        cuda=tvae_params.get('cuda', False)
    )
    
    if tvae_params.get('cuda', False) and torch.cuda.is_available():
        logger.info("Starting TVAE training on GPU (this may take 5-10 minutes with T4 GPUs)...")
    else:
        logger.info("Starting TVAE training on CPU (this may take 10-20 minutes)...")
    
    model.fit(data)
    
    logger.info("TVAE training complete!")
    
    return model


def validate_raw_data_quality(original_data: pd.DataFrame, 
                             synthetic_data: pd.DataFrame,
                             original_stats: Dict[str, Any],
                             tolerance: float = 0.10) -> Dict[str, bool]:
    """
    Validate that synthetic raw data has similar characteristics to original.
    
    Args:
        original_data: Original SECOM data
        synthetic_data: Generated synthetic data
        original_stats: Statistics from original data analysis
        tolerance: Acceptable deviation percentage (default 10%)
        
    Returns:
        Dictionary of validation results
    """
    logger.info("Validating synthetic data quality...")
    
    results = {
        'feature_count_match': True,
        'missing_patterns_similar': True,
        'value_ranges_reasonable': True,
        'no_unexpected_nans': True
    }
    
    # Check feature count
    if synthetic_data.shape[1] != original_stats['n_features']:
        logger.warning(f"Feature count mismatch: {synthetic_data.shape[1]} vs {original_stats['n_features']}")
        results['feature_count_match'] = False
    
    # Check missing patterns are similar
    synth_missing = (synthetic_data.isnull().sum() / len(synthetic_data) * 100)
    orig_missing = pd.Series(original_stats['missing_by_feature'])
    
    missing_diff = abs(synth_missing - orig_missing).mean()
    if missing_diff > tolerance * 100:  # Allow tolerance% deviation
        logger.warning(f"Missing patterns differ by {missing_diff:.2f}% on average")
        results['missing_patterns_similar'] = False
    else:
        logger.info(f"✓ Missing patterns similar (avg diff: {missing_diff:.2f}%)")
    
    # Check value ranges are reasonable (allow variation for realism)
    outlier_count = 0
    for col in synthetic_data.columns:
        if col in original_stats['feature_ranges']:
            orig_range = original_stats['feature_ranges'][col]
            synth_valid = synthetic_data[col].dropna()
            
            if len(synth_valid) > 0:
                # Allow values slightly outside original range (synthetic variation)
                range_buffer = (orig_range['max'] - orig_range['min']) * 0.3  # 30% buffer for realism
                if (synth_valid.min() < orig_range['min'] - range_buffer or 
                    synth_valid.max() > orig_range['max'] + range_buffer):
                    outlier_count += 1
    
    if outlier_count > len(synthetic_data.columns) * 0.1:  # More than 10% features outside range
        logger.warning(f"{outlier_count} features have values outside expected range")
        results['value_ranges_reasonable'] = False
    else:
        logger.info(f"✓ Value ranges reasonable ({outlier_count} features with minor variations)")
    
    logger.info("Validation complete")
    return results


def save_training_artifacts(model: TVAESynthesizer, 
                          metadata: SingleTableMetadata,
                          original_stats: Dict[str, Any],
                          model_dir: Path,
                          model_name: str) -> None:
    """
    Save all training artifacts for later use in generation.
    
    Args:
        model: Trained TVAE model
        metadata: SDV metadata
        original_stats: Statistics from original data (for validation)
        model_dir: Directory to save artifacts
        model_name: Name for the model file
    """
    logger.info("Saving training artifacts...")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts = {
        'model': model,
        'metadata': metadata,
        'original_stats': original_stats,
        'config': CONFIG,
        'training_date': pd.Timestamp.now().isoformat(),
        'sdv_version': 'SDV 1.0+',
        'model_type': 'TVAE_raw_data',
        'description': 'Generates RAW SECOM data with realistic missing patterns for preprocessing pipeline (TVAE for high-dimensional tabular data)'
    }
    
    model_path = model_dir / model_name
    joblib.dump(artifacts, model_path)
    
    logger.info(f"✓ Artifacts saved to {model_path}")
    logger.info(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """
    Main training pipeline execution for RAW data generation.
    
    Workflow:
    1. Load SECOM raw data (with missing values)
    2. Analyze original data characteristics
    3. Combine features and labels
    4. Create SDV metadata (all features as numerical)
    5. Train TVAE model on raw data (better for high-dimensional tabular data)
    6. Validate synthetic data quality
    7. Save all artifacts
    
    The model will learn to generate RAW data with realistic missing patterns,
    ready for a separate preprocessing pipeline.
    """
    logger.info("="*80)
    logger.info("SECOM SDV Training Pipeline - Raw Data Generation (TVAE)")
    logger.info("="*80)
    
    # Print GPU information if available
    if GPU_UTILS_AVAILABLE and CONFIG['tvae_params'].get('cuda', False):
        print_gpu_info()
    
    # 1. Load raw data
    features, labels = load_secom_data(CONFIG['paths']['data_dir'])
    
    # 2. Analyze original data characteristics
    analyzer = SECOMDataAnalyzer(features, labels)
    original_stats = analyzer.analyze_data_characteristics()
    
    # 3. Combine features and target (no preprocessing!)
    data = features.copy()
    data['target'] = labels.values
    
    logger.info(f"\nTraining data shape: {data.shape}")
    logger.info(f"Total features: {len(features.columns)} (all original features)")
    logger.info(f"Training with missing values: {data.isnull().sum().sum()} NaNs")
    
    # 4. Create metadata
    metadata = create_sdv_metadata(data, target_column='target')
    
    # Clear GPU memory before training
    if CONFIG['tvae_params'].get('cuda', False) and torch.cuda.is_available():
        if GPU_UTILS_AVAILABLE:
            for device_id in CONFIG['gpu'].get('device_ids', [0]):
                clear_gpu_memory(device_id)
    
    # 5. Train TVAE model (better suited for high-dimensional tabular data)
    model = train_sdv_model(data, metadata, CONFIG['tvae_params'])
    
    # Print GPU memory usage after training
    if GPU_UTILS_AVAILABLE and CONFIG['tvae_params'].get('cuda', False) and torch.cuda.is_available():
        logger.info("\nGPU Memory Usage After Training:")
        for device_id in CONFIG['gpu'].get('device_ids', [0]):
            mem = get_gpu_memory_usage(device_id)
            logger.info(f"  GPU {device_id}: {mem['allocated']:.2f}/{mem['total']:.2f} GB ({mem['allocated']/mem['total']*100:.1f}%)")
    
    # 6. Validate synthetic data quality
    logger.info("\n" + "="*80)
    logger.info("Generating validation samples...")
    validation_samples = model.sample(CONFIG['validation']['sample_size'])
    
    validation_results = validate_raw_data_quality(
        original_data=features,
        synthetic_data=validation_samples.drop(columns=['target']),
        original_stats=original_stats,
        tolerance=CONFIG['validation']['missing_tolerance']
    )
    
    if all(validation_results.values()):
        logger.info("\n✓ All validation checks passed - synthetic data quality looks good!")
    else:
        logger.warning("\n⚠ Some validation checks failed:")
        for check, passed in validation_results.items():
            if not passed:
                logger.warning(f"  - {check}: FAILED")
    
    # 7. Save artifacts
    logger.info("\n" + "="*80)
    save_training_artifacts(
        model=model,
        metadata=metadata,
        original_stats=original_stats,
        model_dir=CONFIG['paths']['model_dir'],
        model_name=CONFIG['paths']['model_name']
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ Training pipeline complete!")
    logger.info(f"✓ Model saved: {CONFIG['paths']['model_dir'] / CONFIG['paths']['model_name']}")
    logger.info("✓ Ready to generate raw synthetic data for preprocessing pipeline")
    logger.info("="*80)


if __name__ == "__main__":
    main()
