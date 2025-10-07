"""
SECOM Raw Data Generator - Production Grade
============================================

High-performance synthetic data generator for SECOM manufacturing dataset.
Generates RAW data with realistic missing patterns for preprocessing pipeline.

Key Features:
- Generates 1000+ samples per second
- Outputs all 591 original features with realistic missing patterns
- Fault-tolerant with automatic retry
- Multiple output formats (JSON, CSV, Kafka-ready)
- Comprehensive logging and monitoring
- NO preprocessing applied - raw data for separate pipeline

Usage:
    from secom_raw_generator import SECOMGenerator, GeneratorConfig
    
    config = GeneratorConfig(batch_size=1000, target_throughput=1000)
    generator = SECOMGenerator('models/sdv_secom_raw.joblib', config)
    
    # Generate batches
    for batch in generator.generate_stream(total_samples=100000):
        # Process batch
        pass
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Generator, Any
from dataclasses import dataclass
import joblib
import json
import time
from datetime import datetime
from loguru import logger
from tqdm import tqdm


@dataclass
class GeneratorConfig:
    """Configuration for SECOM data generator."""
    batch_size: int = 1000
    target_throughput: int = 1000  # Samples per second
    output_format: str = 'json'  # 'json', 'csv', 'kafka'
    output_dir: str = 'generated_data/secom'
    enable_validation: bool = True
    max_retries: int = 3
    add_metadata: bool = True


class DataValidator:
    """Validates generated raw data quality."""
    
    def __init__(self, original_stats: Dict[str, Any], tolerance: float = 0.20):
        """
        Initialize validator with original dataset statistics.
        
        Args:
            original_stats: Statistics from training (feature count, missing patterns, etc.)
            tolerance: Acceptable deviation from original patterns (20% for raw data)
        """
        self.original_stats = original_stats
        self.tolerance = tolerance
        
    def validate_batch(self, batch: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a batch of generated raw data.
        
        Args:
            batch: Generated data batch (with missing values)
            
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check 1: Feature count
        if len(batch.columns) - 1 != self.original_stats['n_features']:  # -1 for target
            issues.append(f"Feature count mismatch: {len(batch.columns)-1} vs {self.original_stats['n_features']}")
        
        # Check 2: No infinite values
        inf_count = np.isinf(batch.select_dtypes(include=[np.number]).values).sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
        
        # Check 3: Target distribution (allow variation for realism)
        if 'target' in batch.columns:
            fail_rate = (batch['target'] == 1).sum() / len(batch)
            expected_fail_rate = self.original_stats['class_distribution']['fail_pct'] / 100
            
            if abs(fail_rate - expected_fail_rate) > self.tolerance:
                issues.append(
                    f"Target distribution off: {fail_rate:.2%} vs {expected_fail_rate:.2%} expected"
                )
        
        # Check 4: Missing patterns (should be realistic, not all zeros)
        feature_cols = [col for col in batch.columns if col != 'target']
        batch_missing_pct = (batch[feature_cols].isnull().sum() / len(batch) * 100).mean()
        expected_missing_pct = self.original_stats['overall_missing_pct']
        
        if abs(batch_missing_pct - expected_missing_pct) > self.tolerance * 100:
            issues.append(
                f"Missing pattern deviation: {batch_missing_pct:.2f}% vs {expected_missing_pct:.2f}% expected"
            )
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'batch_missing_pct': batch_missing_pct,
            'target_distribution': (batch['target'] == 1).sum() / len(batch) if 'target' in batch.columns else None
        }


class SECOMGenerator:
    """
    High-performance generator for RAW SECOM synthetic data.
    
    Generates realistic manufacturing data with natural missing patterns,
    ready for preprocessing pipeline.
    """
    
    def __init__(self, model_path: str, config: GeneratorConfig):
        """
        Initialize generator with trained model.
        
        Args:
            model_path: Path to trained SDV model artifacts
            config: Generator configuration
        """
        self.config = config
        self.model_path = model_path
        
        logger.info(f"Loading model from {model_path}...")
        artifacts = joblib.load(model_path)
        
        self.model = artifacts['model']
        self.metadata = artifacts['metadata']
        self.original_stats = artifacts.get('original_stats', {})
        self.model_config = artifacts.get('config', {})
        
        self.validator = DataValidator(self.original_stats) if config.enable_validation else None
        
        self.stats = {
            'total_generated': 0,
            'total_batches': 0,
            'failed_validations': 0,
            'retry_count': 0
        }
        
        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Original features: {self.original_stats.get('n_features', 'unknown')}")
        logger.info(f"  Missing pattern: {self.original_stats.get('overall_missing_pct', 0):.2f}%")
        logger.info(f"  Batch size: {config.batch_size}")
        
    def _generate_batch_with_retry(self, batch_size: int) -> pd.DataFrame:
        """
        Generate a batch with automatic retry on failure.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Generated batch DataFrame
        """
        for attempt in range(self.config.max_retries):
            try:
                batch = self.model.sample(num_rows=batch_size)
                
                # Validate if enabled
                if self.validator:
                    validation = self.validator.validate_batch(batch)
                    
                    if not validation['valid']:
                        logger.warning(f"Validation failed (attempt {attempt + 1}): {validation['issues']}")
                        self.stats['failed_validations'] += 1
                        
                        if attempt < self.config.max_retries - 1:
                            self.stats['retry_count'] += 1
                            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                            continue
                    else:
                        logger.debug(f"✓ Validation passed - Missing: {validation['batch_missing_pct']:.2f}%")
                
                return batch
                
            except Exception as e:
                logger.error(f"Generation failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    self.stats['retry_count'] += 1
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise
        
        raise RuntimeError(f"Failed to generate batch after {self.config.max_retries} attempts")
    
    def _add_metadata_fields(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Add metadata fields to batch."""
        if self.config.add_metadata:
            batch = batch.copy()
            batch['generated_at'] = datetime.now().isoformat()
            batch['batch_id'] = batch_id
            batch['device_id'] = [f"DEV_{batch_id:06d}_{i:04d}" for i in range(len(batch))]
        
        return batch
    
    def _save_batch(self, batch: pd.DataFrame, batch_id: int) -> None:
        """Save batch to configured output format."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == 'json':
            output_file = output_dir / f"batch_{batch_id:06d}.json"
            batch.to_json(output_file, orient='records', lines=True)
            
        elif self.config.output_format == 'csv':
            output_file = output_dir / f"batch_{batch_id:06d}.csv"
            batch.to_csv(output_file, index=False)
            
        elif self.config.output_format == 'kafka':
            # Kafka integration placeholder
            logger.debug(f"Kafka output for batch {batch_id} (implement kafka_producer.send())")
            # kafka_producer.send('secom-raw-data', batch.to_dict('records'))
        
        logger.debug(f"✓ Saved batch {batch_id} ({len(batch)} samples)")
    
    def generate_stream(self, 
                       total_samples: Optional[int] = None,
                       duration_seconds: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Generate continuous stream of raw data batches.
        
        Args:
            total_samples: Total number of samples to generate (None = infinite)
            duration_seconds: Duration to generate for (None = infinite)
            
        Yields:
            DataFrames containing generated raw batches
        """
        start_time = time.time()
        samples_generated = 0
        batch_id = 1
        
        # Setup progress bar
        if total_samples:
            pbar = tqdm(total=total_samples, desc="Generating samples", unit="samples")
        else:
            pbar = None
        
        logger.info(f"Starting generation stream...")
        logger.info(f"  Target: {total_samples or 'infinite'} samples")
        logger.info(f"  Duration: {duration_seconds or 'infinite'} seconds")
        logger.info(f"  Throughput: {self.config.target_throughput} samples/sec")
        
        try:
            while True:
                # Check stopping conditions
                if total_samples and samples_generated >= total_samples:
                    break
                
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                
                # Calculate batch size for final batch
                batch_size = self.config.batch_size
                if total_samples:
                    remaining = total_samples - samples_generated
                    batch_size = min(batch_size, remaining)
                
                # Generate batch
                batch_start = time.time()
                batch = self._generate_batch_with_retry(batch_size)
                
                # Add metadata
                batch = self._add_metadata_fields(batch, batch_id)
                
                # Save batch
                if self.config.output_format in ['json', 'csv']:
                    self._save_batch(batch, batch_id)
                
                # Update stats
                self.stats['total_generated'] += len(batch)
                self.stats['total_batches'] += 1
                samples_generated += len(batch)
                
                # Throttle to target throughput
                batch_duration = time.time() - batch_start
                target_duration = batch_size / self.config.target_throughput
                
                if batch_duration < target_duration:
                    time.sleep(target_duration - batch_duration)
                
                # Update progress
                if pbar:
                    pbar.update(len(batch))
                
                yield batch
                
                batch_id += 1
                
        finally:
            if pbar:
                pbar.close()
            
            # Log final stats
            elapsed = time.time() - start_time
            actual_throughput = samples_generated / elapsed if elapsed > 0 else 0
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Generation Complete!")
            logger.info(f"{'='*80}")
            logger.info(f"  Total samples: {samples_generated:,}")
            logger.info(f"  Total batches: {self.stats['total_batches']:,}")
            logger.info(f"  Elapsed time: {elapsed:.2f}s")
            logger.info(f"  Throughput: {actual_throughput:.0f} samples/sec")
            logger.info(f"  Failed validations: {self.stats['failed_validations']}")
            logger.info(f"  Retries: {self.stats['retry_count']}")
            logger.info(f"{'='*80}")
    
    def generate_batches(self, num_batches: int) -> Generator[pd.DataFrame, None, None]:
        """
        Generate a fixed number of batches.
        
        Args:
            num_batches: Number of batches to generate
            
        Yields:
            DataFrames containing generated batches
        """
        total_samples = num_batches * self.config.batch_size
        yield from self.generate_stream(total_samples=total_samples)


def main():
    """Example usage of the generator."""
    
    # Configure generator
    config = GeneratorConfig(
        batch_size=1000,
        target_throughput=1000,
        output_format='json',
        output_dir='generated_data/secom_raw',
        enable_validation=True,
        add_metadata=True
    )
    
    # Initialize generator
    generator = SECOMGenerator(
        model_path='models/sdv_secom_raw.joblib',
        config=config
    )
    
    # Generate 10,000 samples
    logger.info("Generating 10,000 raw SECOM samples...")
    
    for batch in generator.generate_stream(total_samples=10000):
        logger.info(f"Generated batch: {len(batch)} samples, {batch.isnull().sum().sum()} NaNs")
        
        # Show sample of first batch
        if generator.stats['total_batches'] == 1:
            logger.info(f"\nSample data (first 3 rows):")
            logger.info(f"\n{batch.head(3)}")


if __name__ == "__main__":
    main()
