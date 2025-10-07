"""
CLI for SECOM Raw Data Generation
==================================

Command-line interface for training and generating raw SECOM manufacturing data.

Uses TVAE (Variational Autoencoder) synthesizer instead of CTGAN because:
- SECOM has 591 high-dimensional features (TVAE handles this better)
- Heavy missing data patterns (TVAE is more robust)
- Imbalanced classes (TVAE preserves minority class better)
- Faster training on high-dimensional tabular data

Commands:
    train     - Train TVAE model on raw SECOM data
    generate  - Generate fixed number of raw samples
    stream    - Generate continuous stream of raw data
    validate  - Validate trained model quality

Examples:
    # Train model (TVAE needs fewer epochs than CTGAN)
    python raw_cli.py train --epochs 300

    # Generate 100,000 samples
    python raw_cli.py generate --samples 100000 --batch-size 1000

    # Stream for 5 minutes at 1500 samples/sec
    python raw_cli.py stream --duration 300 --throughput 1500

    # Validate model
    python raw_cli.py validate --sample-size 1000
"""

import click
from pathlib import Path
from loguru import logger
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from secom_raw_trainer import main as train_main, CONFIG as TRAIN_CONFIG
from secom_raw_generator import SECOMGenerator, GeneratorConfig


@click.group()
def cli():
    """SECOM Raw Data Generation CLI"""
    pass


@cli.command()
@click.option('--epochs', default=300, help='Number of training epochs (TVAE needs fewer than CTGAN)')
@click.option('--batch-size', default=500, help='Training batch size')
@click.option('--cuda/--no-cuda', default=True, help='Use CUDA GPU acceleration (default: enabled)')
@click.option('--gpu-ids', default='0,1', help='GPU device IDs to use (comma-separated, e.g., "0,1" for both T4s)')
def train(epochs, batch_size, cuda, gpu_ids):
    """
    Train SDV model on raw SECOM data.
    
    This trains a TVAE model that generates RAW data with realistic
    missing patterns, ready for a separate preprocessing pipeline.
    
    TVAE is used instead of CTGAN because:
    - Better for high-dimensional tabular data (591 features)
    - More robust with heavy missing data patterns
    - Faster training on high-dimensional datasets
    
    GPU Support:
    - Automatically uses available GPUs (T4 recommended)
    - Use --gpu-ids to specify which GPUs to use
    - Example: --gpu-ids "0,1" uses both T4 GPUs
    """
    logger.info(f"Training TVAE with epochs={epochs}, batch_size={batch_size}, cuda={cuda}")
    
    if cuda:
        import torch
        if torch.cuda.is_available():
            # Parse GPU IDs
            device_ids = [int(x.strip()) for x in gpu_ids.split(',')]
            logger.info(f"GPU acceleration enabled - Using GPUs: {device_ids}")
            TRAIN_CONFIG['gpu']['device_ids'] = device_ids
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            cuda = False
    
    # Update config
    TRAIN_CONFIG['tvae_params']['epochs'] = epochs
    TRAIN_CONFIG['tvae_params']['batch_size'] = batch_size
    TRAIN_CONFIG['tvae_params']['cuda'] = cuda
    
    # Run training
    train_main()


@cli.command()
@click.option('--samples', default=10000, help='Number of samples to generate')
@click.option('--batch-size', default=1000, help='Batch size for generation')
@click.option('--output-format', default='json', type=click.Choice(['json', 'csv', 'kafka']),
              help='Output format')
@click.option('--output-dir', default='generated_data/secom_raw', help='Output directory')
@click.option('--validate/--no-validate', default=True, help='Enable validation')
@click.option('--model-path', default='models/sdv_secom_raw.joblib', help='Path to trained model')
def generate(samples, batch_size, output_format, output_dir, validate, model_path):
    """
    Generate a fixed number of raw SECOM samples.
    
    Outputs all 591 original features with realistic missing patterns.
    """
    logger.info(f"Generating {samples:,} raw samples...")
    
    config = GeneratorConfig(
        batch_size=batch_size,
        target_throughput=1000,
        output_format=output_format,
        output_dir=output_dir,
        enable_validation=validate,
        add_metadata=True
    )
    
    generator = SECOMGenerator(model_path, config)
    
    # Generate samples
    for batch in generator.generate_stream(total_samples=samples):
        pass  # Batches are saved automatically
    
    logger.info(f"✓ Generation complete! Output: {output_dir}")


@cli.command()
@click.option('--duration', default=60, help='Duration in seconds')
@click.option('--throughput', default=1000, help='Target samples per second')
@click.option('--batch-size', default=1000, help='Batch size for generation')
@click.option('--output-format', default='json', type=click.Choice(['json', 'csv', 'kafka']),
              help='Output format')
@click.option('--output-dir', default='generated_data/secom_raw', help='Output directory')
@click.option('--model-path', default='models/sdv_secom_raw.joblib', help='Path to trained model')
def stream(duration, throughput, batch_size, output_format, output_dir, model_path):
    """
    Generate continuous stream of raw SECOM data.
    
    Useful for real-time simulation of manufacturing processes.
    """
    logger.info(f"Streaming raw data for {duration}s at {throughput} samples/sec...")
    
    config = GeneratorConfig(
        batch_size=batch_size,
        target_throughput=throughput,
        output_format=output_format,
        output_dir=output_dir,
        enable_validation=True,
        add_metadata=True
    )
    
    generator = SECOMGenerator(model_path, config)
    
    # Stream for duration
    for batch in generator.generate_stream(duration_seconds=duration):
        logger.info(f"Streamed batch: {len(batch)} samples, "
                   f"{batch.isnull().sum().sum()} NaN values")
    
    logger.info(f"✓ Streaming complete!")


@cli.command()
@click.option('--sample-size', default=1000, help='Number of validation samples')
@click.option('--model-path', default='models/sdv_secom_raw.joblib', help='Path to trained model')
def validate(sample_size, model_path):
    """
    Validate trained model by generating samples and checking quality.
    
    Reports on:
    - Feature count matching
    - Missing data patterns
    - Value range distributions
    - Target class balance
    """
    import joblib
    from secom_raw_trainer import validate_raw_data_quality
    
    logger.info(f"Validating model with {sample_size} samples...")
    
    # Load model
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    original_stats = artifacts['original_stats']
    
    # Generate samples
    logger.info("Generating validation samples...")
    samples = model.sample(sample_size)
    
    # Extract features only
    feature_cols = [col for col in samples.columns if col != 'target']
    
    # Validate
    results = validate_raw_data_quality(
        original_data=None,  # Not needed for basic validation
        synthetic_data=samples[feature_cols],
        original_stats=original_stats,
        tolerance=0.10
    )
    
    # Report results
    logger.info("\n" + "="*80)
    logger.info("Validation Results:")
    logger.info("="*80)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {check}: {status}")
    
    # Show sample statistics
    logger.info("\nSample Statistics:")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Missing values: {samples[feature_cols].isnull().sum().sum()} "
               f"({samples[feature_cols].isnull().sum().sum() / (len(samples) * len(feature_cols)) * 100:.2f}%)")
    
    if 'target' in samples.columns:
        fail_rate = (samples['target'] == 1).sum() / len(samples)
        logger.info(f"  Fail rate: {fail_rate:.2%}")
    
    logger.info("="*80)
    
    if all(results.values()):
        logger.info("✓ Model validation PASSED!")
    else:
        logger.warning("⚠ Model validation has issues - review above")


if __name__ == '__main__':
    cli()
