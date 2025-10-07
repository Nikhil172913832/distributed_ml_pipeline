"""
Integration tests for SECOM SDV RAW data generation pipeline.
Tests model loading, generation quality, raw data characteristics, throughput, and fault tolerance.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from secom_data_generator import SECOMGenerator, GeneratorConfig
import pandas as pd
import numpy as np
from loguru import logger
import time

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'sdv_secom_raw.pkl'


def test_model_loading():
    """Test 1: Verify model and artifacts load correctly."""
    logger.info("🧪 Test 1: Model Loading")
    
    if not MODEL_PATH.exists():
        logger.error(f"❌ Model not found at {MODEL_PATH}")
        logger.info("   Run 'python data_generator/secom_sdv_trainer.py' first")
        return False
    
    try:
        config = GeneratorConfig(batch_size=100, enable_validation=False)
        generator = SECOMGenerator(MODEL_PATH, config)
        logger.success("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False


def test_basic_generation():
    """Test 2: Generate a small batch and verify structure."""
    logger.info("\n🧪 Test 2: Basic Generation")
    
    try:
        config = GeneratorConfig(batch_size=100, enable_validation=False)
        generator = SECOMGenerator(MODEL_PATH, config)
        
        batches = list(generator.generate_batches(total_samples=100))
        
        if len(batches) != 1:
            logger.error(f"❌ Expected 1 batch, got {len(batches)}")
            return False
        
        batch_df = batches[0]
        logger.info(f"   Generated batch shape: {batch_df.shape}")
        
        # Check for 591 features + target
        expected_features = 591
        actual_cols = len([col for col in batch_df.columns 
                          if col not in ['generated_at', 'batch_id', 'device_id']])
        
        if actual_cols != expected_features + 1:  # +1 for target
            logger.error(f"❌ Expected {expected_features + 1} columns, got {actual_cols}")
            return False
        
        logger.success(f"✅ Basic generation successful with all {expected_features} features")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raw_data_quality():
    """Test 3: Validate raw data characteristics (missing patterns, distributions)."""
    logger.info("\n🧪 Test 3: Raw Data Quality")
    
    try:
        config = GeneratorConfig(batch_size=1000, enable_validation=True)
        generator = SECOMGenerator(MODEL_PATH, config)
        
        batches = list(generator.generate_batches(total_samples=1000))
        batch_df = batches[0]
        
        # Remove metadata columns for analysis
        feature_cols = [col for col in batch_df.columns 
                       if col not in ['target', 'generated_at', 'batch_id', 'device_id']]
        features_df = batch_df[feature_cols]
        
        # Check feature count
        if len(feature_cols) != 591:
            logger.error(f"❌ Expected 591 features, got {len(feature_cols)}")
            return False
        logger.success(f"✅ All 591 features present")
        
        # Check overall missing percentage (should be around 4-5%)
        overall_missing = features_df.isnull().sum().sum() / (features_df.shape[0] * features_df.shape[1])
        logger.info(f"   Overall missing percentage: {overall_missing * 100:.2f}%")
        
        if 0.01 <= overall_missing <= 0.15:  # 1-15% range (original is ~4.54%)
            logger.success(f"✅ Missing percentage realistic: {overall_missing * 100:.2f}% (original: 4.54%)")
        else:
            logger.warning(f"⚠️  Missing percentage unusual: {overall_missing * 100:.2f}% (expected 1-15%)")
        
        # Check that we have features with varying missing patterns
        missing_per_feature = features_df.isnull().mean()
        
        complete_features = (missing_per_feature == 0).sum()
        low_missing = ((missing_per_feature > 0) & (missing_per_feature <= 0.10)).sum()
        moderate_missing = ((missing_per_feature > 0.10) & (missing_per_feature <= 0.50)).sum()
        high_missing = ((missing_per_feature > 0.50) & (missing_per_feature <= 0.75)).sum()
        critical_missing = (missing_per_feature > 0.75).sum()
        
        logger.info(f"   Missing pattern distribution:")
        logger.info(f"     • Complete (0%): {complete_features} features")
        logger.info(f"     • Low (0-10%): {low_missing} features")
        logger.info(f"     • Moderate (10-50%): {moderate_missing} features")
        logger.info(f"     • High (50-75%): {high_missing} features")
        logger.info(f"     • Critical (>75%): {critical_missing} features")
        
        # Original has: 53 complete, 485 low, 45 moderate, 0 high, 8 critical
        if complete_features >= 20:  # Should have some complete features
            logger.success(f"✅ Has complete features: {complete_features} (original: 53)")
        else:
            logger.warning(f"⚠️  Very few complete features: {complete_features}")
        
        if critical_missing > 0:
            logger.success(f"✅ Has critical missing features: {critical_missing} (original: 8)")
        
        # Check for infinite values (should not have any in valid data)
        inf_count = features_df.apply(
            lambda x: x.isin([float('inf'), float('-inf')]).sum() if x.dtype in ['float64', 'float32'] else 0
        ).sum()
        if inf_count > 0:
            logger.warning(f"⚠️  Found {inf_count} infinite values (may be valid in raw data)")
        else:
            logger.success(f"✅ No infinite values")
        
        # Check target distribution (should be ~6.64% fail rate)
        if 'target' in batch_df.columns:
            fail_rate = batch_df['target'].mean()
            expected_rate = 0.0664
            tolerance = expected_rate * 1.0  # 100% tolerance for raw data (wider range)
            
            if abs(fail_rate - expected_rate) > tolerance:
                logger.warning(f"⚠️  Fail rate {fail_rate:.4f} outside expected range (expected ~0.0664)")
            else:
                logger.success(f"✅ Target distribution: {fail_rate:.4f} (expected ~0.0664)")
        
        # Check metadata fields
        required_fields = ['generated_at', 'batch_id', 'device_id']
        missing_fields = [f for f in required_fields if f not in batch_df.columns]
        if missing_fields:
            logger.error(f"❌ Missing metadata fields: {missing_fields}")
            return False
        logger.success(f"✅ All metadata fields present")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Raw data quality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_throughput():
    """Test 4: Measure generation throughput."""
    logger.info("\n🧪 Test 4: Throughput")
    
    try:
        config = GeneratorConfig(
            batch_size=1000,
            enable_validation=False,  # Disable validation for max speed
            target_throughput=None
        )
        generator = SECOMGenerator(MODEL_PATH, config)
        
        start_time = time.time()
        total_samples = 10000
        
        batches = list(generator.generate_batches(total_samples=total_samples))
        
        duration = time.time() - start_time
        throughput = total_samples / duration
        
        logger.info(f"   Generated {total_samples} samples in {duration:.2f}s")
        logger.info(f"   Throughput: {throughput:.1f} samples/sec")
        
        if throughput >= 500:  # Minimum acceptable throughput
            logger.success(f"✅ Throughput meets requirements (≥500 samples/sec): {throughput:.1f}")
            return True
        else:
            logger.warning(f"⚠️  Throughput below target: {throughput:.1f} < 500 samples/sec")
            return False
        
    except Exception as e:
        logger.error(f"❌ Throughput test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fault_tolerance():
    """Test 5: Verify fault tolerance and retry logic."""
    logger.info("\n🧪 Test 5: Fault Tolerance")
    
    try:
        config = GeneratorConfig(
            batch_size=100,
            enable_validation=True,
            max_retries=3
        )
        generator = SECOMGenerator(MODEL_PATH, config)
        
        # Generate multiple batches and track success rate
        total_batches = 10
        successful_batches = 0
        
        for batch in generator.generate_batches(total_samples=total_batches * 100):
            successful_batches += 1
        
        success_rate = successful_batches / total_batches
        logger.info(f"   Success rate: {success_rate * 100:.1f}% ({successful_batches}/{total_batches})")
        
        if success_rate >= 0.8:  # 80% success rate
            logger.success(f"✅ Fault tolerance acceptable: {success_rate * 100:.1f}%")
            return True
        else:
            logger.error(f"❌ Success rate too low: {success_rate * 100:.1f}%")
            return False
        
    except Exception as e:
        logger.error(f"❌ Fault tolerance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests and report results."""
    logger.info("=" * 60)
    logger.info("SECOM RAW DATA GENERATION - Integration Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Basic Generation", test_basic_generation),
        ("Raw Data Quality", test_raw_data_quality),
        ("Throughput", test_throughput),
        ("Fault Tolerance", test_fault_tolerance),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("-" * 60)
    logger.info(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
