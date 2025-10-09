#!/usr/bin/env python3
"""
Pipeline Performance Benchmark Script

This script measures the actual throughput and latency of the ML pipeline
by querying the database and Prometheus metrics.

Usage:
    python benchmarks/benchmark_pipeline.py [--duration 60]
"""

import argparse
import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.database import Database, PredictionRepository, PreprocessedDataRepository


class PipelineBenchmark:
    def __init__(self):
        self.db = Database()
        self.pred_repo = PredictionRepository(self.db.get_connection())
        self.prep_repo = PreprocessedDataRepository(self.db.get_connection())
    
    def measure_prediction_throughput(self, duration_seconds=60):
        """
        Measure prediction throughput over a time period.
        
        Returns:
            dict: Benchmark results
        """
        print(f"Measuring prediction throughput for {duration_seconds} seconds...")
        print("Make sure producer, consumer, and inference are running!")
        print()
        
        # Get initial count
        start_time = time.time()
        initial_count = self._get_prediction_count()
        
        if initial_count == 0:
            print("WARNING: No predictions found. Is the inference service running?")
        
        print(f"Initial predictions: {initial_count}")
        print(f"Waiting {duration_seconds} seconds...")
        
        # Wait
        time.sleep(duration_seconds)
        
        # Get final count
        end_time = time.time()
        final_count = self._get_prediction_count()
        
        # Calculate metrics
        elapsed = end_time - start_time
        predictions_made = final_count - initial_count
        throughput = predictions_made / elapsed if elapsed > 0 else 0
        
        return {
            'duration_seconds': elapsed,
            'predictions_made': predictions_made,
            'throughput_per_second': throughput,
            'initial_count': initial_count,
            'final_count': final_count,
        }
    
    def measure_preprocessing_throughput(self, duration_seconds=60):
        """
        Measure preprocessing throughput over a time period.
        
        Returns:
            dict: Benchmark results
        """
        print(f"Measuring preprocessing throughput for {duration_seconds} seconds...")
        
        # Get initial count
        start_time = time.time()
        initial_count = self._get_preprocessed_count()
        
        print(f"Initial preprocessed records: {initial_count}")
        print(f"Waiting {duration_seconds} seconds...")
        
        # Wait
        time.sleep(duration_seconds)
        
        # Get final count
        end_time = time.time()
        final_count = self._get_preprocessed_count()
        
        # Calculate metrics
        elapsed = end_time - start_time
        records_processed = final_count - initial_count
        throughput = records_processed / elapsed if elapsed > 0 else 0
        
        return {
            'duration_seconds': elapsed,
            'records_processed': records_processed,
            'throughput_per_second': throughput,
            'initial_count': initial_count,
            'final_count': final_count,
        }
    
    def measure_latency(self, sample_size=100):
        """
        Estimate end-to-end latency by comparing timestamps.
        
        This is an approximation based on database timestamps.
        
        Returns:
            dict: Latency statistics
        """
        print(f"Measuring latency (sample size: {sample_size})...")
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Query to get latency between preprocessing and prediction
        query = """
            SELECT 
                EXTRACT(EPOCH FROM (p.created_at - prep.created_at)) as latency_seconds
            FROM secom.predictions p
            JOIN secom.preprocessed_data prep ON prep.id = p.preprocessed_data_id
            WHERE p.created_at > NOW() - INTERVAL '1 hour'
            ORDER BY p.created_at DESC
            LIMIT %s
        """
        
        cursor.execute(query, (sample_size,))
        results = cursor.fetchall()
        
        if not results:
            print("WARNING: No predictions found in the last hour")
            return {
                'sample_size': 0,
                'mean_latency_ms': 0,
                'p50_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
            }
        
        # Calculate statistics
        latencies = [row[0] * 1000 for row in results]  # Convert to ms
        latencies.sort()
        
        n = len(latencies)
        
        return {
            'sample_size': n,
            'mean_latency_ms': sum(latencies) / n,
            'min_latency_ms': latencies[0],
            'max_latency_ms': latencies[-1],
            'p50_latency_ms': latencies[int(n * 0.50)],
            'p95_latency_ms': latencies[int(n * 0.95)],
            'p99_latency_ms': latencies[int(n * 0.99)],
        }
    
    def get_model_performance(self):
        """
        Get current model performance metrics.
        
        Returns:
            dict: Model performance
        """
        print("Querying model performance...")
        
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                accuracy,
                precision,
                recall,
                f1_score
            FROM secom.model_performance_metrics
            ORDER BY window_start DESC
            LIMIT 1
        """
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        if not result:
            print("WARNING: No model performance metrics found")
            return {}
        
        return {
            'accuracy': float(result[0]) if result[0] else 0,
            'precision': float(result[1]) if result[1] else 0,
            'recall': float(result[2]) if result[2] else 0,
            'f1_score': float(result[3]) if result[3] else 0,
        }
    
    def _get_prediction_count(self):
        """Get total number of predictions."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM secom.predictions")
        return cursor.fetchone()[0]
    
    def _get_preprocessed_count(self):
        """Get total number of preprocessed records."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM secom.preprocessed_data")
        return cursor.fetchone()[0]


def print_results(results):
    """Pretty print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    for section, data in results.items():
        print(f"\n{section.upper().replace('_', ' ')}:")
        print("-" * 60)
        for key, value in data.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)


def save_results(results, output_file):
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'benchmark_script': 'benchmark_pipeline.py',
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ML Pipeline Performance')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration to measure throughput (seconds, default: 60)')
    parser.add_argument('--output', type=str,
                        default='benchmarks/results/benchmark_latest.json',
                        help='Output file for results')
    parser.add_argument('--no-latency', action='store_true',
                        help='Skip latency measurement')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SECOM ML Pipeline - Performance Benchmark")
    print("=" * 60)
    print(f"Duration: {args.duration}s")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()
    
    benchmark = PipelineBenchmark()
    results = {}
    
    try:
        # Measure preprocessing throughput
        results['preprocessing'] = benchmark.measure_preprocessing_throughput(args.duration)
        print()
        
        # Measure prediction throughput
        results['inference'] = benchmark.measure_prediction_throughput(args.duration)
        print()
        
        # Measure latency
        if not args.no_latency:
            results['latency'] = benchmark.measure_latency()
            print()
        
        # Get model performance
        results['model_performance'] = benchmark.get_model_performance()
        print()
        
        # Print results
        print_results(results)
        
        # Save results
        save_results(results, args.output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
