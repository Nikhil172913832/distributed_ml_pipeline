"""
Simple load testing utilities for ML pipeline.

Provides basic load test harness for testing inference throughput and latency.
"""

import time
import threading
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import statistics


@dataclass
class LoadTestResult:
    """Results from load test"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    requests_per_second: float
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        """Average latency"""
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def median_latency(self) -> float:
        """Median latency"""
        return statistics.median(self.latencies) if self.latencies else 0
    
    @property
    def p95_latency(self) -> float:
        """95th percentile latency"""
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(0.95 * len(sorted_latencies))
        return sorted_latencies[idx]
    
    @property
    def p99_latency(self) -> float:
        """99th percentile latency"""
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int(0.99 * len(sorted_latencies))
        return sorted_latencies[idx]
    
    def summary(self) -> str:
        """Generate summary report"""
        return f"""
Load Test Results:
==================
Total Requests:     {self.total_requests}
Successful:         {self.successful_requests}
Failed:             {self.failed_requests}
Success Rate:       {self.success_rate * 100:.2f}%

Duration:           {self.total_duration:.2f}s
Throughput:         {self.requests_per_second:.2f} req/s

Latency Statistics:
  Average:          {self.avg_latency * 1000:.2f}ms
  Median:           {self.median_latency * 1000:.2f}ms
  P95:              {self.p95_latency * 1000:.2f}ms
  P99:              {self.p99_latency * 1000:.2f}ms
"""


class LoadTester:
    """
    Simple load testing harness for ML inference endpoints.
    """
    
    def __init__(
        self,
        target_function: Callable,
        num_requests: int = 100,
        num_threads: int = 10,
        timeout: float = 30.0
    ):
        """
        Initialize load tester.
        
        Args:
            target_function: Function to test (should accept no args and return bool for success)
            num_requests: Total number of requests to send
            num_threads: Number of concurrent threads
            timeout: Timeout per request in seconds
        """
        self.target_function = target_function
        self.num_requests = num_requests
        self.num_threads = num_threads
        self.timeout = timeout
        
        self.results = []
        self.lock = threading.Lock()
    
    def _worker(self, num_requests_per_thread: int):
        """Worker thread that sends requests"""
        for _ in range(num_requests_per_thread):
            start_time = time.time()
            success = False
            error = None
            
            try:
                result = self.target_function()
                success = bool(result)
            except Exception as e:
                error = str(e)
            
            latency = time.time() - start_time
            
            with self.lock:
                self.results.append({
                    'success': success,
                    'latency': latency,
                    'error': error
                })
    
    def run(self) -> LoadTestResult:
        """
        Run load test.
        
        Returns:
            LoadTestResult with statistics
        """
        self.results = []
        
        # Calculate requests per thread
        requests_per_thread = self.num_requests // self.num_threads
        
        # Start timer
        start_time = time.time()
        
        # Create and start threads
        threads = []
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self._worker, args=(requests_per_thread,))
            thread.start()
            threads.append(thread)
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=self.timeout * requests_per_thread)
        
        # Calculate duration
        total_duration = time.time() - start_time
        
        # Aggregate results
        successful = sum(1 for r in self.results if r['success'])
        failed = sum(1 for r in self.results if not r['success'])
        latencies = [r['latency'] for r in self.results]
        errors = [r['error'] for r in self.results if r['error']]
        
        return LoadTestResult(
            total_requests=len(self.results),
            successful_requests=successful,
            failed_requests=failed,
            total_duration=total_duration,
            requests_per_second=len(self.results) / total_duration if total_duration > 0 else 0,
            latencies=latencies,
            errors=errors
        )


class InferenceLoadTester:
    """
    Specialized load tester for ML inference endpoints.
    """
    
    def __init__(
        self,
        endpoint_url: str,
        sample_data_generator: Callable[[], Dict[str, Any]],
        num_requests: int = 100,
        num_threads: int = 10
    ):
        """
        Initialize inference load tester.
        
        Args:
            endpoint_url: URL of inference endpoint
            sample_data_generator: Function that generates sample input data
            num_requests: Total number of requests
            num_threads: Number of concurrent threads
        """
        self.endpoint_url = endpoint_url
        self.sample_data_generator = sample_data_generator
        self.num_requests = num_requests
        self.num_threads = num_threads
    
    def _make_request(self) -> bool:
        """Make single inference request"""
        try:
            data = self.sample_data_generator()
            response = requests.post(
                self.endpoint_url,
                json=data,
                timeout=5.0
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def run(self) -> LoadTestResult:
        """Run load test against inference endpoint"""
        tester = LoadTester(
            target_function=self._make_request,
            num_requests=self.num_requests,
            num_threads=self.num_threads
        )
        return tester.run()


# Utility functions for common load test scenarios

def generate_random_features(n_features: int = 10) -> Dict[str, Any]:
    """Generate random feature vector for testing"""
    features = np.random.randn(n_features).tolist()
    return {'features': features}


def _test_model_inference_speed_helper(
    model,
    X_test: np.ndarray,
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    Test model inference speed (helper function).
    
    Args:
        model: Trained model with predict() method
        X_test: Test data
        num_samples: Number of predictions to make
        
    Returns:
        Dictionary with timing statistics
    """
    # Sample random indices
    indices = np.random.choice(len(X_test), size=num_samples, replace=True)
    X_sample = X_test[indices]
    
    # Warmup
    _ = model.predict(X_sample[:10])
    
    # Time predictions
    latencies = []
    
    for i in range(num_samples):
        start = time.time()
        _ = model.predict(X_sample[i:i+1])
        latency = time.time() - start
        latencies.append(latency)
    
    return {
        'avg_latency_ms': np.mean(latencies) * 1000,
        'median_latency_ms': np.median(latencies) * 1000,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
        'p99_latency_ms': np.percentile(latencies, 99) * 1000,
        'throughput_per_sec': 1.0 / np.mean(latencies)
    }


def _test_batch_inference_speed_helper(
    model,
    X_test: np.ndarray,
    batch_sizes: List[int] = [1, 8, 16, 32, 64, 128]
) -> Dict[int, Dict[str, float]]:
    """
    Test model inference speed with different batch sizes (helper function).
    
    Args:
        model: Trained model
        X_test: Test data
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary mapping batch size to timing statistics
    """
    results = {}
    
    for batch_size in batch_sizes:
        # Sample data
        indices = np.random.choice(len(X_test), size=batch_size * 10, replace=True)
        X_sample = X_test[indices]
        
        # Warmup
        _ = model.predict(X_sample[:batch_size])
        
        # Time batches
        latencies = []
        
        for i in range(0, len(X_sample), batch_size):
            batch = X_sample[i:i+batch_size]
            
            start = time.time()
            _ = model.predict(batch)
            latency = time.time() - start
            latencies.append(latency)
        
        results[batch_size] = {
            'avg_batch_latency_ms': np.mean(latencies) * 1000,
            'avg_per_sample_latency_ms': np.mean(latencies) / batch_size * 1000,
            'throughput_samples_per_sec': batch_size / np.mean(latencies)
        }
    
    return results


# Test functions

import pytest


class TestLoadTester:
    """Test load testing utilities"""
    
    def test_simple_load_test(self):
        """Test basic load test functionality"""
        def simple_function():
            time.sleep(0.01)  # Simulate 10ms latency
            return True
        
        tester = LoadTester(
            target_function=simple_function,
            num_requests=50,
            num_threads=5
        )
        
        result = tester.run()
        
        assert result.total_requests == 50
        assert result.successful_requests == 50
        assert result.failed_requests == 0
        assert result.success_rate == 1.0
        assert result.avg_latency >= 0.01  # At least 10ms
    
    def test_load_test_with_failures(self):
        """Test load test with some failures"""
        call_count = [0]
        
        def flaky_function():
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise Exception("Simulated failure")
            return True
        
        tester = LoadTester(
            target_function=flaky_function,
            num_requests=30,
            num_threads=3
        )
        
        result = tester.run()
        
        assert result.total_requests == 30
        assert result.failed_requests > 0
        assert result.success_rate < 1.0
    
    def test_load_test_result_stats(self):
        """Test LoadTestResult statistics"""
        result = LoadTestResult(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_duration=10.0,
            requests_per_second=10.0,
            latencies=[0.1] * 50 + [0.2] * 30 + [0.5] * 20
        )
        
        assert result.success_rate == 0.95
        assert result.avg_latency > 0
        assert result.median_latency > 0
        assert result.p95_latency >= result.median_latency
        assert result.p99_latency >= result.p95_latency
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        def fast_function():
            return True
        
        tester = LoadTester(
            target_function=fast_function,
            num_requests=100,
            num_threads=10
        )
        
        result = tester.run()
        
        # Should have high throughput for fast function
        assert result.requests_per_second > 10


class TestInferencePerformance:
    """Test inference performance utilities"""
    
    def test_model_inference_speed(self):
        """Test single-sample inference speed measurement"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20)
        model = RandomForestClassifier(n_estimators=10, max_depth=5)
        model.fit(X[:800], y[:800])
        
        stats = _test_model_inference_speed_helper(model, X[800:], num_samples=100)
        
        assert 'avg_latency_ms' in stats
        assert 'throughput_per_sec' in stats
        assert stats['avg_latency_ms'] > 0
        assert stats['throughput_per_sec'] > 0
    
    def test_batch_inference_speed(self):
        """Test batch inference speed measurement"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=500, n_features=10)
        model = LogisticRegression()
        model.fit(X[:400], y[:400])
        
        results = _test_batch_inference_speed_helper(
            model, X[400:],
            batch_sizes=[1, 8, 16]
        )
        
        assert len(results) == 3
        assert 1 in results
        assert 'throughput_samples_per_sec' in results[1]
        
        # Larger batches should have better per-sample throughput
        assert results[16]['throughput_samples_per_sec'] > results[1]['throughput_samples_per_sec']


class TestLoadScenarios:
    """Test various load scenarios"""
    
    def test_sustained_load(self):
        """Test sustained load over time"""
        request_count = [0]
        
        def counting_function():
            request_count[0] += 1
            time.sleep(0.001)
            return True
        
        tester = LoadTester(
            target_function=counting_function,
            num_requests=100,
            num_threads=10
        )
        
        result = tester.run()
        
        assert request_count[0] == 100
        assert result.requests_per_second > 0
    
    def test_latency_percentiles(self):
        """Test latency percentile calculations"""
        def variable_latency_function():
            # Random latency between 10-50ms
            latency = np.random.uniform(0.01, 0.05)
            time.sleep(latency)
            return True
        
        tester = LoadTester(
            target_function=variable_latency_function,
            num_requests=100,
            num_threads=5
        )
        
        result = tester.run()
        
        # P99 should be higher than P95, which should be higher than median
        assert result.p99_latency >= result.p95_latency
        assert result.p95_latency >= result.median_latency
        assert result.median_latency >= result.avg_latency * 0.5  # Reasonable lower bound
