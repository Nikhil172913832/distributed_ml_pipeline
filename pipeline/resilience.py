"""
Resilience patterns for distributed ML pipeline.

Provides retry logic, circuit breakers, and graceful degradation patterns
for production-grade error handling.
"""

from typing import Callable, TypeVar, Optional, Any, Type
from functools import wraps
import time
import logging
from contextlib import contextmanager

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
from circuitbreaker import circuit, CircuitBreakerError

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ==========================================
# RETRY DECORATORS
# ==========================================

def retry_on_connection_error(
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
    multiplier: int = 2
):
    """
    Retry decorator for connection errors with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        multiplier: Exponential backoff multiplier
        
    Usage:
        @retry_on_connection_error(max_attempts=5)
        def connect_to_kafka():
            # Connection logic
            pass
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO)
    )


def retry_on_transient_error(
    max_attempts: int = 3,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator for transient errors with fixed backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exception types to retry on
        
    Usage:
        @retry_on_transient_error(max_attempts=3, exceptions=(ValueError, KeyError))
        def process_data(data):
            # Processing logic
            pass
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )


# ==========================================
# CIRCUIT BREAKER DECORATORS
# ==========================================

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: Type[Exception] = Exception
):
    """
    Circuit breaker decorator to prevent cascade failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type that triggers circuit breaker
        
    Usage:
        @circuit_breaker(failure_threshold=5, recovery_timeout=60)
        def load_model(model_path):
            # Model loading logic
            pass
    """
    return circuit(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception
    )


# ==========================================
# COMBINED PATTERNS
# ==========================================

def resilient_operation(
    max_retries: int = 3,
    circuit_failures: int = 5,
    circuit_timeout: int = 60
):
    """
    Combines retry and circuit breaker patterns for maximum resilience.
    
    Args:
        max_retries: Maximum retry attempts
        circuit_failures: Failures before circuit opens
        circuit_timeout: Circuit recovery timeout
        
    Usage:
        @resilient_operation(max_retries=3, circuit_failures=5)
        def critical_operation():
            # Critical operation logic
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply circuit breaker first (outer)
        func = circuit(
            failure_threshold=circuit_failures,
            recovery_timeout=circuit_timeout
        )(func)
        
        # Then apply retry (inner)
        func = retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )(func)
        
        return func
    return decorator


# ==========================================
# GRACEFUL DEGRADATION
# ==========================================

class DegradedModeError(Exception):
    """Raised when service enters degraded mode."""
    pass


@contextmanager
def graceful_degradation(
    fallback_value: Optional[Any] = None,
    log_error: bool = True,
    raise_on_failure: bool = False
):
    """
    Context manager for graceful degradation.
    
    Args:
        fallback_value: Value to return on failure
        log_error: Whether to log errors
        raise_on_failure: Whether to raise exception after logging
        
    Usage:
        with graceful_degradation(fallback_value=[], log_error=True):
            result = risky_operation()
        # If risky_operation fails, result will be []
    """
    try:
        yield
    except Exception as e:
        if log_error:
            logger.error(f"Operation failed, entering degraded mode: {e}", exc_info=True)
        
        if raise_on_failure:
            raise DegradedModeError(f"Degraded mode: {e}") from e
        
        return fallback_value


def with_fallback(fallback_value: Any = None):
    """
    Decorator that returns fallback value on exception.
    
    Args:
        fallback_value: Value to return if function raises exception
        
    Usage:
        @with_fallback(fallback_value=[])
        def get_predictions():
            # May fail
            return predictions
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Function {func.__name__} failed, returning fallback value: {e}"
                )
                return fallback_value
        return wrapper
    return decorator


# ==========================================
# TIMEOUT HANDLING
# ==========================================

class TimeoutError(Exception):
    """Raised when operation times out."""
    pass


def timeout(seconds: int):
    """
    Decorator to add timeout to function execution.
    
    Args:
        seconds: Maximum execution time in seconds
        
    Note: This is a simple implementation. For production, consider using
    signal-based timeouts or threading.
        
    Usage:
        @timeout(30)
        def long_running_operation():
            # Operation logic
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")
            
            # Set the signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        return wrapper
    return decorator


# ==========================================
# HEALTH CHECK UTILITIES
# ==========================================

class HealthCheck:
    """
    Health check utility for monitoring service health.
    
    Usage:
        health = HealthCheck()
        health.register_check("database", check_database_connection)
        health.register_check("kafka", check_kafka_connection)
        
        if health.is_healthy():
            # Proceed with operation
            pass
    """
    
    def __init__(self):
        self.checks: dict[str, Callable[[], bool]] = {}
        self.last_results: dict[str, bool] = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_checks(self) -> dict[str, bool]:
        """Run all registered health checks."""
        results = {}
        for name, check_func in self.checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = False
        
        self.last_results = results
        return results
    
    def is_healthy(self) -> bool:
        """Check if all health checks pass."""
        results = self.run_checks()
        return all(results.values())
    
    def get_status(self) -> dict[str, Any]:
        """Get detailed health status."""
        results = self.run_checks()
        return {
            "healthy": all(results.values()),
            "checks": results,
            "timestamp": time.time()
        }


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # Example: Kafka connection with retry and circuit breaker
    @resilient_operation(max_retries=3, circuit_failures=5, circuit_timeout=60)
    def connect_to_kafka(bootstrap_servers: str):
        """Connect to Kafka with resilience patterns."""
        from kafka import KafkaProducer
        return KafkaProducer(bootstrap_servers=bootstrap_servers.split(','))
    
    # Example: Model loading with fallback
    @with_fallback(fallback_value=None)
    def load_model(model_path: str):
        """Load model with fallback to None on failure."""
        import joblib
        return joblib.load(model_path)
    
    # Example: Database query with timeout
    @timeout(30)
    def execute_query(query: str):
        """Execute database query with 30s timeout."""
        # Query execution logic
        pass
