"""Performance profiling utilities for distributed training."""
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from typing import Optional, Callable, Dict, Any
import logging
from pathlib import Path
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Profile training performance and identify bottlenecks.
    
    Supports CPU and GPU profiling with automatic report generation.
    """
    
    def __init__(self,
                 enabled: bool = True,
                 trace_dir: str = "traces",
                 profile_memory: bool = True,
                 profile_shapes: bool = True,
                 with_stack: bool = False):
        """
        Initialize performance profiler.
        
        Args:
            enabled: Enable profiling
            trace_dir: Directory to save traces
            profile_memory: Profile memory usage
            profile_shapes: Record tensor shapes
            with_stack: Record stack traces
        """
        self.enabled = enabled
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.profile_memory = profile_memory
        self.profile_shapes = profile_shapes
        self.with_stack = with_stack
        
        logger.info(f"PerformanceProfiler initialized: enabled={enabled}")
    
    @contextmanager
    def profile_section(self, name: str):
        """
        Profile a specific code section.
        
        Args:
            name: Name of the section being profiled
        """
        if not self.enabled:
            yield
            return
        
        with record_function(name):
            yield
    
    def profile_training(self,
                        train_fn: Callable,
                        num_steps: int = 100,
                        warmup_steps: int = 10,
                        active_steps: int = 20,
                        *args,
                        **kwargs) -> Any:
        """
        Profile training function with warmup and active profiling phases.
        
        Args:
            train_fn: Training function to profile
            num_steps: Total number of steps to run
            warmup_steps: Steps before starting profiling
            active_steps: Steps to actively profile
            *args: Arguments to pass to train_fn
            **kwargs: Keyword arguments to pass to train_fn
        
        Returns:
            Result from train_fn
        """
        if not self.enabled:
            return train_fn(*args, **kwargs)
        
        # Configure profiler schedule
        prof_schedule = schedule(
            skip_first=warmup_steps,
            wait=1,
            warmup=2,
            active=active_steps,
            repeat=1
        )
        
        # Activities to profile
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        result = None
        
        with profile(
            activities=activities,
            schedule=prof_schedule,
            on_trace_ready=self._trace_handler,
            record_shapes=self.profile_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
        ) as prof:
            for step in range(num_steps):
                with record_function(f"step_{step}"):
                    result = train_fn(*args, **kwargs)
                prof.step()
        
        # Print summary
        self._print_summary(prof)
        
        return result
    
    def _trace_handler(self, prof):
        """Handle trace ready event."""
        # Export Chrome trace
        trace_file = self.trace_dir / f"trace_{int(time.time())}.json"
        prof.export_chrome_trace(str(trace_file))
        logger.info(f"Trace saved to {trace_file}")
        
        # Export stack traces if enabled
        if self.with_stack:
            stack_file = self.trace_dir / f"stacks_{int(time.time())}.txt"
            with open(stack_file, 'w') as f:
                f.write(prof.key_averages(group_by_stack_n=5).table(
                    sort_by="self_cuda_time_total", row_limit=20
                ))
            logger.info(f"Stack traces saved to {stack_file}")
    
    def _print_summary(self, prof):
        """Print profiling summary."""
        logger.info("\n" + "="*80)
        logger.info("PROFILING SUMMARY")
        logger.info("="*80)
        
        # CPU time summary
        logger.info("\nTop 10 operations by CPU time:")
        logger.info(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=10
        ))
        
        # GPU time summary
        if torch.cuda.is_available():
            logger.info("\nTop 10 operations by CUDA time:")
            logger.info(prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=10
            ))
        
        # Memory summary
        if self.profile_memory:
            logger.info("\nTop 10 operations by memory usage:")
            logger.info(prof.key_averages().table(
                sort_by="self_cuda_memory_usage",
                row_limit=10
            ))
        
        logger.info("="*80)
    
    def profile_model(self, model: torch.nn.Module, input_shape: tuple):
        """
        Profile model forward pass.
        
        Args:
            model: Model to profile
            input_shape: Shape of input tensor (excluding batch dimension)
        """
        if not self.enabled:
            return
        
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=self.profile_memory
        ) as prof:
            with record_function("model_forward"):
                model(dummy_input)
        
        logger.info("\nModel Forward Pass Profiling:")
        logger.info(prof.key_averages().table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=20
        ))


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, enabled: bool = True):
        """
        Initialize timing context.
        
        Args:
            name: Name of the code block
            enabled: Enable timing
        """
        self.name = name
        self.enabled = enabled
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """End timing and log result."""
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            logger.info(f"{self.name}: {elapsed:.4f}s")
    
    @property
    def elapsed(self) -> Optional[float]:
        """Get elapsed time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class MemoryTracker:
    """Track GPU memory usage during training."""
    
    def __init__(self, device: int = 0, enabled: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            device: CUDA device ID
            enabled: Enable tracking
        """
        self.device = device
        self.enabled = enabled and torch.cuda.is_available()
        self.peak_memory = 0
        
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(device)
    
    def log_memory(self, step: str = ""):
        """
        Log current memory usage.
        
        Args:
            step: Description of current step
        """
        if not self.enabled:
            return
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
        
        self.peak_memory = max(self.peak_memory, allocated)
        
        logger.info(
            f"Memory {step}: "
            f"Allocated={allocated:.2f}GB, "
            f"Reserved={reserved:.2f}GB, "
            f"Peak={max_allocated:.2f}GB"
        )
    
    def get_memory_summary(self) -> Dict[str, float]:
        """
        Get memory usage summary.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.enabled:
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1024**3,
            'peak_gb': self.peak_memory
        }
    
    def reset(self):
        """Reset memory tracking."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.peak_memory = 0


class ThroughputMeter:
    """Measure training throughput (samples/sec)."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize throughput meter.
        
        Args:
            window_size: Number of recent samples to average
        """
        self.window_size = window_size
        self.samples = []
        self.times = []
    
    def update(self, batch_size: int):
        """
        Update with new batch.
        
        Args:
            batch_size: Number of samples in batch
        """
        self.samples.append(batch_size)
        self.times.append(time.time())
        
        # Keep only recent samples
        if len(self.samples) > self.window_size:
            self.samples.pop(0)
            self.times.pop(0)
    
    def get_throughput(self) -> Optional[float]:
        """
        Calculate current throughput.
        
        Returns:
            Samples per second
        """
        if len(self.samples) < 2:
            return None
        
        total_samples = sum(self.samples)
        total_time = self.times[-1] - self.times[0]
        
        if total_time > 0:
            return total_samples / total_time
        return None
    
    def log_throughput(self):
        """Log current throughput."""
        throughput = self.get_throughput()
        if throughput:
            logger.info(f"Throughput: {throughput:.2f} samples/sec")


def benchmark_dataloader(dataloader, num_batches: int = 100):
    """
    Benchmark dataloader performance.
    
    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to test
    """
    logger.info(f"Benchmarking dataloader for {num_batches} batches...")
    
    start_time = time.time()
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        if isinstance(batch, dict):
            batch_size = next(iter(batch.values())).size(0)
        else:
            batch_size = batch[0].size(0)
        
        total_samples += batch_size
    
    elapsed = time.time() - start_time
    throughput = total_samples / elapsed
    
    logger.info(f"DataLoader throughput: {throughput:.2f} samples/sec")
    logger.info(f"Time per batch: {elapsed/num_batches:.4f}s")
