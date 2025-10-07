"""
GPU Utilities for TVAE Training
================================

Helper functions for GPU detection, monitoring, and optimization.
"""

import torch
from loguru import logger
from typing import List, Dict, Optional


def check_gpu_availability() -> Dict[str, any]:
    """
    Check GPU availability and return detailed information.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': 0,
        'gpus': []
    }
    
    if info['cuda_available']:
        info['num_gpus'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        
        for i in range(info['num_gpus']):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': props.total_memory / 1024**3,
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            }
            info['gpus'].append(gpu_info)
    
    return info


def print_gpu_info():
    """Print formatted GPU information."""
    info = check_gpu_availability()
    
    logger.info("="*80)
    logger.info("GPU Information")
    logger.info("="*80)
    
    if info['cuda_available']:
        logger.info(f"CUDA Available: ✓ Yes (CUDA {info['cuda_version']})")
        logger.info(f"Number of GPUs: {info['num_gpus']}")
        logger.info("")
        
        for gpu in info['gpus']:
            logger.info(f"GPU {gpu['id']}: {gpu['name']}")
            logger.info(f"  Memory: {gpu['total_memory_gb']:.1f} GB")
            logger.info(f"  Compute Capability: {gpu['compute_capability']}")
            logger.info(f"  Multiprocessors: {gpu['multi_processor_count']}")
            logger.info("")
    else:
        logger.warning("CUDA Available: ✗ No")
        logger.warning("Training will use CPU (slower)")
    
    logger.info("="*80)


def get_gpu_memory_usage(device_id: int = 0) -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    free = total - allocated
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


def print_gpu_memory(device_id: int = 0):
    """Print current GPU memory usage."""
    mem = get_gpu_memory_usage(device_id)
    
    logger.info(f"GPU {device_id} Memory Usage:")
    logger.info(f"  Allocated: {mem['allocated']:.2f} GB")
    logger.info(f"  Reserved:  {mem['reserved']:.2f} GB")
    logger.info(f"  Free:      {mem['free']:.2f} GB")
    logger.info(f"  Total:     {mem['total']:.2f} GB")
    logger.info(f"  Usage:     {(mem['allocated']/mem['total']*100):.1f}%")


def clear_gpu_memory(device_id: Optional[int] = None):
    """
    Clear GPU cache to free memory.
    
    Args:
        device_id: Specific GPU to clear, or None for all
    """
    if not torch.cuda.is_available():
        return
    
    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
        logger.info(f"✓ Cleared GPU {device_id} cache")
    else:
        torch.cuda.empty_cache()
        logger.info("✓ Cleared all GPU caches")


def optimize_batch_size_for_gpu(
    feature_count: int,
    gpu_memory_gb: float,
    default_batch_size: int = 500
) -> int:
    """
    Suggest optimal batch size based on GPU memory and dataset.
    
    Args:
        feature_count: Number of features in dataset
        gpu_memory_gb: Available GPU memory in GB
        default_batch_size: Default batch size
        
    Returns:
        Recommended batch size
    """
    # T4 has 16GB, estimate memory usage
    # TVAE with 591 features needs ~8-10GB for batch_size=500
    
    memory_per_sample_mb = (feature_count * 4) / 1024  # 4 bytes per float
    overhead_factor = 4  # Account for gradients, optimizer state, etc.
    
    # Estimate max batch size for available memory
    available_memory_mb = gpu_memory_gb * 1024 * 0.8  # Use 80% of GPU memory
    max_batch_size = int(available_memory_mb / (memory_per_sample_mb * overhead_factor))
    
    # Round to nearest 100
    suggested_batch_size = min(max_batch_size, 1000)
    suggested_batch_size = (suggested_batch_size // 100) * 100
    
    if suggested_batch_size < default_batch_size:
        logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) may be limited for batch_size={default_batch_size}")
        logger.info(f"Recommended batch_size: {suggested_batch_size}")
        return suggested_batch_size
    
    return default_batch_size


def setup_gpu_for_training(device_ids: List[int] = [0, 1]) -> Dict:
    """
    Setup GPUs for optimal training.
    
    Args:
        device_ids: List of GPU IDs to use
        
    Returns:
        Configuration dictionary
    """
    info = check_gpu_availability()
    
    if not info['cuda_available']:
        logger.warning("No CUDA GPUs available, using CPU")
        return {'use_cuda': False, 'device_ids': [], 'primary_device': None}
    
    # Validate device IDs
    valid_devices = []
    for device_id in device_ids:
        if device_id < info['num_gpus']:
            valid_devices.append(device_id)
        else:
            logger.warning(f"GPU {device_id} not available (only {info['num_gpus']} GPU(s) found)")
    
    if not valid_devices:
        logger.warning("No valid GPU devices, using CPU")
        return {'use_cuda': False, 'device_ids': [], 'primary_device': None}
    
    # Set primary device
    primary_device = valid_devices[0]
    torch.cuda.set_device(primary_device)
    
    # Clear GPU cache
    for device_id in valid_devices:
        clear_gpu_memory(device_id)
    
    logger.info(f"✓ GPU setup complete: Using GPU(s) {valid_devices}")
    logger.info(f"  Primary device: GPU {primary_device}")
    
    return {
        'use_cuda': True,
        'device_ids': valid_devices,
        'primary_device': primary_device
    }


if __name__ == "__main__":
    # Test GPU utilities
    print_gpu_info()
    
    if torch.cuda.is_available():
        print("\nMemory Usage:")
        for i in range(torch.cuda.device_count()):
            print_gpu_memory(i)
        
        print("\nOptimal Batch Size Suggestions:")
        for i in range(torch.cuda.device_count()):
            mem = get_gpu_memory_usage(i)
            batch_size = optimize_batch_size_for_gpu(591, mem['total'])
            print(f"GPU {i}: Recommended batch_size = {batch_size}")
