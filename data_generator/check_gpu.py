#!/usr/bin/env python3
"""
Quick GPU Check Script
======================

Run this to verify your T4 GPUs are detected and ready for training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gpu_utils import print_gpu_info, print_gpu_memory, optimize_batch_size_for_gpu
import torch


def main():
    print("\n🔍 Checking GPU Configuration for TVAE Training\n")
    
    # Print GPU information
    print_gpu_info()
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        print("\n📊 Memory Status:")
        print("="*80)
        for i in range(num_gpus):
            print(f"\nGPU {i}:")
            print_gpu_memory(i)
        
        print("\n⚙️  Recommended Settings for SECOM (591 features):")
        print("="*80)
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / 1024**3
            batch_size = optimize_batch_size_for_gpu(591, total_mem_gb)
            print(f"\nGPU {i} ({torch.cuda.get_device_name(i)}):")
            print(f"  Total Memory: {total_mem_gb:.1f} GB")
            print(f"  Recommended batch_size: {batch_size}")
        
        print("\n✅ Training Commands:")
        print("="*80)
        print("\n# Train with both T4 GPUs (recommended):")
        print("python data_generator/raw_cli.py train --epochs 300 --cuda --gpu-ids '0,1'")
        
        print("\n# Train with single GPU:")
        print("python data_generator/raw_cli.py train --epochs 300 --cuda --gpu-ids '0'")
        
        print("\n# Train with custom batch size:")
        print("python data_generator/raw_cli.py train --epochs 300 --batch-size 600 --cuda")
        
        print("\n✨ Expected Performance with T4 GPUs:")
        print("="*80)
        print("  • Training time: ~5-10 minutes (vs 10-20 min on CPU)")
        print("  • GPU utilization: 60-80%")
        print("  • Memory usage: ~8-12 GB per GPU")
        print("  • Speedup: 2-4x faster than CPU")
        
    else:
        print("\n❌ No CUDA GPUs detected!")
        print("\nPossible issues:")
        print("  1. NVIDIA drivers not installed")
        print("  2. PyTorch not built with CUDA support")
        print("  3. GPU not properly configured")
        print("\nTo verify:")
        print("  nvidia-smi  # Check GPU visibility")
        print("  pip show torch  # Check PyTorch CUDA version")
        
        print("\n📝 CPU Training Command:")
        print("python data_generator/raw_cli.py train --epochs 300 --no-cuda")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
