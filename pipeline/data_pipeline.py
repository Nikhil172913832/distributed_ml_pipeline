"""Enhanced data pipeline for distributed training."""
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from typing import Optional, Callable, Tuple
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DistributedDataPipeline:
    """
    Distributed data pipeline with optimized loading and augmentation.
    
    Handles data loading, preprocessing, augmentation, and distributed sampling
    for efficient multi-GPU training.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 num_workers: int = 4,
                 rank: int = 0,
                 world_size: int = 1,
                 pin_memory: bool = True,
                 prefetch_factor: int = 2,
                 drop_last: bool = True,
                 augmentation_fn: Optional[Callable] = None):
        """
        Initialize distributed data pipeline.
        
        Args:
            dataset: PyTorch Dataset object
            batch_size: Batch size per GPU
            num_workers: Number of data loading workers
            rank: Process rank in distributed setting
            world_size: Total number of processes
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch
            drop_last: Drop last incomplete batch
            augmentation_fn: Optional data augmentation function
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.augmentation_fn = augmentation_fn
        
        logger.info(f"DataPipeline initialized: rank={rank}/{world_size}, "
                   f"batch_size={batch_size}, workers={num_workers}")
    
    def get_dataloader(self, shuffle: bool = True, sampler: Optional[DistributedSampler] = None) -> DataLoader:
        """
        Create dataloader with proper distributed sampling.
        
        Args:
            shuffle: Whether to shuffle data (only applies if sampler is None)
            sampler: Optional custom sampler (if None, creates DistributedSampler)
        
        Returns:
            Configured DataLoader
        """
        # Create distributed sampler if not provided and world_size > 1
        if sampler is None and self.world_size > 1:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=self.drop_last
            )
            # Don't shuffle in DataLoader when using sampler
            shuffle = False
        
        # Create DataLoader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn if self.augmentation_fn else None
        )
        
        logger.info(f"DataLoader created: {len(dataloader)} batches, "
                   f"shuffle={shuffle}, sampler={type(sampler).__name__ if sampler else None}")
        
        return dataloader
    
    def _collate_fn(self, batch):
        """
        Custom collate function with augmentation.
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Collated and augmented batch
        """
        if self.augmentation_fn:
            batch = [self.augmentation_fn(sample) for sample in batch]
        
        # Default collate behavior
        return torch.utils.data.dataloader.default_collate(batch)
    
    def set_epoch(self, epoch: int):
        """
        Set epoch for distributed sampler (important for proper shuffling).
        
        Args:
            epoch: Current epoch number
        """
        if hasattr(self.dataset, 'sampler') and isinstance(self.dataset.sampler, DistributedSampler):
            self.dataset.sampler.set_epoch(epoch)


class AugmentationPipeline:
    """
    Data augmentation pipeline with common transformations.
    
    Provides flexible augmentation with support for different data types.
    """
    
    def __init__(self, 
                 augmentations: Optional[list] = None,
                 probability: float = 0.5):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations: List of augmentation functions
            probability: Probability of applying augmentations
        """
        self.augmentations = augmentations or []
        self.probability = probability
        
        logger.info(f"AugmentationPipeline initialized with {len(self.augmentations)} transforms")
    
    def __call__(self, sample):
        """
        Apply augmentations to sample.
        
        Args:
            sample: Input sample (dict with 'data' and 'label' keys or tensor)
        
        Returns:
            Augmented sample
        """
        if np.random.random() < self.probability:
            for aug in self.augmentations:
                sample = aug(sample)
        
        return sample
    
    @staticmethod
    def random_noise(sample, noise_std: float = 0.01):
        """Add random Gaussian noise."""
        if isinstance(sample, dict) and 'data' in sample:
            sample['data'] = sample['data'] + torch.randn_like(sample['data']) * noise_std
        else:
            sample = sample + torch.randn_like(sample) * noise_std
        return sample
    
    @staticmethod
    def random_scale(sample, scale_range: Tuple[float, float] = (0.9, 1.1)):
        """Random scaling."""
        scale = np.random.uniform(*scale_range)
        if isinstance(sample, dict) and 'data' in sample:
            sample['data'] = sample['data'] * scale
        else:
            sample = sample * scale
        return sample
    
    @staticmethod
    def normalize(sample, mean: float = 0.0, std: float = 1.0):
        """Normalize data."""
        if isinstance(sample, dict) and 'data' in sample:
            sample['data'] = (sample['data'] - mean) / std
        else:
            sample = (sample - mean) / std
        return sample


class DatasetSplitter:
    """
    Utility for splitting datasets into train/val/test sets.
    """
    
    @staticmethod
    def split_dataset(dataset: Dataset, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Calculate lengths
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
        
        logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def create_pipelines(train_dataset: Dataset,
                        val_dataset: Dataset,
                        test_dataset: Dataset,
                        batch_size: int,
                        num_workers: int = 4,
                        rank: int = 0,
                        world_size: int = 1,
                        augmentation_fn: Optional[Callable] = None) -> Tuple[DistributedDataPipeline, 
                                                                             DistributedDataPipeline,
                                                                             DistributedDataPipeline]:
        """
        Create data pipelines for train, val, and test sets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size per GPU
            num_workers: Number of data loading workers
            rank: Process rank
            world_size: Total number of processes
            augmentation_fn: Augmentation function (only applied to training)
        
        Returns:
            Tuple of (train_pipeline, val_pipeline, test_pipeline)
        """
        train_pipeline = DistributedDataPipeline(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            augmentation_fn=augmentation_fn
        )
        
        val_pipeline = DistributedDataPipeline(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            augmentation_fn=None  # No augmentation for validation
        )
        
        test_pipeline = DistributedDataPipeline(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            augmentation_fn=None  # No augmentation for testing
        )
        
        logger.info("Created train, val, and test pipelines")
        
        return train_pipeline, val_pipeline, test_pipeline
