"""Checkpoint management for distributed training."""
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints with automatic cleanup and recovery.
    
    Features:
    - Save/load checkpoints with full training state
    - Automatic cleanup of old checkpoints
    - Support for best model tracking
    - Distributed training compatibility
    """
    
    def __init__(self,
                 checkpoint_dir: Path,
                 keep_last_n: int = 5,
                 save_best: bool = True,
                 metric_for_best: str = 'val_loss',
                 mode: str = 'min',
                 rank: int = 0):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep (0 = keep all)
            save_best: Whether to track and save best model
            metric_for_best: Metric to use for determining best model
            mode: 'min' or 'max' for best metric
            rank: Process rank (only rank 0 saves checkpoints)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.metric_for_best = metric_for_best
        self.mode = mode
        self.rank = rank
        
        # Track best model
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_checkpoint_path = None
        
        logger.info(f"CheckpointManager initialized: dir={checkpoint_dir}, "
                   f"keep_last={keep_last_n}, metric={metric_for_best}")
    
    def save_checkpoint(self,
                       epoch: int,
                       model_state: Dict[str, Any],
                       optimizer_state: Dict[str, Any],
                       scheduler_state: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       extra_state: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Save checkpoint to disk.
        
        Args:
            epoch: Current epoch number
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Optional scheduler state dict
            metrics: Dictionary of metrics
            extra_state: Any additional state to save
        
        Returns:
            Path to saved checkpoint (None if not rank 0)
        """
        # Only rank 0 saves checkpoints in distributed setting
        if self.rank != 0:
            return None
        
        # Prepare checkpoint dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'timestamp': datetime.now().isoformat(),
        }
        
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Check if this is the best model
        if self.save_best and metrics and self.metric_for_best in metrics:
            current_metric = metrics[self.metric_for_best]
            is_best = self._is_better(current_metric, self.best_metric)
            
            if is_best:
                self.best_metric = current_metric
                best_path = self.checkpoint_dir / 'best_model.pt'
                
                # Copy checkpoint to best model
                shutil.copy2(checkpoint_path, best_path)
                self.best_checkpoint_path = best_path
                
                logger.info(f"New best model saved: {self.metric_for_best}={current_metric:.4f}")
        
        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load checkpoint from disk.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
        return checkpoint
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Returns:
            Checkpoint dictionary or None if no checkpoints exist
        """
        checkpoints = self.get_checkpoint_list()
        
        if not checkpoints:
            logger.info("No checkpoints found")
            return None
        
        latest = checkpoints[-1]
        return self.load_checkpoint(latest)
    
    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the best model checkpoint.
        
        Returns:
            Checkpoint dictionary or None if no best checkpoint exists
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        
        if not best_path.exists():
            logger.warning("No best model checkpoint found")
            return None
        
        return self.load_checkpoint(best_path)
    
    def get_checkpoint_list(self) -> list:
        """
        Get sorted list of checkpoint paths.
        
        Returns:
            List of checkpoint paths sorted by epoch
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        return checkpoints
    
    def resume_training(self,
                       model,
                       optimizer,
                       scheduler=None) -> int:
        """
        Resume training from latest checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Optional scheduler to load state into
        
        Returns:
            Epoch to resume from (0 if no checkpoint)
        """
        checkpoint = self.load_latest_checkpoint()
        
        if checkpoint is None:
            logger.info("No checkpoint found, starting from scratch")
            return 0
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        logger.info(f"Resumed training from epoch {epoch}")
        
        return epoch
    
    def _is_better(self, current: float, best: float) -> bool:
        """
        Check if current metric is better than best.
        
        Args:
            current: Current metric value
            best: Best metric value so far
        
        Returns:
            True if current is better
        """
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.get_checkpoint_list()
        
        if len(checkpoints) > self.keep_last_n:
            # Remove oldest checkpoints
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {ckpt}")
    
    def delete_all_checkpoints(self):
        """Delete all checkpoints (use with caution)."""
        if self.rank != 0:
            return
        
        for ckpt in self.checkpoint_dir.glob('*.pt'):
            ckpt.unlink()
            logger.info(f"Deleted checkpoint: {ckpt}")
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Get information about a checkpoint without fully loading it.
        
        Args:
            checkpoint_path: Path to checkpoint
        
        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'path': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'metrics': checkpoint.get('metrics', {}),
        }
        
        return info
    
    def list_all_checkpoints(self) -> list:
        """
        List all checkpoints with their information.
        
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = self.get_checkpoint_list()
        return [self.get_checkpoint_info(ckpt) for ckpt in checkpoints]


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}")
    
    def __call__(self, metric: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self._is_improvement(metric):
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered")
                return True
        
        return False
    
    def _is_improvement(self, metric: float) -> bool:
        """Check if metric improved."""
        if self.mode == 'min':
            return metric < self.best_score - self.min_delta
        else:
            return metric > self.best_score + self.min_delta
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
