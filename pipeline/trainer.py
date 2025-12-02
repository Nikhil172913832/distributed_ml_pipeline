"""Advanced distributed trainer with modern optimization techniques."""
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class AdvancedDistributedTrainer:
    """
    Advanced trainer with mixed precision, gradient accumulation, and optimization.
    
    Supports:
    - Automatic Mixed Precision (AMP) training
    - Gradient accumulation for larger effective batch sizes
    - Gradient clipping for training stability
    - Learning rate scheduling
    - Distributed training synchronization
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 use_amp: bool = True,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Initialize advanced trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimizer instance
            criterion: Loss function
            scheduler: Learning rate scheduler
            device: Device for training ('cuda' or 'cpu')
            use_amp: Enable automatic mixed precision
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            rank: Process rank in distributed setting
            world_size: Total number of processes
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.rank = rank
        self.world_size = world_size
        
        # Initialize gradient scaler for AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Wrap model with DistributedDataParallel if multi-GPU
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[rank] if device == 'cuda' else None,
                output_device=rank if device == 'cuda' else None
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        logger.info(f"Trainer initialized: rank={rank}/{world_size}, "
                   f"AMP={use_amp}, grad_accum={gradient_accumulation_steps}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute single training step with gradient accumulation and mixed precision.
        
        Args:
            batch: Dictionary containing 'input' and 'target' tensors
        
        Returns:
            Dictionary of metrics for this step
        """
        self.model.train()
        
        # Move batch to device
        inputs = batch['input'].to(self.device, non_blocking=True)
        targets = batch['target'].to(self.device, non_blocking=True)
        
        # Forward pass with automatic mixed precision
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Metrics
        metrics = {
            'loss': loss.item() * self.gradient_accumulation_steps,
        }
        
        # Compute accuracy if classification task
        if outputs.dim() > 1 and outputs.size(1) > 1:
            _, predicted = outputs.max(1)
            if targets.dim() > 1:
                _, targets = targets.max(1)
            correct = predicted.eq(targets).sum().item()
            metrics['accuracy'] = correct / targets.size(0)
        
        # Optimizer step after accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self._optimizer_step()
        
        self.global_step += 1
        
        return metrics
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Clip gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
    
    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute validation step.
        
        Args:
            batch: Dictionary containing 'input' and 'target' tensors
        
        Returns:
            Dictionary of metrics for this step
        """
        self.model.eval()
        
        # Move batch to device
        inputs = batch['input'].to(self.device, non_blocking=True)
        targets = batch['target'].to(self.device, non_blocking=True)
        
        # Forward pass
        with autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # Metrics
        metrics = {
            'val_loss': loss.item(),
        }
        
        # Compute accuracy if classification task
        if outputs.dim() > 1 and outputs.size(1) > 1:
            _, predicted = outputs.max(1)
            if targets.dim() > 1:
                _, targets = targets.max(1)
            correct = predicted.eq(targets).sum().item()
            metrics['val_accuracy'] = correct / targets.size(0)
        
        return metrics
    
    def train_epoch(self, 
                   dataloader,
                   metrics_callback: Optional[Callable] = None) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            metrics_callback: Optional callback for logging metrics
        
        Returns:
            Average metrics for the epoch
        """
        self.model.train()
        epoch_metrics = {}
        num_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0) + value
            num_batches += 1
            
            # Log metrics via callback
            if metrics_callback and batch_idx % 10 == 0:
                metrics_callback(step_metrics, self.global_step)
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        avg_metrics['epoch_time'] = time.time() - epoch_start_time
        avg_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        self.epoch += 1
        
        return avg_metrics
    
    @torch.no_grad()
    def validate_epoch(self, 
                      dataloader,
                      metrics_callback: Optional[Callable] = None) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation data loader
            metrics_callback: Optional callback for logging metrics
        
        Returns:
            Average metrics for the epoch
        """
        self.model.eval()
        epoch_metrics = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Validation step
            step_metrics = self.validate_step(batch)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0) + value
            num_batches += 1
        
        # Average metrics
        avg_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        # Synchronize metrics across GPUs
        if self.world_size > 1:
            avg_metrics = self._sync_metrics(avg_metrics)
        
        return avg_metrics
    
    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Synchronize metrics across distributed processes.
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            Averaged metrics across all processes
        """
        synced_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value).to(self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            synced_metrics[key] = (tensor / self.world_size).item()
        
        return synced_metrics
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def set_lr(self, lr: float):
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get model state dict (unwrap DDP if needed)."""
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()
    
    def load_model_state(self, state_dict: Dict[str, Any]):
        """Load model state dict (handle DDP wrapper)."""
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
