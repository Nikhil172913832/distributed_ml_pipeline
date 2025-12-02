# API Documentation

## Configuration API

### ConfigManager

Central configuration management system.

```python
from config import ConfigManager

# Initialize
config = ConfigManager('config/config.yaml')

# Access configurations
model_cfg = config.model
training_cfg = config.training
data_cfg = config.data

# Update configuration
config.update_from_dict({
    'training': {'epochs': 200}
})

# Save configuration
config.save_config('output.yaml')
```

### Configuration Classes

#### ModelConfig
```python
ModelConfig(
    model_name: str,
    version: str,
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    hyperparameters: Dict[str, Any] = {}
)
```

#### TrainingConfig
```python
TrainingConfig(
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0001,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
)
```

## Training API

### AdvancedDistributedTrainer

```python
from pipeline.trainer import AdvancedDistributedTrainer

trainer = AdvancedDistributedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=loss_fn,
    use_amp=True,
    gradient_accumulation_steps=2
)

# Training
train_metrics = trainer.train_epoch(train_loader)

# Validation
val_metrics = trainer.validate_epoch(val_loader)
```

#### Methods

- `train_step(batch)` - Single training step
- `validate_step(batch)` - Single validation step
- `train_epoch(dataloader, callback)` - Full epoch training
- `validate_epoch(dataloader)` - Full epoch validation
- `get_lr()` - Get current learning rate
- `set_lr(lr)` - Set learning rate

## Data Pipeline API

### DistributedDataPipeline

```python
from pipeline.data_pipeline import DistributedDataPipeline

pipeline = DistributedDataPipeline(
    dataset=dataset,
    batch_size=64,
    num_workers=4,
    rank=0,
    world_size=4
)

dataloader = pipeline.get_dataloader(shuffle=True)
```

### AugmentationPipeline

```python
from pipeline.data_pipeline import AugmentationPipeline

aug = AugmentationPipeline(
    augmentations=[
        lambda x: AugmentationPipeline.random_noise(x, 0.01),
        lambda x: AugmentationPipeline.random_scale(x, (0.9, 1.1))
    ]
)
```

## Checkpoint API

### CheckpointManager

```python
from pipeline.checkpoint import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    keep_last_n=5,
    save_best=True
)

# Save
manager.save_checkpoint(
    epoch=epoch,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict(),
    metrics={'val_loss': 0.5}
)

# Load
checkpoint = manager.load_latest_checkpoint()
best_checkpoint = manager.load_best_checkpoint()

# Resume training
start_epoch = manager.resume_training(model, optimizer)
```

### EarlyStopping

```python
from pipeline.checkpoint import EarlyStopping

early_stop = EarlyStopping(patience=10, mode='min')

for epoch in range(epochs):
    val_loss = validate()
    if early_stop(val_loss):
        break
```

## Monitoring API

### MetricsTracker

```python
from monitoring import MetricsTracker

tracker = MetricsTracker(
    experiment_name='experiment',
    enable_mlflow=True,
    enable_wandb=True
)

# Log metrics
tracker.log_metrics({'loss': 0.5, 'accuracy': 0.9}, step=epoch)

# Log parameters
tracker.log_parameters({'lr': 0.001, 'batch_size': 64})

# Log model
tracker.log_model(model, model_path='model.pt')

# Finish
tracker.finish()
```

## Profiling API

### PerformanceProfiler

```python
from pipeline.profiler import PerformanceProfiler

profiler = PerformanceProfiler(
    enabled=True,
    trace_dir='traces'
)

# Profile training
result = profiler.profile_training(
    train_fn=training_function,
    num_steps=100
)

# Profile model
profiler.profile_model(model, input_shape=(3, 224, 224))
```

### MemoryTracker

```python
from pipeline.profiler import MemoryTracker

tracker = MemoryTracker(device=0)
tracker.log_memory("start")
# ... code ...
tracker.log_memory("end")
summary = tracker.get_memory_summary()
```

### ThroughputMeter

```python
from pipeline.profiler import ThroughputMeter

meter = ThroughputMeter()
for batch in dataloader:
    meter.update(batch_size=len(batch))
meter.log_throughput()
```

## Best Practices

### 1. Configuration Management
- Use YAML files for configuration
- Validate configs before training
- Version control your configs

### 2. Distributed Training
- Always use DistributedSampler
- Synchronize metrics across GPUs
- Save checkpoints only from rank 0

### 3. Checkpointing
- Save regularly (every N epochs)
- Keep multiple checkpoints
- Track best model separately

### 4. Monitoring
- Log metrics frequently
- Track system metrics (GPU, memory)
- Use multiple monitoring backends

### 5. Performance
- Enable mixed precision (AMP)
- Use gradient accumulation for large batches
- Profile regularly to find bottlenecks
