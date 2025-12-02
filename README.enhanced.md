# Distributed ML Pipeline

[![CI/CD](https://github.com/Nikhil172913832/distributed_ml_pipeline/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/Nikhil172913832/distributed_ml_pipeline/actions)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Production-ready MLOps system for semiconductor manufacturing quality control with real-time inference, automated retraining, drift detection, and comprehensive monitoring.

## ✨ Key Features

### Core ML Pipeline
- **Real-time Inference** with confidence scoring and latency monitoring
- **Automated Retraining** triggered by performance degradation or data drift
- **Statistical Drift Detection** using Kolmogorov-Smirnov test
- **Model Versioning** with metadata tracking and rollback support
- **Distributed Training** with PyTorch DDP and multi-GPU support

### Production Features
- **Advanced Configuration** with YAML-based config management
- **Comprehensive Monitoring** via MLflow, Prometheus, and W&B
- **Checkpoint Management** with automatic cleanup and recovery
- **Performance Profiling** with memory and throughput tracking
- **CI/CD Pipelines** for automated testing and deployment
- **Kubernetes Ready** with full manifests and auto-scaling

### Enhanced Data Pipeline
- **Distributed Sampling** for multi-GPU training
- **Data Augmentation** with configurable transformations
- **Optimized Loading** with prefetching and persistent workers

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Distributed ML Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Data Sources                                                     │
│      ↓                                                            │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐            │
│  │ Producer │ →→→ │  Kafka   │ →→→ │  Consumer    │            │
│  └──────────┘     └──────────┘     └──────────────┘            │
│                                            ↓                      │
│                                     ┌──────────────┐            │
│                                     │  PostgreSQL  │            │
│                                     └──────────────┘            │
│                                            ↓                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Distributed Training Cluster                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │ Worker  │  │ Worker  │  │ Worker  │  │ Worker  │   │   │
│  │  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │   │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│                  ┌──────────────┐                               │
│                  │  Checkpoints │                               │
│                  └──────────────┘                               │
│                          ↓                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Inference   │  │  Retrainer   │  │   Monitor    │         │
│  │   Service    │  │   Service    │  │   Service    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│         ↓                  ↓                  ↓                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │         Monitoring Stack (Prometheus + Grafana)        │   │
│  └────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- CUDA 11.8+ (for GPU training)
- Kubernetes cluster (optional, for K8s deployment)

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/Nikhil172913832/distributed_ml_pipeline.git
cd distributed_ml_pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure pipeline
cp config/default_config.yaml config/config.yaml
# Edit config/config.yaml with your settings

# 4. Start services
make start

# 5. Verify deployment
make status
```

### Docker Deployment

```bash
# Start all services with Docker Compose
docker-compose -f docker-compose.enhanced.yml up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f
```

### Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f k8s/

# Monitor training
kubectl logs -n ml-pipeline -l app=distributed-training --follow

# Check inference service
kubectl get svc -n ml-pipeline inference-service
```

See [k8s/README.md](k8s/README.md) for detailed instructions.

## 📊 Usage Examples

### Configuration Management

```python
from config import ConfigManager

# Load configuration
config_manager = ConfigManager('config/config.yaml')

# Access configurations
model_config = config_manager.model
training_config = config_manager.training

# Update configuration
config_manager.update_from_dict({
    'training': {'epochs': 200, 'batch_size': 64}
})

# Save configuration
config_manager.save_config('config/updated_config.yaml')
```

### Distributed Training

```python
from pipeline.trainer import AdvancedDistributedTrainer
from pipeline.data_pipeline import DistributedDataPipeline
from pipeline.checkpoint import CheckpointManager
from monitoring import MetricsTracker
import torch.nn as nn

# Initialize components
model = YourModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Setup trainer with advanced features
trainer = AdvancedDistributedTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    use_amp=True,  # Mixed precision
    gradient_accumulation_steps=2,
    max_grad_norm=1.0
)

# Setup checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    keep_last_n=5,
    save_best=True
)

# Setup metrics tracking
metrics_tracker = MetricsTracker(
    experiment_name='secom_training',
    enable_mlflow=True,
    enable_wandb=True
)

# Training loop
for epoch in range(num_epochs):
    # Train
    train_metrics = trainer.train_epoch(train_loader)
    metrics_tracker.log_metrics(train_metrics, epoch, prefix='train')
    
    # Validate
    val_metrics = trainer.validate_epoch(val_loader)
    metrics_tracker.log_metrics(val_metrics, epoch, prefix='val')
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(
        epoch=epoch,
        model_state=trainer.get_model_state(),
        optimizer_state=optimizer.state_dict(),
        metrics=val_metrics
    )
```

### Data Pipeline with Augmentation

```python
from pipeline.data_pipeline import (
    DistributedDataPipeline,
    AugmentationPipeline,
    DatasetSplitter
)

# Create augmentation pipeline
augmentation = AugmentationPipeline(
    augmentations=[
        lambda x: AugmentationPipeline.random_noise(x, 0.01),
        lambda x: AugmentationPipeline.random_scale(x, (0.9, 1.1))
    ],
    probability=0.5
)

# Split dataset
train_ds, val_ds, test_ds = DatasetSplitter.split_dataset(
    dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)

# Create distributed pipelines
train_pipeline, val_pipeline, test_pipeline = DatasetSplitter.create_pipelines(
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    batch_size=64,
    rank=rank,
    world_size=world_size,
    augmentation_fn=augmentation
)

# Get dataloaders
train_loader = train_pipeline.get_dataloader(shuffle=True)
val_loader = val_pipeline.get_dataloader(shuffle=False)
```

### Performance Profiling

```python
from pipeline.profiler import (
    PerformanceProfiler,
    MemoryTracker,
    ThroughputMeter,
    benchmark_dataloader
)

# Setup profiler
profiler = PerformanceProfiler(
    enabled=True,
    trace_dir='traces',
    profile_memory=True
)

# Profile training
result = profiler.profile_training(
    train_fn=train_step,
    num_steps=100,
    warmup_steps=10,
    active_steps=20
)

# Track memory
memory_tracker = MemoryTracker(device=0)
memory_tracker.log_memory("before_training")
# ... training code ...
memory_tracker.log_memory("after_training")
summary = memory_tracker.get_memory_summary()

# Measure throughput
throughput_meter = ThroughputMeter()
for batch in dataloader:
    throughput_meter.update(batch_size=batch['input'].size(0))
throughput_meter.log_throughput()

# Benchmark dataloader
benchmark_dataloader(train_loader, num_batches=100)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test
pytest tests/test_components.py::TestCheckpointManager -v

# Run tests in parallel
pytest tests/ -n auto
```

## 📈 Monitoring & Metrics

### Prometheus Metrics

Access: `http://localhost:9090`

```promql
# Prediction throughput
rate(secom_predictions_made_total[5m])

# Model accuracy
secom_model_accuracy

# Training loss
training_loss

# GPU memory usage
gpu_memory_allocated_mb
```

### Grafana Dashboards

Access: `http://localhost:3000` (default: admin/admin)

- **ML Performance Dashboard**: Model metrics, accuracy, loss
- **System Metrics Dashboard**: GPU usage, memory, throughput
- **Pipeline Health Dashboard**: Service status, Kafka lag

### MLflow Tracking

```python
# Access MLflow UI
# http://localhost:5000

# Log custom metrics
metrics_tracker.log_metrics({
    'custom_metric': value
}, step=epoch)

# Log model
metrics_tracker.log_model(
    model=model,
    model_path=Path('models/model.pt')
)
```

## 🔧 Configuration

### YAML Configuration

```yaml
# config/config.yaml
model:
  model_name: "secom_classifier"
  version: "1.0.0"
  architecture: "resnet50"
  num_classes: 2

distributed:
  num_workers: 4
  backend: "nccl"  # or "gloo" for CPU
  world_size: 4

training:
  epochs: 100
  learning_rate: 0.001
  use_amp: true
  gradient_accumulation_steps: 2

monitoring:
  enable_mlflow: true
  enable_prometheus: true
  enable_wandb: false
```

### Environment Variables

```bash
# Distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0

# Kafka
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Database
export DATABASE_URL=postgresql://user:pass@localhost:5432/ml_pipeline
```

## 📚 Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture details
- [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md) - ML operations guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment instructions
- [BENCHMARKING.md](BENCHMARKING.md) - Performance benchmarking
- [k8s/README.md](k8s/README.md) - Kubernetes deployment

## 🛠️ Development

### Project Structure

```
distributed_ml_pipeline/
├── config/              # Configuration management
│   ├── config.py
│   └── default_config.yaml
├── pipeline/            # Core pipeline components
│   ├── trainer.py      # Advanced trainer
│   ├── data_pipeline.py # Data loading
│   ├── checkpoint.py   # Checkpoint management
│   └── profiler.py     # Performance profiling
├── monitoring/          # Monitoring and metrics
│   └── metrics_tracker.py
├── tests/              # Test suite
│   ├── test_components.py
│   └── conftest.py
├── k8s/                # Kubernetes manifests
├── .github/workflows/  # CI/CD pipelines
└── docker-compose.enhanced.yml
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8 .
pylint pipeline/ config/ monitoring/

# Type check
mypy .
```

## 🚢 Deployment Options

### 1. Local Development
```bash
python pipeline/model_trainer.py
```

### 2. Docker
```bash
docker-compose -f docker-compose.enhanced.yml up
```

### 3. Kubernetes
```bash
kubectl apply -f k8s/
```

### 4. Cloud (AWS/GCP/Azure)
- See [DEPLOYMENT.md](DEPLOYMENT.md) for cloud-specific instructions

## 🔍 Troubleshooting

### Common Issues

**GPU not detected**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Verify NVIDIA driver
nvidia-smi
```

**Out of memory**
```python
# Reduce batch size or enable gradient accumulation
training:
  batch_size: 32  # Reduce
  gradient_accumulation_steps: 4  # Increase
```

**Checkpoint loading fails**
```python
# Resume from latest checkpoint
checkpoint_manager.resume_training(model, optimizer, scheduler)
```

See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting) for more solutions.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Built with PyTorch, Scikit-learn, Kafka, and PostgreSQL
- Inspired by modern MLOps best practices
- Thanks to the open-source community

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for production ML systems**
