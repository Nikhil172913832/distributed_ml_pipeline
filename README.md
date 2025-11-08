# 🚀 Distributed ML Pipeline with Continuous Learning

**Production-ready MLOps system** for semiconductor manufacturing quality control with real-time inference, automated retraining, and drift detection.

## Key Features

- **Real-time Inference**: Predictions with confidence scoring
- **Continuous Learning**: Auto-retraining when accuracy < 85% or drift detected
- **Drift Detection**: Statistical monitoring (KS-test) every 6 hours
- **Model Registry**: Complete version control with metadata
- **Observability**: Prometheus + Grafana dashboards
- **Production-Ready**: Full Docker orchestration

## Architecture

```
Data → Kafka → Preprocessing → PostgreSQL → Inference → Continuous Learning
         ↓                         ↓            ↓              ↓
    Monitoring ←──────────────────────────────────────────────┘
```

**Services**: Producer | Consumer | Inference | Retrainer | Kafka | PostgreSQL | Redis | Prometheus | Grafana

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams.

## �️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | Scikit-learn (LogisticRegression, RandomForest, GradientBoosting) |
| Streaming | Apache Kafka 7.5 (KRaft) |
| Database | PostgreSQL 16 (13 tables) |
| Monitoring | Prometheus + Grafana |
| Orchestration | Docker Compose |

## Quick Start

```bash
# 1. Setup environment
./setup.sh

# 2. Start all services
make start

# 3. Monitor
open http://localhost:3000  # Grafana (admin/admin)
```

**Verify deployment:**
```bash
make status      # Service health
make metrics     # Prometheus metrics
make performance # Model accuracy/F1
```


## ML Operations

### Training
```bash
python pipeline/model_trainer.py --data-days 7 --auto-deploy
```

### Monitoring
```bash
make model-info      # Current model details
make performance     # Accuracy & F1 scores
make drift-status    # Drift detection status
make logs-inference  # Inference logs
```

### Manual Retraining
```bash
make trigger-retrain
```

**Automated triggers**: Performance < 85% accuracy/80% F1, or drift detected (p-value < 0.05)



## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete system architecture
- **[ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md)**: ML operations deep dive
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Quick deployment guide
- **[BENCHMARKING.md](BENCHMARKING.md)**: Performance measurement guide
- **[benchmarks/](benchmarks/)**: Reproducible benchmark scripts
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Executive overview

## Performance Benchmarking

Measure actual performance on your hardware:

```bash
# Run k6 load test (recommended)
make bench

# Run Python benchmark
make bench-python

# Run all benchmarks
make bench-all
```

**Example output:**
```
✓ Prediction rate: 24.5 predictions/sec
✓ Inference latency (p95): 67ms
✓ Model accuracy: 87.3%
```

Results saved to `benchmarks/results/` with timestamps. See **[benchmarks/README.md](benchmarks/README.md)** for detailed usage and **[BENCHMARKING.md](BENCHMARKING.md)** for comprehensive measurement guide.

**Note:** Performance varies by hardware/configuration. Always benchmark your deployment!

## Testing

```bash
pytest tests/test_ml_pipeline.py -v  # ML integration tests
pytest tests/ -v --cov=pipeline      # Full test suite
```

## Troubleshooting

```bash
make status          # Check all services
make logs            # View all logs
docker logs <service> # Specific service logs
make clean           # Reset everything
```

**Common issues**: See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting)

## Monitoring

### Prometheus Metrics

Access Prometheus: `http://localhost:9090`

Key metrics:
```promql
# Prediction throughput
rate(secom_predictions_made_total[5m])

# Model accuracy
secom_model_accuracy

# Inference latency (p95)
histogram_quantile(0.95, secom_inference_duration_seconds_bucket)

# Drift events
secom_drift_detected_total

# Retraining jobs
secom_retraining_triggered_total
```




### 7. Monitoring Stack
- **Prometheus**: Scrapes metrics from all services (8000-8003)
- **Grafana**: ML Performance + Pipeline Health dashboards
- **Kafka UI**: Topic monitoring and management
- **pgAdmin**: Database administration



```

### View Logs
```bash
# All services
make logs

# Specific service
make logs-inference
make logs-retrainer

# Producer/Consumer
tail -f logs/producer_*.log
tail -f logs/consumer_*.log

# Docker logs
docker-compose logs -f [service_name]
```
### Reset Everything
```bash
# Stop and remove all data
make clean

# Restart fresh
./setup.sh
make up
```

**Built with for robust ML pipelines in production**
