# Distributed ML Pipeline

End-to-end MLOps pipeline demonstrating continuous learning patterns for binary classification on the SECOM manufacturing dataset.

## What This Is

A learning project showing how to build an ML pipeline with:
- Kafka streaming for data ingestion
- PostgreSQL for data storage and model registry
- Automated retraining based on performance degradation
- Basic drift detection using statistical tests
- REST API for predictions

This uses synthetic data generated from the original SECOM dataset. It's designed to demonstrate MLOps concepts, not for production manufacturing use.

## Stack

- **ML**: scikit-learn (LogisticRegression, RandomForest, GradientBoosting)
- **Data**: Kafka, PostgreSQL, Redis
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker Compose

## Quick Start

```bash
./setup.sh
make start
```

Access Grafana at http://localhost:3000 (admin/admin)

Check status:
```bash
make status
make performance
```

## API Usage

The prediction API runs on port 8004:

```bash
# Single prediction
curl -X POST http://localhost:8004/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_0": 1.2, "feature_1": -0.5, ...}}'

# Health check
curl http://localhost:8004/health

# Model info
curl http://localhost:8004/model/info
```

## How It Works

1. Producer generates synthetic SECOM data every 5 seconds
2. Consumer preprocesses and stores in PostgreSQL
3. Inference service makes predictions and tracks performance
4. Retrainer automatically triggers when accuracy drops below 85%
5. Drift detector runs KS-test every 6 hours

## Training

```bash
# Train new model
python pipeline/model_trainer.py --data-days 7 --auto-deploy

# Trigger retraining
make trigger-retrain
```

## Testing

```bash
pytest tests/ -v
```

Note: Some tests require running services (Kafka, PostgreSQL).

## Known Limitations

- Uses synthetic data only
- Polling-based inference (not event-driven)
- No authentication or rate limiting
- Single-instance deployment (no horizontal scaling configured)
- Model loading from local filesystem (won't work in distributed setup)
- No automated rollback if new model performs worse

## Project Structure

```
pipeline/          # Core services (producer, consumer, inference, retrainer)
database/          # Schema and database utilities
tests/             # Integration and unit tests
models/            # Trained model artifacts
config/            # Configuration management
```

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Key metrics tracked:
- Prediction throughput
- Model accuracy and F1 score
- Inference latency
- Drift detection events

## Troubleshooting

```bash
make logs              # View all logs
make clean             # Reset everything
docker-compose restart # Restart services
```

Common issues:
- Services not starting: Check Docker has enough memory (8GB recommended)
- No predictions: Train a model first with `make train`
- Database errors: Run `./setup.sh` to reinitialize

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and data flow
- [ASSUMPTIONS.md](ASSUMPTIONS.md) - Design decisions and constraints

## What I Learned Building This

- Setting up Kafka with Docker can be tricky (KRaft mode helps)
- Database schema design for ML metadata is non-trivial
- Drift detection needs careful threshold tuning
- Synthetic data is useful for demos but has limitations
- Monitoring is essential but easy to over-engineer

## License

MIT
