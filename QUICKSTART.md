# Quick Start Guide: Updated for New Features

This guide has been updated to reflect the new configuration system and features.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Git

## Setup (5 minutes)

### 1. Clone and Install

```bash
git clone <repository-url>
cd distributed_ml_pipeline

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration (optional, defaults work for local development)
nano .env
```

### 3. Start Services

```bash
# Start infrastructure services
docker compose up -d

# Wait for services to be ready (~30 seconds)
docker compose ps

# Initialize database schema (if not auto-initialized)
docker exec -i postgres psql -U ml_user -d secom_pipeline -f /docker-entrypoint-initdb.d/01_schema.sql
```

### 4. Verify Setup

```bash
# Check service health
make health

# View service status
make status
```

## Running the Pipeline

### Option 1: Using Make Commands (Recommended)

```bash
# Start all pipeline services
make start

# Monitor logs
make logs

# Check metrics
make metrics

# View model performance
make performance
```

### Option 2: Manual Service Start

```bash
# Terminal 1: Producer
python pipeline/producer.py

# Terminal 2: Consumer
python pipeline/consumer.py

# Terminal 3: Inference
python pipeline/inference.py

# Terminal 4: Retrainer
python pipeline/retrainer.py
```

## New Features

### MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns --port 5000

# Access at http://localhost:5000
```

### Feature Store (Redis)

The feature store is automatically enabled when Redis is running. Features are cached with 24-hour TTL.

### Data Validation

Data validation runs automatically in the consumer and inference services. Check logs for validation results.

## Development Workflow

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=pipeline --cov-report=html

# Run specific test file
pytest tests/test_ml_pipeline.py -v

# View coverage report
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black pipeline/ config/ monitoring/ tests/

# Lint code
ruff check pipeline/ config/ monitoring/

# Type check
mypy pipeline/ config/ monitoring/
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Monitoring

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

**Key Queries:**
```promql
# Prediction throughput
rate(secom_predictions_made_total[5m])

# Model accuracy
secom_model_accuracy

# Inference latency (p95)
histogram_quantile(0.95, secom_inference_duration_seconds_bucket)
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin)

- ML Performance Dashboard
- Pipeline Health Dashboard

### MLflow Tracking

Access MLflow at `http://localhost:5000`

- Compare experiment runs
- View model metrics and parameters
- Download model artifacts

## Training Models

### Manual Training

```bash
# Train with default settings
make train

# Train with custom parameters
python pipeline/model_trainer.py --data-days 14 --auto-deploy
```

### Trigger Retraining

```bash
# Manual trigger
make trigger-retrain

# Check retraining status
make logs-retrainer
```

## Configuration

### Using New Configuration System

```python
from config.settings import get_settings

# Get all settings
settings = get_settings()

# Access specific settings
kafka_servers = settings.kafka.bootstrap_servers
db_connection = settings.database.connection_string

# Override for testing
from config.settings import override_settings
test_settings = override_settings(
    kafka__bootstrap_servers="test:9092"
)
```

### Environment Variables

All configuration can be set via environment variables:

```bash
# Kafka
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_RAW_TOPIC=secom-raw-data

# Database
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=secom_pipeline

# Inference
export INFERENCE_BATCH_SIZE=100
export INFERENCE_CONFIDENCE_THRESHOLD=0.7

# Monitoring
export MONITORING_ENABLE_MLFLOW=true
export MONITORING_MLFLOW_TRACKING_URI=http://localhost:5000
```

## Troubleshooting

### Services Not Starting

```bash
# Check Docker logs
docker compose logs

# Restart specific service
docker compose restart <service-name>

# Clean restart
make clean
make start
```

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker exec postgres pg_isready -U ml_user

# Reinitialize database
docker compose down -v
docker compose up -d postgres
# Wait 10 seconds
docker exec -i postgres psql -U ml_user -d secom_pipeline -f /docker-entrypoint-initdb.d/01_schema.sql
```

### Kafka Issues

```bash
# List topics
make kafka-topics

# Check Kafka logs
docker logs kafka

# Recreate topics
docker exec kafka kafka-topics --delete --topic secom-raw-data --bootstrap-server localhost:9092
./setup.sh  # Recreates topics
```

## Next Steps

1. **Explore MLflow:** View experiment tracking at http://localhost:5000
2. **Check Metrics:** View Grafana dashboards at http://localhost:3000
3. **Run Tests:** Execute `pytest tests/ -v` to verify everything works
4. **Read Documentation:** Check `docs/` directory for detailed guides

## Useful Commands

```bash
# Service management
make start          # Start all services
make stop           # Stop all services
make restart        # Restart all services
make clean          # Clean and remove volumes

# Monitoring
make status         # Service status
make health         # Health checks
make metrics        # Prometheus metrics
make performance    # Model performance

# Development
make test           # Run tests
make lint           # Run linter
make format         # Format code

# Logs
make logs           # All logs
make logs-inference # Inference logs
make logs-retrainer # Retrainer logs
```

## Additional Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [ML Pipeline Guide](ML_PIPELINE_GUIDE.md)
- [Model Card](docs/MODEL_CARD.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Architecture Decision Records](docs/adr/)
