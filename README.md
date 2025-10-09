# 🚀 Distributed ML Pipeline with Continuous Learning

A **production-ready, end-to-end machine learning system** for semiconductor manufacturing quality control. This comprehensive pipeline features real-time inference, automated model retraining, drift detection, and complete observability - built for the SECOM manufacturing dataset.

## 🌟 What Makes This Special?

This isn't just a data pipeline - it's a **complete MLOps solution** with:

- ✅ **Real-time Inference**: Sub-5ms prediction latency with confidence scoring
- ✅ **Continuous Learning**: Automated model retraining when performance degrades
- ✅ **Drift Detection**: Statistical monitoring of feature and prediction distributions
- ✅ **Model Registry**: Version control for all trained models with full metadata
- ✅ **Performance Monitoring**: Hourly accuracy/F1 tracking with automatic alerts
- ✅ **Full Observability**: Prometheus metrics + Grafana dashboards
- ✅ **Production-Grade**: Docker orchestration, health checks, graceful shutdown

## 📋 Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [ML Operations](#ml-operations)
- [Monitoring](#monitoring)
- [Documentation](#documentation)
- [Development](#development)
- [Testing](#testing)

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│          SECOM ML PIPELINE - CONTINUOUS LEARNING SYSTEM           │
└───────────────────────────────────────────────────────────────────┘

  Data Generation          Streaming           Processing
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Producer   │─────▶│    Kafka     │─────▶│   Consumer   │
│  (SDV TVAE)  │      │   (KRaft)    │      │(Preprocess)  │
└──────────────┘      └──────────────┘      └──────┬───────┘
                                                    │
                      ┌──────────────────────────────┘
                      │
                      v
┌───────────────────────────────────────────────────────────────────┐
│                        PostgreSQL Database                        │
│  ┌─────────────┬──────────────┬────────────────┬───────────────┐ │
│  │  Raw Data   │ Preprocessed │ Model Registry │  Predictions  │ │
│  └─────────────┴──────────────┴────────────────┴───────────────┘ │
└───────────────────────────────────────────────────────────────────┘
         │                          │                       │
         v                          v                       v
┌──────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Inference   │      │  Model Trainer   │      │   Retrainer      │
│              │      │                  │      │                  │
│ • Predict    │      │ • GridSearchCV   │◀─────│ • Monitor        │
│ • Monitor    │──────│ • Compare Models │      │ • Auto-retrain   │
│ • Drift      │      │ • Register       │      │ • Deploy         │
└──────────────┘      └──────────────────┘      └──────────────────┘
         │                                                │
         └────────────────────┬──────────────────────────┘
                              v
                  ┌──────────────────────┐
                  │ Prometheus + Grafana │
                  │   (Observability)    │
                  └──────────────────────┘
```

## ✨ Features

### 🎯 ML Operations
- **Inference Engine**:
  - Real-time predictions with confidence scores
  - Batch prediction support
  - Low confidence alerts
  - Automatic model loading from registry

- **Continuous Learning**:
  - Automated retraining on performance degradation
  - Multiple model types (LogisticRegression, RandomForest, GradientBoosting)
  - Hyperparameter optimization via GridSearchCV
  - Model comparison and auto-deployment

- **Drift Detection**:
  - Feature distribution monitoring (KS-test)
  - Prediction distribution tracking
  - Configurable sensitivity thresholds
  - Automatic retraining triggers

- **Performance Monitoring**:
  - Hourly accuracy, precision, recall, F1 calculation
  - Sliding window metrics
  - Performance degradation alerts
  - Model comparison dashboards

### 🔄 Data Pipeline
- **Real-time Data Streaming**: Kafka-based event streaming with KRaft mode
- **Synthetic Data Generation**: SDV (TVAE) model for realistic SECOM data
- **Robust Preprocessing**: Median imputation + standardization
- **Data Quality Tracking**: Automated quality metrics and validation

### 📊 Observability
- **Comprehensive Monitoring**: Prometheus metrics + Grafana dashboards
- **Pipeline Audit Logging**: Complete event tracking
- **Error Handling**: Dead Letter Queue (DLQ) for failed messages
- **Health Checks**: Service health endpoints

### 🚀 Production-Ready
- ✅ Docker containerization for all services
- ✅ Connection pooling and retry logic
- ✅ Graceful shutdown handling
- ✅ Environment-based configuration
- ✅ Structured logging with Loguru
- ✅ Integration test suite

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Streaming** | Apache Kafka 7.5 (KRaft) | Event streaming and message broker |
| **Database** | PostgreSQL 16 | Persistent data storage |
| **ML Framework** | Scikit-learn | Model training and inference |
| **Data Generation** | SDV (TVAE) | Synthetic data generation |
| **Preprocessing** | pandas, NumPy | Data processing |
| **Monitoring** | Prometheus + Grafana | Metrics and visualization |
| **Language** | Python 3.11+ | Primary development language |
| **Orchestration** | Docker Compose | Container orchestration |

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- 8GB+ RAM recommended
- Linux/macOS (Windows with WSL2)

### Installation & Deployment

1. **Clone the repository**
   ```bash
   cd /home/darklord/Projects/distributed_ml_pipeline
   ```

2. **Run setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This will:
   - Create Python virtual environment
   - Install dependencies
   - Start Docker containers (Kafka, PostgreSQL, Redis, monitoring)
   - Create Kafka topics
   - Initialize database schema with ML operations tables

3. **Start all services**
   ```bash
   make up
   ```

4. **Verify deployment**
   ```bash
   make status
   ```

   Check service health:
   - Producer: `http://localhost:8000/metrics`
   - Consumer: `http://localhost:8001/metrics`
   - Inference: `http://localhost:8002/metrics`
   - Retrainer: `http://localhost:8003/metrics`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000` (admin/admin)

### Running the Complete Pipeline

**Option 1: Docker Compose (Recommended)**
```bash
# Start all services
make up

# View logs
make logs

# Monitor ML metrics
make metrics

# Check model performance
make performance
```

**Option 2: Manual Execution**
```bash
# Terminal 1 - Producer
source .venv/bin/activate
python pipeline/producer.py

# Terminal 2 - Consumer
source .venv/bin/activate
python pipeline/consumer.py

# Terminal 3 - Inference
source .venv/bin/activate
python pipeline/inference.py

# Terminal 4 - Retrainer
source .venv/bin/activate
python pipeline/retrainer.py
```

## 🤖 ML Operations

### Model Training

Train initial models:
```bash
# Train all model types
python pipeline/model_trainer.py --data-days 7 --auto-deploy

# Train specific model
python pipeline/model_trainer.py --model-type logistic_regression
```

### Inference

The inference service automatically:
- Loads the active model from registry
- Makes predictions on new preprocessed data
- Tracks performance hourly
- Detects drift every 6 hours
- Triggers retraining when needed

Monitor inference:
```bash
# View inference logs
make logs-inference

# Check current model
make model-info

# View drift status
make drift-status
```

### Continuous Learning

The retrainer service handles:
- Monitoring retraining triggers
- Executing training pipeline
- Comparing model versions
- Auto-deploying best models

Trigger manual retraining:
```bash
make trigger-retrain

# View retraining logs
make logs-retrainer
```

### Performance Monitoring

Check model performance:
```bash
# Current model metrics
make performance

# View in Grafana
open http://localhost:3000
# Navigate to "ML Performance Dashboard"
```

## 📊 Monitoring

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

### Grafana Dashboards

Access Grafana: `http://localhost:3000` (admin/admin)

**ML Performance Dashboard**:
- Real-time accuracy and F1 score
- Prediction distribution
- Inference latency percentiles
- Low confidence alerts
- Drift detection events
- Retraining job status

**Pipeline Health Dashboard**:
- Service status
- Kafka lag
- Database connections
- Error rates

## 📚 Documentation

Comprehensive guides available:

- **[ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md)**: Complete ML operations guide
  - Continuous learning workflow
  - Configuration reference
  - Monitoring setup
  - Troubleshooting

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture
  - Component details
  - Data flow diagrams
  - Scalability considerations
  - Failure modes & recovery

- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Quick deployment guide
  - Fast startup
  - Service verification
  - Common issues

- **[QUICKSTART.md](QUICKSTART.md)**: Original quick start
  - Basic pipeline setup
  - Initial configuration

## 📦 Pipeline Components

### 1. Data Generator (`producer.py`)
- Loads trained SDV (TVAE) model
- Generates batches of synthetic SECOM data
- Publishes to Kafka topic `secom-raw-data`
- Exposes Prometheus metrics on port 8000

### 2. Preprocessor (`consumer.py`)
- Consumes raw data from Kafka
- Applies preprocessing pipeline:
  - Missing value imputation (median strategy)
  - Feature standardization (mean=0, std=1)
  - Data quality checks
- Stores results in PostgreSQL
- Exposes Prometheus metrics on port 8001

### 3. Inference Engine (`inference.py`)
- Loads active model from model registry
- Makes real-time predictions on preprocessed data
- Tracks prediction confidence scores
- **Performance monitoring**: Hourly accuracy/F1 calculation
- **Drift detection**: KS-test on feature distributions (every 6h)
- Triggers retraining on degradation or drift
- Exposes Prometheus metrics on port 8002

### 4. Model Trainer (`model_trainer.py`)
- Trains multiple model types with GridSearchCV:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- 5-fold stratified cross-validation
- Compares models by F1 score
- Registers models with full metadata
- Saves model artifacts to disk

### 5. Retrainer Service (`retrainer.py`)
- Polls for retraining triggers every 5 minutes
- Executes model training pipeline
- Manages training job lifecycle
- Auto-deploys best models
- Enforces 6-hour cooldown period
- Exposes Prometheus metrics on port 8003

### 6. Database Layer (`database.py`)
- Thread-safe connection pooling
- Comprehensive repository pattern:
  - **Core**: Raw data, preprocessed data, batches, DLQ, audit logs
  - **ML Ops**: Model registry, predictions, performance metrics, drift metrics, retraining triggers
- PostgreSQL functions for automated calculations

### 7. Monitoring Stack
- **Prometheus**: Scrapes metrics from all services (8000-8003)
- **Grafana**: ML Performance + Pipeline Health dashboards
- **Kafka UI**: Topic monitoring and management
- **pgAdmin**: Database administration

## ⚙️ Configuration

### Environment Variables

Key configurations in `.env`:

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_RAW_TOPIC=secom-raw-data
BATCH_SIZE=100
GENERATION_INTERVAL_SECONDS=5

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=secom_pipeline
POSTGRES_USER=ml_user
POSTGRES_PASSWORD=your_secure_password

# ML Operations
INFERENCE_BATCH_SIZE=100
PERFORMANCE_CHECK_INTERVAL=3600  # 1 hour
DRIFT_CHECK_INTERVAL=21600       # 6 hours
RETRAINING_COOLDOWN=21600        # 6 hours
PERFORMANCE_THRESHOLD_ACCURACY=0.85
PERFORMANCE_THRESHOLD_F1=0.80
DRIFT_P_VALUE_THRESHOLD=0.05

# Models
SDV_MODEL_PATH=./models/sdv_secom_raw.joblib
PREPROCESSING_PIPELINE_PATH=./models/preprocessing_pipeline.joblib
MODEL_ARTIFACTS_DIR=./models

# Monitoring
PROMETHEUS_PORT_PRODUCER=8000
PROMETHEUS_PORT_CONSUMER=8001
PROMETHEUS_PORT_INFERENCE=8002
PROMETHEUS_PORT_RETRAINER=8003
LOG_LEVEL=INFO
```

### Database Schema

The pipeline uses a comprehensive schema in the `secom` schema:

**Core Tables**:
- **`raw_data`**: Raw synthetic SECOM samples (590 features + target)
- **`preprocessed_data`**: Preprocessed samples ready for ML
- **`batch_metadata`**: Batch processing statistics
- **`dead_letter_queue`**: Failed message tracking
- **`pipeline_audit_log`**: Complete event tracking

**ML Operations Tables**:
- **`model_registry`**: All trained models with versions and metadata
- **`predictions`**: All predictions with confidence scores
- **`model_performance_metrics`**: Time-windowed performance tracking
- **`data_drift_metrics`**: Drift detection results
- **`retraining_triggers`**: Retraining events and outcomes
- **`feature_importance`**: Model feature importance

**Views**:
- `active_model_performance`: Current model metrics
- `recent_predictions_summary`: Latest predictions
- `drift_detection_summary`: Drift overview
- `model_comparison`: Compare model versions

## 🧪 Testing

Run comprehensive test suite:
```bash
source .venv/bin/activate

# Unit tests
pytest tests/test_pipeline.py -v

# ML pipeline integration tests
pytest tests/test_ml_pipeline.py -v --cov=pipeline

# Full coverage report
pytest tests/ -v --cov=pipeline --cov-report=html
```

Health check:
```bash
python pipeline/health_check.py
```

## 🐛 Troubleshooting

### Common Issues

**1. Kafka Connection Issues**
```bash
# Check Kafka status
docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Verify topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Check logs
docker logs kafka
```

**2. PostgreSQL Issues**
```bash
# Check PostgreSQL logs
docker logs postgres

# Connect to database
docker exec -it postgres psql -U ml_user -d secom_pipeline

# Verify tables
\dt secom.*
```

**3. Model Not Found**
```bash
# Check model registry
make model-info

# Train initial model
python pipeline/model_trainer.py --data-days 7 --auto-deploy
```

**4. No Predictions Being Made**
```bash
# Check inference logs
make logs-inference

# Verify preprocessed data exists
docker exec -it postgres psql -U ml_user -d secom_pipeline \
  -c "SELECT COUNT(*) FROM secom.preprocessed_data;"
```

**5. Retraining Not Triggering**
```bash
# Check trigger status
docker exec -it postgres psql -U ml_user -d secom_pipeline \
  -c "SELECT * FROM secom.retraining_triggers ORDER BY created_at DESC LIMIT 5;"

# Manual trigger
make trigger-retrain

# Check retrainer logs
make logs-retrainer
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

## � Performance Tuning

### Kafka Optimization
```bash
# Increase partitions for parallelism
docker exec kafka kafka-topics --alter --topic secom-raw-data \
  --partitions 6 --bootstrap-server localhost:9092
```

### Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_preprocessed_created ON secom.preprocessed_data(created_at);
CREATE INDEX idx_predictions_model ON secom.predictions(model_id);
CREATE INDEX idx_performance_window ON secom.model_performance_metrics(window_start);
```

### Inference Optimization
- Increase `INFERENCE_BATCH_SIZE` for higher throughput
- Adjust `PERFORMANCE_CHECK_INTERVAL` based on data volume
- Use model quantization for faster inference

## 🚀 Production Deployment

### Pre-Production Checklist

- [ ] Set strong passwords in `.env`
- [ ] Configure Kafka authentication (SASL/SSL)
- [ ] Enable PostgreSQL SSL connections
- [ ] Set up secrets management (Vault/AWS Secrets Manager)
- [ ] Configure network segmentation
- [ ] Enable audit logging
- [ ] Set up automated backups
- [ ] Configure alerting (PagerDuty, Slack)
- [ ] Load test the system
- [ ] Document runbooks

### Scaling Considerations

**Horizontal Scaling**:
- Add consumer replicas (Kafka handles partitioning)
- Deploy multiple inference instances
- PostgreSQL read replicas for queries

**Vertical Scaling**:
- Increase database connection pool size
- GPU acceleration for model training
- Redis caching layer

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed scalability guidelines.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SECOM Dataset**: UCI Machine Learning Repository
- **SDV**: Synthetic Data Vault for realistic data generation
- **Kafka**: Apache Kafka for streaming infrastructure
- **PostgreSQL**: Robust data persistence
- **Prometheus & Grafana**: Comprehensive monitoring

## 📞 Support

For issues and questions:
- Check [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md) for detailed troubleshooting
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for common deployment issues
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system design details

---

**Built with ❤️ for production ML systems**

*A complete MLOps solution demonstrating continuous learning, drift detection, and automated model management.*

## 📈 Performance Tuning

### Kafka Optimization
- Adjust `BATCH_SIZE` for throughput vs latency
- Increase `max_poll_records` for batch processing
- Configure `compression_type` (gzip, snappy, lz4)

### Database Optimization
- Tune connection pool size (min/max connections)
- Adjust `page_size` for batch inserts
- Create additional indexes for query patterns

### Preprocessing
- Enable GPU acceleration for TVAE model
- Parallelize preprocessing with multiprocessing
- Cache preprocessing pipeline in memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🔗 Related Projects

- [SDV (Synthetic Data Vault)](https://github.com/sdv-dev/SDV)
- [Apache Kafka](https://kafka.apache.org/)
- [scikit-learn](https://scikit-learn.org/)

## 📧 Support

For issues and questions:
- Create an issue in GitHub
- Check existing documentation
- Review logs for error details

---

**Built with ❤️ for robust ML pipelines in production**
