# 🚀 Distributed ML Pipeline with Continuous Learning

**Production-ready MLOps system** for semiconductor manufacturing quality control with real-time inference, automated retraining, and drift detection.

## ✨ Key Features

- 🎯 **Real-time Inference**: Sub-5ms predictions with confidence scoring
- 🔄 **Continuous Learning**: Auto-retraining when accuracy < 85% or drift detected
- 📊 **Drift Detection**: Statistical monitoring (KS-test) every 6 hours
- 🗂️ **Model Registry**: Complete version control with metadata
- 📈 **Observability**: Prometheus + Grafana dashboards
- 🐳 **Production-Ready**: Full Docker orchestration

## 🏗️ Architecture

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

## 🚀 Quick Start

```bash
# 1. Setup environment
./setup.sh

# 2. Start all services
make up

# 3. Monitor
open http://localhost:3000  # Grafana (admin/admin)
```

**Verify deployment:**
```bash
make status      # Service health
make metrics     # Prometheus metrics
make performance # Model accuracy/F1
```


## 🤖 ML Operations

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

## 📊 Monitoring

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| Kafka UI | http://localhost:8080 | - |

**Grafana Dashboards**:
- ML Performance: Accuracy, F1, drift events, retraining status
- Pipeline Health: Service status, throughput, errors

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete system architecture (600+ lines)
- **[ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md)**: ML operations deep dive (300+ lines)
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Quick deployment guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Executive overview

## 🧪 Testing

```bash
pytest tests/test_ml_pipeline.py -v  # ML integration tests
pytest tests/ -v --cov=pipeline      # Full test suite
```

## 🐛 Troubleshooting

```bash
make status          # Check all services
make logs            # View all logs
docker logs <service> # Specific service logs
make clean           # Reset everything
```

**Common issues**: See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting)

## 📈 Performance

- **Throughput**: 500-2000 predictions/sec
- **Latency**: <5ms (p95)
- **Accuracy**: >85% maintained via continuous learning
- **Training**: 5-30 minutes per retraining cycle

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

**Built with ❤️ for production ML systems** | Complete MLOps solution with continuous learning

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
