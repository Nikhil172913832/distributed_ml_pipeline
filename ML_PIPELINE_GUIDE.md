# SECOM ML Pipeline Guide

## Overview

Distributed machine learning pipeline for SECOM semiconductor manufacturing data.

Features:
- Real-time data ingestion via Kafka
- Automated preprocessing and data quality monitoring
- Real-time predictions with confidence scoring
- Automated model retraining based on performance and drift
- Prometheus metrics and Grafana dashboards
- PostgreSQL data persistence

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SECOM ML Pipeline Architecture                    │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌─────────┐     ┌─────────────┐     ┌──────────────┐
│   Producer   │────▶│  Kafka  │────▶│  Consumer   │────▶│  PostgreSQL  │
│  (Synthetic  │     │ (Queue) │     │(Preprocessor)│     │  (Storage)   │
│    Data)     │     └─────────┘     └─────────────┘     └──────────────┘
└──────────────┘                            │                     │
                                            │                     │
                                            ▼                     ▼
                                   ┌─────────────────┐   ┌──────────────┐
                                   │   Inference     │◀──│  Preprocessed│
                                   │    Engine       │   │     Data     │
                                   └─────────────────┘   └──────────────┘
                                            │
                                            ├──▶ Performance Monitoring
                                            ├──▶ Drift Detection
                                            └──▶ Predictions Storage
                                                      │
                                                      ▼
                                            ┌─────────────────┐
                                            │   Retrainer     │
                                            │   (Continuous   │
                                            │    Learning)    │
                                            └─────────────────┘
                                                      │
                                                      ├──▶ Model Training
                                                      ├──▶ Hyperparameter Tuning
                                                      └──▶ Model Deployment

┌──────────────────────────────────────────────────────────────────────────┐
│                          Monitoring Stack                                 │
│  Prometheus (Metrics) ──▶ Grafana (Visualization) ──▶ Alerts             │
└──────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Producer (`pipeline/producer.py`)
- Generates synthetic SECOM data using SDV
- Publishes to Kafka in configurable batches
- Metrics: batch generation rate, Kafka throughput

### 2. Consumer/Preprocessor (`pipeline/consumer.py`)
- Consumes raw data from Kafka
- Applies preprocessing pipeline (imputation, scaling)
- Stores raw and preprocessed data to PostgreSQL
- Dead Letter Queue for failed messages

### 3. Inference Engine (`pipeline/inference.py`)
- Loads active model from registry
- Makes predictions on preprocessed data
- Tracks prediction confidence and uncertainty
- Performance Monitoring: Calculates hourly accuracy, precision, recall, F1
- Drift Detection: Statistical tests (KS-test) for feature and prediction drift
- Triggers Retraining: When performance degrades or drift detected

### 4. Model Trainer (`pipeline/model_trainer.py`)
- Automated training with multiple algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Grid Search with cross-validation
- Model comparison and selection
- Registers models in database with full metadata

### 5. Retrainer (`pipeline/retrainer.py`)
- Monitors retraining triggers
- Executes training pipeline when triggered
- Auto-deploys best model (configurable)
- Cooldown period to prevent excessive retraining

## Database Schema

### ML Operations Tables

1. `model_registry`: All trained models with metadata
2. `predictions`: All predictions with confidence scores
3. `model_performance_metrics`: Time-windowed performance tracking
4. `data_drift_metrics`: Drift detection results
5. `retraining_triggers`: Retraining events and reasons
6. `feature_importance`: Feature importance across model versions

## Quick Start

### Prerequisites
```bash
# Docker and Docker Compose
docker --version
docker-compose --version

# Python 3.11+ (for local development)
python --version
```

### 1. Setup Environment
```bash
# Clone repository
git clone <repo-url>
cd distributed_ml_pipeline

# Create environment file
cp .env.example .env  # Edit as needed

# Create necessary directories
mkdir -p logs models/artifacts
```

### 2. Start Infrastructure
```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f inference
docker-compose logs -f retrainer
```

### 3. Initialize Database
The database schema is automatically created on first startup via `database/init/01_init_schema.sql`.

### 4. Train Initial Model
```bash
# Run initial training (inside retrainer container or locally)
python pipeline/model_trainer.py --triggered-by manual --auto-deploy

# Or manually register existing model
# This requires adding a record to model_registry table
```

### 5. Monitor the Pipeline

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Kafka UI**: http://localhost:8080
- **pgAdmin**: http://localhost:8081

## Monitoring Metrics

### Inference Service (Port 8002)
- `secom_predictions_made_total`: Total predictions
- `secom_model_accuracy`: Current accuracy
- `secom_model_f1_score`: Current F1 score
- `secom_inference_duration_seconds`: Inference latency
- `secom_drift_detected_total`: Drift events
- `secom_low_confidence_predictions_total`: Low confidence count

### Retrainer Service (Port 8003)
- `secom_retraining_jobs_started_total`: Training jobs initiated
- `secom_retraining_jobs_completed_total`: Jobs completed (success/failure)
- `secom_active_training_jobs`: Currently training
- `secom_models_deployed_total`: Models deployed
- `secom_pending_retraining_triggers`: Waiting triggers

## Configuration

### Environment Variables

#### Inference Service
```bash
INFERENCE_BATCH_SIZE=100
INFERENCE_POLL_INTERVAL=5
CONFIDENCE_THRESHOLD=0.7

# Performance Monitoring
PERFORMANCE_WINDOW_HOURS=1
ACCURACY_THRESHOLD=0.85
F1_THRESHOLD=0.80
DEGRADATION_TOLERANCE=0.05

# Drift Detection
DRIFT_DETECTION_ENABLED=true
DRIFT_CHECK_INTERVAL_HOURS=6
KS_TEST_THRESHOLD=0.05
DRIFT_SCORE_THRESHOLD=0.3
BASELINE_DAYS=7
```

#### Retrainer Service
```bash
RETRAINER_CHECK_INTERVAL=300  # seconds
RETRAINER_AUTO_DEPLOY=true
MAX_CONCURRENT_TRAINING=1
RETRAINING_COOLDOWN_HOURS=6

# Training Configuration
TRAIN_TEST_SPLIT=0.2
RANDOM_STATE=42
CV_FOLDS=5
N_JOBS=-1
```

## Continuous Learning Workflow

### 1. Performance Degradation Trigger
```
Inference Engine monitors hourly performance
  ↓
If Accuracy < 0.85 OR F1 < 0.80
  ↓
Create retraining trigger
  ↓
Retrainer picks up trigger
  ↓
Train new model with latest data
  ↓
Compare with current model
  ↓
Deploy if better (auto-deploy enabled)
```

### 2. Data Drift Trigger
```
Inference Engine checks drift every 6 hours
  ↓
Run KS-test on sample features
  ↓
If significant drift detected (p < 0.05)
  ↓
Create retraining trigger
  ↓
Retraining workflow initiated
```

### 3. Manual/Scheduled Trigger
```bash
# Manually trigger retraining
INSERT INTO secom.retraining_triggers (
    trigger_type, trigger_reason, status
) VALUES (
    'manual', 'Scheduled weekly retraining', 'pending'
);
```

## Testing

### Run Integration Tests
```bash
# Test producer-consumer flow
pytest tests/test_pipeline.py -v

# Test inference engine
pytest tests/test_inference.py -v

# Test retraining workflow
pytest tests/test_retrainer.py -v
```

### Manual Testing
```bash
# Check database records
docker-compose exec postgres psql -U ml_user -d secom_pipeline

# Query recent predictions
SELECT * FROM secom.predictions ORDER BY created_at DESC LIMIT 10;

# Check model performance
SELECT * FROM secom.active_model_performance;

# View drift events
SELECT * FROM secom.drift_detection_summary;
```

## Performance Optimization

### Batch Size Tuning
- **Producer**: Larger batches = higher throughput, higher latency
- **Consumer**: Balance between processing speed and memory
- **Inference**: Optimize for GPU utilization if available

### Database Optimization
```sql
-- Create additional indexes for frequent queries
CREATE INDEX idx_predictions_timestamp 
ON secom.predictions(prediction_timestamp DESC);

-- Partition large tables by time
-- (Advanced: implement table partitioning for predictions)
```

### Model Optimization
- Use lightweight models for low-latency requirements
- Implement model quantization for reduced size
- Cache predictions for duplicate inputs (Redis)

## Production Considerations

### Security
- [ ] Enable Kafka authentication (SASL/SSL)
- [ ] PostgreSQL SSL connections
- [ ] Secrets management (Vault, AWS Secrets Manager)
- [ ] Network segmentation
- [ ] Rate limiting on API endpoints

### Scalability
- [ ] Horizontal scaling of consumers (Kafka consumer groups)
- [ ] Multiple inference instances behind load balancer
- [ ] Database read replicas for analytics
- [ ] Redis for caching and session management

### Reliability
- [ ] Implement circuit breakers
- [ ] Exponential backoff for retries
- [ ] Health checks and auto-restart
- [ ] Backup and disaster recovery
- Monitoring and alerting (PagerDuty, Slack)

## Troubleshooting

### Issue: No predictions being made
```bash
# Check if preprocessed data exists
docker-compose exec postgres psql -U ml_user -d secom_pipeline \
  -c "SELECT COUNT(*) FROM secom.preprocessed_data;"

# Check if active model exists
docker-compose exec postgres psql -U ml_user -d secom_pipeline \
  -c "SELECT * FROM secom.model_registry WHERE is_active = TRUE;"
```

### Issue: Retraining not triggering
```bash
# Check for pending triggers
docker-compose exec postgres psql -U ml_user -d secom_pipeline \
  -c "SELECT * FROM secom.retraining_triggers WHERE status = 'pending';"

# Check retrainer logs
docker-compose logs -f retrainer
```

### Issue: High memory usage
```bash
# Check batch sizes in environment variables
# Reduce INFERENCE_BATCH_SIZE or BATCH_SIZE

# Monitor container resources
docker stats
```

## Additional Resources

- [SECOM Dataset Documentation](secom/secom.names)
- [Model Comparison Results](models/model_comparison.csv)
- [Architecture Details](ARCHITECTURE.md)

## License

See [LICENSE](LICENSE) file for details.

