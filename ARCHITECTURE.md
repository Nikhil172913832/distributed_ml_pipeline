# SECOM ML Pipeline - System Architecture

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│           SECOM DISTRIBUTED ML PIPELINE WITH CONTINUOUS LEARNING          │
└───────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  Data Generation   │     ┌────────────────────────────────────────────┐
│  (Producer)        │────▶│       KAFKA MESSAGE BROKER (KRaft)         │
│                    │     │                                            │
│  • SDV TVAE Model  │     │  Topics:                                   │
│  • Synthetic Data  │     │  ├─ secom-raw-data      (streaming)       │
│  • Batch Creation  │     │  └─ secom-dlq           (errors)          │
└────────────────────┘     └────────────────────────────────────────────┘
                                              │
                                              v
┌──────────────────────────────────────────────────────────────────────────┐
│                       DATA PROCESSING PIPELINE                           │
├────────────────────┬─────────────────────────────────────────────────────┤
│  Preprocessing     │              POSTGRESQL DATABASE                    │
│  (Consumer)        │                                                     │
│                    │  Core Tables:                                       │
│  • Consume Batch   │  ├─ raw_data           (590 features + target)     │
│  • Impute Missing  │  ├─ preprocessed_data  (cleaned features)          │
│  • Standardize     │  ├─ batch_metadata     (processing stats)          │
│  • Quality Check   │  ├─ dead_letter_queue  (error tracking)            │
│  • Store to DB     │  └─ pipeline_audit_log (event tracking)            │
└────────────────────┴─────────────────────────────────────────────────────┘
                                              │
                                              v
┌──────────────────────────────────────────────────────────────────────────┐
│                       ML OPERATIONS PIPELINE                             │
├────────────────────┬─────────────────────────────────────────────────────┤
│  Inference Engine  │            ML OPERATIONS TABLES                     │
│                    │                                                     │
│  • Load Model      │  ├─ model_registry      (all trained models)       │
│  • Make Prediction │  ├─ predictions         (all predictions)          │
│  • Track Confidence│  ├─ model_performance_metrics (hourly metrics)     │
│  • Monitor Drift   │  ├─ data_drift_metrics  (drift detection)          │
│  • Trigger Retrain │  ├─ retraining_triggers (retrain events)           │
└────────────────────┤  └─ feature_importance  (model insights)           │
                     │                                                     │
┌────────────────────┴─────────────────────────────────────────────────────┤
│  Continuous Learning (Retrainer)                                         │
│                                                                          │
│  • Monitor Triggers                                                      │
│  • Execute Training ──▶ Model Trainer                                   │
│  • Compare Models        │                                               │
│  • Auto-Deploy           ├─ Logistic Regression                         │
│                          ├─ Random Forest                               │
│                          └─ Gradient Boosting                           │
└──────────────────────────────────────────────────────────────────────────┘
                                              │
                                              v
┌──────────────────────────────────────────────────────────────────────────┐
│                       MONITORING & OBSERVABILITY                         │
├────────────────────┬─────────────────────────────────────────────────────┤
│   Prometheus       │              Grafana Dashboards                     │
│                    │                                                     │
│  Metrics:          │  Visualizations:                                    │
│  • Throughput      │  • Model Performance (Accuracy, F1)                │
│  • Latency         │  • Prediction Distribution                         │
│  • Accuracy        │  • Drift Detection Alerts                          │
│  • Drift Score     │  • Retraining Job Status                           │
│  • Job Status      │  • Pipeline Health                                 │
└────────────────────┴─────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Generation Layer
**Service**: `producer`  
**Port**: 8000 (Prometheus metrics)  
**Language**: Python

**Responsibilities**:
- Generate synthetic SECOM data using trained SDV TVAE model
- Maintain realistic feature distributions and correlations
- Publish batches to Kafka with configurable rate
- Track generation metrics (throughput, data quality)

**Key Technologies**:
- SDV (Synthetic Data Vault) for data generation
- Kafka Producer for streaming
- Prometheus client for metrics

---

### 2. Data Processing Layer
**Service**: `consumer`  
**Port**: 8001 (Prometheus metrics)  
**Language**: Python

**Responsibilities**:
- Consume raw data from Kafka topic
- Apply preprocessing pipeline:
  - Median imputation for missing values
  - Standardization (z-score normalization)
  - Feature engineering
- Store both raw and preprocessed data
- Track data quality metrics
- Handle errors via Dead Letter Queue

**Key Technologies**:
- Kafka Consumer with consumer groups
- Pandas & NumPy for data processing
- PostgreSQL for persistence
- Scikit-learn for preprocessing

---

### 3. ML Inference Layer
**Service**: `inference`  
**Port**: 8002 (Prometheus metrics)  
**Language**: Python

**Responsibilities**:
- Load active model from registry
- Make real-time predictions on preprocessed data
- Calculate prediction confidence and uncertainty
- **Performance Monitoring**:
  - Track hourly accuracy, precision, recall, F1
  - Detect performance degradation
  - Trigger retraining when accuracy < 85% or F1 < 80%
- **Drift Detection**:
  - Run KS-test on feature distributions every 6 hours
  - Compare prediction distributions
  - Alert on significant drift (p-value < 0.05)
- Store all predictions with metadata

**Key Features**:
- Configurable confidence thresholds
- Sliding window performance calculation
- Statistical drift detection (Kolmogorov-Smirnov test)
- Automated trigger generation

---

### 4. Continuous Learning Layer

#### 4a. Model Trainer
**Service**: `model_trainer.py` (invoked by retrainer)  
**Language**: Python

**Responsibilities**:
- Train multiple model types in parallel:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Hyperparameter optimization via GridSearchCV
- 5-fold stratified cross-validation
- Model comparison based on F1 score
- Register models with full metadata
- Save model artifacts to disk

**Training Process**:
```python
1. Load data from PostgreSQL (last N days)
2. Split into train/test sets (80/20)
3. For each model type:
   a. Grid search with CV
   b. Train on full training set
   c. Evaluate on test set
   d. Register in database
4. Select best model by F1 score
5. Deploy if auto-deploy enabled
```

#### 4b. Retrainer Service
**Service**: `retrainer`  
**Port**: 8003 (Prometheus metrics)  
**Language**: Python

**Responsibilities**:
- Poll for retraining triggers every 5 minutes
- Execute training pipeline when triggered
- Manage training job lifecycle
- Enforce cooldown period (6 hours)
- Auto-deploy best model (configurable)
- Track retraining history

**Trigger Types**:
- `performance_degradation`: Model metrics below threshold
- `data_drift`: Significant distribution shift detected
- `scheduled`: Periodic retraining (cron-like)
- `manual`: User-initiated

---

### 5. Data Persistence Layer
**Service**: `postgres`  
**Port**: 5432  
**Database**: PostgreSQL 16

**Schema Design**:

#### Core Tables
- `raw_data`: Original features (JSONB), target, Kafka metadata
- `preprocessed_data`: Cleaned features, quality metrics
- `batch_metadata`: Batch statistics, processing duration
- `dead_letter_queue`: Failed messages for debugging
- `pipeline_audit_log`: Event tracking

#### ML Operations Tables  
- `model_registry`: Model metadata, hyperparameters, test metrics
- `predictions`: All predictions with confidence scores
- `model_performance_metrics`: Time-windowed performance
- `data_drift_metrics`: Drift detection results
- `retraining_triggers`: Retraining events and outcomes
- `feature_importance`: Feature importance tracking

#### Views
- `active_model_performance`: Current model metrics
- `recent_predictions_summary`: Latest predictions
- `drift_detection_summary`: Drift overview
- `model_comparison`: Compare model versions

**Indexes**: Optimized for time-series queries, feature lookups

---

### 6. Monitoring & Observability

#### 6a. Prometheus
**Service**: `prometheus`  
**Port**: 9090  
**Purpose**: Metrics collection

**Scraped Services**:
- Producer (8000)
- Consumer (8001)
- Inference (8002)
- Retrainer (8003)

**Key Metrics**:
```promql
# Throughput
rate(secom_batches_generated_total[5m])
rate(secom_predictions_made_total[5m])

# Model Performance
secom_model_accuracy
secom_model_f1_score

# Latency
histogram_quantile(0.95, secom_inference_duration_seconds_bucket)

# Drift & Retraining
secom_drift_detected_total
secom_retraining_triggered_total
secom_models_deployed_total
```

#### 6b. Grafana
**Service**: `grafana`  
**Port**: 3000  
**Purpose**: Visualization

**Dashboards**:
1. **ML Performance Dashboard**:
   - Real-time accuracy/F1 score graphs
   - Prediction throughput
   - Inference latency (p50, p95, p99)
   - Low confidence prediction alerts
   - Drift detection events
   - Retraining job status

2. **Pipeline Health Dashboard**:
   - Service status
   - Kafka lag
   - Database connections
   - Error rates

---

## Data Flow

### Normal Operation

```
1. Producer generates batch (100 samples)
   ↓
2. Publishes to Kafka topic 'secom-raw-data'
   ↓
3. Consumer reads batch
   ↓
4. Preprocessing applied
   ↓
5. Store raw + preprocessed to PostgreSQL
   ↓
6. Inference polls for new preprocessed data
   ↓
7. Load active model
   ↓
8. Make predictions
   ↓
9. Store predictions with confidence scores
   ↓
10. (Hourly) Calculate performance metrics
    ↓
11. (Every 6h) Check for data drift
```

### Continuous Learning

```
1. Inference detects performance degradation
   (Accuracy < 85% or F1 < 80%)
   ↓
2. Create retraining trigger in database
   ↓
3. Retrainer polls and picks up trigger
   ↓
4. Execute training pipeline:
   - Load last 7 days of data
   - Train 3 model types
   - Compare results
   ↓
5. Select best model (highest F1)
   ↓
6. Register new model version
   ↓
7. Auto-deploy (activate model)
   ↓
8. Inference automatically switches to new model
   ↓
9. Update trigger status to 'completed'
   ↓
10. Enforce 6-hour cooldown period
```

### Drift Detection

```
1. Inference service (every 6 hours):
   ↓
2. Query baseline data (last 7 days)
   ↓
3. Query current window data (last 6 hours)
   ↓
4. For each feature (sample):
   - Run Kolmogorov-Smirnov test
   - Calculate drift score
   ↓
5. If drift detected (p-value < 0.05):
   - Record drift metric
   - Increment drift counter
   ↓
6. If significant drift (>10% features):
   - Create retraining trigger
   - Alert via Grafana
```

---

## Security Considerations

### Current Implementation
- Environment-based configuration
- Database connection pooling
- Kafka consumer groups for load balancing

### Production Enhancements
- [ ] Kafka SASL/SSL authentication
- [ ] PostgreSQL SSL connections
- [ ] API authentication (JWT)
- [ ] Secrets management (Vault/AWS Secrets Manager)
- [ ] Network segmentation
- [ ] Rate limiting
- [ ] Audit logging encryption

---

## Scalability

### Horizontal Scaling
- **Consumer**: Add replicas (Kafka consumer group handles partitioning)
- **Inference**: Multiple instances behind load balancer
- **PostgreSQL**: Read replicas for analytics queries

### Vertical Scaling
- **Database**: Increase connection pool size
- **Inference**: GPU acceleration for larger models
- **Training**: Distributed training (Dask, Ray)

### Performance Optimization
- Batch size tuning
- Database query optimization
- Caching layer (Redis)
- Model quantization
- Feature selection

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Message Queue | Apache Kafka (KRaft) | 7.5.0 |
| Database | PostgreSQL | 16 |
| ML Framework | Scikit-learn | Latest |
| Data Processing | Pandas, NumPy | Latest |
| Synthetic Data | SDV | Latest |
| Monitoring | Prometheus + Grafana | Latest |
| Container Runtime | Docker Compose | Latest |
| Language | Python | 3.11+ |

---

## Performance Characteristics

Performance varies based on hardware and configuration. Use the monitoring tools to measure actual performance.

Monitoring:
- Throughput via Prometheus counters
- Latency via Prometheus histograms
- Accuracy/F1 scores in database

Storage:
- Raw Data: ~50KB per batch (100 samples)
- Predictions: ~5KB per batch
- Models: ~5-50MB per model

---

## Failure Modes & Recovery

### Producer Failure
- Impact: No new data generated
- Recovery: Auto-restart via Docker
- Mitigation: Kafka retains last N hours of data

### Consumer Failure
- Impact: Data backlog in Kafka
- Recovery: Consumer catches up on restart (offset tracking)
- Mitigation: Dead Letter Queue for bad messages

### Inference Failure
- Impact: No predictions made
- Recovery: Auto-restart, pending data queued
- Mitigation: Multiple inference instances

### Database Failure
- Impact: Pipeline halts
- Recovery: PostgreSQL auto-restart
- Mitigation: Regular backups, connection retry logic

### Model Performance Degradation
- Impact: Poor predictions
- Recovery: Automatic retraining triggered
- Mitigation: Performance monitoring, drift detection
