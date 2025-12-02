# Assumptions and Limitations

This document outlines the key assumptions, constraints, and limitations of the distributed ML pipeline for the SECOM manufacturing dataset.

## Table of Contents
- [Data Assumptions](#data-assumptions)
- [Model Assumptions](#model-assumptions)
- [System Assumptions](#system-assumptions)
- [Limitations](#limitations)
- [Production Considerations](#production-considerations)

---

## Data Assumptions

### Data Distribution

**Expected Input Format:**
- **Features:** 590 numeric features representing semiconductor manufacturing sensor readings
- **Target:** Binary classification (pass/fail, encoded as 0/1)
- **Data Type:** All features are continuous numerical values (floats)
- **Class Imbalance:** Highly imbalanced dataset (~93% negative class, ~7% positive class)

**Statistical Properties:**
- Features are assumed to be independently measured sensor readings
- Values may contain missing data (NaN) which is common in manufacturing sensor logs
- No strict value ranges enforced (sensors may have different scales)
- Temporal dependencies are not explicitly modeled (each sample treated independently)

**Missing Data:**
- Missing values can occur due to sensor failures or data collection issues
- Assumed to be Missing At Random (MAR) or Missing Completely At Random (MCAR)
- Imputation strategy: Median imputation for numerical features by default
- Columns with >80% missing values may indicate faulty sensors

**Outliers:**
- Sensor readings may contain outliers due to:
  - Equipment malfunction
  - Calibration errors
  - Physical anomalies in manufacturing process
- Outlier handling: IQR-based clipping by default (configurable)

---

## Model Assumptions

### Feature Assumptions

**Feature Engineering:**
- No domain-specific feature engineering is applied by default
- Raw sensor readings are assumed to contain sufficient signal
- Feature scaling (StandardScaler) is applied to all features
- No feature selection by default (all 590 features used)

**Feature Importance:**
- Not all 590 features are expected to be equally informative
- Some features may be redundant or correlated
- Model selection (Random Forest, Logistic Regression) handles feature importance internally

**Temporal Aspects:**
- Samples are assumed to be independent (no time series modeling)
- No concept of "manufacturing batch" or temporal ordering
- Retraining occurs periodically but doesn't use temporal patterns

### Model Selection

**Default Models:**
- **Logistic Regression:** Baseline linear model
- **Random Forest:** Primary production model (better handling of non-linear patterns)
- **XGBoost/LightGBM:** Available but not default (computational cost)

**Training Assumptions:**
- Training data is representative of production data distribution
- Class imbalance is handled via `class_weight='balanced'` or SMOTE
- Model performance is evaluated using F1-score (balanced for imbalanced classes)
- Validation set is 20% of training data (stratified split)

**Performance Expectations:**
- **Minimum acceptable F1-score:** 0.75
- **Target F1-score:** 0.80+
- **Inference latency:** <100ms per batch (batch size 32)
- **Model size:** <500MB serialized

---

## System Assumptions

### Infrastructure

**Compute Resources:**
- **Training:** 4-8 CPU cores, 16GB RAM minimum
- **Inference:** 2 CPU cores, 4GB RAM per instance
- **GPU:** Optional for training (not required for baseline models)
- **Distributed Training:** Assumes homogeneous compute nodes

**Storage:**
- **Model Registry:** Local filesystem or S3-compatible storage
- **Data Lake:** Kafka topics for streaming, PostgreSQL for metadata
- **Logs:** 90-day retention, compressed after 7 days

**Network:**
- Kafka brokers accessible at low latency (<10ms)
- Database connections stable (connection pooling enabled)
- External API calls (MLflow, Slack webhooks) have timeouts (5-10s)

### Scalability

**Throughput Assumptions:**
- **Inference:** 1000-10000 requests/second (horizontally scaled)
- **Training:** Full retraining every 7 days or on drift detection
- **Data Ingestion:** 100-1000 samples/second via Kafka

**Horizontal Scaling:**
- Inference service: Stateless, scales with Kubernetes HPA
- Consumer service: Multiple consumers per Kafka partition (consumer group)
- Kafka: Minimum 3 brokers for production (replication factor 3)

**Resource Limits:**
- Kubernetes pods limited to 2GB RAM, 1 CPU per instance (configurable)
- Auto-scaling triggers at 70% CPU utilization
- Maximum 10 inference replicas (configurable)

---

## Limitations

### Data Quality

**Known Issues:**
1. **High Missing Data Rate:** Some features have >50% missing values
   - *Impact:* Reduces effective feature set
   - *Mitigation:* Imputation or feature removal

2. **Feature Redundancy:** Many features likely correlated
   - *Impact:* Model complexity, potential overfitting
   - *Mitigation:* Feature selection, regularization

3. **Class Imbalance:** 93:7 class ratio
   - *Impact:* Model bias toward majority class
   - *Mitigation:* Class weighting, SMOTE, F1-score optimization

### Model Limitations

**Interpretability:**
- Random Forest provides feature importance but not direct causality
- Deep learning models (if used) lack interpretability
- No SHAP/LIME explanations implemented by default

**Drift Detection:**
- Statistical drift detection (KS test, PSI) may have false positives
- Concept drift (changed relationship X→y) harder to detect than covariate drift
- No continuous drift monitoring (batch-based checks only)

**Retraining:**
- Full retraining required (no incremental learning)
- Retraining latency: 30-60 minutes for full dataset
- Manual approval required before deploying new models (SafeRetrainer)

### Operational Limitations

**Monitoring:**
- Alerting requires external services (Slack, email)
- No built-in anomaly detection for system metrics
- Prometheus metrics require Grafana for visualization

**Disaster Recovery:**
- Model registry backups not automated
- Kafka topic replication assumes 3+ brokers
- No multi-region failover

**Security:**
- Authentication not implemented (assumes internal network)
- No encryption at rest for models or data
- Secrets managed via environment variables (not Vault)

---

## Production Considerations

### Performance Tuning

**Inference Optimization:**
- Batch predictions recommended (batch size 16-64)
- Model quantization not implemented (ONNX/TensorRT potential improvement)
- Caching predictions not implemented (stateless by default)

**Training Optimization:**
- Distributed training requires careful data partitioning
- Hyperparameter tuning not automated (manual grid search)
- GPU training supported but not required

### Data Versioning

**Assumptions:**
- Data hash (SHA256) uniquely identifies dataset versions
- Config hash tracks preprocessing/training configuration
- Git commit SHA tracks code version
- No Delta Lake or data catalog integration

### Monitoring Requirements

**Required Metrics:**
- **Model Performance:** F1, precision, recall, AUC-ROC
- **System Health:** Inference latency, throughput, error rate
- **Data Quality:** Missing value rate, drift scores, schema violations

**Alerting Thresholds:**
- F1-score drop >5%: Warning
- F1-score drop >10%: Critical
- Inference latency >500ms (P95): Warning
- Error rate >1%: Critical
- Data drift PSI >0.25: Warning

---

## Future Improvements

### Short-Term (3-6 months)
1. Implement SHAP/LIME for model explanations
2. Add incremental learning for faster retraining
3. Automate hyperparameter tuning (Optuna/Ray Tune)
4. Implement model A/B testing framework

### Long-Term (6-12 months)
1. Deep learning models (Transformers for tabular data)
2. Multi-model ensemble
3. Real-time drift detection (online algorithms)
4. Federated learning for multi-site manufacturing

---

## References

- **Dataset:** [SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM)
- **Drift Detection:** Kolmogorov-Smirnov test, Population Stability Index (PSI)
- **Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Model Registry:** MLflow integration (optional)

---

**Last Updated:** 2025-05-28

**Maintained By:** ML Platform Team
