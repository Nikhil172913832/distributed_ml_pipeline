# Design Decisions and Constraints

## What This Project Is

A demonstration MLOps pipeline using synthetic SECOM manufacturing data. Built to learn and show understanding of:
- Event streaming with Kafka
- ML model lifecycle management
- Automated retraining workflows
- Basic drift detection

## Key Design Choices

### Why Synthetic Data?

The original SECOM dataset is public but small. I trained an SDV TVAE model to generate realistic samples for continuous pipeline operation. This means:
- Data distribution matches the original
- Can generate unlimited samples for testing
- No real manufacturing insights (it's fake data)

### Why Kafka?

Honestly, for this scale, Kafka is overkill. Direct database writes would work fine. But:
- Kafka is common in real MLOps systems
- Good learning experience
- Shows I can work with message queues

### Why Polling Instead of Events?

The inference service polls the database every 5 seconds instead of using Kafka events. This was simpler to implement and debug, though less efficient.

### Model Selection

Using basic sklearn models (LogisticRegression, RandomForest, GradientBoosting) because:
- Fast to train
- Easy to interpret
- Good enough for binary classification
- No GPU required

Deep learning would be overkill for 590 features and wouldn't add much value.

## Known Issues

### Performance

- Haven't load tested beyond a few hundred samples/sec
- Database will bottleneck before Kafka does
- Model loading from disk is slow (should use shared storage)

### Scalability

- Everything assumes single-instance deployment
- No session affinity or sticky routing
- Model files stored locally (breaks with multiple inference instances)

### Monitoring

- Metrics exist but no alerting configured
- Grafana dashboards are basic
- No distributed tracing

### Security

- No authentication
- No encryption
- No rate limiting
- Secrets in environment variables

This is fine for a demo but would need serious work for production.

## What I'd Do Differently

If rebuilding from scratch:
1. Skip Kafka, use database triggers or Redis pub/sub
2. Add proper API from the start (not as an afterthought)
3. Use MinIO or S3 for model storage
4. Implement proper feature store
5. Add A/B testing framework
6. Use managed services where possible

## Assumptions About Data

- 590 features, all numeric
- Binary classification (pass/fail)
- Highly imbalanced (93% pass, 7% fail)
- Missing values handled with median imputation
- No temporal dependencies between samples

## Assumptions About Deployment

- Running on single machine with Docker Compose
- 8GB RAM available
- Local development only
- No high availability requirements

## Future Improvements

Things I'd add if this were a real project:
- Proper authentication and authorization
- Model performance attribution (which features cause drift)
- Automated rollback on model degradation
- Shadow mode for testing new models
- Cost tracking and optimization
- Better error handling and retry logic

## References

- SECOM Dataset: https://archive.ics.uci.edu/ml/datasets/SECOM
- SDV for synthetic data: https://sdv.dev
- Drift detection: Kolmogorov-Smirnov test
