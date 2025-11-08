# Deployment Guide

## Prerequisites

- Docker & Docker Compose installed
- At least 8GB RAM available
- 10GB free disk space

## Quick Start

### 1. Clone and Setup
```bash
git clone <repo-url>
cd distributed_ml_pipeline

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d
```

### 2. Wait for Services to Initialize
```bash
# Check service health
docker-compose ps
```

### 3. Verify the Pipeline

Check Data Ingestion:
```bash
# View producer logs
docker-compose logs producer | tail -20

# Check Kafka UI
open http://localhost:8080
```

Check Database:
```bash
# Connect to database
docker-compose exec postgres psql -U ml_user -d secom_pipeline

# Check raw data
SELECT COUNT(*) FROM secom.raw_data;

# Check preprocessed data
SELECT COUNT(*) FROM secom.preprocessed_data;

# Exit
\q
```

### 4. Train Initial Model
```bash
# Option 1: Train locally (if you have Python 3.11+)
pip install -r requirements.txt
python pipeline/model_trainer.py --auto-deploy

# Option 2: Train in container
docker-compose exec retrainer python model_trainer.py --auto-deploy
```

### 5. Monitor the Pipeline

- Grafana Dashboard: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Kafka UI: http://localhost:8080
- pgAdmin: http://localhost:8081

## Verify Status

### Check Current Status
```bash
make status
```

### Check Model Performance
```bash
make model-info
```

### View Predictions
```bash
make performance
```

## Pipeline Operation

1. Producer generates synthetic SECOM data every 5 seconds
2. Consumer processes and stores the data in PostgreSQL
3. Inference makes predictions on new data
4. Performance Monitor tracks model accuracy every hour
5. Drift Detector checks for data drift every 6 hours
6. Retrainer automatically retrains when:
   - Accuracy drops below 85%
   - F1 score drops below 80%
   - Significant data drift detected

## Common Operations

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f inference
docker-compose logs -f retrainer
```

### Manually Trigger Retraining
```bash
make trigger-retrain

# Or via database
docker-compose exec postgres psql -U ml_user -d secom_pipeline -c \
  "INSERT INTO secom.retraining_triggers (trigger_type, trigger_reason, status) 
   VALUES ('manual', 'Testing retraining', 'pending');"
```

### Stop the Pipeline
```bash
# Stop all services (keeps data)
docker-compose down

# Stop and remove all data
docker-compose down -v
```

### Restart Everything
```bash
docker-compose restart
```

## Monitoring

### Grafana

1. Open http://localhost:3000
2. Login with admin/admin
3. Navigate to Dashboards

### Prometheus

Access http://localhost:9090 and query:

```promql
# Model accuracy
secom_model_accuracy

# Predictions per second
rate(secom_predictions_made_total[5m])

# Drift detections
sum(increase(secom_drift_detected_total[24h]))

# Retraining triggers
sum(increase(secom_retraining_triggered_total[24h]))
```

## Troubleshooting

### Services Won't Start
```bash
# Check Docker resources
docker stats

# Check logs for errors
docker-compose logs | grep -i error

# Restart specific service
docker-compose restart <service-name>
```

### No Data in Database
```bash
# Check if producer is running
docker-compose logs producer

# Check if Kafka is receiving messages
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic secom-raw-data \
  --from-beginning \
  --max-messages 5
```

### No Predictions Being Made
```bash
# Check if active model exists
make model-info

# If no active model, train one
docker-compose exec retrainer python model_trainer.py --auto-deploy
```

### High Memory Usage
```bash
# Reduce batch sizes in .env
BATCH_SIZE=50
INFERENCE_BATCH_SIZE=50

# Restart services
docker-compose restart
```

## Next Steps

1. Customize Configuration: Edit `.env` to tune thresholds and intervals
2. Add Custom Models: Modify `pipeline/model_trainer.py`
3. Create Alerts: Set up Prometheus alerting rules
4. Scale Up: Add more consumer/inference instances

## More Information

- [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md): Full documentation
- [ARCHITECTURE.md](ARCHITECTURE.md): Architecture details
