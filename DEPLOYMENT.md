# 🚀 Quick Deployment Guide

This guide will help you get the SECOM ML Pipeline up and running in minutes.

## Prerequisites

- Docker & Docker Compose installed
- At least 8GB RAM available
- 10GB free disk space

## 🎯 Quick Start (5 Minutes)

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

# Wait for all services to be "healthy" or "running"
# This usually takes 30-60 seconds
```

### 3. Verify the Pipeline

#### Check Data Ingestion
```bash
# View producer logs (should show batches being generated)
docker-compose logs producer | tail -20

# Check Kafka UI
open http://localhost:8080
# You should see the 'secom-raw-data' topic with messages
```

#### Check Database
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

Open these URLs in your browser:

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
  - Go to Dashboards → SECOM ML Pipeline
- **Prometheus**: http://localhost:9090
- **Kafka UI**: http://localhost:8080
- **pgAdmin**: http://localhost:8081

## 🔍 Verify Everything Works

### Check Current Status
```bash
make status
```

Expected output:
```
=== SECOM ML Pipeline Status ===

Services:
NAME                COMMAND              SERVICE     STATUS
kafka              ...                  kafka       running (healthy)
postgres           ...                  postgres    running (healthy)
producer           ...                  producer    running
consumer           ...                  consumer    running
inference          ...                  inference   running
retrainer          ...                  retrainer   running

Recent Activity:
event_type        | event_status | component  | created_at
------------------+--------------+------------+------------------------
batch_processing  | success      | consumer   | 2025-10-09 10:30:45
inference         | success      | inference  | 2025-10-09 10:30:50
```

### Check Model Performance
```bash
make model-info
```

Expected output:
```
=== Active Model Information ===
model_name         | model_version    | model_type          | test_accuracy | test_f1_score | deployed_at
-------------------+------------------+---------------------+---------------+---------------+-------------
secom_classifier  | 20251009_103000  | logistic_regression | 0.89          | 0.87          | 2025-10-09
```

### View Predictions
```bash
make performance
```

## 🎨 What's Happening?

Once the pipeline is running, here's what happens automatically:

1. **Producer** generates synthetic SECOM data every 5 seconds
2. **Consumer** processes and stores the data in PostgreSQL
3. **Inference** makes predictions on new data
4. **Performance Monitor** tracks model accuracy every hour
5. **Drift Detector** checks for data drift every 6 hours
6. **Retrainer** automatically retrains the model when:
   - Accuracy drops below 85%
   - F1 score drops below 80%
   - Significant data drift detected

## 🛠️ Common Operations

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

## 📊 Monitoring Dashboards

### Grafana Setup

1. Open http://localhost:3000
2. Login with admin/admin
3. Navigate to Dashboards → SECOM ML Pipeline - Model Performance

You'll see:
- Real-time accuracy and F1 score
- Prediction throughput
- Inference latency (p95, p99)
- Drift detection alerts
- Retraining job status

### Prometheus Metrics

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

## 🐛 Troubleshooting

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

## 🔄 Next Steps

1. **Customize Configuration**: Edit `.env` to tune thresholds and intervals
2. **Add Custom Models**: Modify `pipeline/model_trainer.py` to add new algorithms
3. **Create Alerts**: Set up Prometheus alerting rules
4. **Scale Up**: Add more consumer/inference instances
5. **Production Setup**: Enable authentication, SSL, and monitoring

## 📚 More Information

- Full documentation: [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md)
- Architecture details: [ARCHITECTURE.md](ARCHITECTURE.md)
- API Reference: [API.md](API.md) *(coming soon)*

## 🎉 Success!

Your ML pipeline is now running with:
- ✅ Real-time data ingestion
- ✅ Automated preprocessing
- ✅ Model inference
- ✅ Performance monitoring
- ✅ Drift detection
- ✅ Continuous learning

Monitor it via Grafana and watch as it automatically retrains when needed!
