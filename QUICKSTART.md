# Quick Reference - SECOM ML Pipeline

## Quick Commands

### Setup & Start
```bash
# Initial setup
./setup.sh

# Or using Make
make setup

# Start services
make start

# Check health
make health
```

### Run Pipeline
```bash
# Terminal 1 - Producer
make producer

# Terminal 2 - Consumer  
make consumer
```

### Monitoring
```bash
# View all logs
make logs

# Producer logs only
make logs-producer

# Consumer logs only
make logs-consumer

# Health check
python pipeline/health_check.py
```

### Database Operations
```bash
# Connect to PostgreSQL
make db-shell

# View recent batches
psql> SELECT * FROM secom.recent_batches_summary LIMIT 5;

# Check data quality
psql> SELECT * FROM secom.data_quality_summary;

# View audit logs
psql> SELECT * FROM secom.pipeline_audit_log ORDER BY created_at DESC LIMIT 10;
```

### Kafka Operations
```bash
# List topics
make kafka-topics

# Consume messages (console)
make kafka-console

# Create new topic
docker exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partitions 3 \
  --replication-factor 1
```

## Dashboard URLs

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| Kafka UI | http://localhost:8080 | - |
| pgAdmin | http://localhost:8081 | admin@secom.local / admin |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |
| Producer Metrics | http://localhost:8000/metrics | - |
| Consumer Metrics | http://localhost:8001/metrics | - |

## Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables |
| `docker-compose.yml` | Service orchestration |
| `monitoring/prometheus.yml` | Metrics scraping |
| `database/init/01_init_schema.sql` | Database schema |

## Common Tasks

### Reset Everything
```bash
make clean
make setup
```

### Update Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests
```bash
make test
```

### Format Code
```bash
make format
make lint
```

## Troubleshooting

### Kafka Not Starting
```bash
# Check logs
docker logs kafka

# Restart Kafka
docker restart kafka

# Recreate cluster ID
docker-compose down -v
make start
```

### Database Connection Error
```bash
# Check PostgreSQL status
docker exec postgres pg_isready

# View logs
docker logs postgres

# Restart database
docker restart postgres
```

### Producer/Consumer Not Working
```bash
# Check if models exist
ls -lh models/

# Train SDV model
python data_generator/secom_raw_trainer.py

# Check environment variables
cat .env

# View detailed logs
tail -f logs/producer_*.log
tail -f logs/consumer_*.log
```

## Performance Tips

### Increase Throughput
```env
# In .env
BATCH_SIZE=500                    # Larger batches
GENERATION_INTERVAL_SECONDS=1     # Faster generation
```

### Reduce Latency
```env
BATCH_SIZE=10                     # Smaller batches
GENERATION_INTERVAL_SECONDS=10    # Slower, steady flow
```

### Database Optimization
```sql
-- Create indexes for your queries
CREATE INDEX idx_custom ON secom.raw_data(your_column);

-- Analyze tables
ANALYZE secom.raw_data;
```

## Useful SQL Queries

```sql
-- Total samples processed
SELECT COUNT(*) FROM secom.raw_data;

-- Batches by status
SELECT processing_status, COUNT(*) 
FROM secom.batch_metadata 
GROUP BY processing_status;

-- Recent errors
SELECT * FROM secom.dead_letter_queue 
WHERE status = 'failed' 
ORDER BY created_at DESC 
LIMIT 10;

-- Average processing time
SELECT AVG(total_processing_duration_ms) as avg_ms
FROM secom.batch_metadata
WHERE processing_status = 'completed';

-- Class distribution
SELECT target, COUNT(*) as count
FROM secom.raw_data
GROUP BY target;
```

## Key Metrics to Monitor

### Producer
- Batch generation rate (batches/minute)
- Kafka send success rate
- Generation duration (should be <1s)

### Consumer
- Message processing rate
- Preprocessing duration
- Database insert duration
- Error rate (<1%)

### System
- Kafka lag (should be near 0)
- Database connection pool usage
- Memory usage
- Disk space

## Getting Help

1. Check logs: `make logs`
2. Run health check: `make health`
3. Review documentation: `README.md`, `ARCHITECTURE.md`
4. Verify configuration in `.env`
