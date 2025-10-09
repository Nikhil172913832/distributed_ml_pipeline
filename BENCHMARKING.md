# 📊 Benchmarking Guide

This guide explains how to measure the actual performance of your deployment. Performance characteristics vary based on hardware, configuration, and workload.

## ⚠️ Important Note

Performance claims in documentation should be verified through benchmarking on your specific hardware and configuration. This guide provides tools and methods to establish your own baselines.

## 🎯 What to Measure

### 1. Inference Latency
- **Metric**: Time from receiving data to returning prediction
- **Target**: As measured by your requirements
- **Tools**: Prometheus histograms, custom timing scripts

### 2. Throughput
- **Metric**: Predictions per second
- **Target**: Depends on your workload
- **Tools**: Prometheus counters, load testing tools

### 3. Model Accuracy
- **Metric**: Accuracy, Precision, Recall, F1
- **Target**: Domain-specific (configured as 85% accuracy, 80% F1)
- **Tools**: Database queries, Grafana dashboards

### 4. End-to-End Latency
- **Metric**: Data generation → prediction storage
- **Target**: Depends on use case
- **Tools**: Distributed tracing, timestamps

## 🛠️ Benchmarking Tools

### Option 1: Prometheus Metrics (Built-in)

The system already exposes metrics. Query them:

```bash
# Access Prometheus
open http://localhost:9090

# Example queries:
# - Inference latency (p50, p95, p99):
histogram_quantile(0.95, secom_inference_duration_seconds_bucket)

# - Prediction rate:
rate(secom_predictions_made_total[5m])

# - Current model accuracy:
secom_model_accuracy
```

### Option 2: Database Queries

Check actual performance from stored data:

```sql
-- Average predictions per hour (last 24h)
SELECT 
    date_trunc('hour', created_at) as hour,
    COUNT(*) as predictions
FROM secom.predictions
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- Model performance over time
SELECT 
    window_start,
    accuracy,
    f1_score,
    precision,
    recall
FROM secom.model_performance_metrics
ORDER BY window_start DESC
LIMIT 20;

-- Inference latency (if you add timing columns)
SELECT 
    AVG(EXTRACT(EPOCH FROM (prediction_time - data_arrival_time))) as avg_latency_seconds,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (prediction_time - data_arrival_time))) as p95_latency
FROM secom.predictions
WHERE created_at > NOW() - INTERVAL '1 hour';
```

### Option 3: Load Testing with Python

Create a simple load test script:

```python
# benchmark_inference.py
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pipeline.database import Database, PreprocessedDataRepository

def benchmark_inference_throughput(duration_seconds=60, num_threads=4):
    """
    Measure how many predictions are made over a time period.
    
    Note: This measures the OVERALL system throughput, not just inference.
    """
    db = Database()
    pred_repo = PreprocessedDataRepository(db.get_connection())
    
    start_time = time.time()
    initial_count = pred_repo.count_all()
    
    time.sleep(duration_seconds)
    
    end_time = time.time()
    final_count = pred_repo.count_all()
    
    predictions_made = final_count - initial_count
    elapsed = end_time - start_time
    throughput = predictions_made / elapsed
    
    print(f"Duration: {elapsed:.2f}s")
    print(f"Predictions: {predictions_made}")
    print(f"Throughput: {throughput:.2f} predictions/sec")
    
    return throughput

if __name__ == "__main__":
    print("Benchmarking inference throughput...")
    print("Make sure producer, consumer, and inference are running!")
    print()
    
    throughput = benchmark_inference_throughput(duration_seconds=60)
```

### Option 4: HTTP Load Testing (if you add REST API)

If you expose an HTTP endpoint for inference:

```bash
# Using Apache Bench
ab -n 1000 -c 10 -p data.json -T application/json http://localhost:8002/predict

# Using hey
hey -n 1000 -c 10 -m POST -D data.json http://localhost:8002/predict

# Using k6
k6 run load-test.js
```

Example k6 script:

```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 10,
  duration: '30s',
};

export default function () {
  const payload = JSON.stringify({
    features: Array(590).fill(0).map(() => Math.random())
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  let res = http.post('http://localhost:8002/predict', payload, params);
  
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  sleep(0.1);
}
```

## 📈 Establishing Baselines

### Step 1: Measure Current Performance

Run your benchmarks and record results:

```bash
# Create a results file
cat > benchmark_results.txt <<EOF
Benchmark Date: $(date)
Hardware: [Your CPU/RAM/Disk]
Configuration: [Batch sizes, worker counts, etc.]

Results:
========
Inference Latency (p50): [X]ms
Inference Latency (p95): [X]ms
Inference Latency (p99): [X]ms
Throughput: [X] predictions/sec
Model Accuracy: [X]%
Model F1 Score: [X]
EOF
```

### Step 2: Document Configuration

Record all relevant settings:

```bash
# System info
docker stats --no-stream

# Service configurations
grep -E "BATCH_SIZE|WORKERS|POOL" .env

# Model info
make model-info
```

### Step 3: Run Multiple Tests

Get consistent results:

```bash
# Run 5 benchmark iterations
for i in {1..5}; do
  echo "Run $i:"
  python benchmark_inference.py
  sleep 60
done
```

### Step 4: Analyze Results

Calculate statistics:

```python
import numpy as np

results = [45.2, 47.1, 46.8, 45.9, 46.3]  # Your measurements

print(f"Mean: {np.mean(results):.2f}")
print(f"Std Dev: {np.std(results):.2f}")
print(f"Min: {np.min(results):.2f}")
print(f"Max: {np.max(results):.2f}")
print(f"95% CI: {np.percentile(results, [2.5, 97.5])}")
```

## 🎯 Performance Tuning

Once you have baselines, tune for better performance:

### Inference Optimization

```bash
# 1. Increase batch size (trades latency for throughput)
export INFERENCE_BATCH_SIZE=1000

# 2. Reduce monitoring frequency
export PERFORMANCE_CHECK_INTERVAL=7200  # 2 hours instead of 1

# 3. Use model quantization
# (Implement in model_trainer.py if needed)
```

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_predictions_created ON secom.predictions(created_at);
CREATE INDEX idx_preprocessed_created ON secom.preprocessed_data(created_at);

-- Tune PostgreSQL settings (postgresql.conf)
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 16MB
```

### Kafka Optimization

```bash
# Increase partitions for parallelism
docker exec kafka kafka-topics --alter \
  --topic secom-raw-data \
  --partitions 6 \
  --bootstrap-server localhost:9092

# Increase consumer instances
docker-compose up --scale consumer=3
```

## 📊 Monitoring Performance Over Time

### Grafana Dashboard Queries

Add these to your dashboard:

```promql
# Latency trend (7-day moving average)
avg_over_time(secom_inference_duration_seconds_bucket[7d])

# Throughput trend
rate(secom_predictions_made_total[1h])

# Accuracy trend
secom_model_accuracy
```

### Alerts

Set up alerts for performance degradation:

```yaml
# prometheus/alerts.yml
groups:
  - name: performance
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, secom_inference_duration_seconds_bucket) > 0.1
        for: 5m
        annotations:
          summary: "Inference latency is high (p95 > 100ms)"
      
      - alert: LowThroughput
        expr: rate(secom_predictions_made_total[5m]) < 10
        for: 10m
        annotations:
          summary: "Prediction throughput is low (<10/sec)"
```

## 🔬 Reproducible Benchmarks

To make benchmarks reproducible:

### 1. Document Everything

```markdown
## Benchmark Report

**Date**: 2024-12-10
**Git Commit**: abc123def
**Hardware**: 
- CPU: Intel i7-9700K @ 3.60GHz (8 cores)
- RAM: 32GB DDR4
- Disk: NVMe SSD

**Configuration**:
- INFERENCE_BATCH_SIZE=100
- KAFKA_PARTITIONS=3
- POSTGRES_POOL_SIZE=20

**Results**:
- Inference Latency (p95): 45ms
- Throughput: 150 predictions/sec
- Model Accuracy: 87.3%
```

### 2. Version Control Benchmarks

```bash
# Create benchmarks directory
mkdir -p benchmarks/
cat > benchmarks/2024-12-10_baseline.json <<EOF
{
  "date": "2024-12-10",
  "commit": "abc123",
  "hardware": {
    "cpu": "Intel i7-9700K",
    "ram_gb": 32,
    "disk": "NVMe SSD"
  },
  "config": {
    "batch_size": 100,
    "kafka_partitions": 3
  },
  "results": {
    "inference_p95_ms": 45,
    "throughput_per_sec": 150,
    "accuracy": 0.873
  }
}
EOF

git add benchmarks/
git commit -m "Add baseline benchmark"
```

### 3. Automate Benchmarking

```bash
# scripts/run_benchmarks.sh
#!/bin/bash
set -e

echo "Running automated benchmarks..."

# Wait for services to be ready
sleep 30

# Run benchmarks
python benchmarks/benchmark_inference.py > results.txt
python benchmarks/benchmark_training.py >> results.txt

# Save results with timestamp
mv results.txt "benchmarks/results_$(date +%Y%m%d_%H%M%S).txt"

echo "Benchmarks complete!"
```

## 🎓 Best Practices

1. **Always benchmark on production-like hardware**: Dev machine results don't predict production performance
2. **Run multiple iterations**: Single runs can be misleading due to variance
3. **Measure under load**: Performance degrades under realistic workloads
4. **Document everything**: Configuration, hardware, git commit, date
5. **Automate where possible**: Regular benchmarks catch performance regressions
6. **Set realistic SLAs**: Based on actual measurements, not aspirational goals

## 📚 Further Reading

- [Prometheus Query Examples](https://prometheus.io/docs/prometheus/latest/querying/examples/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
- [Load Testing Best Practices](https://k6.io/docs/test-authoring/test-builder/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)

---

**Remember**: Performance numbers without reproducible methodology are just marketing. Always measure, document, and validate on your specific deployment.
