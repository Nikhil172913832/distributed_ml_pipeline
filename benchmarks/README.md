# Benchmarking

This directory contains performance benchmarking tools for the SECOM ML Pipeline.

## 📊 Available Benchmarks

### 1. k6 Load Test (Recommended)
Monitors the inference pipeline by querying Prometheus metrics.

**Prerequisites:**
- Install k6: `brew install k6` (macOS) or see [k6.io](https://k6.io/docs/getting-started/installation/)
- All services running: `make up`
- Services actively processing data

**Run:**
```bash
# Via Makefile (recommended)
make bench

# Or directly
k6 run benchmarks/k6-inference.js
```

**Measured Metrics:**
- Prediction throughput (predictions/sec)
- Inference latency (p95 in milliseconds)
- Model accuracy
- Error rates

### 2. Python Benchmark Script
Direct database queries for accurate measurements.

**Prerequisites:**
- Services running: `make up`
- Python environment: `source .venv/bin/activate`

**Run:**
```bash
# Via Makefile
make bench-python

# Or directly
python benchmarks/benchmark_pipeline.py --duration 60

# Custom duration
python benchmarks/benchmark_pipeline.py --duration 120 --output benchmarks/results/my_test.json

# Skip latency measurement
python benchmarks/benchmark_pipeline.py --no-latency
```

**Measured Metrics:**
- Preprocessing throughput
- Inference throughput
- End-to-end latency (p50, p95, p99)
- Model performance (accuracy, precision, recall, F1)

## 📁 Results

Benchmark results are saved to `benchmarks/results/` with timestamps:

```bash
benchmarks/results/
├── benchmark_latest.json          # Latest run
├── benchmark_2024-12-10_143022.json
└── benchmark_2024-12-10_150133.json
```

### Result Format

```json
{
  "preprocessing": {
    "duration_seconds": 60.02,
    "records_processed": 1543,
    "throughput_per_second": 25.71
  },
  "inference": {
    "duration_seconds": 60.01,
    "predictions_made": 1489,
    "throughput_per_second": 24.81
  },
  "latency": {
    "sample_size": 100,
    "mean_latency_ms": 45.23,
    "p50_latency_ms": 42.10,
    "p95_latency_ms": 67.89,
    "p99_latency_ms": 89.12
  },
  "model_performance": {
    "accuracy": 0.873,
    "precision": 0.865,
    "recall": 0.881,
    "f1_score": 0.873
  },
  "metadata": {
    "timestamp": "2024-12-10T14:30:22",
    "benchmark_script": "benchmark_pipeline.py"
  }
}
```

## 🎯 Interpreting Results

### Throughput
- **Good**: Consistently processes batches without backlog
- **Warning**: Throughput drops significantly under load
- **Issue**: Consumer/inference can't keep up with producer

### Latency
- **Good**: p95 < 100ms for end-to-end pipeline
- **Warning**: p95 > 500ms indicates bottlenecks
- **Issue**: p95 > 1000ms requires investigation

### Model Performance
- **Good**: Accuracy > 85%, F1 > 80% (configured thresholds)
- **Warning**: Metrics dropping, may trigger retraining
- **Issue**: Metrics below thresholds, check model quality

## 🔧 Tuning for Better Performance

### If throughput is low:

```bash
# Increase batch sizes
export BATCH_SIZE=200
export INFERENCE_BATCH_SIZE=200

# Add more Kafka partitions
docker exec kafka kafka-topics --alter \
  --topic secom-raw-data \
  --partitions 6 \
  --bootstrap-server localhost:9092

# Scale consumers
docker-compose up -d --scale consumer=3
```

### If latency is high:

```bash
# Reduce batch sizes (trades throughput for latency)
export BATCH_SIZE=50
export INFERENCE_BATCH_SIZE=50

# Optimize database
# Add indexes (see BENCHMARKING.md)

# Reduce monitoring frequency
export PERFORMANCE_CHECK_INTERVAL=7200
```

## 📈 Continuous Benchmarking

Run benchmarks regularly to catch performance regressions:

```bash
# Run daily benchmark
0 2 * * * cd /path/to/project && make bench-python

# Or create a cron script
cat > scripts/daily_benchmark.sh <<'EOF'
#!/bin/bash
cd /path/to/project
source .venv/bin/activate

# Run benchmark
timestamp=$(date +%Y%m%d_%H%M%S)
python benchmarks/benchmark_pipeline.py \
  --duration 300 \
  --output "benchmarks/results/benchmark_${timestamp}.json"

# Commit results
git add benchmarks/results/
git commit -m "Benchmark results: ${timestamp}"
EOF

chmod +x scripts/daily_benchmark.sh
```

## 🚀 Baseline Performance

Establish your own baselines:

1. **Run initial benchmark** on your hardware
2. **Document configuration** (CPU, RAM, batch sizes)
3. **Save results** with git commit
4. **Monitor trends** over time

**Do NOT use arbitrary numbers** - measure on your specific deployment!

## 📚 See Also

- [BENCHMARKING.md](../BENCHMARKING.md) - Comprehensive benchmarking guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System architecture and scaling
- Grafana dashboards: http://localhost:3000

---

**Remember:** Performance numbers are meaningless without context. Always document your hardware, configuration, and measurement methodology.
