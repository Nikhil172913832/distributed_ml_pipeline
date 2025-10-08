# 🚀 Distributed ML Pipeline for SECOM Manufacturing Data

A **production-ready, scalable machine learning pipeline** for semiconductor manufacturing quality control using synthetic SECOM data. This system demonstrates end-to-end ML operations with real-time data streaming, preprocessing, and storage.

## 📋 Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECOM ML Pipeline Architecture                │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│     SDV      │──┬──>│    Kafka     │──┬──>│  Consumer &  │
│  Generator   │  │   │   (KRaft)    │  │   │Preprocessor  │
│   (TVAE)     │  │   │              │  │   │              │
└──────────────┘  │   └──────────────┘  │   └──────────────┘
                  │                      │            │
                  │   ┌──────────────┐   │            │
                  └──>│ Raw Data     │   │            v
                      │   Topic      │   │   ┌──────────────┐
                      └──────────────┘   │   │  PostgreSQL  │
                                         │   │   Database   │
                      ┌──────────────┐   │   │              │
                      │ Preprocessed │<──┘   └──────────────┘
                      │   Topic      │
                      └──────────────┘            │
                                                  │
┌──────────────┐      ┌──────────────┐           v
│  Prometheus  │<────>│   Grafana    │   ┌──────────────┐
│   Metrics    │      │  Dashboard   │   │ Audit Logs & │
└──────────────┘      └──────────────┘   │    DLQ       │
                                         └──────────────┘
```

## ✨ Features

### Core Capabilities
- **🔄 Real-time Data Streaming**: Kafka-based event streaming with KRaft mode (no Zookeeper)
- **🤖 Synthetic Data Generation**: SDV (TVAE) model for realistic SECOM manufacturing data
- **🔧 Robust Preprocessing**: Automated missing value imputation and feature scaling
- **💾 Persistent Storage**: PostgreSQL with optimized schema for ML workloads
- **📊 Comprehensive Monitoring**: Prometheus metrics + Grafana dashboards
- **🔍 Data Quality Tracking**: Automated quality metrics and anomaly detection
- **⚠️ Error Handling**: Dead Letter Queue (DLQ) for failed messages
- **📝 Audit Trail**: Complete pipeline audit logging

### Production-Ready Features
- ✅ Connection pooling for database efficiency
- ✅ Batch processing for optimal throughput
- ✅ Graceful shutdown handling
- ✅ Health check endpoints
- ✅ Configurable retry logic
- ✅ Structured logging with Loguru
- ✅ Docker containerization
- ✅ Environment-based configuration

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Streaming** | Apache Kafka (KRaft) | Event streaming and message broker |
| **Database** | PostgreSQL 16 | Persistent data storage |
| **Data Generation** | SDV (TVAE) | Synthetic data generation |
| **Preprocessing** | scikit-learn, pandas | ML preprocessing pipeline |
| **Monitoring** | Prometheus + Grafana | Metrics and visualization |
| **Language** | Python 3.10+ | Primary development language |
| **Orchestration** | Docker Compose | Container orchestration |

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- 8GB+ RAM recommended
- Linux/macOS (Windows with WSL2)

### Installation

1. **Clone the repository**
   ```bash
   cd /home/darklord/Projects/distributed_ml_pipeline
   ```

2. **Run setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   This will:
   - Create Python virtual environment
   - Install dependencies
   - Start Docker containers (Kafka, PostgreSQL, monitoring tools)
   - Create Kafka topics
   - Initialize database schema

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Train the SDV model** (if not already trained)
   ```bash
   source .venv/bin/activate
   python data_generator/secom_raw_trainer.py
   ```

### Running the Pipeline

**Terminal 1 - Start Producer:**
```bash
source .venv/bin/activate
python pipeline/producer.py
```

**Terminal 2 - Start Consumer:**
```bash
source .venv/bin/activate
python pipeline/consumer.py
```

**Terminal 3 - Monitor Health:**
```bash
source .venv/bin/activate
python pipeline/health_check.py
```

## 📦 Pipeline Components

### 1. Data Generator (`producer.py`)
- Loads trained SDV (TVAE) model
- Generates batches of synthetic SECOM data
- Publishes to Kafka topic `secom-raw-data`
- Exposes Prometheus metrics on port 8000

### 2. Preprocessor Consumer (`consumer.py`)
- Consumes raw data from Kafka
- Applies preprocessing pipeline:
  - Missing value imputation (median strategy)
  - Feature standardization (mean=0, std=1)
  - Data quality checks
- Stores results in PostgreSQL
- Exposes Prometheus metrics on port 8001

### 3. Database Layer (`database.py`)
- Thread-safe connection pooling
- Repositories for each data type:
  - Raw data storage
  - Preprocessed data storage
  - Batch metadata tracking
  - Dead letter queue
  - Audit logging

### 4. Monitoring Stack
- **Prometheus**: Scrapes metrics from producer/consumer
- **Grafana**: Visualization dashboards
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

# Models
SDV_MODEL_PATH=./models/sdv_secom_raw.joblib
PREPROCESSING_PIPELINE_PATH=./models/preprocessing_pipeline.joblib

# Monitoring
PROMETHEUS_PORT=8000  # Producer metrics
LOG_LEVEL=INFO
```

### Database Schema

The pipeline uses a comprehensive schema in the `secom` schema:

- **`raw_data`**: Raw synthetic SECOM samples (590 features + target)
- **`preprocessed_data`**: Preprocessed samples ready for ML
- **`batch_metadata`**: Batch processing statistics
- **`dead_letter_queue`**: Failed message tracking
- **`data_quality_metrics`**: Quality metrics over time
- **`pipeline_audit_log`**: Complete audit trail

## 📊 Monitoring

### Access Dashboards

| Service | URL | Credentials |
|---------|-----|-------------|
| Kafka UI | http://localhost:8080 | - |
| pgAdmin | http://localhost:8081 | admin@secom.local / admin |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin |

### Key Metrics

**Producer Metrics:**
- `secom_batches_generated_total`: Total batches generated
- `secom_samples_generated_total`: Total samples created
- `secom_kafka_messages_sent_total`: Messages published
- `secom_generation_duration_seconds`: Generation time

**Consumer Metrics:**
- `secom_messages_consumed_total`: Messages processed
- `secom_batches_processed_total`: Batches completed
- `secom_preprocessing_duration_seconds`: Preprocessing time
- `secom_preprocessing_errors_total`: Error count
- `secom_dlq_messages_total`: DLQ messages

### Database Queries

View recent batches:
```sql
SELECT * FROM secom.recent_batches_summary LIMIT 10;
```

Check data quality:
```sql
SELECT * FROM secom.data_quality_summary;
```

Pipeline health:
```sql
SELECT * FROM secom.pipeline_health;
```

## 🧪 Testing

Run unit tests:
```bash
source .venv/bin/activate
pytest tests/ -v --cov=pipeline
```

Run integration tests:
```bash
pytest tests/ -v -m integration
```

Health check:
```bash
python pipeline/health_check.py
```

## 🐛 Troubleshooting

### Kafka Connection Issues
```bash
# Check Kafka status
docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Verify topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

### PostgreSQL Issues
```bash
# Check PostgreSQL logs
docker logs postgres

# Connect to database
docker exec -it postgres psql -U ml_user -d secom_pipeline
```

### View Logs
```bash
# Producer logs
tail -f logs/producer_*.log

# Consumer logs
tail -f logs/consumer_*.log

# Docker logs
docker-compose logs -f
```

### Reset Everything
```bash
# Stop and remove all data
docker-compose down -v

# Restart
./setup.sh
```

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
