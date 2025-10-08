# SECOM ML Pipeline - System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SECOM ML PIPELINE ECOSYSTEM                       │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  Data Generation   │
│                    │
│  1. SDV Model      │──┐
│     (TVAE)         │  │
│  2. Feature        │  │
│     Engineering    │  │
│  3. Batch          │  │
│     Creation       │  │
└────────────────────┘  │
                        │
                        v
┌────────────────────────────────────────────┐
│         KAFKA MESSAGE BROKER               │
│                                            │
│  Topics:                                   │
│  ├─ secom-raw-data         (3 partitions) │
│  ├─ secom-preprocessed-data (3 partitions)│
│  └─ secom-dead-letter-queue (1 partition) │
└────────────────────────────────────────────┘
                        │
                        v
┌────────────────────┐
│  Preprocessing     │
│                    │
│  1. Consume Batch  │
│  2. Impute Missing │
│  3. Standardize    │
│  4. Quality Check  │
│  5. Store Results  │
└────────────────────┘
                        │
                        v
┌─────────────────────────────────────────────┐
│         POSTGRESQL DATABASE                 │
│                                             │
│  Tables:                                    │
│  ├─ raw_data            (JSONB features)   │
│  ├─ preprocessed_data   (JSONB features)   │
│  ├─ batch_metadata      (statistics)       │
│  ├─ dead_letter_queue   (error tracking)   │
│  ├─ data_quality_metrics (monitoring)      │
│  └─ pipeline_audit_log  (audit trail)      │
└─────────────────────────────────────────────┘
                        │
                        v
┌────────────────────────────────────────────┐
│         MONITORING & OBSERVABILITY         │
│                                            │
│  ┌──────────────┐    ┌──────────────┐    │
│  │  Prometheus  │───>│   Grafana    │    │
│  │   Metrics    │    │  Dashboards  │    │
│  └──────────────┘    └──────────────┘    │
│                                            │
│  ┌──────────────┐    ┌──────────────┐    │
│  │   Loguru     │    │   Kafka UI   │    │
│  │   Logging    │    │   Monitor    │    │
│  └──────────────┘    └──────────────┘    │
└────────────────────────────────────────────┘
```

## Data Flow Diagram

```
1. GENERATION PHASE
   ┌─────────┐
   │   SDV   │
   │  Model  │ → Generate 590 features + target
   └─────────┘
       ↓
   ┌─────────┐
   │  Batch  │ → Group samples (default: 100)
   │  Create │
   └─────────┘
       ↓
   ┌─────────┐
   │  Kafka  │ → Publish to raw-data topic
   │Producer │
   └─────────┘

2. INGESTION PHASE
   ┌─────────┐
   │  Kafka  │ → Consume from raw-data topic
   │Consumer │
   └─────────┘
       ↓
   ┌─────────┐
   │  Store  │ → Save to raw_data table
   │   Raw   │
   └─────────┘

3. PREPROCESSING PHASE
   ┌─────────────┐
   │   Impute    │ → Median imputation
   │   Missing   │
   └─────────────┘
         ↓
   ┌─────────────┐
   │ Standardize │ → Mean=0, Std=1
   │  Features   │
   └─────────────┘
         ↓
   ┌─────────────┐
   │   Quality   │ → Check for anomalies
   │    Check    │
   └─────────────┘
         ↓
   ┌─────────────┐
   │    Store    │ → Save to preprocessed_data
   │ Preprocessed│
   └─────────────┘

4. MONITORING PHASE
   ┌─────────────┐
   │  Metrics    │ → Prometheus counters/histograms
   │ Collection  │
   └─────────────┘
         ↓
   ┌─────────────┐
   │  Grafana    │ → Real-time dashboards
   │Visualization│
   └─────────────┘
         ↓
   ┌─────────────┐
   │   Alerts    │ → Anomaly detection
   │   & Logs    │
   └─────────────┘
```

## Component Interaction

```
┌──────────┐  HTTP:8000  ┌────────────┐
│ Producer │──metrics───>│ Prometheus │
└──────────┘             └────────────┘
     │                         │
     │ Kafka                   │
     v                         v
┌──────────┐             ┌────────────┐
│  Topic   │             │  Grafana   │
│Raw Data  │             └────────────┘
└──────────┘
     │
     │ Consume
     v
┌──────────┐  HTTP:8001  ┌────────────┐
│ Consumer │──metrics───>│ Prometheus │
└──────────┘             └────────────┘
     │
     │ Store
     v
┌──────────┐  TCP:5432   ┌────────────┐
│PostgreSQL│<────────────│  pgAdmin   │
└──────────┘             └────────────┘
```

## Database Schema

```sql
secom (schema)
│
├── raw_data
│   ├── id (UUID, PK)
│   ├── batch_id (VARCHAR)
│   ├── sample_index (INTEGER)
│   ├── features (JSONB)  -- 590 features
│   ├── target (INTEGER)  -- -1 or 1
│   └── metadata (timestamps, kafka info)
│
├── preprocessed_data
│   ├── id (UUID, PK)
│   ├── raw_data_id (UUID, FK)
│   ├── features (JSONB)  -- preprocessed features
│   ├── target (INTEGER)
│   └── quality_metrics
│
├── batch_metadata
│   ├── id (UUID, PK)
│   ├── batch_id (VARCHAR, UNIQUE)
│   ├── statistics (counts, distributions)
│   └── processing_info (status, duration)
│
├── dead_letter_queue
│   ├── id (UUID, PK)
│   ├── message_data
│   ├── error_info
│   └── retry_tracking
│
├── data_quality_metrics
│   ├── id (UUID, PK)
│   ├── batch_id (VARCHAR, FK)
│   ├── metric_name
│   ├── metric_value
│   └── anomaly_detection
│
└── pipeline_audit_log
    ├── id (UUID, PK)
    ├── event_type
    ├── event_status
    ├── metadata (JSONB)
    └── timestamp
```

## Metrics Collected

### Producer Metrics
- `secom_batches_generated_total` - Counter
- `secom_samples_generated_total` - Counter
- `secom_kafka_messages_sent_total` - Counter
- `secom_kafka_send_errors_total` - Counter
- `secom_generation_duration_seconds` - Histogram
- `secom_active_batches` - Gauge

### Consumer Metrics
- `secom_messages_consumed_total` - Counter
- `secom_batches_processed_total` - Counter
- `secom_preprocessing_errors_total` - Counter
- `secom_preprocessing_duration_seconds` - Histogram
- `secom_db_insert_duration_seconds` - Histogram
- `secom_dlq_messages_total` - Counter
- `secom_active_processing_batches` - Gauge

## Technology Stack Details

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Generation** | SDV (TVAE) | Synthetic SECOM data with realistic missing patterns |
| **Message Queue** | Apache Kafka 7.5 (KRaft) | Distributed streaming platform |
| **Database** | PostgreSQL 16 | ACID-compliant data storage |
| **Preprocessing** | scikit-learn, pandas | ML preprocessing operations |
| **Monitoring** | Prometheus 2.x | Metrics collection |
| **Visualization** | Grafana | Dashboard and alerting |
| **Logging** | Loguru | Structured logging |
| **Containerization** | Docker, Docker Compose | Service orchestration |

## Scalability Considerations

1. **Horizontal Scaling**
   - Kafka: Add partitions for parallel processing
   - Consumer: Run multiple consumer instances in same group
   - Database: Read replicas for query load

2. **Vertical Scaling**
   - Increase Kafka broker resources
   - Optimize PostgreSQL connection pool
   - GPU acceleration for preprocessing

3. **Performance Optimization**
   - Batch processing for efficiency
   - Connection pooling for database
   - Compression for Kafka messages
   - Indexing for database queries
