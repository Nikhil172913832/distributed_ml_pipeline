-- Initialize SECOM ML Pipeline Database
-- This script creates all necessary tables for the distributed ML pipeline

-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for better organization
CREATE SCHEMA IF NOT EXISTS secom;

-- ==========================================
-- RAW DATA TABLE
-- ==========================================
-- Stores raw synthetic SECOM data from Kafka producer
CREATE TABLE IF NOT EXISTS secom.raw_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id VARCHAR(100) NOT NULL,
    sample_index INTEGER NOT NULL,
    
    -- All 590 SECOM features (feature_0 to feature_589)
    features JSONB NOT NULL,
    
    -- Target variable
    target INTEGER NOT NULL CHECK (target IN (-1, 1)),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    kafka_offset BIGINT,
    kafka_partition INTEGER,
    kafka_topic VARCHAR(255),
    
    -- Indexing for quick lookups
    UNIQUE(batch_id, sample_index)
);

-- Indexes for raw_data
CREATE INDEX idx_raw_data_batch_id ON secom.raw_data(batch_id);
CREATE INDEX idx_raw_data_created_at ON secom.raw_data(created_at DESC);
CREATE INDEX idx_raw_data_target ON secom.raw_data(target);
CREATE INDEX idx_raw_data_features_gin ON secom.raw_data USING GIN(features);

-- ==========================================
-- PREPROCESSED DATA TABLE
-- ==========================================
-- Stores preprocessed data after feature engineering and scaling
CREATE TABLE IF NOT EXISTS secom.preprocessed_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raw_data_id UUID NOT NULL REFERENCES secom.raw_data(id) ON DELETE CASCADE,
    
    -- Preprocessed features (after imputation, scaling, feature engineering)
    features JSONB NOT NULL,
    
    -- Target variable (copied for convenience)
    target INTEGER NOT NULL,
    
    -- Data quality metrics
    missing_count INTEGER,
    imputation_applied BOOLEAN DEFAULT FALSE,
    feature_count INTEGER,
    
    -- Processing metadata
    preprocessing_version VARCHAR(50),
    processing_duration_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(raw_data_id)
);

-- Indexes for preprocessed_data
CREATE INDEX idx_preprocessed_data_raw_id ON secom.preprocessed_data(raw_data_id);
CREATE INDEX idx_preprocessed_data_created_at ON secom.preprocessed_data(created_at DESC);
CREATE INDEX idx_preprocessed_data_target ON secom.preprocessed_data(target);

-- ==========================================
-- BATCH METADATA TABLE
-- ==========================================
-- Tracks batch processing statistics
CREATE TABLE IF NOT EXISTS secom.batch_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Batch statistics
    total_samples INTEGER NOT NULL,
    pass_samples INTEGER NOT NULL,
    fail_samples INTEGER NOT NULL,
    
    -- Processing stats
    processing_status VARCHAR(50) DEFAULT 'pending',
    raw_ingestion_time TIMESTAMP WITH TIME ZONE,
    preprocessing_start_time TIMESTAMP WITH TIME ZONE,
    preprocessing_end_time TIMESTAMP WITH TIME ZONE,
    total_processing_duration_ms FLOAT,
    
    -- Error tracking
    error_count INTEGER DEFAULT 0,
    error_details TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for batch_metadata
CREATE INDEX idx_batch_metadata_status ON secom.batch_metadata(processing_status);
CREATE INDEX idx_batch_metadata_created_at ON secom.batch_metadata(created_at DESC);

-- ==========================================
-- DEAD LETTER QUEUE TABLE
-- ==========================================
-- Stores failed messages for debugging and retry
CREATE TABLE IF NOT EXISTS secom.dead_letter_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Original message data
    message_key VARCHAR(255),
    message_value TEXT NOT NULL,
    kafka_topic VARCHAR(255) NOT NULL,
    kafka_partition INTEGER,
    kafka_offset BIGINT,
    
    -- Error information
    error_type VARCHAR(100),
    error_message TEXT,
    stack_trace TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'failed',
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for DLQ
CREATE INDEX idx_dlq_status ON secom.dead_letter_queue(status);
CREATE INDEX idx_dlq_created_at ON secom.dead_letter_queue(created_at DESC);
CREATE INDEX idx_dlq_topic ON secom.dead_letter_queue(kafka_topic);

-- ==========================================
-- DATA QUALITY METRICS TABLE
-- ==========================================
-- Tracks data quality metrics over time
CREATE TABLE IF NOT EXISTS secom.data_quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id VARCHAR(100) REFERENCES secom.batch_metadata(batch_id),
    
    -- Quality metrics
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_type VARCHAR(50), -- e.g., 'missing_rate', 'outlier_count', 'distribution_shift'
    
    -- Thresholds and alerts
    threshold_value FLOAT,
    is_anomaly BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for quality metrics
CREATE INDEX idx_quality_metrics_batch_id ON secom.data_quality_metrics(batch_id);
CREATE INDEX idx_quality_metrics_name ON secom.data_quality_metrics(metric_name);
CREATE INDEX idx_quality_metrics_anomaly ON secom.data_quality_metrics(is_anomaly) WHERE is_anomaly = TRUE;

-- ==========================================
-- PIPELINE AUDIT LOG
-- ==========================================
-- Comprehensive audit trail for the pipeline
CREATE TABLE IF NOT EXISTS secom.pipeline_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Event information
    event_type VARCHAR(100) NOT NULL, -- e.g., 'data_ingestion', 'preprocessing', 'storage'
    event_status VARCHAR(50) NOT NULL, -- e.g., 'success', 'failure', 'warning'
    
    -- Context
    batch_id VARCHAR(100),
    component VARCHAR(100), -- e.g., 'producer', 'consumer', 'database'
    
    -- Details
    message TEXT,
    metadata JSONB,
    
    -- Performance
    duration_ms FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for audit log
CREATE INDEX idx_audit_log_event_type ON secom.pipeline_audit_log(event_type);
CREATE INDEX idx_audit_log_status ON secom.pipeline_audit_log(event_status);
CREATE INDEX idx_audit_log_created_at ON secom.pipeline_audit_log(created_at DESC);
CREATE INDEX idx_audit_log_batch_id ON secom.pipeline_audit_log(batch_id);

-- ==========================================
-- VIEWS FOR ANALYTICS
-- ==========================================

-- View: Recent batch summary
CREATE OR REPLACE VIEW secom.recent_batches_summary AS
SELECT 
    bm.batch_id,
    bm.total_samples,
    bm.pass_samples,
    bm.fail_samples,
    ROUND((bm.fail_samples::NUMERIC / NULLIF(bm.total_samples, 0) * 100), 2) as failure_rate_pct,
    bm.processing_status,
    bm.total_processing_duration_ms,
    bm.error_count,
    bm.created_at,
    COUNT(pd.id) as preprocessed_count
FROM secom.batch_metadata bm
LEFT JOIN secom.raw_data rd ON bm.batch_id = rd.batch_id
LEFT JOIN secom.preprocessed_data pd ON rd.id = pd.raw_data_id
GROUP BY bm.id, bm.batch_id, bm.total_samples, bm.pass_samples, bm.fail_samples, 
         bm.processing_status, bm.total_processing_duration_ms, bm.error_count, bm.created_at
ORDER BY bm.created_at DESC;

-- View: Data quality summary
CREATE OR REPLACE VIEW secom.data_quality_summary AS
SELECT 
    metric_name,
    COUNT(*) as measurement_count,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    STDDEV(metric_value) as stddev_value,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count,
    MAX(created_at) as last_measured
FROM secom.data_quality_metrics
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY metric_name
ORDER BY metric_name;

-- View: Pipeline health metrics
CREATE OR REPLACE VIEW secom.pipeline_health AS
SELECT 
    DATE_TRUNC('hour', created_at) as time_bucket,
    event_type,
    event_status,
    COUNT(*) as event_count,
    AVG(duration_ms) as avg_duration_ms,
    MAX(duration_ms) as max_duration_ms
FROM secom.pipeline_audit_log
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', created_at), event_type, event_status
ORDER BY time_bucket DESC, event_type;

-- ==========================================
-- FUNCTIONS & TRIGGERS
-- ==========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION secom.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for batch_metadata
CREATE TRIGGER update_batch_metadata_updated_at 
    BEFORE UPDATE ON secom.batch_metadata 
    FOR EACH ROW 
    EXECUTE FUNCTION secom.update_updated_at_column();

-- Function to log pipeline events
CREATE OR REPLACE FUNCTION secom.log_pipeline_event(
    p_event_type VARCHAR,
    p_event_status VARCHAR,
    p_batch_id VARCHAR DEFAULT NULL,
    p_component VARCHAR DEFAULT NULL,
    p_message TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT NULL,
    p_duration_ms FLOAT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_log_id UUID;
BEGIN
    INSERT INTO secom.pipeline_audit_log (
        event_type, event_status, batch_id, component, 
        message, metadata, duration_ms
    ) VALUES (
        p_event_type, p_event_status, p_batch_id, p_component,
        p_message, p_metadata, p_duration_ms
    ) RETURNING id INTO v_log_id;
    
    RETURN v_log_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust user as needed)
GRANT USAGE ON SCHEMA secom TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA secom TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA secom TO ml_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA secom TO ml_user;

-- Insert initial system record
INSERT INTO secom.pipeline_audit_log (event_type, event_status, component, message)
VALUES ('system_init', 'success', 'database', 'Database schema initialized successfully')
ON CONFLICT DO NOTHING;

-- Display summary
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'SECOM ML Pipeline Database Initialized';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Schema: secom';
    RAISE NOTICE 'Tables created: 6';
    RAISE NOTICE 'Views created: 3';
    RAISE NOTICE 'Functions created: 2';
    RAISE NOTICE '========================================';
END $$;
