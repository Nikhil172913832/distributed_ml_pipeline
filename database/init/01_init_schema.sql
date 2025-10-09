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

-- ==========================================
-- MODEL REGISTRY TABLE
-- ==========================================
-- Tracks all trained models and their metadata
CREATE TABLE IF NOT EXISTS secom.model_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- e.g., 'logistic_regression', 'random_forest', 'xgboost'
    
    -- Model artifacts
    model_path VARCHAR(500) NOT NULL,
    preprocessing_pipeline_path VARCHAR(500),
    
    -- Training metadata
    training_dataset_size INTEGER,
    training_start_time TIMESTAMP WITH TIME ZONE,
    training_end_time TIMESTAMP WITH TIME ZONE,
    training_duration_ms FLOAT,
    
    -- Hyperparameters
    hyperparameters JSONB,
    
    -- Performance metrics on test set
    test_accuracy FLOAT,
    test_precision FLOAT,
    test_recall FLOAT,
    test_f1_score FLOAT,
    test_roc_auc FLOAT,
    
    -- Model status
    status VARCHAR(50) DEFAULT 'trained', -- 'trained', 'deployed', 'archived', 'failed'
    is_active BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP WITH TIME ZONE,
    archived_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    training_triggered_by VARCHAR(100), -- 'manual', 'scheduled', 'performance_degradation', 'data_drift'
    notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, model_version)
);

-- Indexes for model_registry
CREATE INDEX idx_model_registry_status ON secom.model_registry(status);
CREATE INDEX idx_model_registry_active ON secom.model_registry(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_model_registry_created_at ON secom.model_registry(created_at DESC);
CREATE INDEX idx_model_registry_name_version ON secom.model_registry(model_name, model_version);

-- ==========================================
-- PREDICTIONS TABLE
-- ==========================================
-- Stores model predictions for tracking and analysis
CREATE TABLE IF NOT EXISTS secom.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    preprocessed_data_id UUID NOT NULL REFERENCES secom.preprocessed_data(id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES secom.model_registry(id),
    
    -- Prediction results
    prediction INTEGER NOT NULL, -- Predicted class: -1 or 1
    prediction_probability FLOAT, -- Probability of predicted class
    prediction_proba_pass FLOAT, -- Probability of pass (-1)
    prediction_proba_fail FLOAT, -- Probability of fail (1)
    
    -- Ground truth (for validation)
    actual_target INTEGER,
    is_correct BOOLEAN,
    
    -- Confidence and uncertainty
    confidence_score FLOAT,
    uncertainty_score FLOAT,
    
    -- Latency tracking
    inference_duration_ms FLOAT,
    
    -- Metadata
    batch_id VARCHAR(100),
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for predictions
CREATE INDEX idx_predictions_preprocessed_id ON secom.predictions(preprocessed_data_id);
CREATE INDEX idx_predictions_model_id ON secom.predictions(model_id);
CREATE INDEX idx_predictions_batch_id ON secom.predictions(batch_id);
CREATE INDEX idx_predictions_created_at ON secom.predictions(created_at DESC);
CREATE INDEX idx_predictions_is_correct ON secom.predictions(is_correct);

-- ==========================================
-- MODEL PERFORMANCE METRICS TABLE
-- ==========================================
-- Tracks model performance over time windows
CREATE TABLE IF NOT EXISTS secom.model_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES secom.model_registry(id),
    
    -- Time window
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    window_type VARCHAR(50) DEFAULT 'hourly', -- 'hourly', 'daily', 'weekly'
    
    -- Sample counts
    total_predictions INTEGER NOT NULL,
    correct_predictions INTEGER,
    incorrect_predictions INTEGER,
    
    -- Performance metrics
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    
    -- Confusion matrix
    true_positives INTEGER,
    true_negatives INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    
    -- Latency statistics
    avg_inference_duration_ms FLOAT,
    p50_inference_duration_ms FLOAT,
    p95_inference_duration_ms FLOAT,
    p99_inference_duration_ms FLOAT,
    
    -- Confidence statistics
    avg_confidence FLOAT,
    avg_uncertainty FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_id, window_start, window_end, window_type)
);

-- Indexes for performance metrics
CREATE INDEX idx_model_perf_model_id ON secom.model_performance_metrics(model_id);
CREATE INDEX idx_model_perf_window ON secom.model_performance_metrics(window_start DESC, window_end DESC);
CREATE INDEX idx_model_perf_accuracy ON secom.model_performance_metrics(accuracy);

-- ==========================================
-- DATA DRIFT DETECTION TABLE
-- ==========================================
-- Tracks data drift and distribution shifts
CREATE TABLE IF NOT EXISTS secom.data_drift_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Time window
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Drift metrics
    drift_type VARCHAR(100) NOT NULL, -- 'feature_drift', 'target_drift', 'prediction_drift'
    feature_name VARCHAR(255), -- NULL for target/prediction drift
    
    -- Statistical tests
    ks_statistic FLOAT, -- Kolmogorov-Smirnov test
    ks_pvalue FLOAT,
    chi2_statistic FLOAT, -- Chi-square test (for categorical)
    chi2_pvalue FLOAT,
    
    -- Distribution statistics
    mean_baseline FLOAT,
    mean_current FLOAT,
    std_baseline FLOAT,
    std_current FLOAT,
    
    -- Drift detection
    is_drift_detected BOOLEAN DEFAULT FALSE,
    drift_score FLOAT, -- 0-1 score indicating severity
    drift_threshold FLOAT DEFAULT 0.05,
    
    -- Reference baseline
    baseline_batch_id VARCHAR(100),
    baseline_start_date TIMESTAMP WITH TIME ZONE,
    baseline_end_date TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for drift metrics
CREATE INDEX idx_drift_window ON secom.data_drift_metrics(window_start DESC, window_end DESC);
CREATE INDEX idx_drift_type ON secom.data_drift_metrics(drift_type);
CREATE INDEX idx_drift_detected ON secom.data_drift_metrics(is_drift_detected) WHERE is_drift_detected = TRUE;
CREATE INDEX idx_drift_feature ON secom.data_drift_metrics(feature_name);

-- ==========================================
-- RETRAINING TRIGGERS TABLE
-- ==========================================
-- Logs when and why model retraining was triggered
CREATE TABLE IF NOT EXISTS secom.retraining_triggers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Trigger information
    trigger_type VARCHAR(100) NOT NULL, -- 'performance_degradation', 'data_drift', 'scheduled', 'manual'
    trigger_reason TEXT NOT NULL,
    
    -- Threshold violations
    performance_threshold_violated FLOAT,
    current_performance_value FLOAT,
    
    drift_threshold_violated FLOAT,
    current_drift_value FLOAT,
    
    -- Associated metrics
    model_id UUID REFERENCES secom.model_registry(id),
    drift_metric_id UUID REFERENCES secom.data_drift_metrics(id),
    performance_metric_id UUID REFERENCES secom.model_performance_metrics(id),
    
    -- Retraining status
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed'
    retraining_started_at TIMESTAMP WITH TIME ZONE,
    retraining_completed_at TIMESTAMP WITH TIME ZONE,
    new_model_id UUID REFERENCES secom.model_registry(id),
    
    -- Results
    retraining_successful BOOLEAN,
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for retraining triggers
CREATE INDEX idx_retraining_trigger_type ON secom.retraining_triggers(trigger_type);
CREATE INDEX idx_retraining_status ON secom.retraining_triggers(status);
CREATE INDEX idx_retraining_created_at ON secom.retraining_triggers(created_at DESC);

-- ==========================================
-- FEATURE IMPORTANCE TABLE
-- ==========================================
-- Tracks feature importance over model versions
CREATE TABLE IF NOT EXISTS secom.feature_importance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES secom.model_registry(id),
    
    feature_name VARCHAR(255) NOT NULL,
    importance_score FLOAT NOT NULL,
    importance_rank INTEGER,
    importance_type VARCHAR(50), -- 'permutation', 'gain', 'split', 'shap'
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_id, feature_name, importance_type)
);

-- Indexes for feature importance
CREATE INDEX idx_feature_importance_model_id ON secom.feature_importance(model_id);
CREATE INDEX idx_feature_importance_score ON secom.feature_importance(importance_score DESC);

-- ==========================================
-- ADDITIONAL VIEWS FOR ML OPERATIONS
-- ==========================================

-- View: Current active model performance
CREATE OR REPLACE VIEW secom.active_model_performance AS
SELECT 
    mr.id as model_id,
    mr.model_name,
    mr.model_version,
    mr.model_type,
    mr.deployed_at,
    mpm.window_start,
    mpm.window_end,
    mpm.total_predictions,
    mpm.accuracy,
    mpm.precision,
    mpm.recall,
    mpm.f1_score,
    mpm.avg_inference_duration_ms
FROM secom.model_registry mr
JOIN secom.model_performance_metrics mpm ON mr.id = mpm.model_id
WHERE mr.is_active = TRUE
ORDER BY mpm.window_start DESC;

-- View: Recent predictions with outcomes
CREATE OR REPLACE VIEW secom.recent_predictions_summary AS
SELECT 
    p.id,
    p.batch_id,
    p.prediction,
    p.actual_target,
    p.is_correct,
    p.confidence_score,
    p.inference_duration_ms,
    mr.model_name,
    mr.model_version,
    p.prediction_timestamp
FROM secom.predictions p
JOIN secom.model_registry mr ON p.model_id = mr.id
ORDER BY p.prediction_timestamp DESC
LIMIT 1000;

-- View: Drift detection summary
CREATE OR REPLACE VIEW secom.drift_detection_summary AS
SELECT 
    drift_type,
    feature_name,
    COUNT(*) as total_checks,
    SUM(CASE WHEN is_drift_detected THEN 1 ELSE 0 END) as drift_detected_count,
    AVG(drift_score) as avg_drift_score,
    MAX(drift_score) as max_drift_score,
    MAX(created_at) as last_checked
FROM secom.data_drift_metrics
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY drift_type, feature_name
ORDER BY drift_detected_count DESC, avg_drift_score DESC;

-- View: Model comparison
CREATE OR REPLACE VIEW secom.model_comparison AS
SELECT 
    model_name,
    model_version,
    model_type,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1_score,
    test_roc_auc,
    is_active,
    status,
    deployed_at,
    created_at
FROM secom.model_registry
ORDER BY created_at DESC;

-- ==========================================
-- ML OPERATIONS FUNCTIONS
-- ==========================================

-- Function to activate a model (deactivates others)
CREATE OR REPLACE FUNCTION secom.activate_model(p_model_id UUID)
RETURNS VOID AS $$
BEGIN
    -- Deactivate all models
    UPDATE secom.model_registry SET is_active = FALSE;
    
    -- Activate the specified model
    UPDATE secom.model_registry 
    SET is_active = TRUE, 
        status = 'deployed',
        deployed_at = CURRENT_TIMESTAMP
    WHERE id = p_model_id;
    
    -- Log the activation
    INSERT INTO secom.pipeline_audit_log (
        event_type, event_status, component, message, metadata
    ) VALUES (
        'model_activation', 
        'success', 
        'model_registry',
        'Model activated',
        jsonb_build_object('model_id', p_model_id)
    );
END;
$$ LANGUAGE plpgsql;

-- Function to calculate model performance for a time window
CREATE OR REPLACE FUNCTION secom.calculate_model_performance(
    p_model_id UUID,
    p_window_start TIMESTAMP WITH TIME ZONE,
    p_window_end TIMESTAMP WITH TIME ZONE,
    p_window_type VARCHAR DEFAULT 'hourly'
)
RETURNS UUID AS $$
DECLARE
    v_metric_id UUID;
    v_total INTEGER;
    v_correct INTEGER;
    v_tp INTEGER;
    v_tn INTEGER;
    v_fp INTEGER;
    v_fn INTEGER;
BEGIN
    -- Calculate metrics from predictions
    SELECT 
        COUNT(*),
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END),
        SUM(CASE WHEN prediction = 1 AND actual_target = 1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN prediction = -1 AND actual_target = -1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN prediction = 1 AND actual_target = -1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN prediction = -1 AND actual_target = 1 THEN 1 ELSE 0 END)
    INTO v_total, v_correct, v_tp, v_tn, v_fp, v_fn
    FROM secom.predictions
    WHERE model_id = p_model_id
        AND prediction_timestamp >= p_window_start
        AND prediction_timestamp < p_window_end
        AND actual_target IS NOT NULL;
    
    -- Insert performance metrics
    INSERT INTO secom.model_performance_metrics (
        model_id, window_start, window_end, window_type,
        total_predictions, correct_predictions, incorrect_predictions,
        accuracy, precision, recall, f1_score,
        true_positives, true_negatives, false_positives, false_negatives
    )
    SELECT
        p_model_id, p_window_start, p_window_end, p_window_type,
        v_total, v_correct, v_total - v_correct,
        CASE WHEN v_total > 0 THEN v_correct::FLOAT / v_total ELSE NULL END,
        CASE WHEN (v_tp + v_fp) > 0 THEN v_tp::FLOAT / (v_tp + v_fp) ELSE NULL END,
        CASE WHEN (v_tp + v_fn) > 0 THEN v_tp::FLOAT / (v_tp + v_fn) ELSE NULL END,
        CASE WHEN (2*v_tp + v_fp + v_fn) > 0 THEN 2.0*v_tp / (2*v_tp + v_fp + v_fn) ELSE NULL END,
        v_tp, v_tn, v_fp, v_fn
    RETURNING id INTO v_metric_id;
    
    RETURN v_metric_id;
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
    RAISE NOTICE 'Tables created: 13 (including ML operations tables)';
    RAISE NOTICE 'Views created: 7 (including ML monitoring views)';
    RAISE NOTICE 'Functions created: 4 (including ML operations functions)';
    RAISE NOTICE '========================================';
END $$;
