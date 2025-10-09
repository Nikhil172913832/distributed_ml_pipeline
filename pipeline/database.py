"""
Database Layer for SECOM ML Pipeline

Provides connection management and data access operations for PostgreSQL.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
from uuid import UUID

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import execute_batch, Json
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """Manages PostgreSQL connections and operations"""
    
    def __init__(self):
        self.config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'secom_pipeline'),
            'user': os.getenv('POSTGRES_USER', 'ml_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'ml_password'),
        }
        
        self.pool = None
        self._create_connection_pool()
    
    def _create_connection_pool(self):
        """Create thread-safe connection pool"""
        try:
            logger.info("Creating database connection pool...")
            self.pool = ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                **self.config
            )
            logger.info("✓ Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self.pool.putconn(conn)
    
    def close_all(self):
        """Close all connections in pool"""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")


class RawDataRepository:
    """Repository for raw SECOM data operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def insert_batch(
        self, 
        samples: List[Dict],
        kafka_metadata: Optional[Dict] = None
    ) -> int:
        """
        Insert a batch of raw samples
        
        Args:
            samples: List of sample dictionaries with features and target
            kafka_metadata: Optional Kafka metadata (topic, partition, offset)
            
        Returns:
            Number of rows inserted
        """
        if not samples:
            return 0
        
        insert_query = """
        INSERT INTO secom.raw_data (
            batch_id, sample_index, features, target,
            kafka_offset, kafka_partition, kafka_topic
        ) VALUES (
            %(batch_id)s, %(sample_index)s, %(features)s, %(target)s,
            %(kafka_offset)s, %(kafka_partition)s, %(kafka_topic)s
        )
        ON CONFLICT (batch_id, sample_index) DO NOTHING
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare data for batch insert
                    insert_data = []
                    for sample in samples:
                        row_data = {
                            'batch_id': sample['batch_id'],
                            'sample_index': sample['sample_index'],
                            'features': Json(sample['features']),
                            'target': sample['target'],
                            'kafka_offset': kafka_metadata.get('offset') if kafka_metadata else None,
                            'kafka_partition': kafka_metadata.get('partition') if kafka_metadata else None,
                            'kafka_topic': kafka_metadata.get('topic') if kafka_metadata else None,
                        }
                        insert_data.append(row_data)
                    
                    # Batch insert
                    execute_batch(cur, insert_query, insert_data, page_size=100)
                    inserted_count = len(insert_data)
                    
                    logger.debug(f"Inserted {inserted_count} raw samples")
                    return inserted_count
                    
        except Exception as e:
            logger.error(f"Error inserting raw data: {e}")
            raise
    
    def get_batch_samples(self, batch_id: str) -> List[Tuple]:
        """Get all samples for a batch"""
        query = """
        SELECT id, batch_id, sample_index, features, target
        FROM secom.raw_data
        WHERE batch_id = %s
        ORDER BY sample_index
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (batch_id,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching batch {batch_id}: {e}")
            raise


class PreprocessedDataRepository:
    """Repository for preprocessed data operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def insert_batch(
        self,
        preprocessed_samples: List[Dict],
        preprocessing_version: str = "1.0"
    ) -> int:
        """
        Insert batch of preprocessed samples
        
        Args:
            preprocessed_samples: List of preprocessed sample dictionaries
            preprocessing_version: Version of preprocessing pipeline
            
        Returns:
            Number of rows inserted
        """
        if not preprocessed_samples:
            return 0
        
        insert_query = """
        INSERT INTO secom.preprocessed_data (
            raw_data_id, features, target, missing_count,
            imputation_applied, feature_count, preprocessing_version,
            processing_duration_ms
        ) VALUES (
            %(raw_data_id)s, %(features)s, %(target)s, %(missing_count)s,
            %(imputation_applied)s, %(feature_count)s, %(preprocessing_version)s,
            %(processing_duration_ms)s
        )
        ON CONFLICT (raw_data_id) DO UPDATE SET
            features = EXCLUDED.features,
            processing_duration_ms = EXCLUDED.processing_duration_ms,
            preprocessing_version = EXCLUDED.preprocessing_version
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare data
                    insert_data = []
                    for sample in preprocessed_samples:
                        row_data = {
                            'raw_data_id': sample['raw_data_id'],
                            'features': Json(sample['features']),
                            'target': sample['target'],
                            'missing_count': sample.get('missing_count', 0),
                            'imputation_applied': sample.get('imputation_applied', False),
                            'feature_count': sample.get('feature_count', 0),
                            'preprocessing_version': preprocessing_version,
                            'processing_duration_ms': sample.get('processing_duration_ms', 0.0),
                        }
                        insert_data.append(row_data)
                    
                    # Batch insert
                    execute_batch(cur, insert_query, insert_data, page_size=100)
                    inserted_count = len(insert_data)
                    
                    logger.debug(f"Inserted {inserted_count} preprocessed samples")
                    return inserted_count
                    
        except Exception as e:
            logger.error(f"Error inserting preprocessed data: {e}")
            raise


class BatchMetadataRepository:
    """Repository for batch metadata operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_batch(
        self,
        batch_id: str,
        total_samples: int,
        pass_samples: int,
        fail_samples: int
    ) -> UUID:
        """Create batch metadata record"""
        insert_query = """
        INSERT INTO secom.batch_metadata (
            batch_id, total_samples, pass_samples, fail_samples,
            processing_status, raw_ingestion_time
        ) VALUES (
            %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (batch_id) DO UPDATE SET
            total_samples = EXCLUDED.total_samples,
            pass_samples = EXCLUDED.pass_samples,
            fail_samples = EXCLUDED.fail_samples,
            raw_ingestion_time = EXCLUDED.raw_ingestion_time
        RETURNING id
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_query,
                        (batch_id, total_samples, pass_samples, fail_samples, 
                         'ingested', datetime.utcnow())
                    )
                    batch_uuid = cur.fetchone()[0]
                    logger.debug(f"Created batch metadata for {batch_id}")
                    return batch_uuid
        except Exception as e:
            logger.error(f"Error creating batch metadata: {e}")
            raise
    
    def update_batch_status(
        self,
        batch_id: str,
        status: str,
        error_details: Optional[str] = None,
        processing_duration_ms: Optional[float] = None
    ):
        """Update batch processing status"""
        update_query = """
        UPDATE secom.batch_metadata
        SET processing_status = %s,
            preprocessing_end_time = %s,
            total_processing_duration_ms = COALESCE(%s, total_processing_duration_ms),
            error_details = COALESCE(%s, error_details),
            error_count = error_count + CASE WHEN %s IS NOT NULL THEN 1 ELSE 0 END
        WHERE batch_id = %s
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        update_query,
                        (status, datetime.utcnow(), processing_duration_ms, 
                         error_details, error_details, batch_id)
                    )
                    logger.debug(f"Updated batch {batch_id} status to {status}")
        except Exception as e:
            logger.error(f"Error updating batch status: {e}")
            raise


class DeadLetterQueueRepository:
    """Repository for dead letter queue operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def insert_failed_message(
        self,
        message_key: str,
        message_value: str,
        kafka_topic: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        kafka_partition: Optional[int] = None,
        kafka_offset: Optional[int] = None
    ):
        """Insert failed message into DLQ"""
        insert_query = """
        INSERT INTO secom.dead_letter_queue (
            message_key, message_value, kafka_topic, kafka_partition, kafka_offset,
            error_type, error_message, stack_trace, status
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_query,
                        (message_key, message_value, kafka_topic, kafka_partition, kafka_offset,
                         error_type, error_message, stack_trace, 'failed')
                    )
                    logger.debug(f"Inserted failed message to DLQ: {message_key}")
        except Exception as e:
            logger.error(f"Error inserting to DLQ: {e}")
            raise


class AuditLogRepository:
    """Repository for pipeline audit log operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def log_event(
        self,
        event_type: str,
        event_status: str,
        batch_id: Optional[str] = None,
        component: Optional[str] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict] = None,
        duration_ms: Optional[float] = None
    ):
        """Log pipeline event"""
        insert_query = """
        INSERT INTO secom.pipeline_audit_log (
            event_type, event_status, batch_id, component,
            message, metadata, duration_ms
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s
        )
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_query,
                        (event_type, event_status, batch_id, component,
                         message, Json(metadata) if metadata else None, duration_ms)
                    )
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
            # Don't raise - audit logging failures shouldn't break the pipeline


class ModelRegistryRepository:
    """Repository for model registry operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        model_path: str,
        preprocessing_pipeline_path: Optional[str] = None,
        hyperparameters: Optional[Dict] = None,
        test_metrics: Optional[Dict] = None,
        training_metadata: Optional[Dict] = None,
        triggered_by: str = 'manual'
    ) -> UUID:
        """Register a new model"""
        insert_query = """
        INSERT INTO secom.model_registry (
            model_name, model_version, model_type, model_path,
            preprocessing_pipeline_path, hyperparameters,
            test_accuracy, test_precision, test_recall, test_f1_score, test_roc_auc,
            training_dataset_size, training_start_time, training_end_time, 
            training_duration_ms, training_triggered_by, status
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING id
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_query,
                        (
                            model_name, model_version, model_type, model_path,
                            preprocessing_pipeline_path, 
                            Json(hyperparameters) if hyperparameters else None,
                            test_metrics.get('accuracy') if test_metrics else None,
                            test_metrics.get('precision') if test_metrics else None,
                            test_metrics.get('recall') if test_metrics else None,
                            test_metrics.get('f1_score') if test_metrics else None,
                            test_metrics.get('roc_auc') if test_metrics else None,
                            training_metadata.get('dataset_size') if training_metadata else None,
                            training_metadata.get('start_time') if training_metadata else None,
                            training_metadata.get('end_time') if training_metadata else None,
                            training_metadata.get('duration_ms') if training_metadata else None,
                            triggered_by, 'trained'
                        )
                    )
                    model_id = cur.fetchone()[0]
                    logger.info(f"Registered model {model_name} v{model_version} with ID {model_id}")
                    return model_id
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def activate_model(self, model_id: UUID):
        """Activate a model (deactivates all others)"""
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT secom.activate_model(%s)", (str(model_id),))
                    logger.info(f"Activated model {model_id}")
        except Exception as e:
            logger.error(f"Error activating model: {e}")
            raise
    
    def get_active_model(self) -> Optional[Tuple]:
        """Get the currently active model"""
        query = """
        SELECT id, model_name, model_version, model_type, model_path, 
               preprocessing_pipeline_path, deployed_at
        FROM secom.model_registry
        WHERE is_active = TRUE
        LIMIT 1
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            raise
    
    def get_model_by_id(self, model_id: UUID) -> Optional[Tuple]:
        """Get model by ID"""
        query = """
        SELECT id, model_name, model_version, model_type, model_path,
               preprocessing_pipeline_path, is_active, status
        FROM secom.model_registry
        WHERE id = %s
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (str(model_id),))
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Error getting model: {e}")
            raise


class PredictionRepository:
    """Repository for prediction operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def insert_predictions(
        self,
        predictions: List[Dict]
    ) -> int:
        """Insert batch of predictions"""
        if not predictions:
            return 0
        
        insert_query = """
        INSERT INTO secom.predictions (
            preprocessed_data_id, model_id, prediction, prediction_probability,
            prediction_proba_pass, prediction_proba_fail, actual_target,
            is_correct, confidence_score, uncertainty_score,
            inference_duration_ms, batch_id
        ) VALUES (
            %(preprocessed_data_id)s, %(model_id)s, %(prediction)s, 
            %(prediction_probability)s, %(prediction_proba_pass)s, 
            %(prediction_proba_fail)s, %(actual_target)s, %(is_correct)s,
            %(confidence_score)s, %(uncertainty_score)s, %(inference_duration_ms)s,
            %(batch_id)s
        )
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    execute_batch(cur, insert_query, predictions, page_size=100)
                    inserted_count = len(predictions)
                    logger.debug(f"Inserted {inserted_count} predictions")
                    return inserted_count
        except Exception as e:
            logger.error(f"Error inserting predictions: {e}")
            raise
    
    def get_batch_predictions(self, batch_id: str) -> List[Tuple]:
        """Get all predictions for a batch"""
        query = """
        SELECT id, preprocessed_data_id, prediction, actual_target, 
               is_correct, confidence_score, prediction_timestamp
        FROM secom.predictions
        WHERE batch_id = %s
        ORDER BY prediction_timestamp
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (batch_id,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error fetching predictions: {e}")
            raise


class ModelPerformanceRepository:
    """Repository for model performance metrics"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def calculate_performance_window(
        self,
        model_id: UUID,
        window_start: datetime,
        window_end: datetime,
        window_type: str = 'hourly'
    ) -> Optional[UUID]:
        """Calculate and store performance metrics for a time window"""
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT secom.calculate_model_performance(%s, %s, %s, %s)",
                        (str(model_id), window_start, window_end, window_type)
                    )
                    result = cur.fetchone()
                    if result and result[0]:
                        logger.debug(f"Calculated performance for model {model_id}")
                        return UUID(result[0])
                    return None
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            raise
    
    def get_latest_performance(
        self,
        model_id: UUID,
        window_type: str = 'hourly'
    ) -> Optional[Tuple]:
        """Get latest performance metrics for a model"""
        query = """
        SELECT accuracy, precision, recall, f1_score, total_predictions,
               window_start, window_end
        FROM secom.model_performance_metrics
        WHERE model_id = %s AND window_type = %s
        ORDER BY window_end DESC
        LIMIT 1
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (str(model_id), window_type))
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            raise
    
    def get_performance_trend(
        self,
        model_id: UUID,
        hours: int = 24,
        window_type: str = 'hourly'
    ) -> List[Tuple]:
        """Get performance trend over time"""
        query = """
        SELECT window_start, window_end, accuracy, precision, recall, 
               f1_score, total_predictions
        FROM secom.model_performance_metrics
        WHERE model_id = %s 
            AND window_type = %s
            AND window_start >= NOW() - INTERVAL '%s hours'
        ORDER BY window_start
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (str(model_id), window_type, hours))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error getting performance trend: {e}")
            raise


class DataDriftRepository:
    """Repository for data drift detection"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def record_drift_metric(
        self,
        window_start: datetime,
        window_end: datetime,
        drift_type: str,
        drift_score: float,
        is_drift_detected: bool,
        feature_name: Optional[str] = None,
        statistical_metrics: Optional[Dict] = None,
        baseline_info: Optional[Dict] = None
    ) -> UUID:
        """Record a drift detection metric"""
        insert_query = """
        INSERT INTO secom.data_drift_metrics (
            window_start, window_end, drift_type, feature_name,
            ks_statistic, ks_pvalue, chi2_statistic, chi2_pvalue,
            mean_baseline, mean_current, std_baseline, std_current,
            is_drift_detected, drift_score,
            baseline_batch_id, baseline_start_date, baseline_end_date
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING id
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_query,
                        (
                            window_start, window_end, drift_type, feature_name,
                            statistical_metrics.get('ks_statistic') if statistical_metrics else None,
                            statistical_metrics.get('ks_pvalue') if statistical_metrics else None,
                            statistical_metrics.get('chi2_statistic') if statistical_metrics else None,
                            statistical_metrics.get('chi2_pvalue') if statistical_metrics else None,
                            statistical_metrics.get('mean_baseline') if statistical_metrics else None,
                            statistical_metrics.get('mean_current') if statistical_metrics else None,
                            statistical_metrics.get('std_baseline') if statistical_metrics else None,
                            statistical_metrics.get('std_current') if statistical_metrics else None,
                            is_drift_detected, drift_score,
                            baseline_info.get('batch_id') if baseline_info else None,
                            baseline_info.get('start_date') if baseline_info else None,
                            baseline_info.get('end_date') if baseline_info else None
                        )
                    )
                    drift_id = cur.fetchone()[0]
                    logger.debug(f"Recorded drift metric: {drift_type} = {drift_score}")
                    return drift_id
        except Exception as e:
            logger.error(f"Error recording drift metric: {e}")
            raise
    
    def get_recent_drift_events(self, hours: int = 24) -> List[Tuple]:
        """Get recent drift detection events"""
        query = """
        SELECT id, drift_type, feature_name, drift_score, 
               is_drift_detected, window_start, window_end
        FROM secom.data_drift_metrics
        WHERE created_at >= NOW() - INTERVAL '%s hours'
            AND is_drift_detected = TRUE
        ORDER BY created_at DESC
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (hours,))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error getting drift events: {e}")
            raise


class RetrainingTriggerRepository:
    """Repository for retraining trigger operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def create_trigger(
        self,
        trigger_type: str,
        trigger_reason: str,
        model_id: Optional[UUID] = None,
        performance_data: Optional[Dict] = None,
        drift_data: Optional[Dict] = None
    ) -> UUID:
        """Create a retraining trigger"""
        insert_query = """
        INSERT INTO secom.retraining_triggers (
            trigger_type, trigger_reason, model_id,
            performance_threshold_violated, current_performance_value,
            drift_threshold_violated, current_drift_value,
            status
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING id
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_query,
                        (
                            trigger_type, trigger_reason, 
                            str(model_id) if model_id else None,
                            performance_data.get('threshold') if performance_data else None,
                            performance_data.get('current_value') if performance_data else None,
                            drift_data.get('threshold') if drift_data else None,
                            drift_data.get('current_value') if drift_data else None,
                            'pending'
                        )
                    )
                    trigger_id = cur.fetchone()[0]
                    logger.info(f"Created retraining trigger: {trigger_type}")
                    return trigger_id
        except Exception as e:
            logger.error(f"Error creating retraining trigger: {e}")
            raise
    
    def update_trigger_status(
        self,
        trigger_id: UUID,
        status: str,
        new_model_id: Optional[UUID] = None,
        error_message: Optional[str] = None
    ):
        """Update retraining trigger status"""
        update_query = """
        UPDATE secom.retraining_triggers
        SET status = %s,
            new_model_id = %s,
            retraining_completed_at = CASE WHEN %s = 'completed' THEN NOW() ELSE NULL END,
            retraining_successful = CASE WHEN %s = 'completed' THEN TRUE ELSE FALSE END,
            error_message = %s
        WHERE id = %s
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        update_query,
                        (status, str(new_model_id) if new_model_id else None, 
                         status, status, error_message, str(trigger_id))
                    )
                    logger.debug(f"Updated trigger {trigger_id} to status {status}")
        except Exception as e:
            logger.error(f"Error updating trigger: {e}")
            raise
    
    def get_pending_triggers(self) -> List[Tuple]:
        """Get all pending retraining triggers"""
        query = """
        SELECT id, trigger_type, trigger_reason, model_id, created_at
        FROM secom.retraining_triggers
        WHERE status = 'pending'
        ORDER BY created_at
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Error getting pending triggers: {e}")
            raise
