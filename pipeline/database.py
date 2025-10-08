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
