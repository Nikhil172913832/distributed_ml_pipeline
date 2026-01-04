"""
Kafka Consumer with Preprocessing Pipeline
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional
import signal

import pandas as pd
import numpy as np
import joblib
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from dotenv import load_dotenv

from database import (
    DatabaseManager, RawDataRepository, PreprocessedDataRepository,
    BatchMetadataRepository, DeadLetterQueueRepository, AuditLogRepository
)

load_dotenv()

CONFIG = {
    'kafka': {
        'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
        'topic': os.getenv('KAFKA_RAW_TOPIC', 'secom-raw-data'),
        'dlq_topic': os.getenv('KAFKA_DLQ_TOPIC', 'secom-dead-letter-queue'),
        'group_id': os.getenv('KAFKA_CONSUMER_GROUP', 'secom-preprocessor-group'),
        'auto_offset_reset': os.getenv('KAFKA_AUTO_OFFSET_RESET', 'earliest'),
        'enable_auto_commit': False,
        'max_poll_records': 500,
    },
    'preprocessing': {
        'pipeline_path': os.getenv('PREPROCESSING_PIPELINE_PATH', './models/preprocessing_pipeline.joblib'),
        'batch_processing': True,
        'batch_size': 100,
    },
    'monitoring': {
        'prometheus_port': int(os.getenv('PROMETHEUS_PORT', 8001)),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    }
}

# ==========================================
# PROMETHEUS METRICS
# ==========================================
metrics = {
    'messages_consumed': Counter(
        'secom_messages_consumed_total',
        'Total messages consumed from Kafka'
    ),
    'batches_processed': Counter(
        'secom_batches_processed_total',
        'Total batches processed'
    ),
    'preprocessing_errors': Counter(
        'secom_preprocessing_errors_total',
        'Total preprocessing errors'
    ),
    'db_insert_errors': Counter(
        'secom_db_insert_errors_total',
        'Total database insert errors'
    ),
    'preprocessing_duration': Histogram(
        'secom_preprocessing_duration_seconds',
        'Time to preprocess a batch'
    ),
    'db_insert_duration': Histogram(
        'secom_db_insert_duration_seconds',
        'Time to insert into database'
    ),
    'active_processing': Gauge(
        'secom_active_processing_batches',
        'Number of batches currently being processed'
    ),
    'dlq_messages': Counter(
        'secom_dlq_messages_total',
        'Total messages sent to dead letter queue'
    )
}

# ==========================================
# LOGGING SETUP
# ==========================================
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=CONFIG['monitoring']['log_level']
)
logger.add(
    "logs/consumer_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    compression="zip",
    level="DEBUG"
)


class PreprocessingPipeline:
    """Handles data preprocessing using trained pipeline"""
    
    def __init__(self, pipeline_path: str):
        self.pipeline_path = pipeline_path
        self.pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the preprocessing pipeline"""
        try:
            logger.info(f"Loading preprocessing pipeline from {self.pipeline_path}")
            
            # For now, we'll use a simple preprocessing approach
            # In production, this would load the actual trained pipeline
            # self.pipeline = joblib.load(self.pipeline_path)
            
            # Placeholder - simple preprocessing logic
            self.pipeline = None
            logger.info("Preprocessing pipeline loaded (using simple imputation)")
            
        except FileNotFoundError:
            logger.warning(f"Pipeline file not found at {self.pipeline_path}")
            logger.warning("Using default preprocessing (median imputation + scaling)")
            self.pipeline = None
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise
    
    def preprocess_batch(self, raw_samples: List[Dict]) -> List[Dict]:
        """
        Preprocess a batch of raw samples
        
        Args:
            raw_samples: List of raw sample dictionaries
            
        Returns:
            List of preprocessed sample dictionaries
        """
        if not raw_samples:
            return []
        
        try:
            start_time = time.time()
            
            # Extract features and metadata
            batch_data = []
            for sample in raw_samples:
                features = sample['features']
                batch_data.append({
                    'raw_data_id': sample.get('id'),
                    'features': features,
                    'target': sample['target'],
                    'batch_id': sample['batch_id'],
                    'sample_index': sample['sample_index']
                })
            
            # Convert features to DataFrame
            feature_dicts = [s['features'] for s in batch_data]
            df = pd.DataFrame(feature_dicts)
            
            # Count missing values before preprocessing
            missing_counts = df.isnull().sum(axis=1).tolist()
            
            # Simple preprocessing: median imputation + normalization
            # In production, this would use the loaded pipeline
            df_processed = df.copy()
            
            # Median imputation for missing values
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    median_val = df_processed[col].median()
                    if pd.notna(median_val):
                        df_processed[col].fillna(median_val, inplace=True)
                    else:
                        df_processed[col].fillna(0, inplace=True)
            
            # Simple standardization (mean=0, std=1)
            df_processed = (df_processed - df_processed.mean()) / (df_processed.std() + 1e-8)
            
            # Replace any remaining NaN with 0
            df_processed.fillna(0, inplace=True)
            
            # Build preprocessed samples
            preprocessed_samples = []
            processing_duration = (time.time() - start_time) * 1000  # ms
            
            for idx, (_, row) in enumerate(df_processed.iterrows()):
                preprocessed_features = row.to_dict()
                
                preprocessed_sample = {
                    'raw_data_id': batch_data[idx].get('raw_data_id'),
                    'features': preprocessed_features,
                    'target': batch_data[idx]['target'],
                    'missing_count': int(missing_counts[idx]),
                    'imputation_applied': missing_counts[idx] > 0,
                    'feature_count': len(preprocessed_features),
                    'processing_duration_ms': processing_duration / len(raw_samples)
                }
                preprocessed_samples.append(preprocessed_sample)
            
            logger.debug(
                f"Preprocessed {len(preprocessed_samples)} samples in {processing_duration:.2f}ms"
            )
            
            return preprocessed_samples
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise


class ConsumerOrchestrator:
    """Orchestrates the consumption and preprocessing pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = True
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.raw_repo = RawDataRepository(self.db_manager)
        self.preprocessed_repo = PreprocessedDataRepository(self.db_manager)
        self.batch_meta_repo = BatchMetadataRepository(self.db_manager)
        self.dlq_repo = DeadLetterQueueRepository(self.db_manager)
        self.audit_repo = AuditLogRepository(self.db_manager)
        
        self.preprocessor = PreprocessingPipeline(config['preprocessing']['pipeline_path'])
        self.consumer = self._create_consumer()
        
        # Batch processing state
        self.current_batch = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _create_consumer(self) -> KafkaConsumer:
        """Create Kafka consumer"""
        try:
            logger.info("Creating Kafka consumer...")
            logger.info(f"  Bootstrap servers: {self.config['kafka']['bootstrap_servers']}")
            logger.info(f"  Topic: {self.config['kafka']['topic']}")
            logger.info(f"  Group ID: {self.config['kafka']['group_id']}")
            
            consumer = KafkaConsumer(
                self.config['kafka']['topic'],
                bootstrap_servers=self.config['kafka']['bootstrap_servers'],
                group_id=self.config['kafka']['group_id'],
                auto_offset_reset=self.config['kafka']['auto_offset_reset'],
                enable_auto_commit=self.config['kafka']['enable_auto_commit'],
                max_poll_records=self.config['kafka']['max_poll_records'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None
            )
            
            logger.info("Kafka consumer created successfully")
            return consumer
            
        except Exception as e:
            logger.error(f"Failed to create consumer: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def _process_batch(self, batch_id: str, samples: List[Dict]):
        """Process a complete batch"""
        try:
            metrics['active_processing'].inc()
            batch_start_time = time.time()
            
            logger.info(f"Processing batch {batch_id} ({len(samples)} samples)")
            
            # Calculate batch statistics
            pass_count = sum(1 for s in samples if s['target'] == -1)
            fail_count = sum(1 for s in samples if s['target'] == 1)
            
            # Store raw data
            with metrics['db_insert_duration'].time():
                self.raw_repo.insert_batch(samples)
            
            # Create batch metadata
            self.batch_meta_repo.create_batch(
                batch_id=batch_id,
                total_samples=len(samples),
                pass_samples=pass_count,
                fail_samples=fail_count
            )
            
            # Retrieve stored raw data with IDs
            raw_data = self.raw_repo.get_batch_samples(batch_id)
            
            # Add IDs to samples
            for sample, (raw_id, _, _, _, _) in zip(samples, raw_data):
                sample['id'] = raw_id
            
            # Preprocess
            with metrics['preprocessing_duration'].time():
                preprocessed_samples = self.preprocessor.preprocess_batch(samples)
            
            # Store preprocessed data
            with metrics['db_insert_duration'].time():
                self.preprocessed_repo.insert_batch(preprocessed_samples)
            
            # Update batch status
            processing_duration_ms = (time.time() - batch_start_time) * 1000
            self.batch_meta_repo.update_batch_status(
                batch_id=batch_id,
                status='completed',
                processing_duration_ms=processing_duration_ms
            )
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='batch_processing',
                event_status='success',
                batch_id=batch_id,
                component='consumer',
                message=f"Batch processed successfully: {len(samples)} samples",
                metadata={
                    'pass_count': pass_count,
                    'fail_count': fail_count,
                    'preprocessing_duration_ms': processing_duration_ms
                },
                duration_ms=processing_duration_ms
            )
            
            metrics['batches_processed'].inc()
            logger.info(
                f"Batch {batch_id} completed in {processing_duration_ms:.2f}ms "
                f"(Pass: {pass_count}, Fail: {fail_count})"
            )
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            metrics['preprocessing_errors'].inc()
            
            # Update batch status as failed
            self.batch_meta_repo.update_batch_status(
                batch_id=batch_id,
                status='failed',
                error_details=str(e)
            )
            
            # Log error
            self.audit_repo.log_event(
                event_type='batch_processing',
                event_status='failure',
                batch_id=batch_id,
                component='consumer',
                message=f"Batch processing failed: {str(e)}",
                metadata={'error': str(e), 'traceback': traceback.format_exc()}
            )
            
        finally:
            metrics['active_processing'].dec()
    
    def run(self):
        """Main execution loop"""
        logger.info("=" * 80)
        logger.info("SECOM Data Consumer Started")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Kafka topic: {self.config['kafka']['topic']}")
        logger.info(f"  Consumer group: {self.config['kafka']['group_id']}")
        logger.info(f"  Batch processing: {self.config['preprocessing']['batch_processing']}")
        logger.info("=" * 80)
        
        try:
            while self.running:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=1000)
                
                if not message_batch:
                    # Process any pending batches
                    for batch_id, samples in list(self.current_batch.items()):
                        if samples:
                            self._process_batch(batch_id, samples)
                            del self.current_batch[batch_id]
                    continue
                
                # Process messages
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            sample = message.value
                            batch_id = sample['batch_id']
                            
                            # Add to current batch
                            if batch_id not in self.current_batch:
                                self.current_batch[batch_id] = []
                            
                            self.current_batch[batch_id].append(sample)
                            metrics['messages_consumed'].inc()
                            
                            # Check if batch is complete (all samples received)
                            # For now, we'll process when we reach batch_size
                            if len(self.current_batch[batch_id]) >= self.config['preprocessing']['batch_size']:
                                self._process_batch(batch_id, self.current_batch[batch_id])
                                del self.current_batch[batch_id]
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            metrics['preprocessing_errors'].inc()
                            
                            # Send to DLQ
                            try:
                                self.dlq_repo.insert_failed_message(
                                    message_key=str(message.key),
                                    message_value=str(message.value),
                                    kafka_topic=message.topic,
                                    kafka_partition=message.partition,
                                    kafka_offset=message.offset,
                                    error_type=type(e).__name__,
                                    error_message=str(e),
                                    stack_trace=traceback.format_exc()
                                )
                                metrics['dlq_messages'].inc()
                            except:
                                logger.error("Failed to write to DLQ")
                
                # Commit offsets
                self.consumer.commit()
                
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down consumer...")
        self.running = False
        
        # Process any remaining batches
        for batch_id, samples in self.current_batch.items():
            if samples:
                logger.info(f"Processing remaining batch {batch_id}")
                self._process_batch(batch_id, samples)
        
        # Close connections
        self.consumer.close()
        self.db_manager.close_all()
        
        logger.info("=" * 80)
        logger.info("Consumer shutdown complete")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    try:
        # Start Prometheus metrics server
        logger.info(f"Starting Prometheus metrics server on port {CONFIG['monitoring']['prometheus_port']}")
        start_http_server(CONFIG['monitoring']['prometheus_port'])
        
        # Create and run orchestrator
        orchestrator = ConsumerOrchestrator(CONFIG)
        orchestrator.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
