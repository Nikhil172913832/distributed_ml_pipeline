"""
Kafka Producer for SECOM Synthetic Data Generation
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import signal

import pandas as pd
import joblib
from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CONFIG = {
    'kafka': {
        'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
        'topic': os.getenv('KAFKA_RAW_TOPIC', 'secom-raw-data'),
        'compression_type': 'gzip',
        'acks': 'all',
        'retries': 3,
        'max_in_flight_requests_per_connection': 1,
    },
    'generation': {
        'batch_size': int(os.getenv('BATCH_SIZE', 100)),
        'interval_seconds': int(os.getenv('GENERATION_INTERVAL_SECONDS', 5)),
        'max_batches': int(os.getenv('MAX_BATCHES', 0)),  # 0 = infinite
    },
    'model': {
        'sdv_model_path': Path(os.getenv('SDV_MODEL_PATH', './models/sdv_secom_raw.joblib')),
    },
    'monitoring': {
        'prometheus_port': int(os.getenv('PROMETHEUS_PORT', 8000)),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    }
}

# ==========================================
# PROMETHEUS METRICS
# ==========================================
metrics = {
    'batches_generated': Counter(
        'secom_batches_generated_total',
        'Total number of batches generated'
    ),
    'samples_generated': Counter(
        'secom_samples_generated_total', 
        'Total number of samples generated'
    ),
    'kafka_messages_sent': Counter(
        'secom_kafka_messages_sent_total',
        'Total messages sent to Kafka'
    ),
    'kafka_send_errors': Counter(
        'secom_kafka_send_errors_total',
        'Total Kafka send errors'
    ),
    'generation_duration': Histogram(
        'secom_generation_duration_seconds',
        'Time to generate a batch'
    ),
    'kafka_send_duration': Histogram(
        'secom_kafka_send_duration_seconds',
        'Time to send message to Kafka'
    ),
    'active_batches': Gauge(
        'secom_active_batches',
        'Number of batches currently being processed'
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
    "logs/producer_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    compression="zip",
    level="DEBUG"
)


class SDVDataGenerator:
    """Generates synthetic SECOM data using trained SDV model"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self._load_model()
        
    def _load_model(self):
        """Load the trained SDV model and metadata"""
        try:
            logger.info(f"Loading SDV model from {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"SDV model not found at {self.model_path}")
            
            artifacts = joblib.load(self.model_path)
            self.model = artifacts['model']
            self.metadata = artifacts.get('metadata')
            
            logger.info("SDV model loaded successfully")
            logger.info(f"  Model type: {artifacts.get('model_type', 'Unknown')}")
            logger.info(f"  Training date: {artifacts.get('training_date', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load SDV model: {e}")
            raise
    
    def generate_batch(self, batch_size: int, batch_id: str) -> List[Dict]:
        """
        Generate a batch of synthetic samples
        
        Args:
            batch_size: Number of samples to generate
            batch_id: Unique identifier for this batch
            
        Returns:
            List of sample dictionaries
        """
        try:
            with metrics['generation_duration'].time():
                logger.debug(f"Generating batch {batch_id} with {batch_size} samples")
                
                # Generate synthetic data
                synthetic_data = self.model.sample(num_rows=batch_size)
                
                # Convert to list of records
                samples = []
                for idx, row in synthetic_data.iterrows():
                    # Separate features and target
                    features = row.drop('target').to_dict()
                    target = int(row['target'])
                    
                    sample = {
                        'batch_id': batch_id,
                        'sample_index': idx,
                        'features': features,
                        'target': target,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    samples.append(sample)
                
                metrics['samples_generated'].inc(len(samples))
                logger.debug(f"Generated {len(samples)} samples for batch {batch_id}")
                
                return samples
                
        except Exception as e:
            logger.error(f"Error generating batch {batch_id}: {e}")
            raise


class KafkaDataProducer:
    """Publishes synthetic data to Kafka"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.producer = None
        self._connect()
        
    def _connect(self):
        """Establish Kafka connection"""
        try:
            logger.info("Connecting to Kafka...")
            logger.info(f"  Bootstrap servers: {self.config['kafka']['bootstrap_servers']}")
            logger.info(f"  Topic: {self.config['kafka']['topic']}")
            
            self.producer = KafkaProducer(
                bootstrap_servers=self.config['kafka']['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                compression_type=self.config['kafka']['compression_type'],
                acks=self.config['kafka']['acks'],
                retries=self.config['kafka']['retries'],
                max_in_flight_requests_per_connection=self.config['kafka']['max_in_flight_requests_per_connection'],
            )
            
            logger.info("Connected to Kafka successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def send_batch(self, samples: List[Dict], batch_id: str):
        """
        Send batch of samples to Kafka
        
        Args:
            samples: List of sample dictionaries
            batch_id: Unique batch identifier
        """
        try:
            logger.info(f"Sending batch {batch_id} ({len(samples)} samples) to Kafka")
            
            futures = []
            for sample in samples:
                with metrics['kafka_send_duration'].time():
                    # Use batch_id as message key for partitioning
                    future = self.producer.send(
                        topic=self.config['kafka']['topic'],
                        key=batch_id,
                        value=sample
                    )
                    futures.append(future)
            
            # Wait for all messages to be sent
            for future in futures:
                try:
                    record_metadata = future.get(timeout=10)
                    metrics['kafka_messages_sent'].inc()
                    
                except KafkaError as e:
                    metrics['kafka_send_errors'].inc()
                    logger.error(f"Kafka send error: {e}")
                    raise
            
            logger.info(f"Batch {batch_id} sent successfully to topic '{self.config['kafka']['topic']}'")
            
        except Exception as e:
            logger.error(f"Error sending batch {batch_id}: {e}")
            raise
    
    def close(self):
        """Close Kafka producer connection"""
        if self.producer:
            logger.info("Closing Kafka producer...")
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")


class ProducerOrchestrator:
    """Orchestrates the data generation and publishing pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = True
        self.generator = SDVDataGenerator(config['model']['sdv_model_path'])
        self.producer = KafkaDataProducer(config)
        self.batches_produced = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def run(self):
        """Main execution loop"""
        logger.info("=" * 80)
        logger.info("SECOM Data Producer Started")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Batch size: {self.config['generation']['batch_size']}")
        logger.info(f"  Generation interval: {self.config['generation']['interval_seconds']}s")
        logger.info(f"  Max batches: {self.config['generation']['max_batches'] or 'Unlimited'}")
        logger.info(f"  Kafka topic: {self.config['kafka']['topic']}")
        logger.info("=" * 80)
        
        try:
            while self.running:
                # Check if max batches reached
                max_batches = self.config['generation']['max_batches']
                if max_batches > 0 and self.batches_produced >= max_batches:
                    logger.info(f"Maximum batches ({max_batches}) reached, stopping...")
                    break
                
                # Generate batch ID
                batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                try:
                    metrics['active_batches'].inc()
                    
                    # Generate synthetic data
                    samples = self.generator.generate_batch(
                        batch_size=self.config['generation']['batch_size'],
                        batch_id=batch_id
                    )
                    
                    # Send to Kafka
                    self.producer.send_batch(samples, batch_id)
                    
                    # Update metrics
                    metrics['batches_generated'].inc()
                    self.batches_produced += 1
                    
                    # Calculate stats
                    pass_count = sum(1 for s in samples if s['target'] == -1)
                    fail_count = sum(1 for s in samples if s['target'] == 1)
                    
                    logger.info(
                        f"Batch {self.batches_produced}: {len(samples)} samples "
                        f"(Pass: {pass_count}, Fail: {fail_count})"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_id}: {e}")
                    
                finally:
                    metrics['active_batches'].dec()
                
                # Wait before next batch
                if self.running:
                    time.sleep(self.config['generation']['interval_seconds'])
                    
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down producer...")
        self.running = False
        self.producer.close()
        logger.info("=" * 80)
        logger.info(f"Producer shutdown complete. Total batches produced: {self.batches_produced}")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    try:
        # Start Prometheus metrics server
        logger.info(f"Starting Prometheus metrics server on port {CONFIG['monitoring']['prometheus_port']}")
        start_http_server(CONFIG['monitoring']['prometheus_port'])
        
        # Create and run orchestrator
        orchestrator = ProducerOrchestrator(CONFIG)
        orchestrator.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
