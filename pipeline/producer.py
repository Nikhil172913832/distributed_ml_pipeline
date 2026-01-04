"""
Kafka Producer for SECOM Real Data Streaming
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import signal

from kafka import KafkaProducer
from kafka.errors import KafkaError
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from data_generator.real_data_loader import SECOMDataLoader, DriftSimulator

load_dotenv()

CONFIG = {
    'kafka': {
        'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
        'topic': os.getenv('KAFKA_RAW_TOPIC', 'secom-raw-data'),
        'compression_type': 'gzip',
        'acks': 'all',
        'retries': 3,
    },
    'data': {
        'batch_size': int(os.getenv('BATCH_SIZE', 100)),
        'interval_seconds': int(os.getenv('GENERATION_INTERVAL_SECONDS', 5)),
        'data_path': os.getenv('SECOM_DATA_PATH', '/tmp/secom.data'),
        'labels_path': os.getenv('SECOM_LABELS_PATH', '/tmp/secom_labels.data'),
        'enable_drift': os.getenv('ENABLE_DRIFT', 'true').lower() == 'true',
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



class ProducerOrchestrator:
    """Orchestrates real SECOM data streaming with optional drift simulation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = True
        self.data_loader = SECOMDataLoader(
            config['data']['data_path'],
            config['data']['labels_path']
        )
        self.drift_simulator = DriftSimulator() if config['data']['enable_drift'] else None
        self.kafka_producer = KafkaDataProducer(config)
        self.batches_produced = 0
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.warning(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def run(self):
        logger.info("=" * 80)
        logger.info("SECOM Real Data Producer Started")
        logger.info("=" * 80)
        logger.info(f"Batch size: {self.config['data']['batch_size']}")
        logger.info(f"Interval: {self.config['data']['interval_seconds']}s")
        logger.info(f"Drift simulation: {self.config['data']['enable_drift']}")
        logger.info("=" * 80)
        
        try:
            while self.running:
                batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                try:
                    metrics['active_batches'].inc()
                    
                    with metrics['generation_duration'].time():
                        samples = self.data_loader.get_batch(
                            batch_size=self.config['data']['batch_size'],
                            batch_id=batch_id
                        )
                        
                        if self.drift_simulator:
                            total_samples = len(self.data_loader.data)
                            for i, sample in enumerate(samples):
                                sample['features'] = self.drift_simulator.apply_drift(
                                    sample['features'],
                                    self.data_loader.current_index - len(samples) + i,
                                    total_samples
                                )
                    
                    self.kafka_producer.send_batch(samples, batch_id)
                    
                    metrics['batches_generated'].inc()
                    metrics['samples_generated'].inc(len(samples))
                    self.batches_produced += 1
                    
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
                
                if self.running:
                    time.sleep(self.config['data']['interval_seconds'])
                    
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            
        finally:
            self.shutdown()
    
    def shutdown(self):
        logger.info("Shutting down producer...")
        self.running = False
        self.kafka_producer.close()
        logger.info(f"Producer shutdown complete. Total batches: {self.batches_produced}")



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




def main():
    """Main entry point"""
    try:
        logger.info(f"Starting Prometheus metrics server on port {CONFIG['monitoring']['prometheus_port']}")
        start_http_server(CONFIG['monitoring']['prometheus_port'])
        
        orchestrator = ProducerOrchestrator(CONFIG)
        orchestrator.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

