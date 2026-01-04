"""
Continuous Learning Retrainer Service
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from typing import List, Tuple
from uuid import UUID
import signal

from loguru import logger
from prometheus_client import Counter, Gauge, start_http_server
from dotenv import load_dotenv

from database import (
    DatabaseManager, RetrainingTriggerRepository,
    ModelRegistryRepository, AuditLogRepository
)
from model_trainer import TrainingOrchestrator, CONFIG as TRAINING_CONFIG

load_dotenv()

CONFIG = {
    'retrainer': {
        'check_interval_seconds': int(os.getenv('RETRAINER_CHECK_INTERVAL', 300)),  # 5 minutes
        'auto_deploy': os.getenv('RETRAINER_AUTO_DEPLOY', 'true').lower() == 'true',
        'max_concurrent_training': int(os.getenv('MAX_CONCURRENT_TRAINING', 1)),
        'cooldown_period_hours': int(os.getenv('RETRAINING_COOLDOWN_HOURS', 6)),
    },
    'monitoring': {
        'prometheus_port': int(os.getenv('RETRAINER_PROMETHEUS_PORT', 8003)),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
    }
}

# ==========================================
# PROMETHEUS METRICS
# ==========================================
metrics = {
    'retraining_jobs_started': Counter(
        'secom_retraining_jobs_started_total',
        'Total retraining jobs started',
        ['trigger_type']
    ),
    'retraining_jobs_completed': Counter(
        'secom_retraining_jobs_completed_total',
        'Total retraining jobs completed',
        ['status']
    ),
    'active_training_jobs': Gauge(
        'secom_active_training_jobs',
        'Number of active training jobs'
    ),
    'pending_triggers': Gauge(
        'secom_pending_retraining_triggers',
        'Number of pending retraining triggers'
    ),
    'models_deployed': Counter(
        'secom_models_deployed_total',
        'Total models deployed'
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
    "logs/retrainer_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="90 days",
    compression="zip",
    level="DEBUG"
)


class RetrainerOrchestrator:
    """Orchestrates continuous learning and retraining"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = True
        
        # Database components
        self.db_manager = DatabaseManager()
        self.retraining_repo = RetrainingTriggerRepository(self.db_manager)
        self.model_registry_repo = ModelRegistryRepository(self.db_manager)
        self.audit_repo = AuditLogRepository(self.db_manager)
        
        # Training orchestrator
        self.training_orchestrator = TrainingOrchestrator(TRAINING_CONFIG)
        
        # State tracking
        self.active_jobs = 0
        self.last_retraining_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def _check_cooldown_period(self) -> bool:
        """Check if we're in cooldown period after last retraining"""
        if self.last_retraining_time is None:
            return False
        
        time_since_last = (datetime.utcnow() - self.last_retraining_time).total_seconds() / 3600
        cooldown_hours = self.config['retrainer']['cooldown_period_hours']
        
        if time_since_last < cooldown_hours:
            logger.debug(
                f"In cooldown period: {time_since_last:.1f}h / {cooldown_hours}h since last retraining"
            )
            return True
        
        return False
    
    def _get_pending_triggers(self) -> List[Tuple]:
        """Get pending retraining triggers"""
        try:
            triggers = self.retraining_repo.get_pending_triggers()
            metrics['pending_triggers'].set(len(triggers))
            return triggers
        except Exception as e:
            logger.error(f"Error getting pending triggers: {e}")
            return []
    
    def _execute_retraining(
        self,
        trigger_id: UUID,
        trigger_type: str,
        trigger_reason: str
    ) -> bool:
        """
        Execute model retraining
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info(f"STARTING RETRAINING JOB")
            logger.info(f"  Trigger ID: {trigger_id}")
            logger.info(f"  Trigger Type: {trigger_type}")
            logger.info(f"  Reason: {trigger_reason}")
            logger.info("=" * 80)
            
            # Update trigger status
            self.retraining_repo.update_trigger_status(
                trigger_id=trigger_id,
                status='in_progress'
            )
            
            # Increment active jobs
            self.active_jobs += 1
            metrics['active_training_jobs'].set(self.active_jobs)
            metrics['retraining_jobs_started'].labels(trigger_type=trigger_type).inc()
            
            # Run training pipeline
            start_time = datetime.utcnow()
            
            result = self.training_orchestrator.run_training_pipeline(
                triggered_by=trigger_type,
                auto_deploy=self.config['retrainer']['auto_deploy']
            )
            
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update trigger with results
            self.retraining_repo.update_trigger_status(
                trigger_id=trigger_id,
                status='completed',
                new_model_id=result['model_id']
            )
            
            metrics['retraining_jobs_completed'].labels(status='success').inc()
            
            if self.config['retrainer']['auto_deploy']:
                metrics['models_deployed'].inc()
            
            logger.info("=" * 80)
            logger.info(f"RETRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"  Duration: {duration_ms/1000:.2f}s")
            logger.info(f"  New Model ID: {result['model_id']}")
            logger.info(f"  Model Type: {result['model_type']}")
            logger.info(f"  F1 Score: {result['test_metrics']['f1_score']:.4f}")
            logger.info(f"  Auto-deployed: {self.config['retrainer']['auto_deploy']}")
            logger.info("=" * 80)
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='retraining_completed',
                event_status='success',
                component='retrainer',
                message=f"Retraining completed successfully: {trigger_type}",
                metadata={
                    'trigger_id': str(trigger_id),
                    'trigger_type': trigger_type,
                    'new_model_id': str(result['model_id']),
                    'model_type': result['model_type'],
                    'test_metrics': result['test_metrics'],
                    'auto_deployed': self.config['retrainer']['auto_deploy']
                },
                duration_ms=duration_ms
            )
            
            # Update last retraining time
            self.last_retraining_time = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Update trigger status as failed
            self.retraining_repo.update_trigger_status(
                trigger_id=trigger_id,
                status='failed',
                error_message=str(e)
            )
            
            metrics['retraining_jobs_completed'].labels(status='failure').inc()
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='retraining_completed',
                event_status='failure',
                component='retrainer',
                message=f"Retraining failed: {str(e)}",
                metadata={
                    'trigger_id': str(trigger_id),
                    'trigger_type': trigger_type,
                    'error': str(e)
                }
            )
            
            return False
            
        finally:
            # Decrement active jobs
            self.active_jobs -= 1
            metrics['active_training_jobs'].set(self.active_jobs)
    
    def _process_triggers(self):
        """Process pending retraining triggers"""
        # Check cooldown
        if self._check_cooldown_period():
            logger.debug("Skipping trigger processing (in cooldown period)")
            return
        
        # Check max concurrent jobs
        if self.active_jobs >= self.config['retrainer']['max_concurrent_training']:
            logger.debug(
                f"Max concurrent training jobs reached ({self.active_jobs}/"
                f"{self.config['retrainer']['max_concurrent_training']})"
            )
            return
        
        # Get pending triggers
        triggers = self._get_pending_triggers()
        
        if not triggers:
            logger.debug("No pending retraining triggers")
            return
        
        logger.info(f"Found {len(triggers)} pending retraining trigger(s)")
        
        # Process triggers (prioritize by creation time)
        for trigger_id, trigger_type, trigger_reason, model_id, created_at in triggers:
            # Check if we can start another job
            if self.active_jobs >= self.config['retrainer']['max_concurrent_training']:
                logger.info("Max concurrent jobs reached, deferring remaining triggers")
                break
            
            logger.info(f"Processing trigger {trigger_id}: {trigger_type}")
            
            # Execute retraining
            success = self._execute_retraining(
                trigger_id=UUID(trigger_id),
                trigger_type=trigger_type,
                trigger_reason=trigger_reason
            )
            
            if success:
                logger.info(f"Trigger {trigger_id} processed successfully")
            else:
                logger.error(f"x Trigger {trigger_id} processing failed")
            
            # If configured for single training at a time, break after first
            if self.config['retrainer']['max_concurrent_training'] == 1:
                break
    
    def run(self):
        """Main execution loop"""
        logger.info("=" * 80)
        logger.info("SECOM Continuous Learning Retrainer Started")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Check interval: {self.config['retrainer']['check_interval_seconds']}s")
        logger.info(f"  Auto-deploy: {self.config['retrainer']['auto_deploy']}")
        logger.info(f"  Max concurrent training: {self.config['retrainer']['max_concurrent_training']}")
        logger.info(f"  Cooldown period: {self.config['retrainer']['cooldown_period_hours']}h")
        logger.info("=" * 80)
        
        try:
            while self.running:
                try:
                    # Process pending triggers
                    self._process_triggers()
                    
                    # Sleep before next check
                    time.sleep(self.config['retrainer']['check_interval_seconds'])
                    
                except Exception as e:
                    logger.error(f"Error in retrainer loop: {e}")
                    time.sleep(60)  # Wait before retry
                    
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down retrainer service...")
        self.running = False
        
        # Wait for active jobs
        if self.active_jobs > 0:
            logger.info(f"Waiting for {self.active_jobs} active training job(s) to complete...")
            # In production, you might want to implement a more sophisticated
            # wait mechanism or job tracking
        
        self.db_manager.close_all()
        
        logger.info("=" * 80)
        logger.info("Retrainer service shutdown complete")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    import traceback
    
    try:
        # Start Prometheus metrics server
        logger.info(f"Starting Prometheus metrics server on port {CONFIG['monitoring']['prometheus_port']}")
        start_http_server(CONFIG['monitoring']['prometheus_port'])
        
        # Create and run orchestrator
        orchestrator = RetrainerOrchestrator(CONFIG)
        orchestrator.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
