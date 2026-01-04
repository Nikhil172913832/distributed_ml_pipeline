"""
Automated model rollback mechanism for production safety.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict
from loguru import logger

from database import (
    DatabaseManager, ModelRegistryRepository, 
    ModelPerformanceRepository, AuditLogRepository
)


class ModelRollbackManager:
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        monitoring_window_hours: int = 1,
        accuracy_drop_threshold: float = 0.05,
        f1_drop_threshold: float = 0.05
    ):
        self.db_manager = db_manager
        self.monitoring_window_hours = monitoring_window_hours
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.f1_drop_threshold = f1_drop_threshold
        
        self.model_registry = ModelRegistryRepository(db_manager)
        self.performance_repo = ModelPerformanceRepository(db_manager)
        self.audit_log = AuditLogRepository(db_manager)
    
    def start_monitoring(self, new_model_id: uuid.UUID, previous_model_id: uuid.UUID):
        """Start monitoring new model performance for potential rollback."""
        logger.info(f"Starting rollback monitoring for model {new_model_id}")
        
        # Record monitoring start
        self.audit_log.log_event(
            event_type='rollback_monitoring_started',
            event_status='info',
            component='rollback_manager',
            message=f'Monitoring new model {new_model_id} for {self.monitoring_window_hours}h',
            metadata={
                'new_model_id': str(new_model_id),
                'previous_model_id': str(previous_model_id),
                'monitoring_window_hours': self.monitoring_window_hours
            }
        )
    
    def check_rollback_needed(
        self, 
        new_model_id: uuid.UUID, 
        previous_model_id: uuid.UUID
    ) -> tuple[bool, Optional[str]]:
        """Check if rollback is needed based on performance comparison."""
        
        # Get baseline performance from previous model
        baseline_perf = self._get_baseline_performance(previous_model_id)
        if not baseline_perf:
            logger.warning("No baseline performance found, skipping rollback check")
            return False, None
        
        # Get current performance of new model
        current_perf = self._get_recent_performance(new_model_id)
        if not current_perf:
            logger.warning("No current performance found, skipping rollback check")
            return False, None
        
        # Compare metrics
        accuracy_drop = baseline_perf['accuracy'] - current_perf['accuracy']
        f1_drop = baseline_perf['f1_score'] - current_perf['f1_score']
        
        logger.info(f"Performance comparison - Accuracy drop: {accuracy_drop:.4f}, F1 drop: {f1_drop:.4f}")
        
        # Check thresholds
        if accuracy_drop > self.accuracy_drop_threshold:
            reason = f"Accuracy dropped by {accuracy_drop:.4f} (threshold: {self.accuracy_drop_threshold})"
            return True, reason
        
        if f1_drop > self.f1_drop_threshold:
            reason = f"F1 score dropped by {f1_drop:.4f} (threshold: {self.f1_drop_threshold})"
            return True, reason
        
        return False, None
    
    def execute_rollback(
        self, 
        new_model_id: uuid.UUID, 
        previous_model_id: uuid.UUID, 
        reason: str
    ):
        """Execute rollback to previous model."""
        logger.warning(f"Executing rollback: {reason}")
        
        try:
            # Deactivate new model
            self.model_registry.deactivate_model(new_model_id)
            
            # Reactivate previous model
            self.model_registry.activate_model(previous_model_id)
            
            # Log rollback event
            self.audit_log.log_event(
                event_type='model_rollback',
                event_status='success',
                component='rollback_manager',
                message=f'Rolled back from {new_model_id} to {previous_model_id}',
                metadata={
                    'new_model_id': str(new_model_id),
                    'previous_model_id': str(previous_model_id),
                    'reason': reason
                }
            )
            
            logger.info(f"Rollback completed successfully to model {previous_model_id}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self.audit_log.log_event(
                event_type='model_rollback',
                event_status='failure',
                component='rollback_manager',
                message=f'Rollback failed: {str(e)}',
                metadata={
                    'new_model_id': str(new_model_id),
                    'previous_model_id': str(previous_model_id),
                    'error': str(e)
                }
            )
            raise
    
    def _get_baseline_performance(self, model_id: uuid.UUID) -> Optional[Dict]:
        """Get baseline performance metrics from previous model."""
        # Get last 24 hours of performance
        perf = self.performance_repo.get_latest_performance(model_id, 'hourly')
        if not perf:
            return None
        
        return {
            'accuracy': perf[0],
            'precision': perf[1],
            'recall': perf[2],
            'f1_score': perf[3]
        }
    
    def _get_recent_performance(self, model_id: uuid.UUID) -> Optional[Dict]:
        """Get recent performance of new model."""
        now = datetime.utcnow()
        window_start = now - timedelta(hours=self.monitoring_window_hours)
        
        # Calculate performance for monitoring window
        metric_id = self.performance_repo.calculate_performance_window(
            model_id=model_id,
            window_start=window_start,
            window_end=now,
            window_type='rollback_check'
        )
        
        if not metric_id:
            return None
        
        # Retrieve calculated metrics
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT accuracy, precision, recall, f1_score
                FROM secom.model_performance_metrics
                WHERE id = %s
            """, (metric_id,))
            result = cursor.fetchone()
        
        if not result:
            return None
        
        return {
            'accuracy': result[0],
            'precision': result[1],
            'recall': result[2],
            'f1_score': result[3]
        }
