"""
Safe Retrainer with Human-in-the-Loop approval workflow.

Extends the automatic retrainer with manual approval gates before model deployment.
Provides review interface and approval tracking for production model updates.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from enum import Enum

from loguru import logger
from dotenv import load_dotenv

from database import (
    DatabaseManager, 
    RetrainingTriggerRepository,
    ModelRegistryRepository,
    AuditLogRepository
)
from model_trainer import TrainingOrchestrator, CONFIG as TRAINING_CONFIG
from monitoring.alerting import AlertManager, AlertSeverity, AlertChannel

load_dotenv()


class ApprovalStatus(str, Enum):
    """Model approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ModelApproval:
    """
    Tracks model approval requests and decisions.
    """
    
    def __init__(
        self,
        model_id: UUID,
        model_type: str,
        metrics: Dict[str, float],
        training_metadata: Dict[str, Any],
        approval_timeout_hours: int = 24
    ):
        """
        Initialize model approval request.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model (e.g., 'logistic_regression', 'random_forest')
            metrics: Model evaluation metrics
            training_metadata: Additional training context
            approval_timeout_hours: Hours before approval request expires
        """
        self.approval_id = uuid4()
        self.model_id = model_id
        self.model_type = model_type
        self.metrics = metrics
        self.training_metadata = training_metadata
        self.status = ApprovalStatus.PENDING
        self.created_at = datetime.utcnow()
        self.approval_timeout_hours = approval_timeout_hours
        self.approved_by: Optional[str] = None
        self.approved_at: Optional[datetime] = None
        self.rejection_reason: Optional[str] = None
        self.notes: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if approval request has expired"""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds() / 3600
        return elapsed > self.approval_timeout_hours
    
    def approve(self, approved_by: str, notes: Optional[str] = None):
        """Approve model for deployment"""
        if self.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot approve model in {self.status} status")
        
        if self.is_expired():
            self.status = ApprovalStatus.EXPIRED
            raise ValueError("Approval request has expired")
        
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approved_by
        self.approved_at = datetime.utcnow()
        self.notes = notes
        
        logger.info(f"Model {self.model_id} approved by {approved_by}")
    
    def reject(self, rejected_by: str, reason: str, notes: Optional[str] = None):
        """Reject model for deployment"""
        if self.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot reject model in {self.status} status")
        
        self.status = ApprovalStatus.REJECTED
        self.approved_by = rejected_by
        self.approved_at = datetime.utcnow()
        self.rejection_reason = reason
        self.notes = notes
        
        logger.info(f"Model {self.model_id} rejected by {rejected_by}: {reason}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'approval_id': str(self.approval_id),
            'model_id': str(self.model_id),
            'model_type': self.model_type,
            'metrics': self.metrics,
            'training_metadata': self.training_metadata,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'approved_by': self.approved_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'rejection_reason': self.rejection_reason,
            'notes': self.notes,
            'is_expired': self.is_expired()
        }


class SafeRetrainer:
    """
    Enhanced retrainer with human-in-the-loop approval.
    
    Trains models automatically but requires manual approval before deployment.
    Provides alerting, review interface, and approval tracking.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        alert_manager: Optional[AlertManager] = None,
        approval_timeout_hours: int = 24
    ):
        """
        Initialize safe retrainer.
        
        Args:
            config: Retrainer configuration
            alert_manager: Alert manager for notifications
            approval_timeout_hours: Hours to wait for approval before expiring
        """
        self.config = config
        self.approval_timeout_hours = approval_timeout_hours
        
        # Database components
        self.db_manager = DatabaseManager()
        self.retraining_repo = RetrainingTriggerRepository(self.db_manager)
        self.model_registry_repo = ModelRegistryRepository(self.db_manager)
        self.audit_repo = AuditLogRepository(self.db_manager)
        
        # Training orchestrator
        self.training_orchestrator = TrainingOrchestrator(TRAINING_CONFIG)
        
        # Alert manager
        self.alert_manager = alert_manager or self._create_default_alert_manager()
        
        # Pending approvals
        self.pending_approvals: Dict[UUID, ModelApproval] = {}
        
        logger.info("SafeRetrainer initialized with human-in-the-loop approval")
    
    def _create_default_alert_manager(self) -> AlertManager:
        """Create default alert manager"""
        channels = [AlertChannel.LOG]
        
        # Add Slack if configured
        if os.getenv('SLACK_WEBHOOK_URL'):
            channels.append(AlertChannel.SLACK)
        
        # Add email if configured
        if os.getenv('SMTP_USER') and os.getenv('ALERT_TO_EMAILS'):
            channels.append(AlertChannel.EMAIL)
        
        return AlertManager(
            channels=channels,
            min_severity=AlertSeverity.INFO
        )
    
    def train_new_model(
        self,
        trigger_id: UUID,
        trigger_type: str,
        trigger_reason: str
    ) -> ModelApproval:
        """
        Train a new model and create approval request.
        
        Args:
            trigger_id: Retraining trigger ID
            trigger_type: Type of trigger (drift, performance, manual, etc.)
            trigger_reason: Human-readable reason
            
        Returns:
            ModelApproval object for review
        """
        logger.info("=" * 80)
        logger.info("TRAINING NEW MODEL (Human Approval Required)")
        logger.info(f"  Trigger: {trigger_type}")
        logger.info(f"  Reason: {trigger_reason}")
        logger.info("=" * 80)
        
        # Update trigger status
        self.retraining_repo.update_trigger_status(
            trigger_id=trigger_id,
            status='in_progress'
        )
        
        # Send alert for training start
        self.alert_manager.alert_retraining_required(
            reason=trigger_reason,
            trigger_type=trigger_type
        )
        
        try:
            # Run training (without auto-deploy)
            start_time = datetime.utcnow()
            
            result = self.training_orchestrator.run_training_pipeline(
                triggered_by=trigger_type,
                auto_deploy=False  # Never auto-deploy in safe mode
            )
            
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Create approval request
            approval = ModelApproval(
                model_id=result['model_id'],
                model_type=result['model_type'],
                metrics=result['test_metrics'],
                training_metadata={
                    'trigger_id': str(trigger_id),
                    'trigger_type': trigger_type,
                    'trigger_reason': trigger_reason,
                    'training_duration_ms': duration_ms,
                    'trained_at': end_time.isoformat(),
                    'training_samples': result.get('training_samples', 0),
                    'validation_samples': result.get('validation_samples', 0)
                },
                approval_timeout_hours=self.approval_timeout_hours
            )
            
            # Store approval request
            self.pending_approvals[approval.model_id] = approval
            
            # Update trigger
            self.retraining_repo.update_trigger_status(
                trigger_id=trigger_id,
                status='awaiting_approval',
                new_model_id=result['model_id']
            )
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='model_training_completed',
                event_status='success',
                component='safe_retrainer',
                message=f"Model trained, awaiting approval: {result['model_type']}",
                metadata=approval.to_dict(),
                duration_ms=duration_ms
            )
            
            # Send alert with model details
            self.alert_manager.alert_retraining_completed(
                model_id=str(result['model_id']),
                model_type=result['model_type'],
                metrics=result['test_metrics'],
                approved=False
            )
            
            logger.info("=" * 80)
            logger.info("MODEL TRAINING COMPLETED - AWAITING APPROVAL")
            logger.info(f"  Model ID: {result['model_id']}")
            logger.info(f"  Approval ID: {approval.approval_id}")
            logger.info(f"  Model Type: {result['model_type']}")
            logger.info(f"  Metrics: {result['test_metrics']}")
            logger.info(f"  Expires in: {self.approval_timeout_hours}h")
            logger.info("=" * 80)
            
            return approval
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            
            # Update trigger as failed
            self.retraining_repo.update_trigger_status(
                trigger_id=trigger_id,
                status='failed',
                error_message=str(e)
            )
            
            # Send error alert
            self.alert_manager.alert_system_health(
                component='safe_retrainer',
                status='error',
                message=f"Model training failed: {str(e)}",
                severity=AlertSeverity.ERROR
            )
            
            raise
    
    def approve_model(
        self,
        model_id: UUID,
        approved_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Approve model for deployment.
        
        Args:
            model_id: Model ID to approve
            approved_by: Username/email of approver
            notes: Optional approval notes
            
        Returns:
            True if deployment successful
        """
        if model_id not in self.pending_approvals:
            raise ValueError(f"No pending approval found for model {model_id}")
        
        approval = self.pending_approvals[model_id]
        
        try:
            # Approve the model
            approval.approve(approved_by=approved_by, notes=notes)
            
            # Deploy to production
            logger.info(f"Deploying approved model {model_id}...")
            
            # Update model registry to mark as production
            self.model_registry_repo.set_production_model(model_id)
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='model_approved_and_deployed',
                event_status='success',
                component='safe_retrainer',
                message=f"Model approved and deployed by {approved_by}",
                metadata=approval.to_dict()
            )
            
            # Send success alert
            self.alert_manager.send_alert(
                title="Model Approved and Deployed",
                message=(
                    f"Model {model_id} has been approved and deployed to production.\n"
                    f"Approved by: {approved_by}\n"
                    f"Model type: {approval.model_type}\n"
                    f"Metrics: {approval.metrics}"
                ),
                severity=AlertSeverity.INFO,
                component='safe_retrainer',
                metadata=approval.to_dict()
            )
            
            # Remove from pending
            del self.pending_approvals[model_id]
            
            logger.info(f"✓ Model {model_id} deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            
            # Log failure
            self.audit_repo.log_event(
                event_type='model_deployment_failed',
                event_status='failure',
                component='safe_retrainer',
                message=f"Failed to deploy approved model: {str(e)}",
                metadata={'model_id': str(model_id), 'error': str(e)}
            )
            
            raise
    
    def reject_model(
        self,
        model_id: UUID,
        rejected_by: str,
        reason: str,
        notes: Optional[str] = None
    ):
        """
        Reject model for deployment.
        
        Args:
            model_id: Model ID to reject
            rejected_by: Username/email of reviewer
            reason: Rejection reason
            notes: Optional rejection notes
        """
        if model_id not in self.pending_approvals:
            raise ValueError(f"No pending approval found for model {model_id}")
        
        approval = self.pending_approvals[model_id]
        approval.reject(rejected_by=rejected_by, reason=reason, notes=notes)
        
        # Log audit event
        self.audit_repo.log_event(
            event_type='model_rejected',
            event_status='success',
            component='safe_retrainer',
            message=f"Model rejected by {rejected_by}: {reason}",
            metadata=approval.to_dict()
        )
        
        # Send alert
        self.alert_manager.send_alert(
            title="Model Rejected",
            message=(
                f"Model {model_id} has been rejected.\n"
                f"Rejected by: {rejected_by}\n"
                f"Reason: {reason}"
            ),
            severity=AlertSeverity.WARNING,
            component='safe_retrainer',
            metadata=approval.to_dict()
        )
        
        # Remove from pending
        del self.pending_approvals[model_id]
        
        logger.info(f"Model {model_id} rejected by {rejected_by}")
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests"""
        return [
            approval.to_dict() 
            for approval in self.pending_approvals.values()
            if approval.status == ApprovalStatus.PENDING and not approval.is_expired()
        ]
    
    def cleanup_expired_approvals(self):
        """Remove expired approval requests"""
        expired = [
            model_id 
            for model_id, approval in self.pending_approvals.items()
            if approval.is_expired()
        ]
        
        for model_id in expired:
            approval = self.pending_approvals[model_id]
            approval.status = ApprovalStatus.EXPIRED
            
            logger.warning(f"Approval request expired for model {model_id}")
            
            # Log audit event
            self.audit_repo.log_event(
                event_type='model_approval_expired',
                event_status='warning',
                component='safe_retrainer',
                message=f"Approval request expired for model {model_id}",
                metadata=approval.to_dict()
            )
            
            # Remove from pending
            del self.pending_approvals[model_id]


# CLI interface for approval management
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SafeRetrainer approval interface")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List pending approvals
    list_parser = subparsers.add_parser('list', help='List pending approvals')
    
    # Approve model
    approve_parser = subparsers.add_parser('approve', help='Approve model')
    approve_parser.add_argument('model_id', help='Model UUID')
    approve_parser.add_argument('--approved-by', required=True, help='Approver name/email')
    approve_parser.add_argument('--notes', help='Approval notes')
    
    # Reject model
    reject_parser = subparsers.add_parser('reject', help='Reject model')
    reject_parser.add_argument('model_id', help='Model UUID')
    reject_parser.add_argument('--rejected-by', required=True, help='Reviewer name/email')
    reject_parser.add_argument('--reason', required=True, help='Rejection reason')
    reject_parser.add_argument('--notes', help='Rejection notes')
    
    args = parser.parse_args()
    
    # Initialize safe retrainer
    CONFIG = {
        'retrainer': {
            'check_interval_seconds': 300,
            'auto_deploy': False,
            'max_concurrent_training': 1,
            'cooldown_period_hours': 6,
        }
    }
    
    retrainer = SafeRetrainer(CONFIG)
    
    if args.command == 'list':
        approvals = retrainer.get_pending_approvals()
        
        if not approvals:
            print("No pending approvals")
        else:
            import json
            print(json.dumps(approvals, indent=2))
    
    elif args.command == 'approve':
        model_id = UUID(args.model_id)
        retrainer.approve_model(
            model_id=model_id,
            approved_by=args.approved_by,
            notes=args.notes
        )
        print(f"✓ Model {model_id} approved and deployed")
    
    elif args.command == 'reject':
        model_id = UUID(args.model_id)
        retrainer.reject_model(
            model_id=model_id,
            rejected_by=args.rejected_by,
            reason=args.reason,
            notes=args.notes
        )
        print(f"✓ Model {model_id} rejected")
    
    else:
        parser.print_help()
