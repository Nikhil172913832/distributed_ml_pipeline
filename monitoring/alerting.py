"""
Alerting utilities for model monitoring and retraining notifications.

Provides alert channels (email, Slack, webhook) and alert rules for:
- Model performance degradation
- Data drift detection
- Retraining triggers
- System health issues
"""

import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Available alert channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    LOG = "log"


@dataclass
class Alert:
    """Alert message structure"""
    title: str
    message: str
    severity: AlertSeverity
    component: str
    timestamp: str = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class AlertManager:
    """
    Central alert management system.
    
    Sends alerts through configured channels (email, Slack, webhooks).
    Supports alert deduplication and rate limiting.
    """
    
    def __init__(
        self,
        channels: Optional[List[AlertChannel]] = None,
        email_config: Optional[Dict[str, str]] = None,
        slack_webhook_url: Optional[str] = None,
        custom_webhook_url: Optional[str] = None,
        min_severity: AlertSeverity = AlertSeverity.INFO
    ):
        """
        Initialize alert manager.
        
        Args:
            channels: List of enabled alert channels (default: [LOG])
            email_config: Email configuration with keys: smtp_host, smtp_port, 
                         smtp_user, smtp_password, from_email, to_emails
            slack_webhook_url: Slack incoming webhook URL
            custom_webhook_url: Custom webhook URL for alert notifications
            min_severity: Minimum severity level to send alerts
        """
        self.channels = channels or [AlertChannel.LOG]
        self.min_severity = min_severity
        
        # Email configuration
        self.email_config = email_config or self._load_email_config()
        
        # Webhook URLs
        self.slack_webhook_url = slack_webhook_url or os.getenv('SLACK_WEBHOOK_URL')
        self.custom_webhook_url = custom_webhook_url or os.getenv('ALERT_WEBHOOK_URL')
        
        # Alert history for deduplication
        self.alert_history: List[Dict[str, Any]] = []
        
        logger.info(f"AlertManager initialized with channels: {[c.value for c in self.channels]}")
    
    def _load_email_config(self) -> Dict[str, str]:
        """Load email configuration from environment"""
        return {
            'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'smtp_user': os.getenv('SMTP_USER', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('ALERT_FROM_EMAIL', ''),
            'to_emails': os.getenv('ALERT_TO_EMAILS', '').split(',')
        }
    
    def _severity_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity level"""
        return {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }.get(severity, "")
    
    def _severity_color(self, severity: AlertSeverity) -> str:
        """Get color code for severity level (Slack)"""
        return {
            AlertSeverity.INFO: "#36a64f",      # green
            AlertSeverity.WARNING: "#ff9900",   # orange
            AlertSeverity.ERROR: "#ff0000",     # red
            AlertSeverity.CRITICAL: "#8b0000"   # dark red
        }.get(severity, "#808080")
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on severity"""
        severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL
        ]
        
        alert_level = severity_order.index(alert.severity)
        min_level = severity_order.index(self.min_severity)
        
        return alert_level >= min_level
    
    def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        component: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
        channels: Optional[List[AlertChannel]] = None
    ) -> bool:
        """
        Send alert through configured channels.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            component: Component generating the alert
            metadata: Additional alert metadata
            channels: Override default channels for this alert
            
        Returns:
            True if alert was sent successfully
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            component=component,
            metadata=metadata or {}
        )
        
        # Check severity filter
        if not self._should_send_alert(alert):
            logger.debug(f"Alert filtered (severity too low): {title}")
            return False
        
        # Store in history
        self.alert_history.append(alert.to_dict())
        
        # Use specified channels or defaults
        target_channels = channels or self.channels
        
        success = True
        for channel in target_channels:
            try:
                if channel == AlertChannel.LOG:
                    self._send_to_log(alert)
                elif channel == AlertChannel.EMAIL:
                    self._send_to_email(alert)
                elif channel == AlertChannel.SLACK:
                    self._send_to_slack(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_to_webhook(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                success = False
        
        return success
    
    def _send_to_log(self, alert: Alert):
        """Send alert to logs"""
        emoji = self._severity_emoji(alert.severity)
        log_msg = f"{emoji} [{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        
        if alert.severity == AlertSeverity.INFO:
            logger.info(log_msg)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        elif alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            logger.error(log_msg)
    
    def _send_to_email(self, alert: Alert):
        """Send alert via email"""
        if not self.email_config.get('smtp_user') or not self.email_config.get('to_emails'):
            logger.warning("Email configuration incomplete, skipping email alert")
            return
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg['From'] = self.email_config['from_email']
        msg['To'] = ', '.join(self.email_config['to_emails'])
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2 style="color: {self._severity_color(alert.severity)}">
                {self._severity_emoji(alert.severity)} {alert.title}
            </h2>
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Component:</strong> {alert.component}</p>
            <p><strong>Time:</strong> {alert.timestamp}</p>
            <hr>
            <p>{alert.message}</p>
            
            {f"<hr><h3>Metadata</h3><pre>{json.dumps(alert.metadata, indent=2)}</pre>" if alert.metadata else ""}
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        try:
            with smtplib.SMTP(self.email_config['smtp_host'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['smtp_user'], self.email_config['smtp_password'])
                server.send_message(msg)
            
            logger.info(f"Alert sent via email: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise
    
    def _send_to_slack(self, alert: Alert):
        """Send alert to Slack"""
        if not self.slack_webhook_url:
            logger.warning("Slack webhook URL not configured, skipping Slack alert")
            return
        
        # Format Slack message
        emoji = self._severity_emoji(alert.severity)
        color = self._severity_color(alert.severity)
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Component",
                            "value": alert.component,
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp,
                            "short": False
                        }
                    ],
                    "footer": "ML Pipeline Monitoring",
                    "ts": int(datetime.fromisoformat(alert.timestamp).timestamp())
                }
            ]
        }
        
        # Add metadata if present
        if alert.metadata:
            payload["attachments"][0]["fields"].append({
                "title": "Metadata",
                "value": f"```{json.dumps(alert.metadata, indent=2)}```",
                "short": False
            })
        
        # Send to Slack
        response = requests.post(
            self.slack_webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        logger.info(f"Alert sent to Slack: {alert.title}")
    
    def _send_to_webhook(self, alert: Alert):
        """Send alert to custom webhook"""
        if not self.custom_webhook_url:
            logger.warning("Custom webhook URL not configured, skipping webhook alert")
            return
        
        # Send alert as JSON
        response = requests.post(
            self.custom_webhook_url,
            json=alert.to_dict(),
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        response.raise_for_status()
        
        logger.info(f"Alert sent to webhook: {alert.title}")
    
    def alert_model_performance_degradation(
        self,
        model_id: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        baseline_value: Optional[float] = None
    ):
        """Send alert for model performance degradation"""
        metadata = {
            'model_id': model_id,
            'metric': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'baseline_value': baseline_value
        }
        
        message = (
            f"Model {model_id} performance has degraded.\n"
            f"Metric '{metric_name}' is {current_value:.4f}, "
            f"below threshold {threshold:.4f}"
        )
        
        if baseline_value:
            message += f" (baseline: {baseline_value:.4f})"
        
        self.send_alert(
            title="Model Performance Degradation Detected",
            message=message,
            severity=AlertSeverity.WARNING,
            component="model_monitor",
            metadata=metadata
        )
    
    def alert_data_drift(
        self,
        feature_name: str,
        drift_score: float,
        threshold: float,
        drift_type: str = "unknown"
    ):
        """Send alert for data drift detection"""
        metadata = {
            'feature': feature_name,
            'drift_score': drift_score,
            'threshold': threshold,
            'drift_type': drift_type
        }
        
        self.send_alert(
            title="Data Drift Detected",
            message=(
                f"Feature '{feature_name}' shows significant drift.\n"
                f"Drift score: {drift_score:.4f} (threshold: {threshold:.4f})\n"
                f"Type: {drift_type}"
            ),
            severity=AlertSeverity.WARNING,
            component="data_monitor",
            metadata=metadata
        )
    
    def alert_retraining_required(
        self,
        reason: str,
        trigger_type: str,
        model_id: Optional[str] = None
    ):
        """Send alert when retraining is triggered"""
        metadata = {
            'trigger_type': trigger_type,
            'reason': reason,
            'model_id': model_id
        }
        
        self.send_alert(
            title="Model Retraining Triggered",
            message=(
                f"Retraining has been triggered.\n"
                f"Trigger: {trigger_type}\n"
                f"Reason: {reason}"
            ),
            severity=AlertSeverity.INFO,
            component="retrainer",
            metadata=metadata
        )
    
    def alert_retraining_completed(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, float],
        approved: bool = False
    ):
        """Send alert when retraining completes"""
        metadata = {
            'model_id': model_id,
            'model_type': model_type,
            'metrics': metrics,
            'approved': approved
        }
        
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        
        self.send_alert(
            title="Model Retraining Completed",
            message=(
                f"New model trained successfully.\n"
                f"Model ID: {model_id}\n"
                f"Type: {model_type}\n"
                f"Metrics: {metrics_str}\n"
                f"Status: {'Approved for deployment' if approved else 'Awaiting approval'}"
            ),
            severity=AlertSeverity.INFO,
            component="retrainer",
            metadata=metadata
        )
    
    def alert_system_health(
        self,
        component: str,
        status: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.ERROR
    ):
        """Send alert for system health issues"""
        metadata = {
            'component': component,
            'status': status
        }
        
        self.send_alert(
            title=f"System Health Issue: {component}",
            message=message,
            severity=severity,
            component=component,
            metadata=metadata
        )
    
    def get_alert_history(
        self,
        limit: Optional[int] = None,
        severity: Optional[AlertSeverity] = None,
        component: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history with optional filters.
        
        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity
            component: Filter by component
            
        Returns:
            List of alert dictionaries
        """
        filtered = self.alert_history
        
        if severity:
            filtered = [a for a in filtered if a['severity'] == severity.value]
        
        if component:
            filtered = [a for a in filtered if a['component'] == component]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered


# Convenience functions for quick alerts
def send_quick_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    slack_webhook: Optional[str] = None
):
    """
    Send a quick alert without initializing full AlertManager.
    
    Useful for one-off alerts or simple notifications.
    """
    channels = [AlertChannel.LOG]
    
    if slack_webhook:
        channels.append(AlertChannel.SLACK)
    
    manager = AlertManager(
        channels=channels,
        slack_webhook_url=slack_webhook
    )
    
    manager.send_alert(
        title=title,
        message=message,
        severity=severity
    )
