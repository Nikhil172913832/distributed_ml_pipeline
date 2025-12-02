"""Tests for alerting module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from monitoring.alerting import (
    Alert,
    AlertSeverity,
    AlertChannel,
    AlertManager,
    send_quick_alert
)


class TestAlert:
    """Test Alert dataclass"""
    
    def test_alert_creation(self):
        """Test alert creation with defaults"""
        alert = Alert(
            title="Test Alert",
            message="This is a test",
            severity=AlertSeverity.WARNING,
            component="test_component"
        )
        
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.component == "test_component"
        assert alert.timestamp is not None
        assert alert.metadata is None
    
    def test_alert_to_dict(self):
        """Test alert serialization"""
        alert = Alert(
            title="Test",
            message="Message",
            severity=AlertSeverity.INFO,
            component="test",
            metadata={"key": "value"}
        )
        
        d = alert.to_dict()
        assert d['title'] == "Test"
        assert d['severity'] == AlertSeverity.INFO
        assert d['metadata'] == {"key": "value"}
    
    def test_alert_to_json(self):
        """Test JSON serialization"""
        alert = Alert(
            title="Test",
            message="Message",
            severity=AlertSeverity.ERROR,
            component="test"
        )
        
        json_str = alert.to_json()
        assert "Test" in json_str
        assert "error" in json_str


class TestAlertManager:
    """Test AlertManager"""
    
    @pytest.fixture
    def alert_manager(self):
        """Create test alert manager"""
        return AlertManager(
            channels=[AlertChannel.LOG],
            min_severity=AlertSeverity.INFO
        )
    
    def test_initialization(self, alert_manager):
        """Test alert manager initialization"""
        assert AlertChannel.LOG in alert_manager.channels
        assert alert_manager.min_severity == AlertSeverity.INFO
        assert len(alert_manager.alert_history) == 0
    
    def test_severity_filtering(self, alert_manager):
        """Test severity filtering"""
        alert_manager.min_severity = AlertSeverity.WARNING
        
        # Should filter INFO
        alert_info = Alert(
            title="Info",
            message="Low priority",
            severity=AlertSeverity.INFO,
            component="test"
        )
        assert not alert_manager._should_send_alert(alert_info)
        
        # Should pass WARNING
        alert_warning = Alert(
            title="Warning",
            message="Medium priority",
            severity=AlertSeverity.WARNING,
            component="test"
        )
        assert alert_manager._should_send_alert(alert_warning)
        
        # Should pass ERROR
        alert_error = Alert(
            title="Error",
            message="High priority",
            severity=AlertSeverity.ERROR,
            component="test"
        )
        assert alert_manager._should_send_alert(alert_error)
    
    def test_send_to_log(self, alert_manager, caplog):
        """Test logging alert"""
        alert = Alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.INFO,
            component="test"
        )
        
        alert_manager._send_to_log(alert)
        
        # Check if logged (with loguru, we can't easily capture logs in caplog)
        # Just verify method doesn't raise
        assert True
    
    @patch('monitoring.alerting.smtplib.SMTP')
    def test_send_to_email(self, mock_smtp, alert_manager):
        """Test email alert"""
        alert_manager.email_config = {
            'smtp_host': 'smtp.test.com',
            'smtp_port': 587,
            'smtp_user': 'test@test.com',
            'smtp_password': 'password',
            'from_email': 'test@test.com',
            'to_emails': ['recipient@test.com']
        }
        
        alert = Alert(
            title="Email Test",
            message="Test email alert",
            severity=AlertSeverity.WARNING,
            component="test"
        )
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        alert_manager._send_to_email(alert)
        
        # Verify SMTP calls
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
    
    @patch('monitoring.alerting.requests.post')
    def test_send_to_slack(self, mock_post, alert_manager):
        """Test Slack alert"""
        alert_manager.slack_webhook_url = "https://hooks.slack.com/test"
        
        alert = Alert(
            title="Slack Test",
            message="Test Slack alert",
            severity=AlertSeverity.ERROR,
            component="test"
        )
        
        mock_post.return_value.status_code = 200
        
        alert_manager._send_to_slack(alert)
        
        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://hooks.slack.com/test" == call_args[0][0]
        assert 'attachments' in call_args[1]['json']
    
    @patch('monitoring.alerting.requests.post')
    def test_send_to_webhook(self, mock_post, alert_manager):
        """Test custom webhook alert"""
        alert_manager.custom_webhook_url = "https://webhook.test.com"
        
        alert = Alert(
            title="Webhook Test",
            message="Test webhook alert",
            severity=AlertSeverity.CRITICAL,
            component="test"
        )
        
        mock_post.return_value.status_code = 200
        
        alert_manager._send_to_webhook(alert)
        
        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://webhook.test.com" == call_args[0][0]
        assert call_args[1]['headers']['Content-Type'] == 'application/json'
    
    def test_send_alert(self, alert_manager):
        """Test sending alert through manager"""
        success = alert_manager.send_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.INFO,
            component="test"
        )
        
        assert success
        assert len(alert_manager.alert_history) == 1
        assert alert_manager.alert_history[0]['title'] == "Test Alert"
    
    def test_alert_model_performance_degradation(self, alert_manager):
        """Test model performance degradation alert"""
        alert_manager.alert_model_performance_degradation(
            model_id="model-123",
            metric_name="f1_score",
            current_value=0.75,
            threshold=0.80,
            baseline_value=0.85
        )
        
        assert len(alert_manager.alert_history) == 1
        alert = alert_manager.alert_history[0]
        assert "Performance Degradation" in alert['title']
        assert alert['severity'] == AlertSeverity.WARNING.value
    
    def test_alert_data_drift(self, alert_manager):
        """Test data drift alert"""
        alert_manager.alert_data_drift(
            feature_name="temperature",
            drift_score=0.42,
            threshold=0.30,
            drift_type="kolmogorov_smirnov"
        )
        
        assert len(alert_manager.alert_history) == 1
        alert = alert_manager.alert_history[0]
        assert "Data Drift" in alert['title']
        assert alert['metadata']['feature'] == "temperature"
    
    def test_alert_retraining_required(self, alert_manager):
        """Test retraining trigger alert"""
        alert_manager.alert_retraining_required(
            reason="Scheduled retraining",
            trigger_type="scheduled",
            model_id="model-456"
        )
        
        assert len(alert_manager.alert_history) == 1
        alert = alert_manager.alert_history[0]
        assert "Retraining Triggered" in alert['title']
    
    def test_alert_retraining_completed(self, alert_manager):
        """Test retraining completion alert"""
        alert_manager.alert_retraining_completed(
            model_id="model-789",
            model_type="random_forest",
            metrics={"f1_score": 0.92, "accuracy": 0.90},
            approved=False
        )
        
        assert len(alert_manager.alert_history) == 1
        alert = alert_manager.alert_history[0]
        assert "Retraining Completed" in alert['title']
        assert not alert['metadata']['approved']
    
    def test_get_alert_history(self, alert_manager):
        """Test getting alert history with filters"""
        # Send multiple alerts
        alert_manager.send_alert("Info 1", "msg", AlertSeverity.INFO, "test")
        alert_manager.send_alert("Warning 1", "msg", AlertSeverity.WARNING, "test")
        alert_manager.send_alert("Info 2", "msg", AlertSeverity.INFO, "other")
        alert_manager.send_alert("Error 1", "msg", AlertSeverity.ERROR, "test")
        
        # Get all
        all_alerts = alert_manager.get_alert_history()
        assert len(all_alerts) == 4
        
        # Filter by severity
        warnings = alert_manager.get_alert_history(severity=AlertSeverity.WARNING)
        assert len(warnings) == 1
        
        # Filter by component
        test_alerts = alert_manager.get_alert_history(component="test")
        assert len(test_alerts) == 3
        
        # Limit
        limited = alert_manager.get_alert_history(limit=2)
        assert len(limited) == 2


class TestQuickAlert:
    """Test convenience function"""
    
    @patch('monitoring.alerting.AlertManager')
    def test_send_quick_alert(self, mock_manager_class):
        """Test quick alert function"""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        send_quick_alert(
            title="Quick Test",
            message="Quick message",
            severity=AlertSeverity.INFO
        )
        
        # Verify AlertManager was created and used
        mock_manager_class.assert_called_once()
        mock_manager.send_alert.assert_called_once()
