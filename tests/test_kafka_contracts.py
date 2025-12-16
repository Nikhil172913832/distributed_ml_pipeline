"""
Kafka Message Schema Validation (Contract Tests).

Ensures message format consistency between producer and consumer
to prevent integration issues.
"""

import pytest
import json
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError, validator
from datetime import datetime


# ==========================================
# MESSAGE SCHEMAS
# ==========================================

class RawDataMessage(BaseModel):
    """Schema for raw data messages published to Kafka."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    sample_id: str = Field(..., description="Unique sample identifier")
    timestamp: str = Field(..., description="ISO format timestamp")
    features: Dict[str, float] = Field(..., description="590 sensor features")
    target: int = Field(..., ge=-1, le=1, description="Quality label (-1, 0, or 1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    
    @validator('features')
    def validate_feature_count(cls, v):
        """Validate 590 features."""
        if len(v) != 590:
            raise ValueError(f"Expected 590 features, got {len(v)}")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate ISO format timestamp."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {v}")
        return v


class PreprocessedDataMessage(BaseModel):
    """Schema for preprocessed data messages."""
    
    sample_id: str
    batch_id: str
    timestamp: str
    features: Dict[str, float]
    preprocessing_version: str = Field(default="1.0")
    quality_score: float = Field(..., ge=0.0, le=1.0)
    
    @validator('features')
    def validate_feature_count(cls, v):
        if len(v) != 590:
            raise ValueError(f"Expected 590 features, got {len(v)}")
        return v


class DeadLetterQueueMessage(BaseModel):
    """Schema for DLQ messages."""
    
    original_message: str
    error_type: str
    error_message: str
    timestamp: str
    kafka_topic: str
    kafka_partition: int
    kafka_offset: int
    retry_count: int = Field(default=0)


# ==========================================
# CONTRACT TESTS
# ==========================================

class TestRawDataMessageContract:
    """Contract tests for raw data messages."""
    
    def test_valid_raw_message(self):
        """Test valid raw data message."""
        message = {
            "batch_id": "batch_001",
            "sample_id": "sample_001",
            "timestamp": "2025-12-16T19:00:00Z",
            "features": {f"feature_{i}": 0.5 for i in range(590)},
            "target": 0,
            "metadata": {"source": "producer"}
        }
        
        # Should not raise
        validated = RawDataMessage(**message)
        assert validated.batch_id == "batch_001"
        assert len(validated.features) == 590
    
    def test_missing_required_fields(self):
        """Test message with missing required fields."""
        message = {
            "batch_id": "batch_001",
            # Missing sample_id, timestamp, features, target
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RawDataMessage(**message)
        
        errors = exc_info.value.errors()
        assert len(errors) > 0
    
    def test_invalid_feature_count(self):
        """Test message with wrong number of features."""
        message = {
            "batch_id": "batch_001",
            "sample_id": "sample_001",
            "timestamp": "2025-12-16T19:00:00Z",
            "features": {f"feature_{i}": 0.5 for i in range(100)},  # Only 100 features
            "target": 0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RawDataMessage(**message)
        
        assert "590 features" in str(exc_info.value)
    
    def test_invalid_target_value(self):
        """Test message with invalid target value."""
        message = {
            "batch_id": "batch_001",
            "sample_id": "sample_001",
            "timestamp": "2025-12-16T19:00:00Z",
            "features": {f"feature_{i}": 0.5 for i in range(590)},
            "target": 5  # Invalid: must be -1, 0, or 1
        }
        
        with pytest.raises(ValidationError):
            RawDataMessage(**message)
    
    def test_invalid_timestamp_format(self):
        """Test message with invalid timestamp."""
        message = {
            "batch_id": "batch_001",
            "sample_id": "sample_001",
            "timestamp": "invalid-timestamp",
            "features": {f"feature_{i}": 0.5 for i in range(590)},
            "target": 0
        }
        
        with pytest.raises(ValidationError):
            RawDataMessage(**message)
    
    def test_json_serialization(self):
        """Test message can be serialized to JSON."""
        message = RawDataMessage(
            batch_id="batch_001",
            sample_id="sample_001",
            timestamp="2025-12-16T19:00:00Z",
            features={f"feature_{i}": 0.5 for i in range(590)},
            target=0
        )
        
        # Should serialize without error
        json_str = message.json()
        assert isinstance(json_str, str)
        
        # Should deserialize back
        parsed = json.loads(json_str)
        assert parsed["batch_id"] == "batch_001"


class TestPreprocessedDataMessageContract:
    """Contract tests for preprocessed data messages."""
    
    def test_valid_preprocessed_message(self):
        """Test valid preprocessed message."""
        message = {
            "sample_id": "sample_001",
            "batch_id": "batch_001",
            "timestamp": "2025-12-16T19:00:00Z",
            "features": {f"feature_{i}": 0.5 for i in range(590)},
            "preprocessing_version": "1.0",
            "quality_score": 0.95
        }
        
        validated = PreprocessedDataMessage(**message)
        assert validated.quality_score == 0.95
    
    def test_invalid_quality_score(self):
        """Test message with invalid quality score."""
        message = {
            "sample_id": "sample_001",
            "batch_id": "batch_001",
            "timestamp": "2025-12-16T19:00:00Z",
            "features": {f"feature_{i}": 0.5 for i in range(590)},
            "quality_score": 1.5  # Invalid: must be 0-1
        }
        
        with pytest.raises(ValidationError):
            PreprocessedDataMessage(**message)


class TestDeadLetterQueueContract:
    """Contract tests for DLQ messages."""
    
    def test_valid_dlq_message(self):
        """Test valid DLQ message."""
        message = {
            "original_message": '{"invalid": "data"}',
            "error_type": "ValidationError",
            "error_message": "Invalid feature count",
            "timestamp": "2025-12-16T19:00:00Z",
            "kafka_topic": "secom-raw-data",
            "kafka_partition": 0,
            "kafka_offset": 12345,
            "retry_count": 1
        }
        
        validated = DeadLetterQueueMessage(**message)
        assert validated.retry_count == 1


# ==========================================
# SCHEMA COMPATIBILITY TESTS
# ==========================================

class TestSchemaEvolution:
    """Test schema evolution and backward compatibility."""
    
    def test_optional_metadata_field(self):
        """Test that metadata field is optional."""
        message = {
            "batch_id": "batch_001",
            "sample_id": "sample_001",
            "timestamp": "2025-12-16T19:00:00Z",
            "features": {f"feature_{i}": 0.5 for i in range(590)},
            "target": 0
            # No metadata field
        }
        
        validated = RawDataMessage(**message)
        assert validated.metadata == {}
    
    def test_default_preprocessing_version(self):
        """Test default preprocessing version."""
        message = {
            "sample_id": "sample_001",
            "batch_id": "batch_001",
            "timestamp": "2025-12-16T19:00:00Z",
            "features": {f"feature_{i}": 0.5 for i in range(590)},
            "quality_score": 0.95
            # No preprocessing_version
        }
        
        validated = PreprocessedDataMessage(**message)
        assert validated.preprocessing_version == "1.0"


# ==========================================
# INTEGRATION HELPERS
# ==========================================

def validate_kafka_message(topic: str, message: Dict[str, Any]) -> bool:
    """
    Validate Kafka message against appropriate schema.
    
    Args:
        topic: Kafka topic name
        message: Message dictionary
        
    Returns:
        True if valid, False otherwise
    """
    schema_map = {
        "secom-raw-data": RawDataMessage,
        "secom-preprocessed-data": PreprocessedDataMessage,
        "secom-dlq": DeadLetterQueueMessage
    }
    
    schema_class = schema_map.get(topic)
    if not schema_class:
        raise ValueError(f"Unknown topic: {topic}")
    
    try:
        schema_class(**message)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False


if __name__ == "__main__":
    # Run contract tests
    pytest.main([__file__, "-v"])
