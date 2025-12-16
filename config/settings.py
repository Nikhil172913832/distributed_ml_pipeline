"""
Unified Configuration Management using Pydantic Settings

This module provides a single source of truth for all configuration across the pipeline.
It replaces the scattered os.getenv() calls with type-safe, validated configuration.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class KafkaSettings(BaseSettings):
    """Kafka configuration settings."""
    
    model_config = SettingsConfigDict(env_prefix='KAFKA_', case_sensitive=False)
    
    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Comma-separated list of Kafka bootstrap servers"
    )
    raw_topic: str = Field(default="secom-raw-data", description="Topic for raw data")
    preprocessed_topic: str = Field(default="secom-preprocessed-data", description="Topic for preprocessed data")
    dlq_topic: str = Field(default="secom-dlq", description="Dead letter queue topic")
    consumer_group: str = Field(default="secom-preprocessor-group", description="Consumer group ID")
    auto_offset_reset: str = Field(default="earliest", description="Offset reset strategy")
    enable_auto_commit: bool = Field(default=True, description="Enable auto commit")
    max_poll_records: int = Field(default=100, description="Max records per poll")
    session_timeout_ms: int = Field(default=30000, description="Session timeout in ms")
    
    @field_validator('bootstrap_servers')
    @classmethod
    def validate_bootstrap_servers(cls, v: str) -> str:
        """Validate bootstrap servers format."""
        if not v or not any(c in v for c in [':', '.']):
            raise ValueError("Invalid bootstrap_servers format")
        return v


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""
    
    model_config = SettingsConfigDict(env_prefix='POSTGRES_', case_sensitive=False)
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="secom_pipeline", description="Database name")
    user: str = Field(default="ml_user", description="Database user")
    password: str = Field(default="ml_password", description="Database password")
    min_connections: int = Field(default=2, ge=1, description="Minimum pool connections")
    max_connections: int = Field(default=10, ge=1, description="Maximum pool connections")
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    
    model_config = SettingsConfigDict(env_prefix='REDIS_', case_sensitive=False)
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    socket_timeout: int = Field(default=5, description="Socket timeout in seconds")
    max_connections: int = Field(default=50, description="Max connection pool size")


class ProducerSettings(BaseSettings):
    """Data producer configuration."""
    
    model_config = SettingsConfigDict(env_prefix='PRODUCER_', case_sensitive=False)
    
    batch_size: int = Field(default=100, ge=1, le=10000, description="Batch size for generation")
    generation_interval_seconds: int = Field(default=5, ge=1, description="Generation interval")
    sdv_model_path: Path = Field(default=Path("./models/sdv_secom_raw.joblib"), description="SDV model path")


class ConsumerSettings(BaseSettings):
    """Data consumer configuration."""
    
    model_config = SettingsConfigDict(env_prefix='CONSUMER_', case_sensitive=False)
    
    preprocessing_pipeline_path: Path = Field(
        default=Path("./models/preprocessing_pipeline.joblib"),
        description="Preprocessing pipeline path"
    )
    batch_timeout_seconds: int = Field(default=10, ge=1, description="Batch collection timeout")


class InferenceSettings(BaseSettings):
    """Inference engine configuration."""
    
    model_config = SettingsConfigDict(env_prefix='INFERENCE_', case_sensitive=False)
    
    batch_size: int = Field(default=100, ge=1, le=10000, description="Inference batch size")
    poll_interval_seconds: int = Field(default=5, ge=1, description="Polling interval")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold")
    
    # Performance monitoring
    accuracy_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Accuracy threshold")
    f1_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="F1 score threshold")
    degradation_tolerance: float = Field(default=0.05, ge=0.0, le=1.0, description="Performance degradation tolerance")
    performance_window_hours: int = Field(default=1, ge=1, description="Performance calculation window")
    
    # Drift detection
    drift_detection_enabled: bool = Field(default=True, description="Enable drift detection")
    drift_check_interval_hours: int = Field(default=6, ge=1, description="Drift check interval")
    ks_test_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="KS test p-value threshold")
    drift_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Drift score threshold")
    baseline_days: int = Field(default=7, ge=1, description="Baseline data window in days")


class TrainingSettings(BaseSettings):
    """Model training configuration."""
    
    model_config = SettingsConfigDict(env_prefix='TRAIN_', case_sensitive=False)
    
    test_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set split ratio")
    random_state: int = Field(default=42, description="Random seed")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation folds")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 = all cores)")
    data_days: int = Field(default=7, ge=1, description="Days of data to use for training")
    max_samples: int = Field(default=50000, ge=100, description="Maximum training samples")


class RetrainerSettings(BaseSettings):
    """Continuous learning retrainer configuration."""
    
    model_config = SettingsConfigDict(env_prefix='RETRAINER_', case_sensitive=False)
    
    check_interval_seconds: int = Field(default=300, ge=60, description="Trigger check interval")
    auto_deploy: bool = Field(default=True, description="Auto-deploy best model")
    max_concurrent_training: int = Field(default=1, ge=1, description="Max concurrent training jobs")
    cooldown_hours: int = Field(default=6, ge=1, description="Cooldown period between retraining")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    model_config = SettingsConfigDict(env_prefix='MONITORING_', case_sensitive=False)
    
    prometheus_port: int = Field(default=8000, ge=1024, le=65535, description="Prometheus metrics port")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_mlflow: bool = Field(default=False, description="Enable MLflow tracking")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(default="secom_pipeline", description="MLflow experiment name")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v_upper


class PathSettings(BaseSettings):
    """File system paths configuration."""
    
    model_config = SettingsConfigDict(env_prefix='PATH_', case_sensitive=False)
    
    models_dir: Path = Field(default=Path("./models"), description="Models directory")
    artifacts_dir: Path = Field(default=Path("./models/artifacts"), description="Artifacts directory")
    logs_dir: Path = Field(default=Path("./logs"), description="Logs directory")
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    
    def ensure_directories(self) -> None:
        """Create directories if they don't exist."""
        for path in [self.models_dir, self.artifacts_dir, self.logs_dir, self.data_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """
    Main settings class that aggregates all configuration.
    
    Usage:
        from config.settings import get_settings
        
        settings = get_settings()
        kafka_servers = settings.kafka.bootstrap_servers
        db_conn = settings.database.connection_string
    """
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # Service-specific settings
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    producer: ProducerSettings = Field(default_factory=ProducerSettings)
    consumer: ConsumerSettings = Field(default_factory=ConsumerSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    retrainer: RetrainerSettings = Field(default_factory=RetrainerSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    
    # Global settings
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to ensure directories exist."""
        self.paths.ensure_directories()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()
    
    def get_service_config(self, service: str) -> BaseSettings:
        """Get configuration for a specific service."""
        service_map = {
            'producer': self.producer,
            'consumer': self.consumer,
            'inference': self.inference,
            'retrainer': self.retrainer,
        }
        if service not in service_map:
            raise ValueError(f"Unknown service: {service}")
        return service_map[service]


# Singleton instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get the global settings instance (singleton pattern).
    
    Args:
        reload: Force reload settings from environment
        
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None or reload:
        _settings = Settings()
    return _settings


def override_settings(**kwargs) -> Settings:
    """
    Override settings for testing purposes.
    
    Usage:
        settings = override_settings(
            kafka__bootstrap_servers="test:9092",
            database__host="testdb"
        )
    """
    return Settings(**kwargs)
