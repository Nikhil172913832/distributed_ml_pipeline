"""Configuration management for distributed ML pipeline."""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
from pathlib import Path
import logging
import os
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str
    version: str
    architecture: str
    num_classes: int
    pretrained: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        logger.info(f"Model config initialized: {self.model_name} v{self.version}")


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    num_workers: int
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    find_unused_parameters: bool = False
    
    def __post_init__(self):
        """Validate distributed configuration."""
        if self.backend not in ["nccl", "gloo", "mpi"]:
            raise ValueError(f"Invalid backend: {self.backend}")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        
        # Set environment variables for distributed training
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['RANK'] = str(self.rank)
        
        logger.info(f"Distributed config: backend={self.backend}, world_size={self.world_size}")


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    dataset_path: str
    batch_size: int
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    drop_last: bool = True
    
    def __post_init__(self):
        """Validate data configuration."""
        total = self.train_split + self.val_split + self.test_split
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Train/val/test splits must sum to 1.0, got {total}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        logger.info(f"Data config: batch_size={self.batch_size}, num_workers={self.num_workers}")


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int
    learning_rate: float
    weight_decay: float = 0.0001
    momentum: float = 0.9
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_epochs: int = 5
    scheduler_type: str = "cosine"
    optimizer: str = "adamw"
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.scheduler_type not in ["cosine", "linear", "step", "exponential"]:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")
        if self.optimizer not in ["adam", "adamw", "sgd"]:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        logger.info(f"Training config: epochs={self.epochs}, lr={self.learning_rate}")


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    enable_mlflow: bool = False
    enable_prometheus: bool = False
    enable_wandb: bool = False
    log_level: str = "INFO"
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_best_only: bool = True
    metric_for_best: str = "val_loss"
    
    def __post_init__(self):
        """Validate monitoring configuration."""
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        
        # Create directories if they don't exist
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Monitoring config: mlflow={self.enable_mlflow}, wandb={self.enable_wandb}")


class ConfigManager:
    """Central configuration manager for the ML pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize all configurations
        self.model = self._init_model_config()
        self.distributed = self._init_distributed_config()
        self.data = self._init_data_config()
        self.training = self._init_training_config()
        self.monitoring = self._init_monitoring_config()
        
        logger.info("ConfigManager initialized successfully")
    
    def _load_config(self, path: Path) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            raise
    
    def _init_model_config(self) -> ModelConfig:
        """Initialize model configuration."""
        if 'model' in self.config:
            return ModelConfig(**self.config['model'])
        return ModelConfig(
            model_name="default_model",
            version="1.0.0",
            architecture="resnet50",
            num_classes=10
        )
    
    def _init_distributed_config(self) -> DistributedConfig:
        """Initialize distributed configuration."""
        if 'distributed' in self.config:
            return DistributedConfig(**self.config['distributed'])
        return DistributedConfig(num_workers=1)
    
    def _init_data_config(self) -> DataConfig:
        """Initialize data configuration."""
        if 'data' in self.config:
            return DataConfig(**self.config['data'])
        return DataConfig(
            dataset_path="data/",
            batch_size=32
        )
    
    def _init_training_config(self) -> TrainingConfig:
        """Initialize training configuration."""
        if 'training' in self.config:
            return TrainingConfig(**self.config['training'])
        return TrainingConfig(
            epochs=100,
            learning_rate=0.001
        )
    
    def _init_monitoring_config(self) -> MonitoringConfig:
        """Initialize monitoring configuration."""
        if 'monitoring' in self.config:
            return MonitoringConfig(**self.config['monitoring'])
        return MonitoringConfig()
    
    def save_config(self, output_path: Path):
        """Save current configuration to YAML file."""
        config_dict = {
            'model': asdict(self.model),
            'distributed': asdict(self.distributed),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'monitoring': asdict(self.monitoring)
        }
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            # Compute and save a simple config hash for versioning
            try:
                serialized = yaml.dump(config_dict, sort_keys=True).encode('utf-8')
                config_hash = hashlib.sha256(serialized).hexdigest()
                meta = {'config_hash': config_hash}
                # Save a small meta file alongside config
                with open(f"{output_path}.meta.json", 'w') as mf:
                    import json

                    json.dump(meta, mf)
                logger.info(f"Configuration saved to {output_path} (hash={config_hash})")
            except Exception:
                logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving config to {output_path}: {e}")
            raise
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in updates.items():
            if section == 'model' and hasattr(self, 'model'):
                for key, value in values.items():
                    setattr(self.model, key, value)
            elif section == 'distributed' and hasattr(self, 'distributed'):
                for key, value in values.items():
                    setattr(self.distributed, key, value)
            elif section == 'data' and hasattr(self, 'data'):
                for key, value in values.items():
                    setattr(self.data, key, value)
            elif section == 'training' and hasattr(self, 'training'):
                for key, value in values.items():
                    setattr(self.training, key, value)
            elif section == 'monitoring' and hasattr(self, 'monitoring'):
                for key, value in values.items():
                    setattr(self.monitoring, key, value)
        
        logger.info("Configuration updated from dictionary")
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary."""
        return {
            'model': asdict(self.model),
            'distributed': asdict(self.distributed),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'monitoring': asdict(self.monitoring)
        }
    
    def validate(self) -> bool:
        """Validate all configurations."""
        try:
            # All validation happens in __post_init__ of each config class
            logger.info("All configurations validated successfully")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
