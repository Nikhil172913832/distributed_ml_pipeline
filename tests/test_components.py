"""Test suite for distributed ML pipeline components."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    ConfigManager,
    ModelConfig,
    DistributedConfig,
    DataConfig,
    TrainingConfig,
    MonitoringConfig
)
from pipeline.checkpoint import CheckpointManager, EarlyStopping
from pipeline.data_pipeline import DistributedDataPipeline, AugmentationPipeline, DatasetSplitter


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=100, input_dim=10, num_classes=2):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1,)).squeeze()
        return {'input': x, 'target': y}


class TestConfigManagement:
    """Test configuration management."""
    
    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(
            model_name="test_model",
            version="1.0",
            architecture="simple",
            num_classes=2
        )
        assert config.model_name == "test_model"
        assert config.num_classes == 2
    
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        with pytest.raises(ValueError):
            ModelConfig(
                model_name="test",
                version="1.0",
                architecture="simple",
                num_classes=-1  # Invalid
            )
    
    def test_distributed_config_creation(self):
        """Test DistributedConfig creation."""
        config = DistributedConfig(num_workers=4, backend="gloo")
        assert config.num_workers == 4
        assert config.backend == "gloo"
    
    def test_data_config_validation(self):
        """Test DataConfig validation."""
        with pytest.raises(ValueError):
            DataConfig(
                dataset_path="data/",
                batch_size=32,
                train_split=0.5,
                val_split=0.3,
                test_split=0.3  # Sum > 1.0
            )
    
    def test_training_config_creation(self):
        """Test TrainingConfig creation."""
        config = TrainingConfig(
            epochs=100,
            learning_rate=0.001,
            use_amp=True
        )
        assert config.epochs == 100
        assert config.use_amp is True
    
    def test_config_manager(self):
        """Test ConfigManager."""
        manager = ConfigManager()
        assert manager.model is not None
        assert manager.training is not None
        assert manager.data is not None


class TestCheckpointManager:
    """Test checkpoint management."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model(self):
        """Create simple model."""
        return SimpleModel()
    
    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer."""
        return torch.optim.Adam(model.parameters())
    
    def test_checkpoint_save_load(self, temp_dir, model, optimizer):
        """Test checkpoint save and load."""
        manager = CheckpointManager(temp_dir, keep_last_n=3)
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            epoch=1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metrics={'val_loss': 0.5}
        )
        
        assert checkpoint_path is not None
        assert checkpoint_path.exists()
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint(checkpoint_path)
        assert checkpoint['epoch'] == 1
        assert 'model_state_dict' in checkpoint
    
    def test_checkpoint_cleanup(self, temp_dir, model, optimizer):
        """Test checkpoint cleanup."""
        manager = CheckpointManager(temp_dir, keep_last_n=2)
        
        # Save multiple checkpoints
        for epoch in range(1, 6):
            manager.save_checkpoint(
                epoch=epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )
        
        # Should only keep last 2
        checkpoints = manager.get_checkpoint_list()
        assert len(checkpoints) == 2
    
    def test_best_checkpoint(self, temp_dir, model, optimizer):
        """Test best model tracking."""
        manager = CheckpointManager(
            temp_dir,
            save_best=True,
            metric_for_best='val_loss',
            mode='min'
        )
        
        # Save checkpoints with different metrics
        manager.save_checkpoint(
            epoch=1,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metrics={'val_loss': 0.5}
        )
        
        manager.save_checkpoint(
            epoch=2,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            metrics={'val_loss': 0.3}  # Better
        )
        
        # Check best model was saved
        best_path = temp_dir / 'best_model.pt'
        assert best_path.exists()
        
        # Load and verify
        best = manager.load_best_checkpoint()
        assert best['metrics']['val_loss'] == 0.3


class TestEarlyStopping:
    """Test early stopping."""
    
    def test_early_stopping_min(self):
        """Test early stopping with min mode."""
        early_stop = EarlyStopping(patience=3, mode='min')
        
        # Improving
        assert not early_stop(1.0)
        assert not early_stop(0.8)
        assert not early_stop(0.6)
        
        # Not improving
        assert not early_stop(0.7)
        assert not early_stop(0.8)
        assert not early_stop(0.9)
        assert early_stop(1.0)  # Should trigger
    
    def test_early_stopping_max(self):
        """Test early stopping with max mode."""
        early_stop = EarlyStopping(patience=2, mode='max')
        
        # Improving
        assert not early_stop(0.5)
        assert not early_stop(0.7)
        
        # Not improving
        assert not early_stop(0.6)
        assert not early_stop(0.5)
        assert early_stop(0.4)  # Should trigger


class TestDataPipeline:
    """Test data pipeline."""
    
    @pytest.fixture
    def dataset(self):
        """Create simple dataset."""
        return SimpleDataset(size=100)
    
    def test_data_pipeline_creation(self, dataset):
        """Test data pipeline creation."""
        pipeline = DistributedDataPipeline(
            dataset=dataset,
            batch_size=16,
            num_workers=0,  # Use 0 for testing
            rank=0,
            world_size=1
        )
        
        dataloader = pipeline.get_dataloader(shuffle=True)
        assert dataloader is not None
        assert len(dataloader) > 0
    
    def test_data_pipeline_batch_size(self, dataset):
        """Test batch size configuration."""
        batch_size = 16
        pipeline = DistributedDataPipeline(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=0,
            rank=0,
            world_size=1,
            drop_last=False
        )
        
        dataloader = pipeline.get_dataloader()
        batch = next(iter(dataloader))
        assert batch['input'].size(0) <= batch_size
    
    def test_augmentation_pipeline(self):
        """Test augmentation pipeline."""
        aug = AugmentationPipeline(
            augmentations=[
                lambda x: AugmentationPipeline.random_noise(x, 0.01),
                lambda x: AugmentationPipeline.random_scale(x, (0.9, 1.1))
            ],
            probability=1.0
        )
        
        sample = torch.randn(10)
        augmented = aug(sample)
        assert augmented.shape == sample.shape
    
    def test_dataset_splitter(self, dataset):
        """Test dataset splitting."""
        train_ds, val_ds, test_ds = DatasetSplitter.split_dataset(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(dataset)


class TestTrainer:
    """Test trainer components."""
    
    @pytest.fixture
    def model(self):
        """Create simple model."""
        return SimpleModel()
    
    @pytest.fixture
    def dataset(self):
        """Create simple dataset."""
        return SimpleDataset(size=32)
    
    @pytest.fixture
    def dataloader(self, dataset):
        """Create dataloader."""
        return torch.utils.data.DataLoader(dataset, batch_size=8)
    
    def test_model_forward_pass(self, model):
        """Test model forward pass."""
        x = torch.randn(8, 10)
        output = model(x)
        assert output.shape == (8, 2)
    
    def test_training_step(self, model, dataloader):
        """Test single training step."""
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        batch = next(iter(dataloader))
        
        # Forward pass
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0


@pytest.mark.parametrize("batch_size,num_workers", [
    (16, 0),
    (32, 0),
    (64, 0),
])
def test_different_batch_sizes(batch_size, num_workers):
    """Test different batch size configurations."""
    dataset = SimpleDataset(size=100)
    pipeline = DistributedDataPipeline(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        rank=0,
        world_size=1
    )
    
    dataloader = pipeline.get_dataloader()
    assert dataloader is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
