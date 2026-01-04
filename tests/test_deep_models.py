"""
Tests for advanced deep learning models.
"""

import pytest
import numpy as np
import torch
from pipeline.deep_models import (
    DeepClassifier, Autoencoder, FocalLoss,
    AttentionLayer, SECOMDataset, ModelTrainer
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_secom_dataset(sample_data):
    X, y = sample_data
    dataset = SECOMDataset(X, y)
    
    assert len(dataset) == 100
    x_sample, y_sample = dataset[0]
    assert x_sample.shape == (50,)
    assert isinstance(y_sample, torch.Tensor)


def test_focal_loss():
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    inputs = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    
    loss = criterion(inputs, targets)
    assert loss.item() > 0


def test_attention_layer():
    attention = AttentionLayer(input_dim=50, attention_dim=32)
    x = torch.randn(10, 50)
    
    output, weights = attention(x)
    assert output.shape == (10, 50)
    assert weights.shape == (10, 1)


def test_deep_classifier():
    model = DeepClassifier(
        input_dim=50,
        hidden_dims=[128, 64],
        dropout=0.3,
        use_attention=False
    )
    
    x = torch.randn(10, 50)
    output = model(x)
    
    assert output.shape == (10, 2)


def test_deep_classifier_with_attention():
    model = DeepClassifier(
        input_dim=50,
        hidden_dims=[128, 64],
        dropout=0.3,
        use_attention=True
    )
    
    x = torch.randn(10, 50)
    output = model(x)
    
    assert output.shape == (10, 2)


def test_autoencoder():
    model = Autoencoder(input_dim=50, latent_dim=16)
    x = torch.randn(10, 50)
    
    reconstructed, latent = model(x)
    
    assert reconstructed.shape == (10, 50)
    assert latent.shape == (10, 16)


def test_autoencoder_reconstruction_error():
    model = Autoencoder(input_dim=50, latent_dim=16)
    x = np.random.randn(10, 50).astype(np.float32)
    
    errors = model.get_reconstruction_error(x)
    
    assert errors.shape == (10,)
    assert np.all(errors >= 0)


def test_model_trainer_initialization(sample_data):
    X, y = sample_data
    
    model = DeepClassifier(input_dim=50, hidden_dims=[64, 32])
    trainer = ModelTrainer(model, use_focal_loss=True)
    
    assert trainer.model is not None
    assert isinstance(trainer.criterion, FocalLoss)


def test_model_training_single_epoch(sample_data):
    X, y = sample_data
    
    model = DeepClassifier(input_dim=50, hidden_dims=[32])
    trainer = ModelTrainer(model, use_focal_loss=False)
    
    dataset = SECOMDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = trainer.train_epoch(loader, optimizer)
    
    assert loss > 0
