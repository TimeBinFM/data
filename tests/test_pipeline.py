"""Tests for the data loading pipeline."""

import pytest
import torch
import numpy as np
from torch.utils.data import Dataset

from preprocessing.config import (
    Config, DatasetConfig, PreprocessingConfig,
    AugmentationConfig, DataLoaderConfig, TransformConfig
)
from preprocessing.dataloader.pipeline import TimeSeriesPipeline


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, n_samples=100, seq_length=10, n_channels=1):
        self.data = torch.randn(n_samples, seq_length, n_channels)
        self.targets = torch.randint(0, 2, (n_samples,))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_pipeline_initialization():
    """Test pipeline initialization with default config."""
    config = Config(
        dataset=DatasetConfig(name="dummy"),
        preprocessing=PreprocessingConfig(
            transforms=[
                TransformConfig(
                    name="MinMaxScaler",
                    params={"feature_range": [0, 1]}
                )
            ]
        ),
        augmentation=AugmentationConfig(enabled=False),
        dataloader=DataLoaderConfig()
    )
    
    pipeline = TimeSeriesPipeline(config)
    assert len(pipeline.transforms) == 1  # MinMaxScaler
    assert len(pipeline.augmentations) == 0


def test_pipeline_with_augmentation():
    """Test pipeline with augmentation enabled."""
    config = Config(
        dataset=DatasetConfig(name="dummy"),
        preprocessing=PreprocessingConfig(
            transforms=[
                TransformConfig(
                    name="MinMaxScaler",
                    params={"feature_range": [0, 1]}
                )
            ]
        ),
        augmentation=AugmentationConfig(
            enabled=True,
            methods=["linear_combo"]
        ),
        dataloader=DataLoaderConfig()
    )
    
    pipeline = TimeSeriesPipeline(config)
    assert len(pipeline.transforms) == 1
    assert len(pipeline.augmentations) == 1


def test_transform_chain():
    """Test pipeline with multiple transforms."""
    config = Config(
        dataset=DatasetConfig(name="dummy"),
        preprocessing=PreprocessingConfig(
            transforms=[
                TransformConfig(
                    name="StandardScaler",
                    params={"epsilon": 1e-8}
                ),
                TransformConfig(
                    name="MinMaxScaler",
                    params={"feature_range": [-1, 1]}
                )
            ]
        ),
        augmentation=AugmentationConfig(enabled=False),
        dataloader=DataLoaderConfig()
    )
    
    dataset = DummyDataset()
    pipeline = TimeSeriesPipeline(config)
    
    # Fit transforms on the dataset
    pipeline.fit_transforms(dataset)
    
    # Create dataloader and get batch
    dataloader = pipeline.create_dataloader(dataset)
    batch = next(iter(dataloader))
    x, _ = batch
    
    # Check that data is scaled to [-1, 1]
    assert torch.all(x >= -1) and torch.all(x <= 1)
    

def test_dataloader_creation():
    """Test dataloader creation and batch generation."""
    config = Config(
        dataset=DatasetConfig(name="dummy"),
        preprocessing=PreprocessingConfig(
            transforms=[
                TransformConfig(
                    name="MinMaxScaler",
                    params={"feature_range": [0, 1]}
                )
            ]
        ),
        augmentation=AugmentationConfig(enabled=False),
        dataloader=DataLoaderConfig(batch_size=32)
    )
    
    dataset = DummyDataset()
    pipeline = TimeSeriesPipeline(config)
    
    # Fit transforms on the dataset
    pipeline.fit_transforms(dataset)
    
    # Create dataloader
    dataloader = pipeline.create_dataloader(dataset)
    
    # Check batch
    batch = next(iter(dataloader))
    x, y = batch
    
    assert x.shape == (32, 10, 1)  # (batch_size, seq_length, n_channels)
    assert y.shape == (32,)  # (batch_size,)
    assert torch.all(x >= 0) and torch.all(x <= 1)  # Check normalization


@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_different_batch_sizes(batch_size):
    """Test pipeline with different batch sizes."""
    config = Config(
        dataset=DatasetConfig(name="dummy"),
        preprocessing=PreprocessingConfig(
            transforms=[
                TransformConfig(
                    name="MinMaxScaler",
                    params={"feature_range": [0, 1]}
                )
            ]
        ),
        augmentation=AugmentationConfig(enabled=False),
        dataloader=DataLoaderConfig(batch_size=batch_size)
    )
    
    dataset = DummyDataset()
    pipeline = TimeSeriesPipeline(config)
    dataloader = pipeline.create_dataloader(dataset)
    
    batch = next(iter(dataloader))
    assert batch[0].shape[0] == batch_size 