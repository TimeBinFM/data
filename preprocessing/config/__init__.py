"""Configuration management for the preprocessing pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str = MISSING  # Dataset name or repo ID
    repo_id: Optional[str] = None  # Optional specific repo ID (e.g., "username/repo")
    files: Optional[List[str]] = None  # Files to download (single file or list)
    subset: Optional[str] = None  # Dataset subset (if using predefined datasets)
    cache_dir: str = "data/cache"
    split: str = "train"
    revision: Optional[str] = None  # Git revision (branch, tag, or commit hash)


@dataclass
class TransformConfig:
    """Configuration for a single transform."""
    name: str = MISSING  # Transform class name
    params: Dict[str, Any] = field(default_factory=dict)  # Transform parameters


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps."""
    transforms: List[TransformConfig] = field(default_factory=list)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    enabled: bool = False
    methods: List[str] = field(default_factory=list)
    linear_combo_ratio: float = 0.5
    jitter_sigma: float = 0.01
    time_warp_sigma: float = 0.1


@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True


@dataclass
class Config:
    """Main configuration class."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


# Register config schema with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# Make config classes available at package level
__all__ = [
    'Config',
    'DatasetConfig',
    'PreprocessingConfig',
    'TransformConfig',
    'AugmentationConfig',
    'DataLoaderConfig',
] 