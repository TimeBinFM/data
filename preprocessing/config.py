"""Configuration management for the preprocessing pipeline."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str = MISSING
    subset: Optional[str] = None
    cache_dir: str = "data/cache"
    split: str = "train"


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
    methods: List[str] = MISSING
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
    dataset: DatasetConfig = DatasetConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    dataloader: DataLoaderConfig = DataLoaderConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config) 