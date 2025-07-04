"""Data loading pipeline for time series preprocessing."""

from typing import Optional, List, Union, Callable
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..transformation import BaseTransform, get_transform
from ..augmentation.linear_combo import LinearCombination


class TimeSeriesPipeline:
    """Pipeline for loading and preprocessing time series data."""
    
    def __init__(self, config: Config):
        """Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.transforms = self._build_transforms()
        self.augmentations = self._build_augmentations()
        
    def _build_transforms(self) -> List[BaseTransform]:
        """Build preprocessing transforms based on config.
        
        Returns:
            List of transform instances that inherit from BaseTransform
        """
        transforms = []
        
        for transform_config in self.config.preprocessing.transforms:
            transform = get_transform(
                name=transform_config.name,
                **transform_config.params
            )
            transforms.append(transform)
            
        return transforms
        
    def _build_augmentations(self) -> List[Callable]:
        """Build augmentation transforms based on config.
        
        Returns:
            list: List of augmentation callables
        """
        augmentations = []
        
        if self.config.augmentation.enabled:
            if "linear_combo" in self.config.augmentation.methods:
                augmentations.append(
                    LinearCombination(
                        alpha=self.config.augmentation.linear_combo_ratio,
                        beta=self.config.augmentation.linear_combo_ratio
                    )
                )
                
        return augmentations
        
    def fit_transforms(self, dataset: Dataset) -> None:
        """Fit preprocessing transforms on the dataset.
        
        Args:
            dataset: Input dataset
        """
        # Get all data at once for fitting
        if isinstance(dataset[0], tuple):
            x = torch.stack([x for x, _ in dataset])
        else:
            x = torch.stack([x for x in dataset])
            
        # Fit each transform sequentially
        for transform in self.transforms:
            if hasattr(transform, 'fit'):
                transform.fit(x)
                x = transform(x)
                
    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: Optional[bool] = None,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """Create a DataLoader with the configured pipeline.
        
        Args:
            dataset: Input dataset
            shuffle: Whether to shuffle the data (overrides config if provided)
            batch_size: Batch size (overrides config if provided)
            
        Returns:
            DataLoader with the preprocessing pipeline
        """
        # Apply preprocessing transforms to the dataset
        transformed_dataset = TransformedDataset(
            dataset,
            transforms=self.transforms,
            augmentations=self.augmentations
        )
        
        return DataLoader(
            transformed_dataset,
            batch_size=batch_size or self.config.dataloader.batch_size,
            shuffle=shuffle if shuffle is not None else self.config.dataloader.shuffle,
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory
        )


class TransformedDataset(Dataset):
    """Dataset wrapper that applies transforms and augmentations."""
    
    def __init__(
        self,
        dataset: Dataset,
        transforms: Optional[List[BaseTransform]] = None,
        augmentations: Optional[List[Callable]] = None
    ):
        """Initialize the transformed dataset.
        
        Args:
            dataset: Base dataset
            transforms: List of preprocessing transforms
            augmentations: List of augmentation transforms
        """
        self.dataset = dataset
        self.transforms = transforms or []
        self.augmentations = augmentations or []
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Union[torch.Tensor, tuple]:
        # Get item from base dataset
        item = self.dataset[idx]
        
        # Apply transforms
        if isinstance(item, tuple):
            x, y = item
            for transform in self.transforms:
                x = transform(x)
            
            # Apply augmentations if any
            if self.augmentations and torch.rand(1).item() < 0.5:  # 50% chance to augment
                for aug in self.augmentations:
                    x, y = aug(x.unsqueeze(0), y.unsqueeze(0))
                    x, y = x.squeeze(0), y.squeeze(0)
                    
            return x, y
        else:
            x = item
            for transform in self.transforms:
                x = transform(x)
                
            # Apply augmentations if any
            if self.augmentations and torch.rand(1).item() < 0.5:
                for aug in self.augmentations:
                    x = aug(x.unsqueeze(0)).squeeze(0)
                    
            return x 