"""Data loading pipeline for time series preprocessing."""

from typing import Optional, List, Union, Callable, Dict, Any, Tuple, Type
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing.config import Config
from preprocessing.transformation import BaseTransform, get_transform
from preprocessing.augmentation.linear_combo import LinearCombination

class TimeSeriesPipeline:
    """Pipeline for loading and preprocessing time series data.
    
    All transforms are applied at batch level and are stateless. Transform parameters
    are stored per batch to support inverse transformation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the pipeline.
        
        Args:
            config: Pipeline configuration. If None, no transforms or augmentations will be applied.
        """
        self.config = config
        self.transforms = self._build_transforms() if config else []
        self.augmentations = self._build_augmentations() if config else []
        
    def _build_transforms(self) -> List[Tuple[Type[BaseTransform], Dict[str, Any]]]:
        """Build preprocessing transforms based on config.
        
        Returns:
            List of tuples (transform_class, transform_params) where transform_class
            contains static transform methods.
        """
        transforms = []
        
        if hasattr(self.config.preprocessing, 'transforms'):
            for transform_config in self.config.preprocessing.transforms:
                transform_cls = get_transform(transform_config.name)
                transforms.append((transform_cls, transform_config.params))
            
        return transforms
        
    def _build_augmentations(self) -> List[Callable]:
        """Build augmentation transforms based on config.
        
        Returns:
            list: List of augmentation callables
        """
        augmentations = []
        
        if (hasattr(self.config, 'augmentation') and 
            self.config.augmentation.enabled and 
            "linear_combo" in self.config.augmentation.methods):
            augmentations.append(
                LinearCombination(
                    alpha=self.config.augmentation.linear_combo_ratio,
                    beta=self.config.augmentation.linear_combo_ratio
                )
            )
                
        return augmentations

    def apply_transforms(self, batch: Union[torch.Tensor, tuple]) -> Tuple[Union[torch.Tensor, tuple], List[Dict[str, Any]]]:
        """Apply transforms to a batch of data.
        
        All transforms are applied at batch level. Each transform is a static method that takes
        a tensor of shape (batch_size, sequence_length, n_channels) and returns both the
        transformed tensor and parameters needed for inverse transformation.
        
        Args:
            batch: Input batch, either a tensor of shape (batch_size, sequence_length, n_channels)
                  or tuple (x, y) where x has that shape
            
        Returns:
            tuple:
                - Transformed batch with same structure as input
                - List of transform parameters for inverse transformation
        """
        if not self.transforms:
            return batch, []

        # Extract x and check if we have labels
        has_labels = isinstance(batch, tuple)
        x = batch[0] if has_labels else batch
        
        # Apply transforms and collect parameters
        transform_params = []
        for transform_cls, params in self.transforms:
            x, stats = transform_cls.transform(x, **params)
            transform_params.append({
                'name': transform_cls.__name__,
                'params': stats
            })
                
        # Return in the same format as input
        return (x, batch[1]) if has_labels else x, transform_params

    def inverse_transforms(self, batch: Union[torch.Tensor, tuple], transform_params: List[Dict[str, Any]]) -> Union[torch.Tensor, tuple]:
        """Apply inverse transforms to recover original scale.
        
        Args:
            batch: Transformed batch, either a tensor of shape (batch_size, sequence_length, n_channels)
                  or tuple (x, y) where x has that shape
            transform_params: List of transform parameters from forward pass
            
        Returns:
            Batch with original scaling, same structure as input
        """
        if not transform_params:
            raise ValueError("transform_params cannot be empty when calling inverse_transforms")

        # Extract x and check if we have labels
        has_labels = isinstance(batch, tuple)
        x = batch[0] if has_labels else batch
        
        # Apply inverse transforms in reverse order
        for transform_info in reversed(transform_params):
            transform_cls = get_transform(transform_info['name'])
            x = transform_cls.inverse_transform(x, transform_info['params'])
                
        # Return in the same format as input
        return (x, batch[1]) if has_labels else x

    def apply_augmentations(self, batch: Union[torch.Tensor, tuple]) -> Union[torch.Tensor, tuple]:
        """Apply augmentations to a batch of data.
        
        Args:
            batch: Input batch, either a tensor of shape (batch_size, sequence_length, n_channels)
                  or tuple (x, y) where x has that shape
            
        Returns:
            Augmented batch with same structure as input
            
        Raises:
            NotImplementedError: This method is not implemented yet
        """
        raise NotImplementedError("Batch-level augmentations not implemented yet")
        
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
            DataLoader with the preprocessing pipeline. If no transforms or augmentations
            are configured, returns a regular DataLoader.
        """
        # Create base dataloader with pass-through dataset
        dataloader = DataLoader(
            dataset,  # Use original dataset directly
            batch_size=batch_size or (self.config.dataloader.batch_size if self.config else 32),
            shuffle=shuffle if shuffle is not None else (self.config.dataloader.shuffle if self.config else True),
            num_workers=self.config.dataloader.num_workers if self.config else 0,
            pin_memory=self.config.dataloader.pin_memory if self.config else False
        )
        
        # Only wrap if we have transforms or augmentations
        if self.transforms or self.augmentations:
            return TransformedDataLoader(dataloader, self)
        return dataloader


class TransformedDataLoader:
    """DataLoader wrapper that applies transforms and augmentations at batch level."""
    
    def __init__(self, dataloader: DataLoader, pipeline: TimeSeriesPipeline):
        """Initialize the transformed dataloader.
        
        Args:
            dataloader: Base dataloader
            pipeline: TimeSeriesPipeline instance containing transforms
        """
        self.dataloader = dataloader
        self.pipeline = pipeline
        
    def __iter__(self):
        """Iterate over batches, applying transforms and augmentations to each batch."""
        for batch in self.dataloader:
            try:
                # First apply augmentations
                batch = self.pipeline.apply_augmentations(batch)
            except NotImplementedError:
                pass
                
            # Then apply transforms (always) and get transform parameters
            # Note: batch[0] is the tensor we want to transform, since we're using TensorDataset
            transformed_batch, transform_params = self.pipeline.apply_transforms(batch[0])
            
            yield transformed_batch, transform_params
            
    def __len__(self):
        """Return the number of batches."""
        return len(self.dataloader) 