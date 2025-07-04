"""Base class for time series datasets."""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Base class for time series datasets."""
    
    def __init__(
        self,
        data: np.ndarray,
        targets: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize TimeSeriesDataset.
        
        Args:
            data: Time series data of shape (n_samples, sequence_length, n_channels)
            targets: Target values of shape (n_samples,) or (n_samples, n_targets)
            transform: Optional transform to be applied on the data
            target_transform: Optional transform to be applied on the targets
            metadata: Optional metadata dictionary
        """
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.targets = torch.as_tensor(targets) if targets is not None else None
        self.transform = transform
        self.target_transform = target_transform
        self.metadata = metadata or {}
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (data, target) where target is None if no targets are available
        """
        x = self.data[idx]
        
        if self.transform is not None:
            x = self.transform(x)
            
        if self.targets is None:
            return x
            
        y = self.targets[idx]
        if self.target_transform is not None:
            y = self.target_transform(y)
            
        return x, y
    
    @property
    def sequence_length(self) -> int:
        """Return the length of each time series sequence."""
        return self.data.shape[1]
    
    @property
    def n_channels(self) -> int:
        """Return the number of channels in the data."""
        return self.data.shape[2] if len(self.data.shape) > 2 else 1
    

    def get_metadata(self, key: str) -> Any:
        """Get metadata value by key."""
        return self.metadata.get(key) 