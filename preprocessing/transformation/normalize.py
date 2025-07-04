"""Normalization transforms for time series data."""

from typing import Optional, Tuple, Union
import torch

from .base import BaseTransform


class MinMaxScaler(BaseTransform):
    """Min-max normalization transform applied per sample."""
    
    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        epsilon: float = 1e-8
    ):
        """Initialize the scaler.
        
        Args:
            feature_range: Target range for scaled data
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.register_buffer('min_vals', None)
        self.register_buffer('range_vals', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each sample independently.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Scaled data with same shape
        """
        # Compute min and max per sample
        self.min_vals = x.amin(dim=1, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        max_vals = x.amax(dim=1, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Handle constant sequences
        self.range_vals = max_vals - self.min_vals
        self.range_vals = torch.where(self.range_vals == 0, self.epsilon, self.range_vals)
        
        # Scale to [0, 1]
        x_std = (x - self.min_vals) / self.range_vals
        
        # Scale to feature range
        x_scaled = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return x_scaled
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Scaled data
            
        Returns:
            Data in original scale
        """
        if self.min_vals is None or self.range_vals is None:
            raise RuntimeError("Scaler must be applied to data before inverse transform")
            
        # Scale back to [0, 1]
        x_std = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        
        # Scale back to original range
        x_original = x_std * self.range_vals + self.min_vals
        
        return x_original


class StandardScaler(BaseTransform):
    """Standardization transform (zero mean, unit variance) applied per sample."""
    
    def __init__(self, epsilon: float = 1e-8):
        """Initialize the scaler.
        
        Args:
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize each sample independently.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Standardized data with same shape
        """
        # Compute mean and std per sample
        self.mean = x.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        self.std = x.std(dim=1, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Handle constant sequences
        self.std = torch.where(self.std == 0, self.epsilon, self.std)
        
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Standardized data
            
        Returns:
            Data in original scale
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be applied to data before inverse transform")
            
        return x * self.std + self.mean


class MeanScaler(BaseTransform):
    """Scaling by mean of absolute values, applied per sample."""
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        center: bool = True
    ):
        """Initialize the scaler.
        
        Args:
            epsilon: Small constant to avoid division by zero
            center: If True, subtract mean before scaling
        """
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer('mean', None)
        self.register_buffer('scale_factor', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each sample by its mean absolute value.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels) or (sequence_length, n_channels)
            
        Returns:
            Scaled data with same shape
        """
        # Compute mean of absolute values per sample
        abs_mean = torch.abs(x).mean(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Add epsilon to handle zero-valued sequences
        self.scale_factor = abs_mean + self.epsilon
        return x / self.scale_factor
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Scaled data
            
        Returns:
            Data in original scale
        """
        if self.scale_factor is None:
            raise RuntimeError("Scaler must be applied to data before inverse transform")
            
        return x * self.scale_factor 