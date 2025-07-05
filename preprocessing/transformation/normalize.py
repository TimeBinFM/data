"""Normalization transforms for time series data."""

from typing import Tuple
import torch

from .base import BaseTransform


class MinMaxScaler(BaseTransform):
    """Min-max normalization transform."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        """Initialize the scaler.
        
        Args:
            feature_range: Target range for scaled data
        """
        self.feature_range = feature_range
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Scale data to target range.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Scaled data with same shape
        """
        # Get dtype-specific epsilon
        eps = torch.finfo(x.dtype).eps
        
        # Compute min and max per sample
        min_vals = x.amin(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        max_vals = x.amax(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Handle constant sequences
        range_vals = max_vals - min_vals
        range_vals = torch.where(range_vals == 0, eps, range_vals)
        
        # Scale to [0, 1]
        x_std = (x - min_vals) / range_vals
        
        # Scale to feature range
        x_scaled = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return x_scaled


class StandardScaler(BaseTransform):
    """Standardization transform (zero mean, unit variance)."""
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize each sample independently.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Standardized data with same shape
        """
        # Get dtype-specific epsilon
        eps = torch.finfo(x.dtype).eps
        
        # Compute mean and std per sample
        mean = x.mean(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        std = x.std(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Handle constant sequences
        std = torch.where(std == 0, eps, std)
        
        return (x - mean) / std


class MeanScaler(BaseTransform):
    """Scaling by mean of absolute values."""
    
    def __init__(self, center: bool = False):
        """Initialize the scaler.
        
        Args:
            center: If True, subtract mean before scaling
        """
        self.center = center
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each sample by its mean absolute value.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Scaled data with same shape
        """
        # Get dtype-specific epsilon
        eps = torch.finfo(x.dtype).eps
        
        # Compute mean for centering if needed
        if self.center:
            mean = x.mean(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
            x = x - mean
            
        # Compute mean of absolute values per sample
        abs_mean = torch.abs(x).mean(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Add epsilon to handle zero-valued sequences
        scale_factor = abs_mean + eps
        
        return x / scale_factor 