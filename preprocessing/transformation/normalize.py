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
        self._min_vals = None
        self._range_vals = None
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each sample independently.
        
        Args:
            x: Input data of shape (sequence_length, n_channels) or 
               (batch_size, sequence_length, n_channels)
            
        Returns:
            Scaled data with same shape
        """
        # Compute min and max per sample
        self._min_vals = x.amin(dim=-2, keepdim=True)  # Shape: (..., 1, n_channels)
        max_vals = x.amax(dim=-2, keepdim=True)  # Shape: (..., 1, n_channels)
        
        # Handle constant sequences
        self._range_vals = max_vals - self._min_vals
        self._range_vals = torch.where(self._range_vals == 0, self.epsilon, self._range_vals)
        
        # Scale to [0, 1]
        x_std = (x - self._min_vals) / self._range_vals
        
        # Scale to feature range
        x_scaled = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        
        return x_scaled
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Scaled data of shape (sequence_length, n_channels) or 
               (batch_size, sequence_length, n_channels)
            
        Returns:
            Data in original scale
        """
        if self._min_vals is None or self._range_vals is None:
            raise RuntimeError("Scaler must be applied to data before inverse transform")
            
        # Scale back to [0, 1]
        x_std = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        
        # Scale back to original range
        x_original = x_std * self._range_vals + self._min_vals
        
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
        self._mean = None
        self._std = None
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize each sample independently.
        
        Args:
            x: Input data of shape (sequence_length, n_channels) or 
               (batch_size, sequence_length, n_channels)
            
        Returns:
            Standardized data with same shape
        """
        # Compute mean and std per sample
        self._mean = x.mean(dim=-2, keepdim=True)  # Shape: (..., 1, n_channels)
        self._std = x.std(dim=-2, keepdim=True)  # Shape: (..., 1, n_channels)
        
        # Handle constant sequences
        self._std = torch.where(self._std == 0, self.epsilon, self._std)
        
        return (x - self._mean) / self._std
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Standardized data of shape (sequence_length, n_channels) or 
               (batch_size, sequence_length, n_channels)
            
        Returns:
            Data in original scale
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("Scaler must be applied to data before inverse transform")
            
        return x * self._std + self._mean


class MeanScaler(BaseTransform):
    """Scaling by mean of absolute values, applied per sample."""
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        center: bool = False  #TODO: or True?
    ):
        """Initialize the scaler.
        
        Args:
            epsilon: Small constant to avoid division by zero
            center: If True, subtract mean before scaling
        """
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self._mean = None
        self._scale_factor = None
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each sample by its mean absolute value.
        
        Args:
            x: Input data of shape (sequence_length, n_channels) or 
               (batch_size, sequence_length, n_channels)
            
        Returns:
            Scaled data with same shape
        """
        # Compute mean for centering if needed
        if self.center:
            self._mean = x.mean(dim=-2, keepdim=True)  # Shape: (..., 1, n_channels)
            x = x - self._mean
            
        # Compute mean of absolute values per sample
        abs_mean = torch.abs(x).mean(dim=-2, keepdim=True)  # Shape: (..., 1, n_channels)
        
        # Add epsilon to handle zero-valued sequences
        self._scale_factor = abs_mean + self.epsilon
        
        return x / self._scale_factor
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Scaled data of shape (sequence_length, n_channels) or 
               (batch_size, sequence_length, n_channels)
            
        Returns:
            Data in original scale
        """
        if self._scale_factor is None:
            raise RuntimeError("Scaler must be applied to data before inverse transform")
            
        x = x * self._scale_factor
        
        if self.center and self._mean is not None:
            x = x + self._mean
            
        return x 