"""Normalization transforms for time series data."""

from typing import Dict, Any, Tuple
import torch

from .base import BaseTransform


class MinMaxScaler(BaseTransform):
    """Min-max normalization transform."""
    
    @staticmethod
    def transform(x: torch.Tensor, feature_range: Tuple[float, float] = (0, 1)) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Scale data to target range.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            feature_range: Target range for scaled data
            
        Returns:
            tuple:
                - Scaled data with same shape
                - Dictionary with parameters for inverse transform
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
        x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        # Store parameters for inverse transform
        stats = {
            'min_vals': min_vals,
            'range_vals': range_vals,
            'feature_range': feature_range
        }
        
        return x_scaled, stats
    
    @staticmethod
    def inverse_transform(x: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Scaled data of shape (batch_size, sequence_length, n_channels)
            stats: Dictionary with parameters from forward transform
            
        Returns:
            Data in original scale
        """
        min_vals = stats['min_vals']
        range_vals = stats['range_vals']
        feature_range = stats['feature_range']
        
        # Scale back to [0, 1]
        x_std = (x - feature_range[0]) / (feature_range[1] - feature_range[0])
        
        # Scale back to original range
        x_original = x_std * range_vals + min_vals
        
        return x_original


class StandardScaler(BaseTransform):
    """Standardization transform (zero mean, unit variance)."""
    
    @staticmethod
    def transform(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Standardize each sample independently.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            tuple:
                - Standardized data with same shape
                - Dictionary with parameters for inverse transform
        """
        # Get dtype-specific epsilon
        eps = torch.finfo(x.dtype).eps
        
        # Compute mean and std per sample
        mean = x.mean(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        std = x.std(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Handle constant sequences
        std = torch.where(std == 0, eps, std)
        
        # Store parameters for inverse transform
        stats = {
            'mean': mean,
            'std': std
        }
        
        return (x - mean) / std, stats
    
    @staticmethod
    def inverse_transform(x: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Standardized data of shape (batch_size, sequence_length, n_channels)
            stats: Dictionary with parameters from forward transform
            
        Returns:
            Data in original scale
        """
        return x * stats['std'] + stats['mean']


class MeanScaler(BaseTransform):
    """Scaling by mean of absolute values."""
    
    @staticmethod
    def transform(x: torch.Tensor, center: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Scale each sample by its mean absolute value.
        
        Args:
            x: Input data of shape (batch_size, sequence_length, n_channels)
            center: If True, subtract mean before scaling
            
        Returns:
            tuple:
                - Scaled data with same shape
                - Dictionary with parameters for inverse transform
        """
        # Get dtype-specific epsilon
        eps = torch.finfo(x.dtype).eps
        
        stats = {}
        
        # Compute mean for centering if needed
        if center:
            mean = x.mean(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
            x = x - mean
            stats['mean'] = mean
            
        # Compute mean of absolute values per sample
        abs_mean = torch.abs(x).mean(dim=-2, keepdim=True)  # Shape: (batch_size, 1, n_channels)
        
        # Add epsilon to handle zero-valued sequences
        scale_factor = abs_mean + eps
        stats['scale_factor'] = scale_factor
        stats['center'] = center
        
        return x / scale_factor, stats
    
    @staticmethod
    def inverse_transform(x: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
        """Inverse transform to recover original scale.
        
        Args:
            x: Scaled data of shape (batch_size, sequence_length, n_channels)
            stats: Dictionary with parameters from forward transform
            
        Returns:
            Data in original scale
        """
        x = x * stats['scale_factor']
        
        if stats['center']:
            x = x + stats['mean']
            
        return x 