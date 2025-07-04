"""Base interface for all transformations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch


class BaseTransform(ABC):
    """Abstract base class for all transformations.
    
    All transforms must:
    1. Handle batched inputs of shape (batch_size, sequence_length, n_channels)
    2. Return both transformed data and parameters needed for inverse transform
    3. Support inverse transformation using stored parameters
    """
    
    @staticmethod
    @abstractmethod
    def transform(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply the transform to input data.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            tuple:
                - Transformed tensor of same shape
                - Dictionary of parameters needed for inverse transform
        """
        pass
    
    @staticmethod
    @abstractmethod
    def inverse_transform(x: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
        """Inverse transform to recover original scale/form.
        
        Args:
            x: Transformed tensor of shape (batch_size, sequence_length, n_channels)
            stats: Dictionary of parameters from the forward transform
            
        Returns:
            Inverse transformed tensor of same shape
        """
        pass
    