"""Base interface for all transformations."""

from abc import ABC, abstractmethod
import torch


class BaseTransform(ABC):
    """Abstract base class for all transformations.
    
    All transforms must:
    1. Handle batched inputs of shape (batch_size, sequence_length, n_channels)
    2. Transform data according to their specific requirements
    """
    
    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to input data.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Transformed tensor
        """
        pass
    