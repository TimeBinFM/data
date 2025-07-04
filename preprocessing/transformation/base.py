"""Base interface for all per-sample transformations."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Transform(ABC, nn.Module):
    """Abstract base class for all per-sample transformations.
    
    All transforms must:
    1. Operate on individual samples independently
    2. Support inverse transformation when applicable
    3. Handle batched inputs of shape (batch_size, sequence_length, n_channels)
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transform to each sample independently.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Transformed tensor of same shape
        """
        pass
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform to recover original scale/form.
        
        Args:
            x: Transformed tensor of shape (batch_size, sequence_length, n_channels)
            
        Returns:
            Inverse transformed tensor of same shape
            
        Raises:
            NotImplementedError: If transform is not invertible
        """
        raise NotImplementedError("This transform does not support inverse transformation")
    
    @property
    def is_invertible(self) -> bool:
        """Whether the transform supports inverse transformation."""
        try:
            self.inverse_transform(torch.ones(1, 1, 1))
            return True
        except NotImplementedError:
            return False 