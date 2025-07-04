"""Linear combination augmentation for time series data using Dirichlet distribution."""

from typing import Optional
import torch
import torch.nn as nn


class LinearCombination(nn.Module):
    """Linear combination augmentation for time series data.
    
    This transform creates new samples by taking linear combinations of existing time series:
    x_new = sum(lambda_i * x_i) where lambda_i are sampled from Dirichlet(alpha)
    and x_i are randomly selected time series.
    
    Input tensor dimensions: [batch_size, n_timeseries, n_elements]
    """
    
    def __init__(
        self,
        n_combine: int = 3,  # Number of time series to combine
        alpha: float = 1.0,  # Dirichlet concentration parameter
        num_samples: Optional[int] = None  # Number of combinations to generate
    ):
        """Initialize the transform.
        
        Args:
            n_combine: Number of time series to combine in each mixture
            alpha: Concentration parameter for Dirichlet distribution
                  alpha > 1: More uniform weights
                  alpha < 1: More sparse weights
                  alpha = 1: Uniform Dirichlet
            num_samples: Number of augmented samples to generate (default: same as input)
        """
        super().__init__()
        self.n_combine = n_combine
        self.alpha = alpha
        self.num_samples = num_samples
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear combination augmentation.
        
        Args:
            x: Input data of shape [batch_size, n_timeseries, n_elements]
            
        Returns:
            torch.Tensor: Augmented data with same shape as input
        """
        batch_size, n_timeseries, n_elements = x.shape
        n_output = self.num_samples or batch_size
        
        # Sample Dirichlet weights for each output sample
        # Shape: [n_output, n_combine]
        weights = torch.distributions.Dirichlet(
            torch.full((n_output, self.n_combine), self.alpha)
        ).sample()
        
        # For each output sample, randomly select n_combine time series
        # Shape: [n_output, n_combine]
        selected_idx = torch.stack([
            torch.randperm(n_timeseries)[:self.n_combine]
            for _ in range(n_output)
        ])
        
        # Gather the selected time series
        # Shape: [n_output, n_combine, n_elements]
        selected_series = x[:, selected_idx.flatten(), :].view(
            batch_size, n_output, self.n_combine, n_elements
        )
        
        # Apply weights to create combinations
        # Shape: [n_output, n_elements]
        mixed_series = torch.sum(
            weights.unsqueeze(-1) * selected_series,
            dim=2
        )
        
        return mixed_series 