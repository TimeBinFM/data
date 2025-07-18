"""Concatenation dataset for sequential stacking of multiple datasets."""

import torch
from typing import Iterator, List, Optional
from preprocessing.common import TensorIterableDataset


class ConcatDataset(TensorIterableDataset):
    """Concatenate multiple datasets sequentially.
    
    This dataset yields all items from the first dataset, then all items from
    the second dataset, and so on. Optionally supports interleaved sampling
    with specified ratios while maintaining streaming capabilities.
    """
    
    def __init__(
        self, 
        datasets: List[TensorIterableDataset],
        sampling_ratios: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        """Initialize the concatenation dataset.
        
        Args:
            datasets: List of datasets to concatenate
            sampling_ratios: Optional list of sampling probabilities for each dataset.
                If None, uses sequential concatenation. If provided, uses interleaved
                sampling with these ratios.
            seed: Random seed for reproducibility when using sampling_ratios
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")
            
        self.datasets = datasets
        self.sampling_ratios = sampling_ratios
        self.seed = seed
        
        if sampling_ratios is not None:
            if len(sampling_ratios) != len(datasets):
                raise ValueError("Number of sampling ratios must match number of datasets")
            if not all(r > 0 for r in sampling_ratios):
                raise ValueError("All sampling ratios must be positive")
            # Normalize ratios to sum to 1
            total = sum(sampling_ratios)
            self.sampling_ratios = [r / total for r in sampling_ratios]
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate through all datasets."""
        if self.sampling_ratios is None:
            # Sequential concatenation
            for dataset in self.datasets:
                for item in dataset:
                    yield item
        else:
            # Interleaved sampling
            yield from self._interleaved_sampling()
    
    def _interleaved_sampling(self) -> Iterator[torch.Tensor]:
        """Sample from datasets without loading everything into memory."""
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        
        # Create iterators for each dataset
        iterators = [iter(dataset) for dataset in self.datasets]
        active_iterators = list(range(len(iterators)))
        
        while active_iterators:
            # Choose which dataset to sample from based on ratios
            weights = [self.sampling_ratios[i] for i in active_iterators]
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            chosen_idx = active_iterators[torch.multinomial(weights_tensor, 1, generator=generator).item()]
            
            try:
                item = next(iterators[chosen_idx])
                yield item
            except StopIteration:
                # Remove exhausted iterator
                active_iterators.remove(chosen_idx)