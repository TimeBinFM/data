import torch
from .tensor_dataset import TensorIterableDataset
from typing import Callable, Iterator, Iterable

class CombiningDataset(TensorIterableDataset):
    def __init__(
        self,
        datasets: Iterable[TensorIterableDataset],
        op: Callable[..., torch.Tensor],
    ):
        self.datasets = list(datasets)
        self.op = op

    def __iter__(self) -> Iterator[torch.Tensor]:
        iterators = [iter(ds) for ds in self.datasets]
        for items in zip(*iterators):
            yield self.op(*items)
