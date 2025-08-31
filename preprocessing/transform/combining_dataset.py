import torch
from torch.utils.data import IterableDataset
from typing import Callable, Iterator, Iterable, TypeVar    

T_in = TypeVar('T_in')
T_out = TypeVar('T_out')

class CombiningDataset(IterableDataset[T_in]):
    def __init__(
        self,
        datasets: Iterable[IterableDataset[T_in]],
        op: Callable[..., T_out],
    ):
        self.datasets = list(datasets)
        self.op = op

    def __iter__(self) -> Iterator[T_out]:
        iterators = [iter(ds) for ds in self.datasets]
        for items in zip(*iterators):
            yield self.op(*items)
