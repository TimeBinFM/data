from ..common import TensorIterableDataset
from typing import Iterator, Iterable

class ConcatDataset(TensorIterableDataset):
    def __init__(self, datasets: Iterable[TensorIterableDataset]):
        self.datasets = list(datasets)

    def __iter__(self) -> Iterator:
        for dataset in self.datasets:
            for item in dataset:
                yield item
