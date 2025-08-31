from torch.utils.data import IterableDataset
from torch import Tensor
from typing import List, Iterator

class UnbatchingIterableDataset(IterableDataset[List[Tensor]]):
    def __init__(self, dataset: IterableDataset[List[Tensor]]):
        self.dataset = dataset

    def __iter__(self) -> Iterator[Tensor]:
        for batch in iter(self.dataset):
            for item in batch:
                yield item
