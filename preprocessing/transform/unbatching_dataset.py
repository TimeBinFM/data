import torch
from torch.utils.data import IterableDataset
from typing import List, Iterator

class UnbatchingIterableDataset(IterableDataset[List[torch.Tensor]]):
    def __init__(self, dataset: IterableDataset[List[torch.Tensor]]):
        self.dataset = dataset

    def __iter__(self) -> Iterator[torch.Tensor]:
        for batch in iter(self.dataset):
            for item in batch:
                yield item
