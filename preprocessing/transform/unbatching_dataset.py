import torch
from .tensor_dataset import TensorIterableDataset
from typing import Iterator

class UnbatchingIterableDataset(TensorIterableDataset):
    def __init__(self, dataset: TensorIterableDataset, dim: int):
        self.dataset = dataset
        self.dim = dim

    def __iter__(self) -> Iterator[torch.Tensor]:
        for batch in iter(self.dataset):
            for item in torch.unbind(batch, self.dim):
                yield item
