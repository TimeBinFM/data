import torch
from .tensor_dataset import TensorIterableDataset
from typing import Callable, Iterator

class TransformingDataset(TensorIterableDataset):
    def __init__(
        self,
        ds: TensorIterableDataset,
        op: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.ds = ds
        self.op = op

    def __iter__(self) -> Iterator[torch.Tensor]:
        for item in iter(self.ds):
            yield self.op(item)
