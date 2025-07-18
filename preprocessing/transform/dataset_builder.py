from torch import Tensor
from typing import Callable
from torch.utils.data import IterableDataset

from .tensor_dataset import TensorIterableDataset
from .transforming_dataset import TransformingDataset
from .batching_dataset import BatchingIterableDataset
from .unbatching_dataset import UnbatchingIterableDataset

class Builder:
    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset

    def map(self, op: Callable[[Tensor], Tensor]) -> "Builder":
        return Builder(TransformingDataset(self.dataset, op=op))

    def batch(self, batch_size: int, include_last_batch: bool = True) -> "Builder":
        return Builder(BatchingIterableDataset(self.dataset, batch_size, include_last_batch))

    def flat(self) -> "Builder":
        return Builder(UnbatchingIterableDataset(self.dataset))

    def build(self) -> TensorIterableDataset:
        return self.dataset
