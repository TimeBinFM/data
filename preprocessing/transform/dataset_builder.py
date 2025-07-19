from torch import Tensor
from typing import Callable, Iterable, Optional

from preprocessing.common import TensorIterableDataset
from .transforming_dataset import TransformingDataset
from .batching_dataset import BatchingIterableDataset
from .unbatching_dataset import UnbatchingIterableDataset
from .combining_dataset import CombiningDataset
from .probabilistic_mixing_dataset import ProbabilisticMixingDataset

class Builder:
    def __init__(self, dataset: TensorIterableDataset):
        self.dataset = dataset

    def map(self, op: Callable[[Tensor], Tensor]) -> "Builder":
        return Builder(TransformingDataset(self.dataset, op=op))

    def batch(self, batch_size: int, include_last_batch: bool = True) -> "Builder":
        return Builder(BatchingIterableDataset(self.dataset, batch_size, include_last_batch))

    def flat(self, dim: int = 0) -> "Builder":
        return Builder(UnbatchingIterableDataset(self.dataset, dim=dim))

    def build(self) -> TensorIterableDataset:
        return self.dataset
