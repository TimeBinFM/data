from torch.utils.data import IterableDataset
from preprocessing.common import TensorIterableDataset
from torch import Tensor
from typing import List, Iterator
from collections import deque

class SlidingWindowIterableDataset(IterableDataset):
    def __init__(self, dataset: TensorIterableDataset, window_size: int, step: int = 1):
        self.dataset = dataset
        self.window_size = window_size
        self.step = step

    def __iter__(self) -> Iterator[List[Tensor]]:
        buffer = deque()
        iterator = iter(self.dataset)

        for item in iterator:
            buffer.append(item)

            if len(buffer) < self.window_size:
                continue

            if len(buffer) == self.window_size:
                yield list(buffer)
                for _ in range(self.step):
                    if buffer:
                        buffer.popleft()
