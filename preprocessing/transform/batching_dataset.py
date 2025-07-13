import torch
from .tensor_dataset import TensorIterableDataset
from typing import Iterator, List

class BatchingIterableDataset(TensorIterableDataset):
    def __init__(self, dataset: TensorIterableDataset, batch_size: int, 
                 include_last_batch: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.include_last_batch = include_last_batch

    def __iter__(self) -> Iterator[List[torch.Tensor]]:
        buffer: List[torch.Tensor] = []
        for item in iter(self.dataset):
            buffer.append(item)
            if len(buffer) == self.batch_size:
                yield buffer
                buffer = []
        
        if len(buffer) > 0 and self.include_last_batch:
            yield buffer
