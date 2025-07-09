import torch
import pytest

from preprocessing.transform.tensor_dataset import TensorIterableDataset
from preprocessing.transform.unbatching_dataset import UnbatchingIterableDataset

# Dummy dataset that yields two batches
class DummyBatchDataset(TensorIterableDataset):
    def __iter__(self):
        yield torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # shape: (2, 2)
        yield torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)  # shape: (2, 2)

def test_unbatching_iterable_dataset_dim0():
    ds = DummyBatchDataset()
    unbatch_ds = UnbatchingIterableDataset(ds, dim=0)

    output = list(unbatch_ds)

    expected = [
        torch.tensor([1, 2], dtype=torch.float32),
        torch.tensor([3, 4], dtype=torch.float32),
        torch.tensor([5, 6], dtype=torch.float32),
        torch.tensor([7, 8], dtype=torch.float32),
    ]

    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"

def test_unbatching_iterable_dataset_dim1():
    # Unbatch across columns instead of rows
    class DummyBatchDim1(TensorIterableDataset):
        def __iter__(self):
            yield torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # shape: (2, 2)

    ds = DummyBatchDim1()
    unbatch_ds = UnbatchingIterableDataset(ds, dim=1)

    output = list(unbatch_ds)
    expected = [
        torch.tensor([1, 3], dtype=torch.float32),
        torch.tensor([2, 4], dtype=torch.float32),
    ]

    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"
