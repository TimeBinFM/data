import torch
from torch.utils.data import IterableDataset
from preprocessing.transform.sliding_window_dataset import SlidingWindowIterableDataset


class DummyTensorDataset(IterableDataset):
    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield torch.tensor([i])


def test_sliding_window_basic():
    dataset = DummyTensorDataset(10)
    windowed = SlidingWindowIterableDataset(dataset, window_size=3, step=2)
    result = list(windowed)

    expected = [
        [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])],
        [torch.tensor([2]), torch.tensor([3]), torch.tensor([4])],
        [torch.tensor([4]), torch.tensor([5]), torch.tensor([6])],
        [torch.tensor([6]), torch.tensor([7]), torch.tensor([8])],
    ]

    assert len(result) == len(expected)
    for actual_window, expected_window in zip(result, expected):
        for actual, expected_tensor in zip(actual_window, expected_window):
            assert torch.equal(actual, expected_tensor)


def test_sliding_window_exact_fit():
    dataset = DummyTensorDataset(6)
    windowed = SlidingWindowIterableDataset(dataset, window_size=3, step=3)
    result = list(windowed)

    expected = [
        [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])],
        [torch.tensor([3]), torch.tensor([4]), torch.tensor([5])]
    ]

    assert len(result) == len(expected)
    for actual_window, expected_window in zip(result, expected):
        for actual, expected_tensor in zip(actual_window, expected_window):
            assert torch.equal(actual, expected_tensor)


def test_sliding_window_too_small():
    dataset = DummyTensorDataset(2)
    windowed = SlidingWindowIterableDataset(dataset, window_size=3, step=1)
    result = list(windowed)
    assert result == []
