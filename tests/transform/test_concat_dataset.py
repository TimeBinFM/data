"""Tests for ConcatDataset."""

import torch
import pytest
from preprocessing.common import TensorIterableDataset
from preprocessing.transform.concat_dataset import ConcatDataset


class DummyDatasetA(TensorIterableDataset):
    """Dummy dataset that emits numbers 0-4."""
    def __iter__(self):
        for i in range(5):
            yield torch.tensor([i], dtype=torch.float32)


class DummyDatasetB(TensorIterableDataset):
    """Dummy dataset that emits numbers 10-12."""
    def __iter__(self):
        for i in range(10, 13):
            yield torch.tensor([i], dtype=torch.float32)


def test_concat_dataset_sequential():
    """Test sequential concatenation (default behavior)."""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    concat_ds = ConcatDataset([ds1, ds2])
    result = list(concat_ds)
    
    # Should get all from ds1, then all from ds2
    expected = [
        torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([2.0]), 
        torch.tensor([3.0]), torch.tensor([4.0]),  # from ds1
        torch.tensor([10.0]), torch.tensor([11.0]), torch.tensor([12.0])  # from ds2
    ]
    
    assert len(result) == 8
    for i, (actual, expected_val) in enumerate(zip(result, expected)):
        assert torch.equal(actual, expected_val), f"Mismatch at index {i}"


def test_concat_dataset_interleaved():
    """Test interleaved sampling with ratios."""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    # Use equal ratios and fixed seed for predictable results
    concat_ds = ConcatDataset([ds1, ds2], sampling_ratios=[0.5, 0.5], seed=42)
    result = list(concat_ds)
    
    # Should have all 8 items but in random order
    assert len(result) == 8
    
    # Check that we have the right values (order doesn't matter)
    result_values = [item.item() for item in result]
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]
    
    assert sorted(result_values) == sorted(expected_values)


def test_concat_dataset_weighted_sampling():
    """Test weighted sampling ratios."""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    # Heavy bias toward ds1
    concat_ds = ConcatDataset([ds1, ds2], sampling_ratios=[0.9, 0.1], seed=123)
    result = list(concat_ds)
    
    assert len(result) == 8
    
    # Check values are present
    result_values = [item.item() for item in result]
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]
    assert sorted(result_values) == sorted(expected_values)


def test_concat_dataset_reproducibility():
    """Test that results are reproducible with same seed."""
    ds1_1 = DummyDatasetA()
    ds2_1 = DummyDatasetB()
    concat_ds_1 = ConcatDataset([ds1_1, ds2_1], sampling_ratios=[0.6, 0.4], seed=456)
    
    ds1_2 = DummyDatasetA()
    ds2_2 = DummyDatasetB()
    concat_ds_2 = ConcatDataset([ds1_2, ds2_2], sampling_ratios=[0.6, 0.4], seed=456)
    
    result1 = list(concat_ds_1)
    result2 = list(concat_ds_2)
    
    assert len(result1) == len(result2)
    for r1, r2 in zip(result1, result2):
        assert torch.equal(r1, r2)


def test_concat_dataset_empty_input():
    """Test error handling for empty dataset list."""
    with pytest.raises(ValueError, match="At least one dataset must be provided"):
        ConcatDataset([])


def test_concat_dataset_ratio_validation():
    """Test validation of sampling ratios."""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    # Wrong number of ratios
    with pytest.raises(ValueError, match="Number of sampling ratios must match"):
        ConcatDataset([ds1, ds2], sampling_ratios=[0.5])
    
    # Negative ratio
    with pytest.raises(ValueError, match="All sampling ratios must be positive"):
        ConcatDataset([ds1, ds2], sampling_ratios=[0.5, -0.1])
    
    # Zero ratio
    with pytest.raises(ValueError, match="All sampling ratios must be positive"):
        ConcatDataset([ds1, ds2], sampling_ratios=[0.5, 0.0])


def test_concat_dataset_ratio_normalization():
    """Test that ratios are normalized to sum to 1."""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    concat_ds = ConcatDataset([ds1, ds2], sampling_ratios=[2.0, 6.0])
    
    # Should normalize to [0.25, 0.75]
    assert abs(concat_ds.sampling_ratios[0] - 0.25) < 1e-6
    assert abs(concat_ds.sampling_ratios[1] - 0.75) < 1e-6
    assert abs(sum(concat_ds.sampling_ratios) - 1.0) < 1e-6


def test_concat_dataset_single_dataset():
    """Test concatenation with single dataset."""
    ds1 = DummyDatasetA()
    concat_ds = ConcatDataset([ds1])
    result = list(concat_ds)
    
    expected = [torch.tensor([float(i)]) for i in range(5)]
    assert len(result) == 5
    for actual, expected_val in zip(result, expected):
        assert torch.equal(actual, expected_val)


def test_concat_dataset_exhausted_datasets():
    """Test behavior when datasets have different lengths."""
    # Create datasets of different lengths
    class ShortDataset(TensorIterableDataset):
        def __iter__(self):
            for i in range(2):
                yield torch.tensor([i + 100], dtype=torch.float32)
    
    ds1 = DummyDatasetA()  # 5 items
    ds2 = ShortDataset()   # 2 items
    
    concat_ds = ConcatDataset([ds1, ds2], sampling_ratios=[0.5, 0.5], seed=789)
    result = list(concat_ds)
    
    # Should still get all 7 items
    assert len(result) == 7
    
    result_values = [item.item() for item in result]
    expected_values = [0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 101.0]
    assert sorted(result_values) == sorted(expected_values)