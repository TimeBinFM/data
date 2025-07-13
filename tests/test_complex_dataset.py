import torch
import pytest
from typing import List

from preprocessing.transform.tensor_dataset import TensorIterableDataset
from preprocessing.transform.transforming_dataset import TransformingDataset
from preprocessing.transform.batching_dataset import BatchingIterableDataset
from preprocessing.transform.probabilistic_mixing_dataset import ProbabilisticMixingDataset

class ConstantDataset(TensorIterableDataset):
    def __init__(self, values: List[torch.Tensor]):
        self.values = values

    def __iter__(self):
        return iter(self.values)

def test_complex_pipeline_reproducible_2d():
    # Step 1: Create two datasets of 2D tensors
    dataset_1 = ConstantDataset([
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
        torch.tensor([[13.0, 14.0], [15.0, 16.0]]),
    ])  # 4 items, each 2x2, will yield 2 batches of shape (2, 2, 2)

    dataset_2 = ConstantDataset([
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
    ])

    # Step 2: Batch dataset_1 in batches of 2
    dataset_1_batched = BatchingIterableDataset(dataset_1, batch_size=2)

    # Step 3: Transform dataset_1 batches
    def transform_dataset_1(batch: torch.Tensor) -> torch.Tensor:
        # batch shape: (2, 2, 2)
        linear_comb = 0.5 * batch[0] + 0.5 * batch[1]  # shape: (2, 2)
        linear_comb_avg = linear_comb.mean()
        return linear_comb - linear_comb_avg

    dataset_1_final = TransformingDataset(dataset_1_batched, op=transform_dataset_1)

    # Step 4: Transform dataset_2
    dataset_2_final = TransformingDataset(
        dataset_2, op=lambda ts: ts - ts.mean()
    )

    # Step 5: Mix the datasets
    final_dataset = ProbabilisticMixingDataset(
        datasets={"ds1": dataset_1_final, "ds2": dataset_2_final},
        probabilities={"ds1": 0.5, "ds2": 0.5},
        seed=42,
    )

    result = list(final_dataset)

    # Expect 4 results (2 from each dataset)
    assert len(result) == 4

    # Check shapes and zero mean
    for tensor in result:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert torch.isclose(tensor.mean(), torch.tensor(0.0), atol=1e-6)

    # Reproducibility check
    final_dataset_2 = ProbabilisticMixingDataset(
        datasets={"ds1": dataset_1_final, "ds2": dataset_2_final},
        probabilities={"ds1": 0.5, "ds2": 0.5},
        seed=42,
    )
    result2 = list(final_dataset_2)
    for t1, t2 in zip(result, result2):
        assert torch.allclose(t1, t2)
