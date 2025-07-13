import torch
import pytest
from collections import Counter
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

def test_complex_pipeline_reproducible():
    # Step 1: Create two datasets
    dataset_1 = ConstantDataset([
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([5.0, 6.0]),
        torch.tensor([7.0, 8.0]),
    ])  # 4 items, will become 2 batches

    dataset_2 = ConstantDataset([
        torch.tensor([10.0, 20.0]),
        torch.tensor([30.0, 40.0]),
    ])

    # Step 2: Batch dataset_1 in batches of 2
    dataset_1_batched = BatchingIterableDataset(dataset_1, batch_size=2)

    # Step 3: Transform dataset_1 batches
    def transform_dataset_1(batch: torch.Tensor) -> torch.Tensor:
        linear_comb = 0.5 * batch[0] + 0.5 * batch[1]
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

    # Check that we got 4 outputs in total (2 from each)
    assert len(result) == 4

    # Check output shapes and means
    for tensor in result:
        assert isinstance(tensor, torch.Tensor)
        assert torch.isclose(tensor.mean(), torch.tensor(0.0), atol=1e-6)

    # Check reproducibility
    final_dataset_2 = ProbabilisticMixingDataset(
        datasets={"ds1": dataset_1_final, "ds2": dataset_2_final},
        probabilities={"ds1": 0.5, "ds2": 0.5},
        seed=42,
    )
    result2 = list(final_dataset_2)
    for t1, t2 in zip(result, result2):
        assert torch.allclose(t1, t2)

