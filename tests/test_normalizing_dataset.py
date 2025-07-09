import torch
import pytest

from preprocessing.transform.tensor_dataset import TensorIterableDataset
from preprocessing.transform.transforming_dataset import TransformingDataset
from preprocessing.transform.batching_dataset import BatchingIterableDataset
from preprocessing.transform.unbatching_dataset import UnbatchingIterableDataset

# Dummy dataset that yields [0], [1], [2], [3]
class DummyInputDataset(TensorIterableDataset):
    def __iter__(self):
        for i in range(4):
            yield torch.tensor([i], dtype=torch.float32)

def test_transform_batch_normalize_unbatch_pipeline():
    # Step 1: Create base dataset
    base_ds = DummyInputDataset()

    # Step 2: Multiply by 2
    doubled_ds = TransformingDataset(base_ds, op=lambda x: x * 2)

    # Step 3: Batch with batch size 2
    batched_ds = BatchingIterableDataset(doubled_ds, batch_size=2)

    # Step 4: Normalize each batch by its mean
    def normalize(batch):
        mean = batch.mean()
        return batch / mean if mean != 0 else batch  # avoid div by zero

    normalized_ds = TransformingDataset(batched_ds, op=normalize)

    # Step 5: Unbatch
    final_ds = UnbatchingIterableDataset(normalized_ds, dim=0)

    # Execute pipeline
    output = list(final_ds)

    # Manually compute expected:
    # Doubled: [0], [2], [4], [6]
    # Batches: [[0, 2]], [[4, 6]]
    # Means: 1.0 and 5.0 => normalized: [[0/1, 2/1]], [[4/5, 6/5]]
    expected = [
        torch.tensor([0.0]),
        torch.tensor([2.0]),
        torch.tensor([0.8]),
        torch.tensor([1.2]),
    ]

    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.allclose(out, exp, atol=1e-5), f"Expected {exp}, got {out}"
