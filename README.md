# Time Series Preprocessing Library

A comprehensive preprocessing pipeline for foundation time series models, providing tools for downloading, preprocessing, and augmenting time series datasets.

## Features

- Download datasets from Hugging Face Hub
- Flexible transform chain with built-in transforms:
  - MinMaxScaler: scales each sample to a specific range independently
  - StandardScaler: standardizes each sample to zero mean and unit variance independently
  - MeanScaler: scales each sample by its mean absolute value independently
- Time series augmentation techniques (e.g., linear combination)
- PyTorch DataLoader integration
- Configurable pipeline using dataclasses

## Installation

Using poetry:
```bash
poetry install
```

Using pip:
```bash
pip install .
```

## Usage

### Basic Example
```python
from preprocessing.config import (
    Config, DatasetConfig, PreprocessingConfig, TransformConfig
)
from preprocessing.dataloader import TimeSeriesPipeline
from preprocessing.downloader import HuggingFaceDownloader

# Configure pipeline
config = Config(
    dataset=DatasetConfig(
        name="air-passengers",
        repo_id="duol/airpassengers",
        files=["AP.csv"]
    ),
    preprocessing=PreprocessingConfig(
        transforms=[
            TransformConfig(
                name="MeanScaler",
                params={
                    "center": True,
                    "epsilon": 1e-8
                }
            )
        ]
    )
)

# Download data
downloader = HuggingFaceDownloader(config.dataset)
data_path = downloader.download()["AP.csv"]

# Create pipeline
pipeline = TimeSeriesPipeline(config)

# Create dataloader with transforms
dataloader = pipeline.create_dataloader(dataset)
```

### Transform Details

All transforms operate on individual samples independently, making them suitable for both batch and online processing:

- **MinMaxScaler**: Scales each sample to a specified range (default [0, 1])
  ```python
  transform = MinMaxScaler(feature_range=(0, 1), epsilon=1e-8)
  ```

- **StandardScaler**: Standardizes each sample independently
  ```python
  transform = StandardScaler(epsilon=1e-8)
  ```

- **MeanScaler**: Scales by mean absolute value
  ```python
  transform = MeanScaler(epsilon=1e-8, center=True)
  ```

### Predefined Datasets

The HuggingFace downloader supports several predefined datasets:

```python
# List available datasets
datasets = HuggingFaceDownloader.list_available_datasets()

# Use predefined dataset
config = Config(
    dataset=DatasetConfig(
        name="UCR",
        subset="ECG200"
    )
)
```

## Development

### Setup Development Environment
```bash
poetry install --with dev
```

### Run Tests
```bash
poetry run pytest
```

### Project Structure
```
preprocessing/
├── augmentation/     # Data augmentation techniques
├── config/          # Configuration system
│   ├── default.yaml # Default configuration
│   └── examples/    # Example configurations
├── dataloader/      # PyTorch data loading pipeline
├── downloader/      # Dataset downloaders
└── transformation/  # Data transforms
```

## License

MIT License - see LICENSE file for details 