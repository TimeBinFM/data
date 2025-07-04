# Time Series Preprocessing Library

A comprehensive preprocessing pipeline for foundation time series models, providing tools for downloading, preprocessing, and augmenting time series datasets.

## Features

- Download datasets from Hugging Face Hub
- Static, batch-level transforms with parameter tracking:
  - MinMaxScaler: scales data to a specific range
  - StandardScaler: standardizes data to zero mean and unit variance
  - MeanScaler: scales data by mean absolute value
- Support for inverse transforms to recover original scale
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
                    "center": True  # epsilon handled automatically based on dtype
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

# Get transformed data and parameters
batch, transform_params = next(iter(dataloader))

# Inverse transform to recover original scale
original_scale = pipeline.inverse_transforms(batch, transform_params)
```

### Transform Details

All transforms are static and operate at batch level with shape (batch_size, sequence_length, n_channels):

- **MinMaxScaler**: Scales data to a specified range (default [0, 1])
  ```python
  # Transform returns both scaled data and parameters
  scaled_data, params = MinMaxScaler.transform(x, feature_range=(0, 1))
  
  # Inverse transform recovers original scale
  original_data = MinMaxScaler.inverse_transform(scaled_data, params)
  ```

- **StandardScaler**: Standardizes data
  ```python
  # Transform and store parameters
  scaled_data, params = StandardScaler.transform(x)
  
  # Inverse transform
  original_data = StandardScaler.inverse_transform(scaled_data, params)
  ```

- **MeanScaler**: Scales by mean absolute value
  ```python
  # Transform with optional centering
  scaled_data, params = MeanScaler.transform(x, center=True)
  
  # Inverse transform
  original_data = MeanScaler.inverse_transform(scaled_data, params)
  ```

Key features of transforms:
- Static methods with no instance state
- Automatic epsilon handling based on dtype
- Parameter tracking for inverse transforms
- Batch-level processing for efficiency

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