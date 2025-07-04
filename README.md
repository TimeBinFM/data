# Time Series Preprocessing Library

A comprehensive preprocessing pipeline for foundation time series models, providing tools for downloading, preprocessing, and augmenting time series datasets.

## Features

- Download datasets from remote sources (e.g., Hugging Face)
- Flexible transform chain with built-in transforms:
  - MinMaxScaler: scales data to a specific range
  - StandardScaler: standardizes to zero mean and unit variance
  - MeanScaler: scales by mean of absolute values
- Time series augmentation techniques
- PyTorch DataLoader integration
- Configurable pipeline using Hydra

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

### Basic Configuration
```yaml
# configs/config.yaml
preprocessing:
  transforms:
    - name: MinMaxScaler
      params:
        feature_range: [0, 1]
    - name: StandardScaler
      params:
        epsilon: 1e-8

augmentation:
  enabled: true
  methods: ["linear_combo"]
  linear_combo_ratio: 0.5
```

### Download Data
```bash
poetry run download-data --dataset ucr --subset ECG200
```

### Run Preprocessing
```bash
poetry run run-preprocessing --config-path configs/preprocessing
```

### Python API
```python
from preprocessing.config import Config
from preprocessing.dataloader import TimeSeriesPipeline

# Create pipeline with config
pipeline = TimeSeriesPipeline(config)

# Create dataloader with transforms
dataloader = pipeline.create_dataloader(dataset)
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

## License

MIT License - see LICENSE file for details 