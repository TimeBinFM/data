"""Hugging Face dataset downloader."""

from typing import Optional, Dict, Any
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
import numpy as np

from ..config import DatasetConfig


class HuggingFaceDownloader:
    """Download datasets from Hugging Face Hub."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize the downloader.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self) -> Dict[str, np.ndarray]:
        """Download the dataset from Hugging Face Hub.
        
        Returns:
            dict: Dictionary containing 'data' and optionally 'targets' arrays
        """
        dataset_path = self._get_dataset_path()
        if dataset_path.exists():
            return self._load_cached_dataset(dataset_path)
            
        return self._download_and_cache_dataset(dataset_path)
    
    def _get_dataset_path(self) -> Path:
        """Get the path where the dataset should be cached."""
        filename = f"{self.config.name}"
        if self.config.subset:
            filename += f"_{self.config.subset}"
        filename += ".npz"
        return self.cache_dir / filename
    
    def _load_cached_dataset(self, path: Path) -> Dict[str, np.ndarray]:
        """Load a cached dataset.
        
        Args:
            path: Path to the cached dataset
            
        Returns:
            dict: Dictionary containing the dataset arrays
        """
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    
    def _download_and_cache_dataset(self, cache_path: Path) -> Dict[str, np.ndarray]:
        """Download dataset from Hugging Face and cache it.
        
        Args:
            cache_path: Path where to cache the downloaded dataset
            
        Returns:
            dict: Dictionary containing the dataset arrays
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Use hf_hub_download to get the data
        # 2. Process it into numpy arrays
        # 3. Save to cache_path
        # 4. Return the arrays
        
        repo_id = f"time-series/{self.config.name}"
        filename = f"{self.config.subset}.npz" if self.config.subset else "data.npz"
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                repo_type="dataset"
            )
            
            return self._load_cached_dataset(Path(downloaded_path))
            
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {str(e)}")
            
    @staticmethod
    def list_available_datasets() -> Dict[str, Any]:
        """List available time series datasets on Hugging Face Hub.
        
        Returns:
            dict: Dictionary of available datasets and their metadata
        """
        # This is a placeholder - in a real implementation you would:
        # 1. Query the Hugging Face API for available time series datasets
        # 2. Return their metadata
        return {
            "UCR": {
                "subsets": ["ECG200", "GunPoint", "CBF"],
                "description": "UCR Time Series Classification Archive"
            },
            "M4": {
                "subsets": ["Hourly", "Daily", "Weekly"],
                "description": "M4 Forecasting Competition Dataset"
            }
        } 