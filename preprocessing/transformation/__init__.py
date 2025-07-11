"""Transformation module for time series data preprocessing."""

from typing import Dict, Type

from .base import BaseTransform
from .normalize import MinMaxScaler, StandardScaler, MeanScaler

__all__ = [
    'BaseTransform',
    'MinMaxScaler', 
    'StandardScaler',
    'MeanScaler',
    'get_transform',
]

# Global registry mapping transform names to their classes
TRANSFORM_REGISTRY: Dict[str, Type[BaseTransform]] = {
    'MinMaxScaler': MinMaxScaler,
    'StandardScaler': StandardScaler,
    'MeanScaler': MeanScaler,
}


def get_transform(name: str) -> Type[BaseTransform]:
    """Get a transform class by name.
    
    Args:
        name: Name of the transform class
        
    Returns:
        Transform class
        
    Raises:
        KeyError: If transform name is not found in registry
    """
    if name not in TRANSFORM_REGISTRY:
        raise KeyError(
            f"Transform '{name}' not found in registry. "
            f"Available transforms: {list(TRANSFORM_REGISTRY.keys())}"
        )
    
    return TRANSFORM_REGISTRY[name]
