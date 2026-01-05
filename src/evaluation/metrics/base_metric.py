"""
Base class for evaluation metrics with caching support.

Provides abstract interface for metrics that can cache real data
computations and reuse them across multiple models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import numpy as np
from typing import Dict, Any, Optional


class CacheableMetric(ABC):
    """Base class for metrics with caching support."""

    def __init__(self, cache_root: Path, dataset_name: str):
        """
        Initialize cacheable metric.

        Args:
            cache_root: Root directory for all evaluation caches
            dataset_name: Name of dataset (e.g., 'toy', 'epure')
        """
        self.cache_root = Path(cache_root)
        self.dataset_name = dataset_name
        self.real_cache_dir = self.cache_root / dataset_name / "real"
        self.real_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Unique identifier for this metric (e.g., 'fid', 'iou_dice')."""
        pass

    @abstractmethod
    def compute_real_data(self, real_dirs: Dict[str, Path], **kwargs) -> Any:
        """
        Compute metric data for real images (to be cached).

        Args:
            real_dirs: {component: Path} or {'full': Path} for real images
            **kwargs: Additional arguments (e.g., components list, batch_size)

        Returns:
            Data to be cached (distributions, features, etc.)
        """
        pass

    @abstractmethod
    def compute_generated_data(self, gen_dirs: Dict[str, Path], **kwargs) -> Any:
        """
        Compute metric data for generated images.

        Args:
            gen_dirs: {component: Path} or {'full': Path} for generated images
            **kwargs: Additional arguments

        Returns:
            Data for generated images (same structure as compute_real_data)
        """
        pass

    @abstractmethod
    def compute_distance(self, real_data: Any, gen_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Compute final metric (e.g., Wasserstein distance) from cached data.

        Args:
            real_data: Cached data from compute_real_data
            gen_data: Data from compute_generated_data
            **kwargs: Additional arguments (e.g., num_bootstrap)

        Returns:
            Dictionary with metric results
        """
        pass

    def get_real_cache_path(self) -> Path:
        """Get cache file path for real data."""
        return self.real_cache_dir / f"{self.metric_name}.pkl"

    def get_model_cache_path(self, model_name: str) -> Path:
        """Get cache file path for model-specific data."""
        model_cache_dir = self.cache_root / self.dataset_name / "models" / model_name
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        return model_cache_dir / f"{self.metric_name}.pkl"

    def load_real_cache(self) -> Optional[Any]:
        """Load cached real data if exists."""
        cache_path = self.get_real_cache_path()
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def save_real_cache(self, data: Any):
        """Save real data to cache."""
        cache_path = self.get_real_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def evaluate(self, real_dirs: Dict[str, Path], gen_dirs: Dict[str, Path],
                 model_name: Optional[str] = None, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Evaluate metric with automatic caching.

        Args:
            real_dirs: {component: Path} or {'full': Path} for real images
            gen_dirs: {component: Path} or {'full': Path} for generated images
            model_name: Model name for generated data cache
            use_cache: Whether to use caching
            **kwargs: Additional arguments for metric computation

        Returns:
            Metric results dictionary
        """
        # Load or compute real data
        if use_cache:
            real_data = self.load_real_cache()
            if real_data is not None:
                print(f"  [{self.metric_name}] Using cached real data")
            else:
                print(f"  [{self.metric_name}] Computing real data (caching for future use)...")
                real_data = self.compute_real_data(real_dirs, **kwargs)
                self.save_real_cache(real_data)
        else:
            real_data = self.compute_real_data(real_dirs, **kwargs)

        # Compute generated data (no cache for generated data for now)
        gen_data = self.compute_generated_data(gen_dirs, **kwargs)

        # Compute final distance/metric
        return self.compute_distance(real_data, gen_data, **kwargs)
