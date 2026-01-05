"""
RCE (Relative Count Error) metric.

Counts connected objects per image and compares to ideal (=1 object).
Uses Wasserstein distance.
"""

import numpy as np
from pathlib import Path
from typing import Union
from scipy.stats import wasserstein_distance
from scipy import ndimage

from .utils import load_image_binary, list_image_files, extract_id_from_filename


def count_connected_objects(image_path: Path, threshold: int = 128) -> int:
    """
    Count number of connected objects in a binary image.

    Args:
        image_path: Path to image file
        threshold: Binarization threshold

    Returns:
        Number of connected components (excluding background)
    """
    mask = load_image_binary(image_path, threshold)

    # Label connected components
    labeled, num_objects = ndimage.label(mask)

    return num_objects


def compute_rce(
    real_full_dir: Union[str, Path],
    gen_full_dir: Union[str, Path]
) -> dict:
    """
    Compute Relative Count Error (RCE).

    Counts connected objects per image, compares to ideal (=1 object).
    Returns Wasserstein distance.

    Args:
        real_full_dir: Directory with real full images
        gen_full_dir: Directory with generated full images

    Returns:
        {
            'real_wd': Wasserstein distance to ideal for real images,
            'gen_wd': Wasserstein distance to ideal for generated images,
            'real_counts': Array of real counts,
            'gen_counts': Array of generated counts
        }
    """
    real_full_dir = Path(real_full_dir)
    gen_full_dir = Path(gen_full_dir)

    real_files = list_image_files(real_full_dir)
    gen_files = list_image_files(gen_full_dir)

    if len(real_files) == 0:
        raise ValueError(f"No images found in real directory: {real_full_dir}")
    if len(gen_files) == 0:
        raise ValueError(f"No images found in generated directory: {gen_full_dir}")

    # Count objects
    real_counts = []
    gen_counts = []

    # Build ID mapping
    gen_ids = {extract_id_from_filename(f.name): f for f in gen_files}
    real_ids = {extract_id_from_filename(f.name): f for f in real_files}

    # Get common IDs
    common_ids = set(gen_ids.keys()).intersection(set(real_ids.keys()))

    if len(common_ids) == 0:
        # If no common IDs, just process all files independently
        real_counts = [count_connected_objects(f) for f in real_files]
        gen_counts = [count_connected_objects(f) for f in gen_files]
    else:
        # Match by ID
        for img_id in sorted(common_ids):
            real_count = count_connected_objects(real_ids[img_id])
            gen_count = count_connected_objects(gen_ids[img_id])

            real_counts.append(real_count)
            gen_counts.append(gen_count)

    real_counts = np.array(real_counts)
    gen_counts = np.array(gen_counts)

    # Wasserstein to ideal (all 1s)
    ideal_dist = np.ones_like(gen_counts)
    wd_gen = wasserstein_distance(gen_counts, ideal_dist)
    wd_real = wasserstein_distance(real_counts, ideal_dist)

    return {
        'real_wd': float(wd_real),
        'gen_wd': float(wd_gen),
        'real_counts': real_counts.tolist(),
        'gen_counts': gen_counts.tolist()
    }


# ============================================================================
# Cacheable Metric Interface
# ============================================================================

from .base_metric import CacheableMetric
from typing import Dict, Any


class RCEMetric(CacheableMetric):
    """RCE metric with automatic caching via CacheableMetric interface."""

    @property
    def metric_name(self) -> str:
        return "rce"

    def compute_real_data(self, real_dirs: Dict[str, Path], **kwargs) -> np.ndarray:
        """
        Count connected objects in real images.

        Args:
            real_dirs: {'full': Path} to real full images directory
            **kwargs: Unused

        Returns:
            Numpy array of object counts per image
        """
        real_dir = real_dirs.get('full')
        if real_dir is None:
            raise ValueError("RCE requires 'full' directory in real_dirs")

        real_files = list_image_files(Path(real_dir))
        if len(real_files) == 0:
            raise ValueError(f"No images found in {real_dir}")

        real_counts = [count_connected_objects(f) for f in real_files]
        return np.array(real_counts)

    def compute_generated_data(self, gen_dirs: Dict[str, Path], **kwargs) -> np.ndarray:
        """
        Count connected objects in generated images.

        Args:
            gen_dirs: {'full': Path} to generated full images directory
            **kwargs: Unused

        Returns:
            Numpy array of object counts per image
        """
        gen_dir = gen_dirs.get('full')
        if gen_dir is None:
            raise ValueError("RCE requires 'full' directory in gen_dirs")

        gen_files = list_image_files(Path(gen_dir))
        if len(gen_files) == 0:
            raise ValueError(f"No images found in {gen_dir}")

        gen_counts = [count_connected_objects(f) for f in gen_files]
        return np.array(gen_counts)

    def compute_distance(self, real_data: np.ndarray, gen_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute Wasserstein distance to ideal (all 1s).

        Args:
            real_data: Real object counts
            gen_data: Generated object counts
            **kwargs: Unused

        Returns:
            {
                'real_wd': WD to ideal for real,
                'gen_wd': WD to ideal for generated,
                'real_counts': List of real counts,
                'gen_counts': List of generated counts
            }
        """
        # Wasserstein to ideal (all 1s)
        ideal_real = np.ones_like(real_data)
        ideal_gen = np.ones_like(gen_data)

        wd_real = wasserstein_distance(real_data, ideal_real)
        wd_gen = wasserstein_distance(gen_data, ideal_gen)

        return {
            'real_wd': float(wd_real),
            'gen_wd': float(wd_gen),
            'real_counts': real_data.tolist(),
            'gen_counts': gen_data.tolist()
        }
