"""
Utility functions for evaluation metrics.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict
from scipy.stats import wasserstein_distance
from tqdm import tqdm


def load_image_binary(path: Path, threshold: int = 128) -> np.ndarray:
    """Load image and convert to binary mask (0 or 1)."""
    img = Image.open(path).convert('L')
    arr = np.array(img, dtype=np.uint8)
    return (arr > threshold).astype(np.uint8)


def extract_id_from_filename(filename: str) -> str:
    """
    Extract ID from filename.

    Examples:
        toy_00001_group_nc.png -> toy_00001
        A0001_full.png -> A0001
    """
    name = Path(filename).stem  # Remove extension
    # Remove component suffix
    for suffix in ['_group_nc', '_group_km', '_fpu', '_bt', '_tpc', '_gi', '_full']:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def list_image_files(directory: Path) -> List[Path]:
    """List all PNG/JPG files in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted([*directory.glob('*.png'), *directory.glob('*.jpg'), *directory.glob('*.jpeg')])


def get_common_ids(dir_dict: Dict[str, Path]) -> List[str]:
    """
    Get common image IDs across multiple directories.

    Args:
        dir_dict: Dictionary mapping component -> directory path

    Returns:
        List of common IDs (sorted)
    """
    # Get IDs from first directory
    first_comp = list(dir_dict.keys())[0]
    first_files = list_image_files(dir_dict[first_comp])
    ids = {extract_id_from_filename(f.name) for f in first_files}

    # Intersect with other directories
    for comp, comp_dir in list(dir_dict.items())[1:]:
        files = list_image_files(comp_dir)
        comp_ids = {extract_id_from_filename(f.name) for f in files}
        ids = ids.intersection(comp_ids)

    return sorted(list(ids))


def bootstrap_wasserstein(
    real_values: np.ndarray,
    gen_values: np.ndarray,
    n_bootstrap: int = 1000,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Bootstrap Wasserstein distance.

    Returns:
        Array of bootstrapped WD values (length n_bootstrap)
    """
    if rng is None:
        rng = np.random.default_rng()

    real_values = np.asarray(real_values)
    gen_values = np.asarray(gen_values)

    wds = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        real_sample = rng.choice(real_values, len(real_values), replace=True)
        gen_sample = rng.choice(gen_values, len(gen_values), replace=True)

        wd = wasserstein_distance(real_sample, gen_sample)
        wds.append(wd)

    return np.array(wds)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total == 0:
        return 0.0

    return float(2 * intersection / total)
