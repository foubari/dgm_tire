"""
Center of Mass (CoM) metrics.

Computes component-wise, overall, and vector Wasserstein distances.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import ndimage
from scipy.stats import wasserstein_distance

from .utils import load_image_binary, get_common_ids, bootstrap_wasserstein


def wasserstein_distance_2d(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    2D Wasserstein distance (treating x and y coordinates separately, then averaging).

    Args:
        points_a: (N, 2) array of 2D points
        points_b: (M, 2) array of 2D points

    Returns:
        Mean of WD for x and y coordinates
    """
    wd_y = wasserstein_distance(points_a[:, 0], points_b[:, 0])
    wd_x = wasserstein_distance(points_a[:, 1], points_b[:, 1])
    return float(np.mean([wd_y, wd_x]))


def compute_com_metrics(
    real_component_dirs: Dict[str, Path],
    gen_component_dirs: Dict[str, Path],
    components: List[str],
    num_bootstrap: int = 1000
) -> dict:
    """
    Compute Center of Mass metrics.

    Returns component-wise, overall, and vector Wasserstein distances.

    Args:
        real_component_dirs: {component: Path} for real images
        gen_component_dirs: {component: Path} for generated images
        components: List of component names
        num_bootstrap: Number of bootstrap iterations

    Returns:
        {
            'component': {
                comp: {
                    'wasserstein': (mean, std)
                },
                ...
            },
            'overall': {
                'wasserstein': float
            },
            'vector_wasserstein': (mean, std)
        }
    """
    from .utils import list_image_files

    # NO MATCHING - compute CoM distributions separately for real and generated
    real_coms = []  # List of (n_components, 2) arrays for each real image
    gen_coms = []   # List of (n_components, 2) arrays for each generated image

    # Get all real image files for first component to determine count
    first_comp = components[0]
    real_files_first = list_image_files(real_component_dirs[first_comp])
    gen_files_first = list_image_files(gen_component_dirs[first_comp])

    # Process all real images
    for i in range(len(real_files_first)):
        real_com = []
        for comp in components:
            real_files = list_image_files(real_component_dirs[comp])
            if i >= len(real_files):
                continue
            real_img = load_image_binary(real_files[i])

            # Compute center of mass (returns (y, x) for 2D)
            real_com_coords = ndimage.center_of_mass(real_img)

            # Handle case where component is empty (CoM will be NaN)
            if np.isnan(real_com_coords[0]):
                real_com_coords = (0.0, 0.0)

            real_com.append(real_com_coords)

        if len(real_com) == len(components):
            real_coms.append(np.array(real_com))  # Shape: (n_components, 2)

    # Process all generated images
    for i in range(len(gen_files_first)):
        gen_com = []
        for comp in components:
            gen_files = list_image_files(gen_component_dirs[comp])
            if i >= len(gen_files):
                continue
            gen_img = load_image_binary(gen_files[i])

            # Compute center of mass
            gen_com_coords = ndimage.center_of_mass(gen_img)

            # Handle empty case
            if np.isnan(gen_com_coords[0]):
                gen_com_coords = (0.0, 0.0)

            gen_com.append(gen_com_coords)

        if len(gen_com) == len(components):
            gen_coms.append(np.array(gen_com))  # Shape: (n_components, 2)

    real_coms = np.stack(real_coms)  # (N_real, n_components, 2)
    gen_coms = np.stack(gen_coms)    # (N_gen, n_components, 2)

    # ──────────────────────────────────────────────────────────────
    # 1. Component-wise metric
    # ──────────────────────────────────────────────────────────────
    component_results = {}
    for i, comp in enumerate(components):
        # Separate Y and X coordinates
        real_y = real_coms[:, i, 0]
        real_x = real_coms[:, i, 1]
        gen_y = gen_coms[:, i, 0]
        gen_x = gen_coms[:, i, 1]

        # Bootstrap WD for each dimension
        wd_y = bootstrap_wasserstein(real_y, gen_y, num_bootstrap)
        wd_x = bootstrap_wasserstein(real_x, gen_x, num_bootstrap)

        # Average across dimensions
        wd_mean = np.mean([np.mean(wd_y), np.mean(wd_x)])
        wd_std = np.std([np.mean(wd_y), np.mean(wd_x)])

        component_results[comp] = {
            'wasserstein': (float(wd_mean), float(wd_std))
        }

    # ──────────────────────────────────────────────────────────────
    # 2. Overall metric (flatten to (N*n_components, 2))
    # ──────────────────────────────────────────────────────────────
    real_flat = real_coms.reshape(-1, 2)
    gen_flat = gen_coms.reshape(-1, 2)
    overall_wd = wasserstein_distance_2d(real_flat, gen_flat)

    # ──────────────────────────────────────────────────────────────
    # 3. Vector metric (reshape to (N, n_components*2))
    # ──────────────────────────────────────────────────────────────
    real_vec = real_coms.reshape(len(real_coms), -1)
    gen_vec = gen_coms.reshape(len(gen_coms), -1)

    vector_wds = [
        wasserstein_distance(real_vec[:, d], gen_vec[:, d])
        for d in range(real_vec.shape[1])
    ]

    return {
        'component': component_results,
        'overall': {'wasserstein': overall_wd},
        'vector_wasserstein': (float(np.mean(vector_wds)), float(np.std(vector_wds)))
    }


# ============================================================================
# Cacheable Metric Interface
# ============================================================================

from .base_metric import CacheableMetric
from typing import Dict, Any


class CoMMetric(CacheableMetric):
    """Center of Mass metric with automatic caching via CacheableMetric interface."""

    @property
    def metric_name(self) -> str:
        return "com"

    def compute_real_data(self, real_dirs: Dict[str, Path], **kwargs) -> np.ndarray:
        """
        Compute CoM positions for all real images.

        Args:
            real_dirs: {component: Path} for real images
            **kwargs: Must include 'components' list

        Returns:
            Numpy array of shape (N_real, n_components, 2) with CoM positions
        """
        from .utils import list_image_files, load_image_binary

        components = kwargs.get('components', list(real_dirs.keys()))
        real_coms = []

        # Get all real image files for first component to determine count
        first_comp = components[0]
        real_files_first = list_image_files(real_dirs[first_comp])

        # Process all real images
        for i in range(len(real_files_first)):
            real_com = []
            for comp in components:
                real_files = list_image_files(real_dirs[comp])
                if i >= len(real_files):
                    continue
                real_img = load_image_binary(real_files[i])

                # Compute center of mass (returns (y, x) for 2D)
                real_com_coords = ndimage.center_of_mass(real_img)

                # Handle case where component is empty (CoM will be NaN)
                if np.isnan(real_com_coords[0]):
                    real_com_coords = (0.0, 0.0)

                real_com.append(real_com_coords)

            if len(real_com) == len(components):
                real_coms.append(np.array(real_com))  # Shape: (n_components, 2)

        return np.stack(real_coms)  # (N_real, n_components, 2)

    def compute_generated_data(self, gen_dirs: Dict[str, Path], **kwargs) -> np.ndarray:
        """
        Compute CoM positions for all generated images.

        Args:
            gen_dirs: {component: Path} for generated images
            **kwargs: Must include 'components' list

        Returns:
            Numpy array of shape (N_gen, n_components, 2) with CoM positions
        """
        from .utils import list_image_files, load_image_binary

        components = kwargs.get('components', list(gen_dirs.keys()))
        gen_coms = []

        # Get all generated image files for first component to determine count
        first_comp = components[0]
        gen_files_first = list_image_files(gen_dirs[first_comp])

        # Process all generated images
        for i in range(len(gen_files_first)):
            gen_com = []
            for comp in components:
                gen_files = list_image_files(gen_dirs[comp])
                if i >= len(gen_files):
                    continue
                gen_img = load_image_binary(gen_files[i])

                # Compute center of mass
                gen_com_coords = ndimage.center_of_mass(gen_img)

                # Handle empty case
                if np.isnan(gen_com_coords[0]):
                    gen_com_coords = (0.0, 0.0)

                gen_com.append(gen_com_coords)

            if len(gen_com) == len(components):
                gen_coms.append(np.array(gen_com))  # Shape: (n_components, 2)

        return np.stack(gen_coms)  # (N_gen, n_components, 2)

    def compute_distance(self, real_data: np.ndarray, gen_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute CoM metrics (component-wise, overall, vector).

        Args:
            real_data: Real CoM positions (N_real, n_components, 2)
            gen_data: Generated CoM positions (N_gen, n_components, 2)
            **kwargs: Can include 'num_bootstrap', 'components'

        Returns:
            {
                'component': {comp: {'wasserstein': (mean, std)}, ...},
                'overall': {'wasserstein': float},
                'vector_wasserstein': (mean, std)
            }
        """
        from .utils import bootstrap_wasserstein

        num_bootstrap = kwargs.get('num_bootstrap', 1000)
        components = kwargs.get('components', [f'comp_{i}' for i in range(real_data.shape[1])])

        # ──────────────────────────────────────────────────────────────
        # 1. Component-wise metric
        # ──────────────────────────────────────────────────────────────
        component_results = {}
        for i, comp in enumerate(components):
            # Separate Y and X coordinates
            real_y = real_data[:, i, 0]
            real_x = real_data[:, i, 1]
            gen_y = gen_data[:, i, 0]
            gen_x = gen_data[:, i, 1]

            # Bootstrap WD for each dimension
            wd_y = bootstrap_wasserstein(real_y, gen_y, num_bootstrap)
            wd_x = bootstrap_wasserstein(real_x, gen_x, num_bootstrap)

            # Average across dimensions
            wd_mean = np.mean([np.mean(wd_y), np.mean(wd_x)])
            wd_std = np.std([np.mean(wd_y), np.mean(wd_x)])

            component_results[comp] = {
                'wasserstein': (float(wd_mean), float(wd_std))
            }

        # ──────────────────────────────────────────────────────────────
        # 2. Overall metric (flatten to (N*n_components, 2))
        # ──────────────────────────────────────────────────────────────
        real_flat = real_data.reshape(-1, 2)
        gen_flat = gen_data.reshape(-1, 2)
        overall_wd = wasserstein_distance_2d(real_flat, gen_flat)

        # ──────────────────────────────────────────────────────────────
        # 3. Vector metric (reshape to (N, n_components*2))
        # ──────────────────────────────────────────────────────────────
        real_vec = real_data.reshape(len(real_data), -1)
        gen_vec = gen_data.reshape(len(gen_data), -1)

        vector_wds = [
            wasserstein_distance(real_vec[:, d], gen_vec[:, d])
            for d in range(real_vec.shape[1])
        ]

        return {
            'component': component_results,
            'overall': {'wasserstein': overall_wd},
            'vector_wasserstein': (float(np.mean(vector_wds)), float(np.std(vector_wds)))
        }
