"""
IoU and Dice metrics with Wasserstein distance.

Computes pairwise IoU/Dice between all component pairs, then measures
distribution distance using Wasserstein.
"""

import numpy as np
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from .utils import (
    load_image_binary,
    get_common_ids,
    bootstrap_wasserstein,
    compute_iou,
    compute_dice,
    extract_id_from_filename
)


def compute_iou_dice_distributions(
    real_component_dirs: Dict[str, Path],
    gen_component_dirs: Dict[str, Path],
    components: List[str],
    num_bootstrap: int = 1000
) -> dict:
    """
    Compute IoU/Dice distributions + Wasserstein distance.

    Args:
        real_component_dirs: {component: Path} for real images
        gen_component_dirs: {component: Path} for generated images
        components: List of component names
        num_bootstrap: Number of bootstrap iterations

    Returns:
        {
            (comp1, comp2): {
                'iou_wd': (mean, std),
                'dice_wd': (mean, std)
            },
            'average': {
                'iou_wd': (mean, std),
                'dice_wd': (mean, std)
            }
        }
    """
    from .utils import list_image_files

    iou_distributions = defaultdict(lambda: {'real': [], 'gen': []})
    dice_distributions = defaultdict(lambda: {'real': [], 'gen': []})

    # NO MATCHING - compute distributions separately for real and generated
    for c1, c2 in itertools.combinations(components, 2):
        # Real images: compute IoU/Dice for all pairs (c1, c2)
        real_files_c1 = list_image_files(real_component_dirs[c1])
        real_files_c2 = list_image_files(real_component_dirs[c2])
        num_real = min(len(real_files_c1), len(real_files_c2))

        for i in range(num_real):
            real1 = load_image_binary(real_files_c1[i])
            real2 = load_image_binary(real_files_c2[i])

            iou_real = compute_iou(real1, real2)
            dice_real = compute_dice(real1, real2)

            iou_distributions[(c1, c2)]['real'].append(iou_real)
            dice_distributions[(c1, c2)]['real'].append(dice_real)

        # Generated images: compute IoU/Dice for all pairs (c1, c2)
        gen_files_c1 = list_image_files(gen_component_dirs[c1])
        gen_files_c2 = list_image_files(gen_component_dirs[c2])
        num_gen = min(len(gen_files_c1), len(gen_files_c2))

        for i in range(num_gen):
            gen1 = load_image_binary(gen_files_c1[i])
            gen2 = load_image_binary(gen_files_c2[i])

            iou_gen = compute_iou(gen1, gen2)
            dice_gen = compute_dice(gen1, gen2)

            iou_distributions[(c1, c2)]['gen'].append(iou_gen)
            dice_distributions[(c1, c2)]['gen'].append(dice_gen)

    # Bootstrap Wasserstein distance for each pair
    results = {}
    all_iou_wds = []
    all_dice_wds = []

    for pair in iou_distributions.keys():
        iou_real = np.array(iou_distributions[pair]['real'])
        iou_gen = np.array(iou_distributions[pair]['gen'])
        dice_real = np.array(dice_distributions[pair]['real'])
        dice_gen = np.array(dice_distributions[pair]['gen'])

        # Bootstrap WD
        iou_wds = bootstrap_wasserstein(iou_real, iou_gen, num_bootstrap)
        dice_wds = bootstrap_wasserstein(dice_real, dice_gen, num_bootstrap)

        # Use string key for JSON serialization: "comp1_comp2"
        pair_key = f"{pair[0]}_{pair[1]}"
        results[pair_key] = {
            'iou_wd': (float(np.mean(iou_wds)), float(np.std(iou_wds))),
            'dice_wd': (float(np.mean(dice_wds)), float(np.std(dice_wds)))
        }

        all_iou_wds.extend(iou_wds)
        all_dice_wds.extend(dice_wds)

    # Compute average across all pairs
    results['average'] = {
        'iou_wd': (float(np.mean(all_iou_wds)), float(np.std(all_iou_wds))),
        'dice_wd': (float(np.mean(all_dice_wds)), float(np.std(all_dice_wds)))
    }

    return results


# ============================================================================
# Cacheable Metric Interface
# ============================================================================

from .base_metric import CacheableMetric
from typing import Dict, Any


class IoUDiceMetric(CacheableMetric):
    """IoU/Dice metric with automatic caching via CacheableMetric interface."""

    @property
    def metric_name(self) -> str:
        return "iou_dice"

    def compute_real_data(self, real_dirs: Dict[str, Path], **kwargs) -> Dict:
        """
        Compute IoU/Dice distributions for real images.

        Args:
            real_dirs: {component: Path} for real images
            **kwargs: Must include 'components' list

        Returns:
            Dict mapping pair keys to {'iou': [...], 'dice': [...]}
        """
        from .utils import list_image_files, load_image_binary, compute_iou, compute_dice
        import itertools
        from collections import defaultdict

        components = kwargs.get('components', list(real_dirs.keys()))
        distributions = defaultdict(lambda: {'iou': [], 'dice': []})

        for c1, c2 in itertools.combinations(components, 2):
            real_files_c1 = list_image_files(real_dirs[c1])
            real_files_c2 = list_image_files(real_dirs[c2])
            num_real = min(len(real_files_c1), len(real_files_c2))

            for i in range(num_real):
                real1 = load_image_binary(real_files_c1[i])
                real2 = load_image_binary(real_files_c2[i])

                iou_val = compute_iou(real1, real2)
                dice_val = compute_dice(real1, real2)

                pair_key = f"{c1}_{c2}"
                distributions[pair_key]['iou'].append(iou_val)
                distributions[pair_key]['dice'].append(dice_val)

        return dict(distributions)

    def compute_generated_data(self, gen_dirs: Dict[str, Path], **kwargs) -> Dict:
        """
        Compute IoU/Dice distributions for generated images.

        Args:
            gen_dirs: {component: Path} for generated images
            **kwargs: Must include 'components' list

        Returns:
            Dict mapping pair keys to {'iou': [...], 'dice': [...]}
        """
        from .utils import list_image_files, load_image_binary, compute_iou, compute_dice
        import itertools
        from collections import defaultdict

        components = kwargs.get('components', list(gen_dirs.keys()))
        distributions = defaultdict(lambda: {'iou': [], 'dice': []})

        for c1, c2 in itertools.combinations(components, 2):
            gen_files_c1 = list_image_files(gen_dirs[c1])
            gen_files_c2 = list_image_files(gen_dirs[c2])
            num_gen = min(len(gen_files_c1), len(gen_files_c2))

            for i in range(num_gen):
                gen1 = load_image_binary(gen_files_c1[i])
                gen2 = load_image_binary(gen_files_c2[i])

                iou_val = compute_iou(gen1, gen2)
                dice_val = compute_dice(gen1, gen2)

                pair_key = f"{c1}_{c2}"
                distributions[pair_key]['iou'].append(iou_val)
                distributions[pair_key]['dice'].append(dice_val)

        return dict(distributions)

    def compute_distance(self, real_data: Dict, gen_data: Dict, **kwargs) -> Dict[str, Any]:
        """
        Compute Wasserstein distance between IoU/Dice distributions.

        Args:
            real_data: Real distributions
            gen_data: Generated distributions
            **kwargs: Can include 'num_bootstrap' (default 1000)

        Returns:
            Dictionary with results per pair and average
        """
        from .utils import bootstrap_wasserstein

        num_bootstrap = kwargs.get('num_bootstrap', 1000)
        results = {}
        all_iou_wds = []
        all_dice_wds = []

        for pair_key in real_data.keys():
            iou_real = np.array(real_data[pair_key]['iou'])
            iou_gen = np.array(gen_data[pair_key]['iou'])
            dice_real = np.array(real_data[pair_key]['dice'])
            dice_gen = np.array(gen_data[pair_key]['dice'])

            iou_wds = bootstrap_wasserstein(iou_real, iou_gen, num_bootstrap)
            dice_wds = bootstrap_wasserstein(dice_real, dice_gen, num_bootstrap)

            results[pair_key] = {
                'iou_wd': (float(np.mean(iou_wds)), float(np.std(iou_wds))),
                'dice_wd': (float(np.mean(dice_wds)), float(np.std(dice_wds)))
            }

            all_iou_wds.extend(iou_wds)
            all_dice_wds.extend(dice_wds)

        # Compute average across all pairs
        results['average'] = {
            'iou_wd': (float(np.mean(all_iou_wds)), float(np.std(all_iou_wds))),
            'dice_wd': (float(np.mean(all_dice_wds)), float(np.std(all_dice_wds)))
        }

        return results
