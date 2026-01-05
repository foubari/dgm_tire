"""
Evaluation metrics for EpureDGM.

Adapted from ICTAI evaluation notebook with support for both EPURE and TOY datasets.
"""

from .fid import compute_fid
from .iou_dice import compute_iou_dice_distributions
from .rce import compute_rce
from .com import compute_com_metrics

__all__ = [
    'compute_fid',
    'compute_iou_dice_distributions',
    'compute_rce',
    'compute_com_metrics',
]
