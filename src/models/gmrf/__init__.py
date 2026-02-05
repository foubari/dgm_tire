"""
GMRF Multimodal VAE with conditioning support.

This module implements the GMRF MVAE architecture with:
- Full covariance matrix assembly with off-diagonal elements
- Learnable GMRF prior p(z)
- Gaussian conditional for cross-modal generation
- Optional conditioning through encoder and decoder MLPs
"""

from .model import GMRF_MVAE, Epure_GMRF_MVAE
from .encoder_decoder import Enc, Dec, GMRF_VAE_Component
from .cov_model import OffDiagonalCov
from .objectives import compute_elbo_dist, kl_divergence_gaussians

__all__ = [
    'GMRF_MVAE',
    'Epure_GMRF_MVAE',
    'Enc',
    'Dec',
    'GMRF_VAE_Component',
    'OffDiagonalCov',
    'compute_elbo_dist',
    'kl_divergence_gaussians',
]
