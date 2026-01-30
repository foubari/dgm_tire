"""
GMRF MVAE for EpureDGM.

Gaussian Markov Random Field Multimodal VAE with cross-modal generation.
Full ICTAI implementation with:
- Complete covariance matrix assembly
- Learnable GMRF prior
- Gaussian conditional for cross-modal generation
"""

from .model import (
    GMRF_MVAE,
    GMRF_VAE,
    GMRF_VAE_EPURE,
    Epure_GMMVAE,
    Encoder,
    Decoder,
    Constants,
    assemble_covariance_matrix_corrected,
)
from .cov_model import OffDiagonalCov, compute_n_off_diag
from .objectives import compute_elbo_dist, kl_divergence_gaussians

__all__ = [
    'GMRF_MVAE',
    'GMRF_VAE',
    'GMRF_VAE_EPURE',
    'Epure_GMMVAE',
    'Encoder',
    'Decoder',
    'OffDiagonalCov',
    'compute_n_off_diag',
    'compute_elbo_dist',
    'kl_divergence_gaussians',
    'Constants',
    'assemble_covariance_matrix_corrected',
]
