"""
GMRF Multimodal VAE model with conditioning support.

This module implements the GMRF MVAE architecture:
- Full covariance matrix assembly with off-diagonal elements
- Learnable GMRF prior p(z)
- Gaussian conditional for cross-modal generation
- Optional conditioning via MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .encoder_decoder import GMRF_VAE_Component
from .cov_model import OffDiagonalCov
from .utils import assemble_covariance_matrix_corrected, Constants


class GMRF_MVAE(nn.Module):
    """
    GMRF Multimodal VAE with optional conditioning.

    This model uses:
    - Full covariance matrix assembly with off-diagonal elements
    - Learnable prior p(z) with full covariance structure
    - Gaussian conditional for cross-modal generation
    - Optional conditioning through encoder and decoder MLPs
    """

    def __init__(self, params):
        """
        Args:
            params: Parameter object with attributes:
                - num_components: int, number of modalities
                - latent_dim: int, latent dimension per modality
                - diagonal_transf: str, transformation for diagonal
                - hidden_dim: int, hidden dim for off-diagonal model
                - n_layers: int, number of layers in off-diagonal model
                - nf, nf_max_e, nf_max_d: filter settings
                - image_size: tuple (height, width)
                - cond_dim: int, conditioning dimension
                - device: str, device to use
                - reduced_diag: bool, whether to reduce diagonal
        """
        super(GMRF_MVAE, self).__init__()

        self.diagonal_transf = params.diagonal_transf
        self.device = params.device
        self.latent_dim = params.latent_dim
        self.cond_dim = getattr(params, 'cond_dim', 0)
        self.num_components = params.num_components

        # Create component VAEs
        self.modality_vaes = nn.ModuleList([
            GMRF_VAE_Component(params).to(params.device)
            for _ in range(self.num_components)
        ])

        # Off-diagonal covariance model
        cov_input_dims = [params.latent_dim for _ in range(self.num_components)]
        encoded_dims = [params.latent_dim for _ in range(self.num_components)]
        self.encoded_dims = encoded_dims

        self.off_diag_cov = OffDiagonalCov(
            input_dims=cov_input_dims,
            encoded_dims=encoded_dims,
            hidden_dim=params.hidden_dim,
            n_layers=params.n_layers
        ).to(self.device)

        # Total latent dimension across all modalities
        total_latent_dim = self.num_components * self.latent_dim

        # Prior parameters: mu_p, diag_p, off_diag_p
        self.mu_p = nn.Parameter(torch.ones(total_latent_dim, device=self.device) * 1e-4)
        self.reduced_diag = params.reduced_diag
        self.diag_p = nn.Parameter(torch.ones(total_latent_dim, device=self.device))

        self.off_diag_scale = 0.1
        self.off_diag_p = nn.Parameter(
            torch.ones(total_latent_dim * (total_latent_dim - 1) // 2, device=self.device) * self.off_diag_scale
        )

        # Storage for forward pass results
        self.recons = None
        self.qz_x = None
        self.z_x = None
        self.Sigmaq = None
        self.muq = None

    def get_sigma_p(self):
        """Compute the prior covariance matrix Sigma_p."""
        if self.reduced_diag:
            diag_elements = self.diag_p
        else:
            if self.diagonal_transf == 'relu':
                diag_elements = F.relu(self.diag_p) + Constants.relu_shift
            elif self.diagonal_transf == 'softplus':
                diag_elements = F.softplus(self.diag_p) + 1e-6
            elif self.diagonal_transf == 'square':
                diag_elements = torch.square(self.diag_p)
            elif self.diagonal_transf == 'exp':
                diag_elements = torch.exp(self.diag_p)
            elif self.diagonal_transf == 'sig':
                diag_elements = torch.sigmoid(self.diag_p)
            else:
                raise ValueError(f"Invalid diagonal_transf: {self.diagonal_transf}")

        # Build lower triangular matrix from off_diag_p
        total_dim = self.mu_p.shape[0]
        lower_matrix = torch.zeros(total_dim, total_dim, device=self.mu_p.device)
        tril_indices = torch.tril_indices(row=total_dim, col=total_dim, offset=-1)
        lower_matrix[tril_indices[0], tril_indices[1]] = self.off_diag_p

        # Make symmetric
        symmetric_matrix = lower_matrix + lower_matrix.T

        # Set diagonal
        sigma_p = symmetric_matrix.clone()
        sigma_p.diagonal(dim1=-2, dim2=-1).copy_(diag_elements)

        return sigma_p

    def get_prior(self):
        """Get the prior distribution p(z)."""
        mu = self.mu_p
        Sigma_p = self.get_sigma_p()
        return MultivariateNormal(mu, covariance_matrix=Sigma_p)

    def sample_from_pz(self, n_samples):
        """Sample from the prior p(z)."""
        distribution = self.get_prior()
        samples = distribution.sample((n_samples,))
        return samples

    def forward(self, x, cond=None, K=1):
        """
        Forward pass through the GMRF MVAE.

        Args:
            x: List of tensors, one per modality, each of shape (B, 1, H, W)
            cond: Conditioning tensor (B, cond_dim) or None
            K: Number of samples (default: 1, not used in current implementation)
        """
        # 1. Encoding Phase
        mus, Sigmas, off_diag_embed = [], [], []

        for x_, vae in zip(x, self.modality_vaes):
            mu, Sigma = vae.enc(x_, cond=cond)  # Pass conditioning
            mus.append(mu)
            Sigmas.append(Sigma)
            off_diag_embed.append(vae.enc.cov_embedding)

        # Calculate off-diagonal elements for q(z|X)
        off_diag_z_x = self.off_diag_cov(*off_diag_embed)

        # Concatenate means from all modalities
        mu_z_x = torch.cat(mus, dim=1)

        # Assemble the full covariance matrix for q(z|X)
        Sigma_x = assemble_covariance_matrix_corrected(mus, Sigmas, off_diag_z_x, self.encoded_dims)

        mu_z_x = mu_z_x.to(self.device)
        Sigma_x = Sigma_x.to(self.device)

        self.Sigmaq = Sigma_x
        self.muq = mu_z_x

        # Define the multivariate normal distribution for q(z|X)
        self.qz_x = MultivariateNormal(mu_z_x, covariance_matrix=Sigma_x)

        # Sample a latent vector using the reparameterization trick
        self.z_x = self.qz_x.rsample().to(self.device)

        # 2. Decoding Phase
        z_splits = torch.split(self.z_x, self.latent_dim, dim=1)

        mus = []
        for z, vae in zip(z_splits, self.modality_vaes):
            mu = vae.dec(z, cond=cond)  # Pass conditioning
            mus.append(mu)

        self.recons = mus

    def decode(self, z, cond=None):
        """Decode a latent vector to reconstructions."""
        z_splits = torch.split(z, self.latent_dim, dim=1)
        mus = []
        for z_split, vae in zip(z_splits, self.modality_vaes):
            mu = vae.dec(z_split, cond=cond)
            mus.append(mu)
        self.recons = mus
        return mus

    def generate(self, num_samples=1, cond=None):
        """
        Generate samples from p(z), optionally conditioned.

        Args:
            num_samples: Number of samples to generate
            cond: Optional conditioning tensor (num_samples, cond_dim)

        Returns:
            List of generated images, one per modality
        """
        z = self.sample_from_pz(num_samples)
        z_splits = torch.split(z, self.latent_dim, dim=1)
        mus = []
        for z_split, vae in zip(z_splits, self.modality_vaes):
            mus.append(vae.dec(z_split, cond=cond))
        return mus

    def conditional_generate(self, cond_img, idx_i, idx_cond, cond=None, n_sample=1):
        """
        Generate modality i given modality j using Gaussian conditional.

        Args:
            cond_img: Observed image for conditioning modality (B, 1, H, W)
            idx_i: Index of target modality to generate
            idx_cond: Index of conditioning modality (observed)
            cond: Optional conditioning tensor (B, cond_dim)
            n_sample: Number of samples to generate

        Returns:
            Generated images for modality i
        """
        # Encode conditioning modality
        m, l = self.modality_vaes[idx_cond].enc(cond_img, cond=cond)
        dist = MultivariateNormal(m, scale_tril=l)
        cond_z = dist.sample([n_sample])  # Shape: [n_sample, batch_size, latent_dim]

        if idx_i == idx_cond:
            return self.modality_vaes[idx_cond].dec(cond_z.squeeze(0), cond=cond)

        # Get prior parameters
        batch_size = cond_img.shape[0]
        mu_p_batch = self.mu_p.repeat(batch_size, 1)
        Sigma = self.get_sigma_p().repeat(batch_size, 1, 1)

        # Indices for slicing
        start_i, end_i = idx_i * self.latent_dim, (idx_i + 1) * self.latent_dim
        start_j, end_j = idx_cond * self.latent_dim, (idx_cond + 1) * self.latent_dim

        # Extract relevant blocks
        mu_i = mu_p_batch[:, start_i:end_i]
        mu_j = mu_p_batch[:, start_j:end_j]

        Sigma_ii = Sigma[:, start_i:end_i, start_i:end_i]
        Sigma_jj = Sigma[:, start_j:end_j, start_j:end_j]
        Sigma_c = Sigma[:, start_i:end_i, start_j:end_j]

        # Invert Sigma_jj
        Sigma_jj_inv = torch.inverse(Sigma_jj)

        # Compute conditional mean and covariance
        mu_cond = mu_i + torch.matmul(
            torch.matmul(Sigma_c, Sigma_jj_inv),
            (cond_z - mu_j).unsqueeze(-1)
        ).squeeze(-1).squeeze(0)

        Sigma_cond = Sigma_ii - torch.matmul(
            torch.matmul(Sigma_c, Sigma_jj_inv),
            Sigma_c.transpose(-2, -1)
        )

        # Generate samples
        samples = []
        for i in range(batch_size):
            cond_dist = MultivariateNormal(mu_cond[i], covariance_matrix=Sigma_cond[i])
            sample = cond_dist.sample((n_sample,))
            samples.append(sample)

        samples = torch.cat(samples, dim=0)
        conditional_generation = self.modality_vaes[idx_i].dec(samples, cond=cond)
        return conditional_generation

    def self_and_cross_modal_generation(self, data, cond=None):
        """
        Generate cross-modal reconstructions matrix.

        Returns a matrix where entry [i][j] is the reconstruction of modality j
        given modality i as input.
        """
        recons = [[None for _ in range(len(self.modality_vaes))] for _ in range(len(self.modality_vaes))]
        self.eval()

        with torch.no_grad():
            for idx_cond in range(len(self.modality_vaes)):
                for idx_i in range(len(self.modality_vaes)):
                    recons[idx_cond][idx_i] = self.conditional_generate(
                        data[idx_cond], idx_i, idx_cond, cond=cond, n_sample=1
                    )

        return recons

    def generate_for_calculating_unconditional_coherence(self, N, cond=None):
        """Generate samples for coherence calculation."""
        samples_list = self.generate(N, cond=cond)
        return [samples.data.cpu() for samples in samples_list]


class Epure_GMRF_MVAE(GMRF_MVAE):
    """
    GMRF MVAE for EPURE dataset with configurable components.

    Default components: group_nc, group_km, bt, fpu, tpc
    """

    def __init__(self, params):
        """
        Args:
            params: Parameter object with all GMRF_MVAE attributes plus:
                - component_names: optional list of component names
        """
        super().__init__(params)

        self.modelName = 'gmrf_mvae_epure'

        # Component names from params or default
        default_names = ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']
        self.components_name = getattr(params, 'component_names', default_names[:params.num_components])

        # Name each VAE
        for vae, comp in zip(self.modality_vaes, self.components_name):
            vae.modelName = comp
            vae.llik_scaling = 1.0
