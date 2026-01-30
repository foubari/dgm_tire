"""
GMRF MVAE - Full implementation matching ICTAI architecture exactly.

Adapted for EPURE dataset with 5 components and 64x32 images.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .cov_model import OffDiagonalCov
from .resnet_block import ResnetBlock, actvn


# Constants (matching ICTAI)
class Constants:
    eta = 1e-6
    relu_shift = 1
    exp_shift = 0
    exp_factor = 1


def assemble_covariance_matrix_corrected(mu_list, sigma_list, off_diag_coeffs, modalities_dim, epsilon=0.9, delta=1e-6):
    """
    Assembles the covariance matrix for a multimodal VAE, ensuring symmetry and positive definiteness.

    Exact copy from ICTAI implementation.

    Parameters:
    - mu_list: List of tensors, each of shape (batch_size, dim_k), mean vectors from modality encoders.
    - sigma_list: List of tensors, each of shape (batch_size, dim_k, dim_k), diagonal covariance matrices.
    - off_diag_coeffs: Tensor of shape (batch_size, num_off_diag_elements), output from the global encoder.
    - modalities_dim: List of ints, dimensions of each modality.
    - epsilon: Scalar or Tensor of shape (total_dim,), with values less than 1.
    - delta: Small positive scalar to prevent division by zero.

    Returns:
    - covariance_matrix: Tensor of shape (batch_size, total_dim, total_dim), the assembled covariance matrix.
    """
    total_dim = sum(modalities_dim)
    batch_size = mu_list[0].shape[0]
    device = mu_list[0].device

    # 1. Assemble Lambda, the big diagonal matrix from sigma_list
    sigma_diags = [torch.diagonal(sigma, dim1=-2, dim2=-1) for sigma in sigma_list]
    v = torch.cat(sigma_diags, dim=1)  # Shape: (batch_size, total_dim)

    # 2. Assemble M from off_diag_coeffs
    M = torch.zeros((batch_size, total_dim, total_dim), device=device)

    # Compute start and end indices for each modality
    start_indices = []
    end_indices = []
    start = 0
    for dim in modalities_dim:
        start_indices.append(start)
        end = start + dim
        end_indices.append(end)
        start = end

    # Prepare to fill M
    num_modalities = len(modalities_dim)
    off_diag_block_sizes = []
    modality_pairs = []
    for i in range(1, num_modalities):
        for j in range(i):
            block_size = modalities_dim[i] * modalities_dim[j]
            off_diag_block_sizes.append(block_size)
            modality_pairs.append((i, j))

    # Compute cumulative sum to get offsets
    off_diag_block_starts = [0]
    for size in off_diag_block_sizes:
        off_diag_block_starts.append(off_diag_block_starts[-1] + size)
    off_diag_block_starts = off_diag_block_starts[:-1]

    # Fill M with off-diagonal blocks
    for block_idx, (i, j) in enumerate(modality_pairs):
        start = off_diag_block_starts[block_idx]
        end = start + off_diag_block_sizes[block_idx]
        block_coeffs = off_diag_coeffs[:, start:end]
        block_coeffs = block_coeffs.view(batch_size, modalities_dim[i], modalities_dim[j])
        M[:, start_indices[i]:end_indices[i], start_indices[j]:end_indices[j]] = block_coeffs
        M[:, start_indices[j]:end_indices[j], start_indices[i]:end_indices[i]] = block_coeffs.transpose(1, 2)

    # 3. Compute s_i = sum_{j != i} |M_{ij}|
    s = torch.sum(torch.abs(M), dim=2) - torch.abs(torch.diagonal(M, dim1=1, dim2=2))
    s = s + delta

    # 4. Compute alpha_i
    if isinstance(epsilon, float) or isinstance(epsilon, int):
        epsilon = torch.full_like(v, epsilon)
    else:
        epsilon = epsilon.to(device)

    alpha = torch.minimum(torch.ones_like(s), (v * epsilon) / s)

    # 5. Compute alpha_{ij} = sqrt(alpha_i * alpha_j)
    alpha_i_sqrt = torch.sqrt(alpha)
    alpha_matrix = alpha_i_sqrt.unsqueeze(2) * alpha_i_sqrt.unsqueeze(1)

    # 6. Scale M symmetrically
    M_adjusted = M * alpha_matrix

    # 7. Construct the covariance matrix
    covariance_matrix = torch.diag_embed(v) + M_adjusted

    return covariance_matrix


class Encoder(nn.Module):
    """
    Encoder network for GMRF VAE - ResNet architecture.

    Adapted for EPURE images (64x32) from ICTAI original (64x64).
    """

    def __init__(self, latent_dim, diagonal_transf, nf=32, nf_max=512):
        super().__init__()
        self.diagonal_transf = diagonal_transf
        self.latent_dim = latent_dim

        # EPURE format: 64x32 -> final size 8x4
        self.s0_h = 8  # Final height
        self.s0_w = 4  # Final width
        self.nf = nf
        self.nf_max = nf_max

        # Number of downsampling layers: 64 -> 32 -> 16 -> 8 (3 layers)
        size = 64  # Use height for nlayers calculation
        nlayers = int(np.log2(size / self.s0_h))  # = 3
        self.nf0 = min(nf_max, nf * 2**nlayers)

        # Initial conv: 1 -> nf
        self.conv_img_z = nn.Conv2d(1, nf, 3, padding=1)

        # ResNet blocks with downsampling
        blocks_z = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.resnet_z = nn.Sequential(*blocks_z)

        # Output layers
        enc_out_dim = self.nf0 * self.s0_h * self.s0_w  # e.g., 256 * 8 * 4 = 8192
        self.fc_mu_z = nn.Linear(enc_out_dim, latent_dim)
        self.lambda_diag_layer = nn.Linear(enc_out_dim, latent_dim)
        self.cov_layer = nn.Linear(enc_out_dim, latent_dim)
        self.cov_embedding = None  # Will be fed to the off diagonal model

    def forward(self, x):
        """
        Args:
            x: (B, 1, 64, 32)

        Returns:
            mu_z: (B, latent_dim)
            lambda_z: (B, latent_dim, latent_dim) diagonal matrix
        """
        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size(0), self.nf0 * self.s0_h * self.s0_w)

        # Store embedding for off-diagonal covariance model
        self.cov_embedding = self.cov_layer(out_z)

        mu_z = self.fc_mu_z(out_z)

        # Diagonal transformation (must be positive)
        raw_diag = self.lambda_diag_layer(out_z)

        if self.diagonal_transf == 'relu':
            lambda_diag = F.relu(raw_diag) + Constants.relu_shift
        elif self.diagonal_transf == 'softplus':
            lambda_diag = F.softplus(raw_diag) + Constants.relu_shift
        elif self.diagonal_transf == 'square':
            lambda_diag = torch.square(raw_diag)
        elif self.diagonal_transf == 'exp':
            lambda_diag = torch.exp(raw_diag)
        elif self.diagonal_transf == 'sig':
            lambda_diag = torch.sigmoid(raw_diag)
        else:
            raise ValueError(f"Invalid diagonal_transf: {self.diagonal_transf}")

        # Construct diagonal matrix
        lambda_z = torch.zeros(out_z.size(0), self.latent_dim, self.latent_dim, device=out_z.device)
        lambda_z.diagonal(dim1=-2, dim2=-1).copy_(lambda_diag)

        return mu_z, lambda_z


class Decoder(nn.Module):
    """
    Decoder network for GMRF VAE - ResNet architecture.

    Adapted for EPURE images (64x32) from ICTAI original (64x64).
    """

    def __init__(self, latent_dim, nf=32, nf_max=256):
        super().__init__()
        self.latent_dim = latent_dim

        # EPURE format: start from 8x4 -> upsample to 64x32
        self.s0_h = 8
        self.s0_w = 4
        self.nf = nf
        self.nf_max = nf_max

        size = 64
        nlayers = int(np.log2(size / self.s0_h))  # = 3
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(latent_dim, self.nf0 * self.s0_h * self.s0_w)

        # ResNet blocks with upsampling
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [ResnetBlock(nf, nf)]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 1, 3, padding=1)

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim)

        Returns:
            out: (B, 1, 64, 32)
        """
        out = self.fc(z).view(-1, self.nf0, self.s0_h, self.s0_w)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))
        return out


class GMRF_VAE(nn.Module):
    """
    Base GMRF VAE for a single modality.
    """

    def __init__(self):
        super().__init__()
        self.enc = None
        self.dec = None
        self.latent_dim = None
        self.modelName = 'gmrf_vae'
        self.llik_scaling = 1.0


class GMRF_VAE_EPURE(GMRF_VAE):
    """
    GMRF VAE for EPURE dataset (64x32 images).
    """

    def __init__(self, params):
        super().__init__()
        self.enc = Encoder(params.latent_dim, params.diagonal_transf, params.nf, params.nf_max_e)
        self.dec = Decoder(params.latent_dim, params.nf, params.nf_max_d)
        self.latent_dim = params.latent_dim
        self.modelName = 'gmrf_vae_epure'
        self.params = params


class GMRF_MVAE(nn.Module):
    """
    GMRF Multimodal VAE - Full implementation matching ICTAI exactly.

    This model uses:
    - Full covariance matrix assembly with off-diagonal elements
    - Learnable prior p(z) with full covariance structure
    - Gaussian conditional for cross-modal generation
    """

    def __init__(self, params, off_diag_cov_class, *modality_vae_classes):
        super().__init__()

        self.diagonal_transf = params.diagonal_transf
        self.device = params.device

        # Individual VAEs for each modality
        self.modality_vaes = nn.ModuleList([
            vae_class(params).to(params.device) for vae_class in modality_vae_classes
        ])

        # Off-diagonal covariance model
        cov_input_dims = [params.latent_dim for _ in range(len(self.modality_vaes))]
        encoded_dims = [params.latent_dim for _ in range(len(self.modality_vaes))]
        self.encoded_dims = encoded_dims

        self.off_diag_cov = off_diag_cov_class(
            input_dims=cov_input_dims,
            encoded_dims=encoded_dims,
            hidden_dim=params.hidden_dim,
            n_layers=params.n_layers
        ).to(self.device)

        self.latent_dim = params.latent_dim

        # Total latent dimension across all modalities
        total_latent_dim = len(self.modality_vaes) * self.latent_dim

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
        """
        Compute the prior covariance matrix Sigma_p.
        """
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

    def forward(self, x, K=1):
        """
        Forward pass through the GMRF MVAE.

        Args:
            x: List of tensors, one per modality, each of shape (B, 1, 64, 32)
            K: Number of samples (default: 1)
        """
        # 1. Encoding Phase
        mus, Sigmas, off_diag_embed = [], [], []

        for x_, vae in zip(x, self.modality_vaes):
            mu, Sigma = vae.enc(x_)
            mus.append(mu)
            Sigmas.append(Sigma)
            cov_embedding = vae.enc.cov_embedding
            off_diag_embed.append(cov_embedding)

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
            mu = vae.dec(z)
            mus.append(mu)

        self.recons = mus

    def decode(self, z):
        """Decode a latent vector to reconstructions."""
        z_splits = torch.split(z, self.latent_dim, dim=1)
        mus = []
        for z_split, vae in zip(z_splits, self.modality_vaes):
            mu = vae.dec(z_split)
            mus.append(mu)
        self.recons = mus
        return mus

    def generate(self, num_samples=1):
        """Generate unconditional samples from p(z)."""
        z = self.sample_from_pz(num_samples)
        z_splits = torch.split(z, self.latent_dim, dim=1)
        mus = []
        for z_split, vae in zip(z_splits, self.modality_vaes):
            mus.append(vae.dec(z_split))
        return mus

    def conditional_generate(self, cond, idx_i, idx_cond, n_sample=1):
        """
        Compute conditional mean and covariance for generating X_i given X_j.

        This implements the Gaussian conditional:
        p(z_i | z_j) = N(mu_cond, Sigma_cond)

        Parameters:
        - cond: Observed data for the conditioning modality (B, 1, 64, 32)
        - idx_i: Index of target modality
        - idx_cond: Index of conditioning modality
        - n_sample: Number of samples to generate
        """
        # Encode conditioning modality
        m, l = self.modality_vaes[idx_cond].enc(cond)
        # ICTAI: l is diagonal matrix, used directly as scale_tril
        dist = MultivariateNormal(m, scale_tril=l)
        cond_z = dist.sample([n_sample])  # Shape: [n_sample, batch_size, latent_dim]

        if idx_i == idx_cond:
            return self.modality_vaes[idx_cond].dec(cond_z)

        # Get prior parameters
        batch_size = cond.shape[0]
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

        Sigma_cond = Sigma_ii - torch.matmul(torch.matmul(Sigma_c, Sigma_jj_inv), Sigma_c.transpose(-2, -1))

        # Generate samples
        samples = []
        for i in range(batch_size):
            cond_dist = MultivariateNormal(mu_cond[i], covariance_matrix=Sigma_cond[i])
            sample = cond_dist.sample((n_sample,))
            samples.append(sample)

        samples = torch.cat(samples, dim=0)
        conditional_generation = self.modality_vaes[idx_i].dec(samples)
        return conditional_generation

    def self_and_cross_modal_generation(self, data):
        """
        Generate cross-modal reconstructions matrix.

        Returns a matrix where entry [i][j] is the reconstruction of modality j
        given modality i as input.
        """
        recons = [[None for _ in range(len(self.modality_vaes))] for _ in range(len(self.modality_vaes))]
        self.eval()

        with torch.no_grad():
            for idx_cond in range(len(self.modality_vaes)):
                for idx_i, vae in enumerate(self.modality_vaes):
                    recons[idx_cond][idx_i] = self.conditional_generate(
                        data[idx_cond], idx_i, idx_cond, n_sample=1
                    )

        return recons


class Epure_GMMVAE(GMRF_MVAE):
    """
    GMRF MVAE for EPURE dataset with 5 components.

    Components: group_nc, group_km, bt, fpu, tpc
    """

    def __init__(self, params):
        # 5 modality VAEs for EPURE (without 'gi')
        super().__init__(
            params,
            OffDiagonalCov,
            GMRF_VAE_EPURE, GMRF_VAE_EPURE, GMRF_VAE_EPURE, GMRF_VAE_EPURE, GMRF_VAE_EPURE
        )
        self.modelName = 'gmrf_mvae_epure'
        self.components_name = ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']

        for vae, comp in zip(self.modality_vaes, self.components_name):
            vae.modelName = comp
            vae.llik_scaling = 1.0

    def generate_for_calculating_unconditional_coherence(self, N):
        """Generate samples for coherence calculation."""
        samples_list = super().generate(N)
        return [samples.data.cpu() for samples in samples_list]
