"""
Encoder and Decoder networks for GMRF MVAE with conditioning support.

This module implements the ResNet-based encoder and decoder architecture,
adapted to support:
- Configurable image sizes (64x64, 64x32, etc.)
- Optional conditioning via MLP (matching mmvaeplus pattern)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet_block import ResnetBlock, actvn
from .utils import Constants


class Enc(nn.Module):
    """
    Encoder network for GMRF VAE with optional conditioning.

    Outputs latent mean and diagonal covariance matrix.
    """

    def __init__(self, latent_dim, diagonal_transf, nf=64, nf_max=1024,
                 image_size=(64, 32), cond_dim=0):
        """
        Args:
            latent_dim: Dimension of the latent space
            diagonal_transf: Transformation for diagonal ('relu', 'softplus', 'square', 'exp', 'sig')
            nf: Base number of filters
            nf_max: Maximum number of filters
            image_size: Tuple of (height, width) for input images
            cond_dim: Dimension of conditioning vector (0 = no conditioning)
        """
        super().__init__()
        self.diagonal_transf = diagonal_transf
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Configurable image size
        self.image_h, self.image_w = image_size
        # After 3 downsampling layers: 64->32->16->8
        self.s0_h = self.image_h // 8
        self.s0_w = self.image_w // 8

        self.nf = nf
        self.nf_max = nf_max

        # Number of downsampling layers
        nlayers = int(np.log2(self.image_h / self.s0_h))  # 3 layers for height 64
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        # ResNet blocks with downsampling
        blocks_z = [ResnetBlock(nf, nf)]
        for i in range(nlayers):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_z = nn.Conv2d(1, nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)

        # Conditioning MLP (matching mmvaeplus pattern)
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
            cond_channels = 256
        else:
            self.cond_mlp = None
            cond_channels = 0

        # FC layers with conditioning
        fc_input_dim = self.nf0 * self.s0_h * self.s0_w + cond_channels
        self.fc_mu_z = nn.Linear(fc_input_dim, latent_dim)
        self.lambda_diag_layer = nn.Linear(fc_input_dim, latent_dim)
        self.cov_layer = nn.Linear(fc_input_dim, latent_dim)

        # Embedding for off-diagonal model (set during forward pass)
        self.cov_embedding = None

    def forward(self, x, cond=None):
        """
        Forward pass.

        Args:
            x: Input image tensor of shape (B, 1, H, W)
            cond: Optional conditioning tensor of shape (B, cond_dim)

        Returns:
            mu_z: Latent mean of shape (B, latent_dim)
            lambda_z: Diagonal covariance matrix of shape (B, latent_dim, latent_dim)
        """
        # Encode through ResNet
        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size(0), self.nf0 * self.s0_h * self.s0_w)

        # Add conditioning if available
        if self.cond_mlp is not None:
            if cond is None:
                cond_emb = torch.zeros(x.size(0), 256, device=x.device, dtype=x.dtype)
            else:
                cond_emb = self.cond_mlp(cond)
            out_z = torch.cat([out_z, cond_emb], dim=1)

        # Store embedding for off-diagonal model
        self.cov_embedding = self.cov_layer(out_z)

        # Compute mean
        mu_z = self.fc_mu_z(out_z)

        # Compute diagonal covariance (must be positive)
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


class Dec(nn.Module):
    """
    Decoder network for GMRF VAE with optional conditioning.

    Generates images from latent vectors.
    """

    def __init__(self, latent_dim, nf=64, nf_max=512, image_size=(64, 32), cond_dim=0):
        """
        Args:
            latent_dim: Dimension of the latent space
            nf: Base number of filters
            nf_max: Maximum number of filters
            image_size: Tuple of (height, width) for output images
            cond_dim: Dimension of conditioning vector (0 = no conditioning)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Configurable image size
        self.image_h, self.image_w = image_size
        self.s0_h = self.image_h // 8
        self.s0_w = self.image_w // 8

        self.nf = nf
        self.nf_max = nf_max

        # Number of upsampling layers
        nlayers = int(np.log2(self.image_h / self.s0_h))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        # Conditioning MLP (matching mmvaeplus pattern)
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
            decoder_input_dim = latent_dim + 256
        else:
            self.cond_mlp = None
            decoder_input_dim = latent_dim

        # FC layer
        self.fc = nn.Linear(decoder_input_dim, self.nf0 * self.s0_h * self.s0_w)

        # ResNet blocks with upsampling
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]
        blocks += [ResnetBlock(nf, nf)]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 1, 3, padding=1)

    def forward(self, u, cond=None):
        """
        Forward pass.

        Args:
            u: Latent vector of shape (B, latent_dim)
            cond: Optional conditioning tensor of shape (B, cond_dim)

        Returns:
            out: Reconstructed image of shape (B, 1, H, W)
        """
        # Add conditioning if available
        if self.cond_mlp is not None:
            if cond is None:
                cond_emb = torch.zeros(u.size(0), 256, device=u.device, dtype=u.dtype)
            else:
                cond_emb = self.cond_mlp(cond)
            u = torch.cat([u, cond_emb], dim=-1)

        # Decode
        out = self.fc(u).view(-1, self.nf0, self.s0_h, self.s0_w)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))

        return out


class GMRF_VAE_Component(nn.Module):
    """
    Single component VAE for GMRF MVAE.

    Wraps encoder and decoder for one modality.
    """

    def __init__(self, params):
        """
        Args:
            params: Parameter object with attributes:
                - latent_dim: int
                - diagonal_transf: str
                - nf: int
                - nf_max_e: int (encoder)
                - nf_max_d: int (decoder)
                - image_size: tuple (height, width)
                - cond_dim: int
        """
        super().__init__()

        image_size = getattr(params, 'image_size', (64, 32))
        cond_dim = getattr(params, 'cond_dim', 0)

        self.enc = Enc(
            latent_dim=params.latent_dim,
            diagonal_transf=params.diagonal_transf,
            nf=params.nf,
            nf_max=params.nf_max_e,
            image_size=image_size,
            cond_dim=cond_dim
        )
        self.dec = Dec(
            latent_dim=params.latent_dim,
            nf=params.nf,
            nf_max=params.nf_max_d,
            image_size=image_size,
            cond_dim=cond_dim
        )

        self.latent_dim = params.latent_dim
        self.modelName = 'gmrf_vae_component'
        self.llik_scaling = 1.0
