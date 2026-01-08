"""
GMRF MVAE - Simplified but functional version.

For full implementation with covariance assembly, refer to ICTAI codebase.
This version provides the core architecture with factorized approximation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .cov_model import OffDiagonalCov
from .resnet_block import ResnetBlock, actvn


class ComponentVAE(nn.Module):
    """VAE for a single component."""

    def __init__(self, latent_dim=16, nf=64, nf_max=512, dropout_p=0.1, cond_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.nf = nf
        self.nf_max = nf_max

        # Calculate channel progression with nf_max capping (ICTAI alignment)
        nf1 = min(nf, nf_max)        # Level 1: min(nf, nf_max)
        nf2 = min(nf*2, nf_max)      # Level 2: min(nf*2, nf_max)
        nf3 = min(nf*4, nf_max)      # Level 3: min(nf*4, nf_max)

        # Encoder: (B, 1, 64, 32) -> (B, latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, nf1, 4, 2, 1), nn.BatchNorm2d(nf1), nn.LeakyReLU(0.2),  # 32x16
            nn.Conv2d(nf1, nf2, 4, 2, 1), nn.BatchNorm2d(nf2), nn.LeakyReLU(0.2),  # 16x8
            nn.Conv2d(nf2, nf3, 4, 2, 1), nn.BatchNorm2d(nf3), nn.LeakyReLU(0.2),  # 8x4
            nn.Flatten(),
            nn.Dropout(dropout_p)
        )

        enc_out_dim = nf3 * 8 * 4  # Use nf3 (capped value)
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        self.fc_cov_emb = nn.Linear(enc_out_dim, latent_dim)  # For OffDiagonalCov

        # Decoder with conditioning
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(),
                nn.Linear(128, 256)
            )
            dec_input_dim = latent_dim + 256
        else:
            self.cond_mlp = None
            dec_input_dim = latent_dim

        self.fc_dec = nn.Linear(dec_input_dim, nf3 * 8 * 4)  # Use nf3 (capped value)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nf3, nf2, 4, 2, 1), nn.BatchNorm2d(nf2), nn.ReLU(),  # 16x8
            nn.ConvTranspose2d(nf2, nf1, 4, 2, 1), nn.BatchNorm2d(nf1), nn.ReLU(),  # 32x16
            nn.ConvTranspose2d(nf1, 1, 4, 2, 1), nn.Sigmoid()  # 64x32
        )

    def encode(self, x):
        """
        Args:
            x: (B, 1, 64, 32)

        Returns:
            mu, logvar, cov_emb
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = F.softplus(logvar)
        cov_emb = self.fc_cov_emb(h)
        return mu, logvar, cov_emb

    def decode(self, z, cond=None):
        """
        Args:
            z: (B, latent_dim)
            cond: (B, cond_dim) or None

        Returns:
            recon: (B, 1, 64, 32)
        """
        if self.cond_mlp is not None:
            if cond is not None:
                cond_emb = self.cond_mlp(cond)
            else:
                # Use zero conditioning for unconditional generation
                cond_emb = torch.zeros(z.size(0), 256, device=z.device, dtype=z.dtype)
            z = torch.cat([z, cond_emb], dim=1)

        h = self.fc_dec(z)
        h = h.view(h.size(0), -1, 8, 4)
        recon = self.decoder(h)
        return recon


class ComponentVAE_Resnet(nn.Module):
    """
    VAE for a single component - ResNet architecture (ICTAI alignment).

    This implementation uses ResnetBlock with skip connections,
    matching the ICTAI original exactly.
    """

    def __init__(
        self,
        latent_dim=4,
        nf=32,
        nf_max_e=512,
        nf_max_d=256,
        diagonal_transf="softplus",
        cond_dim=0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.nf = nf
        self.nf_max_e = nf_max_e
        self.nf_max_d = nf_max_d
        self.diagonal_transf = diagonal_transf

        # Image size: 64x32 → 8x4 (3 downsampling layers)
        dataSize = [1, 64, 32]  # EPURE format (height x width)
        s0_h = self.s0_h = 8  # Final height
        s0_w = self.s0_w = 4  # Final width
        size = dataSize[1]  # Use height for nlayers calculation

        # ENCODER
        nlayers = int(np.log2(size / s0_h))  # 3 layers: 64→32→16→8
        self.nf0_e = min(nf_max_e, nf * 2**nlayers)  # min(512, 32*8) = 256

        # Initial conv: 1 → nf
        self.conv_img_z = nn.Conv2d(1, nf, 3, padding=1)

        # ResNet blocks with downsampling
        blocks_z = [ResnetBlock(nf, nf)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max_e)
            nf1 = min(nf * 2 ** (i + 1), nf_max_e)
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),  # Downsample
                ResnetBlock(nf0, nf1),
            ]

        self.resnet_z = nn.Sequential(*blocks_z)

        # Output layers
        enc_out_dim = self.nf0_e * s0_h * s0_w  # 256 * 8 * 4 = 8192
        self.fc_mu_z = nn.Linear(enc_out_dim, latent_dim)
        self.lambda_diag_layer = nn.Linear(enc_out_dim, latent_dim)
        self.cov_layer = nn.Linear(enc_out_dim, latent_dim)  # For OffDiagonalCov

        # DECODER
        self.nf0_d = min(nf_max_d, nf * 2**nlayers)  # min(256, 32*8) = 256

        # Conditioning MLP (optional)
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128), nn.GELU(), nn.Linear(128, 256)
            )
            dec_input_dim = latent_dim + 256
        else:
            self.cond_mlp = None
            dec_input_dim = latent_dim

        self.fc_dec = nn.Linear(dec_input_dim, self.nf0_d * s0_h * s0_w)

        # ResNet blocks with upsampling
        blocks_dec = []
        for i in range(nlayers):
            nf0 = min(nf * 2 ** (nlayers - i), nf_max_d)
            nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max_d)
            blocks_dec += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2),  # Nearest-neighbor upsampling
            ]

        blocks_dec += [ResnetBlock(nf, nf)]

        self.resnet_dec = nn.Sequential(*blocks_dec)
        self.conv_img_dec = nn.Conv2d(nf, 1, 3, padding=1)

    def encode(self, x):
        """
        Encode input to latent distribution.

        Args:
            x: (B, 1, 64, 32)

        Returns:
            mu_z: Mean (B, latent_dim)
            lambda_z: Diagonal covariance matrix (B, latent_dim, latent_dim)
            cov_emb: Covariance embedding for OffDiagonalCov (B, latent_dim)
        """
        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)
        out_z = out_z.view(out_z.size(0), self.nf0_e * self.s0_h * self.s0_w)

        mu_z = self.fc_mu_z(out_z)
        cov_emb = self.cov_layer(out_z)

        # Diagonal transformation
        raw_diag = self.lambda_diag_layer(out_z)

        if self.diagonal_transf == "softplus":
            lambda_diag = F.softplus(raw_diag) + 1e-6
        elif self.diagonal_transf == "relu":
            lambda_diag = F.relu(raw_diag) + 1.0
        elif self.diagonal_transf == "exp":
            lambda_diag = torch.exp(raw_diag)
        elif self.diagonal_transf == "square":
            lambda_diag = torch.square(raw_diag)
        elif self.diagonal_transf == "sig":
            lambda_diag = torch.sigmoid(raw_diag)
        else:
            raise ValueError(f"Unknown diagonal_transf: {self.diagonal_transf}")

        # Create diagonal matrix
        lambda_z = torch.zeros(
            out_z.size(0), self.latent_dim, self.latent_dim, device=out_z.device
        )
        lambda_z.diagonal(dim1=-2, dim2=-1).copy_(lambda_diag)

        return mu_z, lambda_z, cov_emb

    def decode(self, z, cond=None):
        """
        Decode latent to reconstruction.

        Args:
            z: (B, latent_dim)
            cond: (B, cond_dim) or None

        Returns:
            recon: (B, 1, 64, 32)
        """
        if self.cond_mlp is not None:
            if cond is not None:
                cond_emb = self.cond_mlp(cond)
            else:
                cond_emb = torch.zeros(
                    z.size(0), 256, device=z.device, dtype=z.dtype
                )
            z = torch.cat([z, cond_emb], dim=1)

        out = self.fc_dec(z).view(-1, self.nf0_d, self.s0_h, self.s0_w)
        out = self.resnet_dec(out)
        out = self.conv_img_dec(actvn(out))

        return out


class GMRF_MVAE(nn.Module):
    """
    GMRF Multimodal VAE.

    Simplified version with factorized ELBO for efficiency.
    For full GMRF prior with covariance assembly, see ICTAI implementation.
    """

    def __init__(
        self,
        num_components=5,
        latent_dim=16,
        nf=64,
        nf_max=512,  # Backward compatibility (used when use_resnet=False)
        nf_max_e=512,  # Encoder max filters (ResNet mode)
        nf_max_d=256,  # Decoder max filters (ResNet mode)
        hidden_dim=256,
        n_layers=2,
        beta=1.0,
        cond_dim=2,
        dropout_p=0.1,
        diagonal_transf='softplus',  # For ResNet mode
        use_resnet=True,  # ICTAI alignment: True for ResNet, False for simple Conv
    ):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_resnet = use_resnet

        # Component VAEs - choose architecture
        if use_resnet:
            # ICTAI original: ResNet architecture
            VAEClass = ComponentVAE_Resnet
            vae_kwargs = {
                'latent_dim': latent_dim,
                'nf': nf,
                'nf_max_e': nf_max_e,
                'nf_max_d': nf_max_d,
                'diagonal_transf': diagonal_transf,
                'cond_dim': cond_dim,
            }
        else:
            # Fallback: Simple Conv2d architecture
            VAEClass = ComponentVAE
            vae_kwargs = {
                'latent_dim': latent_dim,
                'nf': nf,
                'nf_max': nf_max,
                'dropout_p': dropout_p,
                'cond_dim': cond_dim,
            }

        self.component_vaes = nn.ModuleList([
            VAEClass(**vae_kwargs)
            for _ in range(num_components)
        ])

        # OffDiagonalCov network (optional, for future full implementation)
        self.off_diag_cov = OffDiagonalCov(
            input_dims=[latent_dim] * num_components,
            encoded_dims=[latent_dim] * num_components,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )

        # Learnable prior (simple factorized version)
        self.register_parameter('mu_p', nn.Parameter(torch.zeros(num_components * latent_dim)))
        self.register_parameter('logvar_p', nn.Parameter(torch.zeros(num_components * latent_dim)))

    def forward(self, x, cond=None):
        """
        Args:
            x: (B, num_components, 64, 32)
            cond: (B, cond_dim) or None

        Returns:
            recons: List of (B, 1, 64, 32) per component
            mus, logvars: Encoded distributions
        """
        B = x.size(0)
        mus, logvars, cov_embs = [], [], []
        recons = []

        # Encode each component
        for i, vae in enumerate(self.component_vaes):
            encoded = vae.encode(x[:, i:i+1])

            if self.use_resnet:
                # ComponentVAE_Resnet returns: mu, lambda_z (matrix), cov_emb
                mu_i, lambda_z_i, cov_emb_i = encoded
                # Convert lambda_z (diagonal matrix) to logvar (vector)
                # lambda_z is diagonal, so extract diagonal elements
                lambda_diag = lambda_z_i.diagonal(dim1=-2, dim2=-1)  # (B, latent_dim)
                # logvar = log(lambda_diag)
                logvar_i = torch.log(lambda_diag + 1e-8)
            else:
                # ComponentVAE returns: mu, logvar, cov_emb
                mu_i, logvar_i, cov_emb_i = encoded

            mus.append(mu_i)
            logvars.append(logvar_i)
            cov_embs.append(cov_emb_i)

            # Reparameterize and decode
            std_i = torch.exp(0.5 * logvar_i)
            eps = torch.randn_like(std_i)
            z_i = mu_i + eps * std_i

            recon_i = vae.decode(z_i, cond)
            recons.append(recon_i)

        return recons, mus, logvars

    def loss_function(self, recons, x, mus, logvars,
                      recon_weights=None, loss_type='mse', alpha_mse=0.5):
        """
        ELBO loss with support for multiple reconstruction loss types.

        Supports two modes:
        1. Simplified factorized KL (default, use_resnet=False)
        2. ICTAI mode with split_l1_mse (use_resnet=True)

        Args:
            recons: List of reconstructions
            x: Input tensor (B, num_components, H, W)
            mus: List of means
            logvars: List of logvars
            recon_weights: Per-component weights (for ICTAI mode)
            loss_type: 'mse', 'l1', 'l1_mse', 'split_l1_mse', 'bce'
            alpha_mse: Weight for MSE in l1_mse mode

        Returns:
            loss, recon_loss, kl_loss
        """
        B = x.size(0)

        # Reconstruction loss
        if recon_weights is not None and loss_type != 'mse':
            # ICTAI mode: use weighted reconstruction loss
            data_list = [x[:, i:i+1] for i in range(self.num_components)]

            if loss_type == 'split_l1_mse':
                # ICTAI original formula
                mse_term = sum(
                    w * F.mse_loss(recon, d)
                    for w, recon, d in zip(recon_weights, recons, data_list)
                )
                l1_term = sum(
                    (1 - w) * F.l1_loss(recon, d)
                    for w, recon, d in zip(recon_weights, recons, data_list)
                )
                recon_loss = mse_term + l1_term
            elif loss_type == 'l1':
                recon_loss = sum(
                    w * F.l1_loss(recon, d)
                    for w, recon, d in zip(recon_weights, recons, data_list)
                )
            elif loss_type == 'l1_mse':
                recon_loss = sum(
                    w * (alpha_mse * F.mse_loss(recon, d) + (1 - alpha_mse) * F.l1_loss(recon, d))
                    for w, recon, d in zip(recon_weights, recons, data_list)
                )
            elif loss_type == 'bce':
                recon_loss = sum(
                    w * F.binary_cross_entropy_with_logits(recon, d)
                    for w, recon, d in zip(recon_weights, recons, data_list)
                )
            else:
                # Default MSE with weights
                recon_loss = sum(
                    w * F.mse_loss(recon, x[:, i:i+1])
                    for i, (w, recon) in enumerate(zip(recon_weights, recons))
                )
        else:
            # Simple mode: MSE without weights (backward compatibility)
            recon_loss = 0
            for i, recon_i in enumerate(recons):
                recon_loss += F.mse_loss(recon_i, x[:, i:i+1], reduction='sum')
            recon_loss = recon_loss / B

        # KL divergence (factorized)
        kl_loss = 0
        for i, (mu_i, logvar_i) in enumerate(zip(mus, logvars)):
            # Prior for component i
            mu_p_i = self.mu_p[i*self.latent_dim:(i+1)*self.latent_dim]
            logvar_p_i = self.logvar_p[i*self.latent_dim:(i+1)*self.latent_dim]

            # KL(q||p) for Gaussian
            kl_i = -0.5 * torch.sum(
                1 + logvar_i - logvar_p_i - (mu_i - mu_p_i).pow(2) / logvar_p_i.exp() - (logvar_i.exp() / logvar_p_i.exp())
            )
            kl_loss += kl_i

        kl_loss = kl_loss / B

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        return loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, num_samples, cond=None, device='cuda'):
        """Sample from prior."""
        samples = []
        for i, vae in enumerate(self.component_vaes):
            mu_p_i = self.mu_p[i*self.latent_dim:(i+1)*self.latent_dim]
            logvar_p_i = self.logvar_p[i*self.latent_dim:(i+1)*self.latent_dim]

            # Sample from prior
            std_p_i = torch.exp(0.5 * logvar_p_i)
            z_i = mu_p_i + torch.randn(num_samples, self.latent_dim).to(device) * std_p_i

            # Decode
            sample_i = vae.decode(z_i, cond)
            samples.append(sample_i)

        # Stack: (num_samples, num_components, 64, 32)
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def cross_modal_generate(self, observed_comp, source_idx, target_idx, device='cuda'):
        """
        Simple cross-modal generation (simplified version).

        For full Gaussian conditional, see ICTAI implementation.
        """
        # Encode observed component
        mu_obs, logvar_obs, _ = self.component_vaes[source_idx].encode(observed_comp.to(device))

        # Sample from posterior
        std_obs = torch.exp(0.5 * logvar_obs)
        z_obs = mu_obs + torch.randn_like(std_obs) * std_obs

        # Simple approach: use shared latent assumption
        # (For full conditional distribution, implement Gaussian conditioning from ICTAI)
        z_target = z_obs  # Simplified: assume shared structure

        # Decode target component
        recon_target = self.component_vaes[target_idx].decode(z_target, cond=None)

        return recon_target
