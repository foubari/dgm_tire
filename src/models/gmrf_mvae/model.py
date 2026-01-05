"""
GMRF MVAE - Simplified but functional version.

For full implementation with covariance assembly, refer to ICTAI codebase.
This version provides the core architecture with factorized approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from .cov_model import OffDiagonalCov


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
        nf_max=512,
        hidden_dim=256,
        n_layers=2,
        beta=1.0,
        cond_dim=2,
        dropout_p=0.1
    ):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.beta = beta

        # Component VAEs
        self.component_vaes = nn.ModuleList([
            ComponentVAE(latent_dim, nf, nf_max, dropout_p, cond_dim)
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
            mu_i, logvar_i, cov_emb_i = vae.encode(x[:, i:i+1])
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

    def loss_function(self, recons, x, mus, logvars):
        """
        Simplified ELBO with factorized KL.

        Returns:
            loss, recon_loss, kl_loss
        """
        B = x.size(0)

        # Reconstruction loss (per component)
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
