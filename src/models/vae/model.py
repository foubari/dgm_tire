"""
Beta-VAE model for EpureDGM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder import Encoder, Decoder


def mask_components(x, p=0.5):
    """
    Component masking for inpainting training (same pattern as VQVAE).

    Args:
        x: (B, C, H, W) - all components
        p: Probability of masking

    Returns:
        Masked x with ONE random component kept, others zeroed
    """
    if p == 0:
        return x

    B, C = x.shape[:2]
    masked_x = torch.zeros_like(x)

    for i in range(B):
        if torch.rand(1).item() < p:
            # Keep only ONE random component
            keep_idx = torch.randint(0, C, (1,)).item()
            masked_x[i, keep_idx] = x[i, keep_idx]
        else:
            # Keep all components (no masking)
            masked_x[i] = x[i]

    return masked_x


class BetaVAE(nn.Module):
    """
    Beta-VAE for multi-component generation.

    Args:
        image_size: (H, W)
        channels: Number of components (5 for EPURE, 3 for TOY)
        cond_dim: Conditioning dimension (2 for width/height)
        latent_dim: Latent space dimensionality
        nf: Base number of filters
        nf_max: Maximum number of filters
        beta: KL weight
        dropout_p: Dropout probability
    """

    def __init__(
        self,
        image_size=(64, 32),
        channels=5,
        cond_dim=2,
        latent_dim=32,
        nf=32,
        nf_max=256,
        beta=4.0,
        dropout_p=0.1
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(channels, latent_dim, nf, nf_max, dropout_p)
        self.decoder = Decoder(latent_dim, channels, cond_dim, nf, nf_max, dropout_p)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond=None, mask_prob=0.0):
        """
        Forward pass with optional masking.

        Args:
            x: (B, C, H, W)
            cond: (B, cond_dim) or None
            mask_prob: Masking probability for training

        Returns:
            recon: (B, C, H, W)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        # Apply masking if specified
        if mask_prob > 0:
            x = mask_components(x, p=mask_prob)

        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode with conditioning
        recon = self.decoder(z, cond)

        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        """
        Compute VAE loss: Reconstruction + beta * KL.

        Returns:
            loss: Total loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        return loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(self, num_samples, cond=None, device='cuda'):
        """
        Sample from prior.

        Args:
            num_samples: Number of samples
            cond: (num_samples, cond_dim) or None
            device: Device

        Returns:
            samples: (num_samples, C, H, W)
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z, cond)
        return samples

    @torch.no_grad()
    def inpaint(self, partial_x, mask, num_samples=10, device='cuda'):
        """
        Inpaint missing components.

        Args:
            partial_x: (B, C, H, W) - some components are zero
            mask: (B, C) - binary mask (1=observed, 0=missing)
            num_samples: Number of posterior samples to average

        Returns:
            recon: (B, C, H, W)
        """
        partial_x = partial_x.to(device)

        # Encode partial observation
        mu, logvar = self.encoder(partial_x)

        # Sample multiple times from posterior
        all_samples = []
        for _ in range(num_samples):
            z = self.reparameterize(mu, logvar)
            sample = self.decoder(z, cond=None)
            all_samples.append(sample)

        # Average over samples
        recon = torch.stack(all_samples).mean(dim=0)

        return recon
