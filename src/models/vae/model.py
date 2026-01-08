"""
Beta-VAE model for multi-component generation.

Uses ResNet encoder/decoder architecture with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder import Encoder, Decoder


def mask_components(x, p=0.5):
    """
    Vectorized component masking for inpainting training.

    With probability p, keeps ONE random component per sample and zeros the rest.
    This encourages the model to learn the joint distribution across components.

    Args:
        x: (B, C, H, W) - all components
        p: Probability of masking

    Returns:
        Masked x with ONE random component kept, others zeroed
    """
    if p == 0:
        return x

    B, C = x.shape[:2]
    device = x.device

    # Generate random masking decisions for each sample
    should_mask = torch.rand(B, device=device) < p

    # For samples that should be masked, select one random component to keep
    keep_indices = torch.randint(0, C, (B,), device=device)

    # Create mask: (B, C, 1, 1)
    mask = torch.zeros(B, C, 1, 1, device=device)
    mask[torch.arange(B, device=device), keep_indices] = 1.0

    # Apply masking only to samples where should_mask is True
    # For others, keep all components (mask = 1 everywhere)
    full_mask = torch.ones(B, C, 1, 1, device=device)
    final_mask = torch.where(should_mask.view(B, 1, 1, 1), mask, full_mask)

    return x * final_mask


class BetaVAE(nn.Module):
    """
    Beta-VAE for multi-component generation with ResNet architecture.

    Args:
        image_size: (H, W) - image dimensions
        channels: Number of components (5 for EPURE, 3 for TOY)
        cond_dim: Conditioning dimension (2 for width/height, 4 for performance)
        latent_dim: Latent space dimensionality
        nf: Base number of filters
        nf_max_encoder: Maximum filters in encoder (for backward compatibility, falls back to nf_max)
        nf_max_decoder: Maximum filters in decoder (for backward compatibility, falls back to nf_max)
        nf_max: Legacy parameter for maximum filters (used if nf_max_encoder/decoder not specified)
        beta: KL divergence weight
        dropout_p: Dropout probability
    """

    def __init__(
        self,
        image_size=(64, 64),
        channels=5,
        cond_dim=2,
        latent_dim=32,
        nf=72,
        nf_max_encoder=None,
        nf_max_decoder=None,
        nf_max=None,
        beta=1.0,
        dropout_p=0.1
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.beta = beta

        # Handle backward compatibility for nf_max parameter
        # Priority: nf_max_encoder/decoder > nf_max > default values
        if nf_max_encoder is None:
            nf_max_encoder = nf_max if nf_max is not None else 1024
        if nf_max_decoder is None:
            nf_max_decoder = nf_max if nf_max is not None else 512

        self.encoder = Encoder(channels, latent_dim, nf, nf_max_encoder, dropout_p, image_size)
        self.decoder = Decoder(latent_dim, channels, cond_dim, nf, nf_max_decoder, dropout_p, image_size)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent distribution.

        Note: logvar already has softplus applied in the encoder output.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond=None, mask_prob=0.0):
        """
        Forward pass with optional component masking.

        Args:
            x: Input components (B, C, H, W)
            cond: Conditioning (B, cond_dim) or None
            mask_prob: Probability of masking (progressive warmup during training)

        Returns:
            recon: Reconstructed components (B, C, H, W)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance with softplus applied (B, latent_dim)
        """
        # Apply component masking for inpainting training
        if mask_prob > 0:
            x = mask_components(x, p=mask_prob)

        # Encode (logvar has softplus applied inside encoder)
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode with conditioning
        recon = self.decoder(z, cond)

        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        """
        Compute VAE loss: Reconstruction + beta * KL divergence.

        Args:
            recon: Reconstructed images (B, C, H, W)
            x: Target images (B, C, H, W)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance with softplus applied (B, latent_dim)

        Returns:
            loss: Total loss
            recon_loss: Reconstruction loss (MSE)
            kl_loss: KL divergence
        """
        # Reconstruction loss: MSE with sum reduction, normalized by batch size
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)

        # KL divergence: uses torch.mean for aggregation
        # Note: logvar already has softplus applied, so it's always positive
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss with beta weighting on KL term
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
    def inpaint(self, partial_x, mask, cond=None, num_samples=10, device='cuda'):
        """
        Inpaint missing components using posterior sampling.

        Args:
            partial_x: Partial observation (B, C, H, W) - missing components are zero
            mask: Binary mask (B, C, H, W) or (B, C, 1, 1) - 1=observed, 0=missing
            cond: Conditioning (B, cond_dim) or None
            num_samples: Number of posterior samples to average

        Returns:
            recon: Inpainted image (B, C, H, W)
        """
        partial_x = partial_x.to(device)

        # Encode partial observation
        mu, logvar = self.encoder(partial_x)

        # Sample multiple times from posterior and average
        all_samples = []
        for _ in range(num_samples):
            z = self.reparameterize(mu, logvar)
            sample = self.decoder(z, cond)
            all_samples.append(sample)

        # Average over samples
        recon = torch.stack(all_samples).mean(dim=0)

        # Optionally: preserve observed components exactly
        # recon = mask * partial_x + (1 - mask) * recon

        return recon
