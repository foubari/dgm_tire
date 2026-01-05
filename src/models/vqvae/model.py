"""
Main VQ-VAE model combining encoder, quantizer, and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer


def mask_components(x, p: float):
    """
    With probability p, keep ONE random component (channel) per sample
    and zero-out the others. Adapted from ICTAI VAE implementation.

    Args:
        x: (B, C, H, W) - input images with C components
        p: float - probability of masking (0.0 = no masking, 1.0 = always mask)

    Returns:
        x_masked: (B, C, H, W) - masked images
    """
    if p == 0:
        return x

    B, C, H, W = x.shape
    keep_mask = torch.rand(B, device=x.device) < p
    if not keep_mask.any():
        return x  # nothing masked this batch

    # build [B,C,1,1] boolean mask that is True only on the kept channel
    idx = torch.randint(0, C, (B,), device=x.device)  # channel to KEEP
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[torch.arange(B, device=x.device), idx] = True

    x_masked = x.clone()
    x_masked[~mask & keep_mask[:, None, None, None]] = 0.
    return x_masked


class VQVAE(nn.Module):
    """
    VQ-VAE model with encoder, vector quantizer, and decoder.
    
    Supports conditional generation via conditioning in the decoder.
    """
    
    def __init__(
        self,
        image_size=(64, 32),
        channels=5,
        cond_dim=2,
        latent_dim=20,
        num_embeddings=512,
        commitment_cost=0.25,
        ema_decay=0.99,
        base_dim=64,
        dim_mults=(1, 2, 4),
    ):
        super().__init__()
        
        self.image_size = image_size
        self.channels = channels
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        # Build components
        self.encoder = Encoder(
            in_channels=channels,
            latent_dim=latent_dim,
            base_dim=base_dim,
            dim_mults=dim_mults
        )
        
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            out_channels=channels,
            cond_dim=cond_dim,
            base_dim=base_dim,
            dim_mults=dim_mults
        )
    
    def forward(self, x, cond=None, target=None):
        """
        Forward pass for training.

        Args:
            x: (B, channels, H, W) - input images (possibly masked)
            cond: (B, cond_dim) or None - conditioning vector
            target: (B, channels, H, W) or None - reconstruction target
                   If None, uses x as target (standard reconstruction)
                   If provided, enables masked training (x=masked, target=full)

        Returns:
            total_loss: scalar tensor
            loss_dict: dict with individual loss components
        """
        # Encode
        z_e = self.encoder(x)  # (B, latent_dim, H//4, W//4)

        # Quantize
        z_q, indices, (vq_loss, commitment_loss) = self.quantizer(z_e)

        # Decode
        x_recon = self.decoder(z_q, cond)

        # Reconstruction loss (against target if provided, else against input)
        if target is None:
            target = x
        recon_loss = F.mse_loss(x_recon, target)

        # Total loss
        total_loss = recon_loss + vq_loss + commitment_loss

        loss_dict = {
            'recon': recon_loss.item(),
            'vq': vq_loss.item(),
            'commit': commitment_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict
    
    def encode(self, x):
        """
        Encode images to discrete latent indices.
        
        Args:
            x: (B, channels, H, W) - input images
        
        Returns:
            indices: (B, H//4, W//4) - discrete code indices
        """
        z_e = self.encoder(x)
        _, indices, _ = self.quantizer(z_e)
        return indices
    
    def decode(self, indices, cond=None):
        """
        Decode discrete latent indices to images.
        
        Args:
            indices: (B, H, W) - discrete code indices
            cond: (B, cond_dim) or None - conditioning vector
        
        Returns:
            x_recon: (B, channels, H*4, W*4) - reconstructed images
        """
        # Lookup embeddings from codebook
        z_q = self.quantizer.embedding(indices)  # (B, H, W, embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2)  # (B, embedding_dim, H, W)
        
        # Decode
        return self.decoder(z_q, cond)
    
    def reconstruct(self, x, cond=None):
        """
        Reconstruct images (encode -> quantize -> decode).
        
        Args:
            x: (B, channels, H, W) - input images
            cond: (B, cond_dim) or None - conditioning vector
        
        Returns:
            x_recon: (B, channels, H, W) - reconstructed images
        """
        z_e = self.encoder(x)
        z_q, _, _ = self.quantizer(z_e)
        return self.decoder(z_q, cond)

