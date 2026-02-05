"""
Standard GAN wrapper combining encoder, generator, and discriminator.
Main interface matching WGAN-GP's API (forward, sample, inpaint).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .encoder import Encoder
from .generator import Generator
from .discriminator import Discriminator


class GAN(nn.Module):
    """
    Standard GAN model wrapper.

    Uses BCE loss with label smoothing and 1:1 D:G update ratio.
    """

    def __init__(
        self,
        encoder,
        generator,
        discriminator,
        *,
        image_size=(64, 32),
        latent_dim=20,
        label_smoothing=0.1,
        cond_drop_prob=0.1,
    ):
        """
        Args:
            encoder: Encoder network (image -> latent)
            generator: Generator network (latent -> image)
            discriminator: Discriminator network (image -> P(real))
            image_size: Tuple (H, W) = (64, 32)
            latent_dim: Total latent dimensions (default 20)
            label_smoothing: Smooth real labels to 1-label_smoothing (default 0.1)
            cond_drop_prob: CFG dropout probability (default 0.1)
        """
        super().__init__()

        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.label_smoothing = label_smoothing
        self.cond_drop_prob = cond_drop_prob

    def forward(self, img, *, cond=None, train_discriminator=True):
        """
        Training step - returns dict of losses.

        Args:
            img: Real images (B, 5, 64, 32) in [0, 1]
            cond: Conditioning (B, cond_dim) or None
            train_discriminator: If True, compute discriminator loss; else generator loss

        Returns:
            Dict with keys:
            - 'discriminator_loss': D loss (when train_discriminator=True)
            - 'generator_loss': G loss (when train_discriminator=False)
            - 'd_real_mean': Mean D(real) score
            - 'd_fake_mean': Mean D(fake) score
        """
        batch_size = img.size(0)

        if train_discriminator:
            # Sample random latent for fake images
            z = torch.randn(batch_size, self.latent_dim, device=img.device)

            # Conditional dropout for CFG
            cond_for_gen = None
            if cond is not None and random.random() >= self.cond_drop_prob:
                cond_for_gen = cond

            # Generate fake images
            with torch.no_grad():
                fake = self.generator(z, cond=cond_for_gen)

            # Discriminator scores
            d_real = self.discriminator(img, cond=cond)
            d_fake = self.discriminator(fake.detach(), cond=cond)

            # BCE loss with label smoothing on real labels
            real_labels = torch.full_like(d_real, 1.0 - self.label_smoothing)
            fake_labels = torch.zeros_like(d_fake)

            loss_real = F.binary_cross_entropy(d_real, real_labels)
            loss_fake = F.binary_cross_entropy(d_fake, fake_labels)
            d_loss = loss_real + loss_fake

            return {
                'discriminator_loss': d_loss,
                'd_real_mean': d_real.mean().item(),
                'd_fake_mean': d_fake.mean().item(),
            }
        else:
            # Generator update
            z = torch.randn(batch_size, self.latent_dim, device=img.device)

            # Conditional dropout
            cond_for_gen = None
            if cond is not None and random.random() >= self.cond_drop_prob:
                cond_for_gen = cond

            # Generate fake images
            fake = self.generator(z, cond=cond_for_gen)

            # Generator loss: wants discriminator to output 1 for fakes
            d_fake = self.discriminator(fake, cond=cond)
            g_loss = F.binary_cross_entropy(d_fake, torch.ones_like(d_fake))

            return {
                'generator_loss': g_loss,
                'd_fake_mean': d_fake.mean().item(),
            }

    @torch.inference_mode()
    def sample(self, *, batch_size=16, cond=None, guidance_scale=0.0):
        """
        Generate samples.

        Args:
            batch_size: Number of samples
            cond: Conditioning (B, cond_dim) or None
            guidance_scale: CFG scale (0=unconditional, >0=conditional)

        Returns:
            Generated images (B, 5, 64, 32) in [0, 1]
        """
        device = next(self.generator.parameters()).device

        # Sample random latent
        z = torch.randn(batch_size, self.latent_dim, device=device)

        if guidance_scale > 0 and cond is not None:
            # Classifier-free guidance
            # Generate unconditional
            fake_uncond = self.generator(z, cond=None)
            # Generate conditional
            fake_cond = self.generator(z, cond=cond)
            # Apply guidance: x = x_uncond + scale * (x_cond - x_uncond)
            fake = fake_uncond + guidance_scale * (fake_cond - fake_uncond)
            fake = torch.clamp(fake, 0., 1.)
        else:
            # Standard generation
            fake = self.generator(z, cond=cond)

        return fake

    @torch.inference_mode()
    def inpaint(self, partial, mask, num_steps=100, lr=0.01):
        """
        Inpaint by optimizing latent code to match known regions.

        NOTE: Inpainting is not yet implemented for Standard GAN.
        This method raises a NotImplementedError.

        Args:
            partial: Partial image (B, 5, 64, 32) in [0, 1]
            mask: Binary mask (B, 5, 64, 32), 1=known, 0=unknown
            num_steps: Optimization iterations (default 100)
            lr: Latent optimization learning rate (default 0.01)

        Returns:
            Inpainted image (B, 5, 64, 32) in [0, 1]
        """
        raise NotImplementedError(
            "Inpainting is not yet implemented for Standard GAN. "
            "Please use 'unconditional' or 'conditional' sampling modes."
        )
