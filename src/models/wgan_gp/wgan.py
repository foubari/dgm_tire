"""
WGAN-GP wrapper combining encoder, generator, and critic.
Main interface matching DDPM's API (forward, sample, inpaint).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .encoder import Encoder
from .generator import Generator
from .critic import Critic


class WGANGP(nn.Module):
    """
    WGAN-GP model wrapper.
    
    Main interface for training and sampling.
    """
    
    def __init__(
        self,
        encoder,
        generator,
        critic,
        *,
        image_size=(64, 32),
        latent_dim=20,
        lambda_gp=10,
        n_critic=5,
        cond_drop_prob=0.1,
    ):
        """
        Args:
            encoder: Encoder network (image → latent)
            generator: Generator network (latent → image)
            critic: Critic network (image → score)
            image_size: Tuple (H, W) = (64, 32)
            latent_dim: Total latent dimensions (default 20)
            lambda_gp: Gradient penalty weight (default 10)
            n_critic: Critic updates per generator update (default 5)
            cond_drop_prob: CFG dropout probability (default 0.1)
        """
        super().__init__()
        
        self.encoder = encoder
        self.generator = generator
        self.critic = critic
        
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.cond_drop_prob = cond_drop_prob
    
    def forward(self, img, *, cond=None, train_critic=True):
        """
        Training step - returns dict of losses.
        
        Args:
            img: Real images (B, 5, 64, 32) in [0, 1]
            cond: Conditioning (B, cond_dim) or None
            train_critic: If True, compute critic loss; else generator loss
        
        Returns:
            Dict with keys:
            - 'critic_loss': Critic loss (when train_critic=True)
            - 'generator_loss': Generator loss (when train_critic=False)
            - 'wasserstein_distance': W-distance estimate
            - 'gradient_penalty': GP term (when train_critic=True)
        """
        batch_size = img.size(0)
        
        if train_critic:
            # Sample random latent for fake images
            z = torch.randn(batch_size, self.latent_dim, device=img.device)
            
            # Conditional dropout for CFG
            cond_for_gen = None
            if cond is not None and random.random() >= self.cond_drop_prob:
                cond_for_gen = cond
            
            # Generate fake images
            with torch.no_grad():
                fake = self.generator(z, cond=cond_for_gen)
            
            # Critic scores
            critic_real = self.critic(img, cond=cond)
            critic_fake = self.critic(fake.detach(), cond=cond)
            
            # Wasserstein distance
            wasserstein_distance = critic_real.mean() - critic_fake.mean()
            
            # Gradient penalty
            gp = self.compute_gradient_penalty(img, fake, cond)
            
            # Total critic loss: maximize W-distance, minimize GP
            critic_loss = -wasserstein_distance + gp
            
            return {
                'critic_loss': critic_loss,
                'wasserstein_distance': wasserstein_distance.item(),
                'gradient_penalty': gp.item(),
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
            
            # Generator loss: maximize critic score on fake
            critic_fake = self.critic(fake, cond=cond)
            generator_loss = -critic_fake.mean()
            
            return {
                'generator_loss': generator_loss,
                'wasserstein_distance': critic_fake.mean().item(),
            }
    
    def compute_gradient_penalty(self, real, fake, cond):
        """
        Compute WGAN-GP gradient penalty.
        
        GP = lambda_gp * E[(||∇D(x_hat)||_2 - 1)^2]
        where x_hat = epsilon * real + (1 - epsilon) * fake
        """
        batch_size = real.size(0)
        device = real.device
        
        # Random interpolation coefficient
        epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Interpolate between real and fake
        x_hat = epsilon * real + (1 - epsilon) * fake
        x_hat.requires_grad_(True)
        
        # Critic score on interpolated samples
        critic_hat = self.critic(x_hat, cond=cond)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_hat,
            inputs=x_hat,
            grad_outputs=torch.ones_like(critic_hat),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Gradient penalty: (||∇D(x_hat)||_2 - 1)^2
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty
    
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
        
        NOTE: Inpainting is not yet implemented for WGAN-GP.
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
            "Inpainting is not yet implemented for WGAN-GP. "
            "Please use 'unconditional' or 'conditional' sampling modes."
        )

