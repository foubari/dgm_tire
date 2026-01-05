"""
Continuous multi-component datasets for image-based generative models.

Used by: DDPM, Flow Matching, VQ-VAE, MM-VAE+, WGAN-GP
"""

from .multi_component import MultiComponentDataset

__all__ = ['MultiComponentDataset']

