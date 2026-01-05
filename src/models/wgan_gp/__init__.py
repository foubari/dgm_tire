"""
WGAN-GP (Wasserstein GAN with Gradient Penalty) model.
"""

from .encoder import Encoder
from .generator import Generator
from .critic import Critic
from .wgan import WGANGP

__all__ = ['Encoder', 'Generator', 'Critic', 'WGANGP']

