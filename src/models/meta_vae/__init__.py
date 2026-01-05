"""
Meta VAE for EpureDGM.

Hierarchical VAE with frozen marginal decoders for cross-modal generation.
Adapted from ICTAI implementation.
"""

from .model import MetaVAE, MetaEncoder, MetaDecoder

__all__ = ['MetaVAE', 'MetaEncoder', 'MetaDecoder']
