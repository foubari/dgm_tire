"""
Utility functions for MMVAE+.
"""

import math
import torch
import torch.nn.functional as F


class Constants(object):
    """Constants for numerical stability."""
    eta = 1e-20


def is_multidata(dataB):
    """Check if data is multimodal (list or tuple)."""
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def get_mean(d, K=100):
    """
    Extract the `mean` parameter for given distribution.
    If attribute not available, estimate from samples.
    """
    try:
        mean = d.mean
    except NotImplementedError:
        samples = d.rsample(torch.Size([K]))
        mean = samples.mean(0)
    return mean


def log_mean_exp(value, dim=0, keepdim=False):
    """Compute log mean exp for numerical stability."""
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))


def unpack_data(batch, device='cuda'):
    """
    Unpack batch from MultiComponentDataset.
    
    Args:
        batch: (data_tuple, conditions) from dataloader
        device: Device to move data to
    
    Returns:
        data: List of 5 tensors on device
        cond: (B, 2) tensor on device or None
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        data_tuple, cond = batch
        data = [d.to(device) for d in data_tuple]
        cond = cond.to(device).float() if cond is not None else None
        return data, cond
    else:
        # Old format (no conditioning) - fallback
        if isinstance(batch, (tuple, list)):
            data = [d.to(device) for d in batch]
        else:
            data = [batch.to(device)]
        return data, None

