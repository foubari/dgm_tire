"""
Discriminator architecture for Standard GAN.
Classifies (B, 5, 64, 32) images as real or fake.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Discriminator: Image -> P(real)

    Architecture:
    - Input: x (B, 5, 64, 32) + optional cond (B, cond_dim)
    - Output: (B, 1) - probability in [0, 1] via Sigmoid

    Uses BatchNorm2d (standard GAN) and optional spectral normalization.

    Parameters: ~3-5M
    """

    def __init__(self, in_channels=5, cond_dim=0, use_spectral_norm=True):
        super().__init__()

        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.use_spectral_norm = use_spectral_norm

        # Helper to optionally wrap layers with spectral norm
        def maybe_sn(layer):
            if use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer

        # Condition embedding (if conditional)
        if cond_dim > 0:
            self.cond_mlp = maybe_sn(nn.Linear(cond_dim, 64))
            # Condition will be broadcast spatially and concatenated
            input_channels = in_channels + 64
        else:
            self.cond_mlp = None
            input_channels = in_channels

        # Conv layers with downsampling
        # Layer 1: (64, 32) -> (32, 16) - no normalization on first layer
        self.conv1 = maybe_sn(nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1))
        self.act1 = nn.LeakyReLU(0.2)

        # Layer 2: (32, 16) -> (16, 8)
        self.conv2 = maybe_sn(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.norm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2)

        # Layer 3: (16, 8) -> (8, 4)
        self.conv3 = maybe_sn(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.norm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2)

        # Layer 4: (8, 4) -> (4, 2)
        self.conv4 = maybe_sn(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        self.norm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2)

        # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final output with Sigmoid (standard GAN outputs probability)
        self.fc = maybe_sn(nn.Linear(512, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond=None):
        """
        Args:
            x: (B, in_channels, 64, 32) - input images
            cond: (B, cond_dim) or None - conditioning vectors

        Returns:
            prob: (B, 1) - probability of being real, in [0, 1]
        """
        # Condition embedding and spatial broadcast
        if self.cond_mlp is not None:
            if cond is None:
                # Use zero conditioning
                cond_emb = torch.zeros(x.size(0), 64, device=x.device, dtype=x.dtype)
            else:
                cond_emb = self.cond_mlp(cond)  # (B, 64)

            # Broadcast condition spatially: (B, 64) -> (B, 64, 64, 32)
            B, C, H, W = x.shape
            cond_spatial = cond_emb.view(B, 64, 1, 1).expand(B, 64, H, W)

            # Concatenate with image
            x = torch.cat([x, cond_spatial], dim=1)  # (B, in_channels + 64, 64, 32)

        # Conv layers
        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act4(x)

        # Global pooling
        x = self.adaptive_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)

        # Final output
        score = self.fc(x)  # (B, 1)
        prob = self.sigmoid(score)  # [0, 1]

        return prob
