"""
Encoder architecture for WGAN-GP.
Compresses (B, 5, 64, 32) images to (B, 20) latent vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder: Image → Latent
    
    Architecture:
    - Input: (B, 5, 64, 32)
    - Output: (B, latent_dim) where latent_dim=20 by default
    
    Parameters: ~1.3M
    """
    
    def __init__(self, in_channels=5, latent_dim=20):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 64)
        self.act1 = nn.ReLU()
        
        # Downsample blocks
        # Block 1: 64x32 -> 32x16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 128)
        self.act2 = nn.ReLU()
        
        # Block 2: 32x16 -> 16x8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, 256)
        self.act3 = nn.ReLU()
        
        # Block 3: 16x8 -> 8x4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.GroupNorm(8, 512)
        self.act4 = nn.ReLU()
        
        # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to latent
        self.fc1 = nn.Linear(512, 256)
        self.act_fc = nn.ReLU()
        self.fc2 = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, 64, 32) - input images
        
        Returns:
            z: (B, latent_dim) - latent vectors
        """
        # Initial conv
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        # Downsample blocks
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
        
        # Projection to latent
        x = self.fc1(x)
        x = self.act_fc(x)
        x = self.fc2(x)  # (B, latent_dim)
        
        return x

