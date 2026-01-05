"""
Encoder architecture for VQ-VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        
        return x + residual


class Downsample(nn.Module):
    """Downsampling layer using stride=2 convolution."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder architecture for VQ-VAE.
    
    Input: (B, channels, 64, 32)
    Output: (B, latent_dim, 16, 8) after 2x downsampling
    """
    
    def __init__(self, in_channels=5, latent_dim=20, base_dim=64, dim_mults=(1, 2, 4)):
        super().__init__()
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=1, padding=1)
        
        # Build encoder layers
        dims = [base_dim * m for m in dim_mults]
        
        self.blocks = nn.ModuleList()
        
        # First ResBlock
        self.blocks.append(ResBlock(base_dim, base_dim))
        
        # Downsample 1: 64x32 -> 32x16
        self.blocks.append(Downsample(base_dim, dims[0]))
        self.blocks.append(ResBlock(dims[0], dims[0]))
        
        # Downsample 2: 32x16 -> 16x8
        self.blocks.append(Downsample(dims[0], dims[1]))
        self.blocks.append(ResBlock(dims[1], dims[1]))
        self.blocks.append(ResBlock(dims[1], dims[1]))
        
        # Final normalization and projection to latent
        self.final_norm = nn.GroupNorm(8, dims[1])
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(dims[1], latent_dim, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)
        
        Returns:
            z_e: (B, latent_dim, H//4, W//4) - continuous latent
        """
        x = self.init_conv(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.final_conv(x)
        
        return x

