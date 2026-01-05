"""
Decoder architecture for VQ-VAE with conditioning support.
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


class Upsample(nn.Module):
    """Upsampling layer using ConvTranspose2d."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    """
    Decoder architecture for VQ-VAE with conditioning support.
    
    Input: (B, latent_dim, 16, 8) [quantized latent]
    Output: (B, out_channels, 64, 32) after 2x upsampling
    """
    
    def __init__(self, latent_dim=20, out_channels=5, cond_dim=2, base_dim=64, dim_mults=(1, 2, 4)):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        
        # Conditioning MLP
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
            # When conditioning is concatenated, first conv receives latent_dim + 256
            first_conv_in = latent_dim + 256
        else:
            self.cond_mlp = None
            first_conv_in = latent_dim
        
        # Build decoder layers
        dims = [base_dim * m for m in dim_mults]
        dims = list(reversed(dims))  # Start from largest (256) and go down
        
        # Initial expansion from latent
        self.init_conv = nn.Conv2d(first_conv_in, dims[0], kernel_size=1)
        
        self.blocks = nn.ModuleList()
        
        # ResBlocks at highest resolution (16x8)
        self.blocks.append(ResBlock(dims[0], dims[0]))
        self.blocks.append(ResBlock(dims[0], dims[0]))
        
        # Upsample 1: 16x8 -> 32x16
        self.blocks.append(Upsample(dims[0], dims[1]))
        self.blocks.append(ResBlock(dims[1], dims[1]))
        
        # Upsample 2: 32x16 -> 64x32
        self.blocks.append(Upsample(dims[1], dims[2]))
        self.blocks.append(ResBlock(dims[2], dims[2]))
        
        # Final projection to output channels
        self.final_conv = nn.Conv2d(dims[2], out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z_q, cond=None):
        """
        Args:
            z_q: (B, latent_dim, H, W) - quantized latent
            cond: (B, cond_dim) or None - conditioning vector
        
        Returns:
            x_recon: (B, out_channels, H*4, W*4) - reconstructed image
        """
        B, C, H, W = z_q.shape
        
        # Apply conditioning if cond_dim > 0
        if self.cond_mlp is not None:
            if cond is not None:
                cond_emb = self.cond_mlp(cond)  # (B, 256)
            else:
                # Use zeros if cond is None (unconditional generation)
                cond_emb = torch.zeros(B, 256, device=z_q.device, dtype=z_q.dtype)
            
            cond_spatial = cond_emb[:, :, None, None].expand(-1, -1, H, W)  # (B, 256, H, W)
            z_combined = torch.cat([z_q, cond_spatial], dim=1)  # (B, latent_dim+256, H, W)
        else:
            z_combined = z_q
        
        x = self.init_conv(z_combined)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)

        # No activation - images are already in [0, 1] from dataset
        # Sigmoid would compress the output range and cause training issues
        return x

