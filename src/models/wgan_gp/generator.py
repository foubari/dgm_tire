"""
Generator architecture for WGAN-GP.
Generates (B, 5, 64, 32) images from (B, 20) latent + optional conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual block for generator.
    
    ResBlock(dim, dim):
    ├── Conv2d(dim → dim, k=3, p=1) + GroupNorm + ReLU
    ├── Conv2d(dim → dim, k=3, p=1) + GroupNorm
    └── Residual connection + ReLU
    """
    
    def __init__(self, dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, dim)
        self.act2 = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = x + residual
        x = self.act2(x)
        
        return x


class Generator(nn.Module):
    """
    Generator: Latent → Image
    
    Architecture:
    - Input: z (B, latent_dim) + optional cond (B, cond_dim)
    - Output: (B, 5, 64, 32) in [0, 1]
    
    Parameters: Target ~10-12M
    """
    
    def __init__(self, latent_dim=20, out_channels=5, cond_dim=0, dim=64, dim_mults=(1, 2, 4)):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.cond_dim = cond_dim
        self.dim = dim
        
        # Condition embedding (if conditional)
        if cond_dim > 0:
            self.cond_mlp = nn.Linear(cond_dim, 128)
            latent_input_dim = latent_dim + 128
        else:
            self.cond_mlp = None
            latent_input_dim = latent_dim
        
        # Initial projection: latent → (512, 8, 4)
        self.fc = nn.Linear(latent_input_dim, 512 * 8 * 4)
        self.norm_init = nn.GroupNorm(8, 512)
        self.act_init = nn.ReLU()
        
        # Build upsampling blocks
        dims = [dim * m for m in dim_mults]
        
        # Upsample Block 1: (512, 8, 4) → (256, 16, 8)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(8, 256)
        self.act1 = nn.ReLU()
        self.res1 = nn.ModuleList([ResBlock(256) for _ in range(2)])
        
        # Upsample Block 2: (256, 16, 8) → (128, 32, 16)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, 128)
        self.act2 = nn.ReLU()
        self.res2 = nn.ModuleList([ResBlock(128) for _ in range(2)])
        
        # Upsample Block 3: (128, 32, 16) → (64, 64, 32)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)
        self.act3 = nn.ReLU()
        self.res3 = nn.ModuleList([ResBlock(64) for _ in range(2)])
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z, cond=None):
        """
        Args:
            z: (B, latent_dim) - latent vectors
            cond: (B, cond_dim) or None - conditioning vectors
        
        Returns:
            x: (B, out_channels, 64, 32) - generated images in [0, 1]
        """
        # Condition embedding
        if self.cond_mlp is not None:
            if cond is None:
                # Use zero conditioning for unconditional generation
                cond_emb = torch.zeros(z.size(0), 128, device=z.device, dtype=z.dtype)
            else:
                cond_emb = self.cond_mlp(cond)  # (B, 128)
            
            # Concatenate latent and condition
            z_cond = torch.cat([z, cond_emb], dim=1)  # (B, latent_dim + 128)
        else:
            z_cond = z
        
        # Project to spatial representation
        x = self.fc(z_cond)  # (B, 512*8*4)
        x = x.view(x.size(0), 512, 8, 4)  # (B, 512, 8, 4)
        x = self.norm_init(x)
        x = self.act_init(x)
        
        # Upsample Block 1
        x = self.up1(x)  # (B, 256, 16, 8)
        x = self.norm1(x)
        x = self.act1(x)
        for res_block in self.res1:
            x = res_block(x)
        
        # Upsample Block 2
        x = self.up2(x)  # (B, 128, 32, 16)
        x = self.norm2(x)
        x = self.act2(x)
        for res_block in self.res2:
            x = res_block(x)
        
        # Upsample Block 3
        x = self.up3(x)  # (B, 64, 64, 32)
        x = self.norm3(x)
        x = self.act3(x)
        for res_block in self.res3:
            x = res_block(x)
        
        # Final output
        x = self.final_conv(x)  # (B, out_channels, 64, 32)
        x = self.sigmoid(x)  # [0, 1]
        
        return x

