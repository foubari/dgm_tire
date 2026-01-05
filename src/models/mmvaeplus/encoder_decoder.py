"""
Encoder and Decoder with ResNet blocks for MMVAE+.
Adapted for 64x32 images with conditioning support.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .utils import Constants


def actvn(x):
    """Activation function: LeakyReLU."""
    return torch.nn.functional.leaky_relu(x, 2e-1)


class ResnetBlock(nn.Module):
    """ResNet block with residual connection."""
    
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class Enc(nn.Module):
    """
    Encoder for 64x32 images with conditioning support.
    
    Architecture:
    - Input: (B, 1, 64, 32)
    - After 3 downsamplings: (B, nf0, 8, 4)
    - Output: mu_w, logvar_w, mu_z, logvar_z
    """
    
    def __init__(self, ndim_w, ndim_z, dist='Normal', nf=32, nf_max=512, cond_dim=0):
        super().__init__()
        self.dist = dist
        self.cond_dim = cond_dim
        
        # Spatial dimensions: 64x32 -> 8x4 after 3 downsamplings
        s0_h, s0_w = 8, 4  # Final spatial size
        nf = self.nf = nf
        nf_max = self.nf_max = nf_max
        size_h, size_w = 64, 32
        
        # Number of downsampling layers
        nlayers_h = int(np.log2(size_h / s0_h))  # 3 layers
        nlayers_w = int(np.log2(size_w / s0_w))  # 3 layers
        assert nlayers_h == nlayers_w, "Height and width downsampling must match"
        nlayers = nlayers_h
        
        self.nf0 = min(nf_max, nf * 2**nlayers)
        self.s0_h = s0_h
        self.s0_w = s0_w
        self.spatial_size = s0_h * s0_w  # 8 * 4 = 32

        # Conditioning MLP (if cond_dim > 0)
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
            # Condition will be concatenated to features before FC layers
            cond_channels = 256
        else:
            self.cond_mlp = None
            cond_channels = 0

        # Blocks for w (private latent)
        blocks_w = [ResnetBlock(nf, nf)]
        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        # Blocks for z (shared latent)
        blocks_z = [ResnetBlock(nf, nf)]
        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_z += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_w = nn.Conv2d(1, 1 * nf, 3, padding=1)
        self.resnet_w = nn.Sequential(*blocks_w)
        # FC layers: input is nf0 * spatial_size + cond_channels
        self.fc_mu_w = nn.Linear(self.nf0 * self.spatial_size + cond_channels, ndim_w)
        self.fc_lv_w = nn.Linear(self.nf0 * self.spatial_size + cond_channels, ndim_w)

        self.conv_img_z = nn.Conv2d(1, 1 * nf, 3, padding=1)
        self.resnet_z = nn.Sequential(*blocks_z)
        self.fc_mu_z = nn.Linear(self.nf0 * self.spatial_size + cond_channels, ndim_z)
        self.fc_lv_z = nn.Linear(self.nf0 * self.spatial_size + cond_channels, ndim_z)

    def forward(self, x, cond=None):
        """
        Args:
            x: (B, 1, 64, 32) - input image
            cond: (B, cond_dim) or None - conditioning vector
        
        Returns:
            mu: (B, ndim_w + ndim_z) - concatenated means
            logvar: (B, ndim_w + ndim_z) - concatenated logvars
        """
        # Encode w (private)
        out_w = self.conv_img_w(x)
        out_w = self.resnet_w(out_w)  # (B, nf0, 8, 4)
        out_w = out_w.view(out_w.size()[0], self.nf0 * self.spatial_size)  # (B, nf0 * 32)
        
        # Encode z (shared)
        out_z = self.conv_img_z(x)
        out_z = self.resnet_z(out_z)  # (B, nf0, 8, 4)
        out_z = out_z.view(out_z.size()[0], self.nf0 * self.spatial_size)  # (B, nf0 * 32)
        
        # Add conditioning if available
        if self.cond_mlp is not None:
            if cond is None:
                # Use zero conditioning
                cond_emb = torch.zeros(x.size(0), 256, device=x.device, dtype=x.dtype)
            else:
                cond_emb = self.cond_mlp(cond)  # (B, 256)
            
            # Concatenate condition to features
            out_w = torch.cat([out_w, cond_emb], dim=1)  # (B, nf0 * 32 + 256)
            out_z = torch.cat([out_z, cond_emb], dim=1)  # (B, nf0 * 32 + 256)
        
        # Project to latent parameters
        mu_w = self.fc_mu_w(out_w)
        lv_w = self.fc_lv_w(out_w)
        mu_z = self.fc_mu_z(out_z)
        lv_z = self.fc_lv_z(out_z)

        # Return concatenated parameters
        if self.dist == 'Normal':
            return torch.cat((mu_w, mu_z), dim=-1), \
                   torch.cat((F.softplus(lv_w) + Constants.eta,
                              F.softplus(lv_z) + Constants.eta), dim=-1)
        else:  # Laplace
            return torch.cat((mu_w, mu_z), dim=-1), \
                   torch.cat((F.softmax(lv_w, dim=-1) * lv_w.size(-1) + Constants.eta,
                              F.softmax(lv_z, dim=-1) * lv_z.size(-1) + Constants.eta), dim=-1)


class Dec(nn.Module):
    """
    Decoder for 64x32 images with conditioning support.
    
    Architecture:
    - Input: (B, latent_dim_u) [+ cond_emb]
    - After FC: (B, nf0, 8, 4)
    - After 3 upsamplings: (B, 1, 64, 32)
    """
    
    def __init__(self, ndim, nf=32, nf_max=256, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        
        # Spatial dimensions
        s0_h, s0_w = 8, 4  # Starting spatial size
        self.nf = nf
        self.nf_max = nf_max
        size_h, size_w = 64, 32
        
        # Number of upsampling layers
        nlayers_h = int(np.log2(size_h / s0_h))  # 3 layers
        nlayers_w = int(np.log2(size_w / s0_w))  # 3 layers
        assert nlayers_h == nlayers_w, "Height and width upsampling must match"
        nlayers = nlayers_h
        
        self.nf0 = min(nf_max, nf * 2**nlayers)
        self.s0_h = s0_h
        self.s0_w = s0_w
        self.spatial_size = s0_h * s0_w  # 8 * 4 = 32

        # Conditioning MLP
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
            decoder_input_dim = ndim + 256
        else:
            self.cond_mlp = None
            decoder_input_dim = ndim

        # First FC layer: projects latent to feature map
        self.fc = nn.Linear(decoder_input_dim, self.nf0 * self.spatial_size)

        # Upsampling blocks
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ]

        blocks += [ResnetBlock(nf, nf)]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 1, 3, padding=1)

    def forward(self, u, cond=None):
        """
        Args:
            u: (B, latent_dim_u) or (K, B, latent_dim_u) - latent codes
            cond: (B, cond_dim) or None - conditioning vector
        
        Returns:
            mean: (B, 1, 64, 32) or (K, B, 1, 64, 32)
            scale: scalar or tensor - length scale
        """
        # Handle K dimension (for reparameterization)
        if u.dim() == 3:
            K, B, D = u.shape
            u_flat = u.view(K * B, D)
            cond_expanded = None
            if cond is not None:
                # cond should be (B, cond_dim), expand to (K, B, cond_dim) then flatten to (K*B, cond_dim)
                cond_expanded = cond.unsqueeze(0).expand(K, -1, -1).contiguous().view(K * B, -1)
        else:
            # u is (B, latent_dim_u)
            u_flat = u
            # cond should be (B, cond_dim) or None
            cond_expanded = cond
        
        # Concatenate conditioning
        if self.cond_mlp is not None:
            if cond_expanded is None:
                # Use zero conditioning
                cond_emb = torch.zeros(u_flat.size(0), 256, device=u_flat.device, dtype=u_flat.dtype)
            else:
                # Verify cond_expanded has correct shape: (B, cond_dim) or (K*B, cond_dim)
                expected_cond_dim = self.cond_dim
                if cond_expanded.shape[-1] != expected_cond_dim:
                    raise ValueError(
                        f"Conditioning shape mismatch: expected last dim={expected_cond_dim}, "
                        f"got {cond_expanded.shape}. cond_expanded shape: {cond_expanded.shape}, "
                        f"u shape: {u.shape}, u_flat shape: {u_flat.shape}"
                    )
                cond_emb = self.cond_mlp(cond_expanded)  # (B, 256) or (K*B, 256)
            
            u_flat = torch.cat([u_flat, cond_emb], dim=-1)
        
        # Project to feature map
        out = self.fc(u_flat).view(-1, self.nf0, self.s0_h, self.s0_w)  # (B, nf0, 8, 4) or (K*B, nf0, 8, 4)
        out = self.resnet(out)  # Upsample to (B, nf, 64, 32)
        out = self.conv_img(actvn(out))  # (B, 1, 64, 32)

        # Reshape if K dimension was present
        if u.dim() == 3:
            out = out.view(K, B, *out.size()[1:])

        # Return mean and scale
        return out, torch.tensor(0.75).to(u.device)  # mean, length scale

