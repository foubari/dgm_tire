"""
ResNet-based encoder and decoder for Beta-VAE.

Implements residual blocks with skip connections and downsampling/upsampling.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def actvn(x):
    """Activation function: LeakyReLU with slope 0.2"""
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    """
    Residual block with skip connections and 0.1 scaling.

    Args:
        fin: Input channels
        fout: Output channels
        fhidden: Hidden channels (default: min(fin, fout))
        is_bias: Use bias in convolutions (default: True)
        dropout_p: Dropout probability (default: 0.1)
    """

    def __init__(self, fin, fout, fhidden=None, is_bias=True, dropout_p=0.1):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.dropout = nn.Dropout2d(dropout_p)

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
        dx = self.dropout(dx)
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx  # CRITICAL: 0.1 scaling factor

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class Encoder(nn.Module):
    """
    ResNet Encoder for VAE with skip connections.

    Args:
        channels: Input channels (5 for EPURE, 3 for TOY)
        latent_dim: Latent space dimensionality
        nf: Base number of filters (default: 72)
        nf_max: Maximum number of filters (default: 1024)
        dropout_p: Dropout probability (default: 0.1)
        image_size: (H, W) tuple for non-square images (default: (64, 64))
    """

    def __init__(self, channels, latent_dim, nf=72, nf_max=1024, dropout_p=0.1, image_size=(64, 64)):
        super().__init__()
        self.nf = nf
        self.nf_max = nf_max

        # Support rectangular images
        h, w = image_size

        # Calculate number of downsampling layers based on height
        # 64 → 32 → 16 → 8 (3 layers)
        self.s0_h = h // 8
        self.s0_w = w // 8

        nlayers = int(np.log2(h / self.s0_h))  # 3 layers for 64x64 or 64x32
        self.nf0 = min(nf_max, nf * 2**nlayers)

        # Initial conv: channels → nf
        self.conv_img = nn.Conv2d(channels, nf, 3, padding=1)

        # ResNet blocks with downsampling
        blocks = [ResnetBlock(nf, nf, dropout_p=dropout_p)]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1, dropout_p=dropout_p),
            ]

        self.resnet = nn.Sequential(*blocks)

        # Output layers - use actual spatial dimensions
        self.fc_mu = nn.Linear(self.nf0 * self.s0_h * self.s0_w, latent_dim)
        self.fc_lv = nn.Linear(self.nf0 * self.s0_h * self.s0_w, latent_dim)

    def forward(self, x):
        """
        Encode input to latent distribution.

        Returns:
            mu: Mean (B, latent_dim)
            logvar: Log variance WITH softplus (B, latent_dim)
        """
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(out.size(0), self.nf0 * self.s0_h * self.s0_w)

        mu = self.fc_mu(out)
        lv = self.fc_lv(out)

        # CRITICAL: Apply softplus to logvar!
        logvar = F.softplus(lv)

        return mu, logvar


class Decoder(nn.Module):
    """
    ResNet Decoder for VAE with skip connections.

    Args:
        latent_dim: Latent space dimensionality
        channels: Output channels (5 for EPURE, 3 for TOY)
        cond_dim: Conditioning dimension (0 if unconditional)
        nf: Base number of filters (default: 72)
        nf_max: Maximum number of filters (default: 512)
        dropout_p: Dropout probability (default: 0.1)
        image_size: (H, W) tuple for non-square images (default: (64, 64))
    """

    def __init__(self, latent_dim, channels, cond_dim=0, nf=72, nf_max=512, dropout_p=0.1, image_size=(64, 64)):
        super().__init__()
        self.nf = nf
        self.nf_max = nf_max

        # Support rectangular images
        h, w = image_size

        self.s0_h = h // 8
        self.s0_w = w // 8

        nlayers = int(np.log2(h / self.s0_h))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        # Conditioning MLP (if needed)
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
            dec_input_dim = latent_dim + 256
        else:
            self.cond_mlp = None
            dec_input_dim = latent_dim

        # FC layer - use actual spatial dimensions
        self.fc = nn.Linear(dec_input_dim, self.nf0 * self.s0_h * self.s0_w)

        # ResNet blocks with upsampling
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1, dropout_p=dropout_p),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [ResnetBlock(nf, nf, dropout_p=dropout_p)]

        self.resnet = nn.Sequential(*blocks)

        # Output conv
        self.conv_img = nn.Conv2d(nf, channels, 3, padding=1)

    def forward(self, z, cond=None):
        """
        Decode latent to reconstruction.

        Args:
            z: Latent code (B, latent_dim)
            cond: Conditioning (B, cond_dim) or None

        Returns:
            recon: Reconstruction (B, channels, H, W)
        """
        # Apply conditioning if available
        if self.cond_mlp is not None:
            if cond is not None:
                cond_emb = self.cond_mlp(cond)
            else:
                # Zero conditioning if not provided
                cond_emb = torch.zeros(z.size(0), 256, device=z.device, dtype=z.dtype)
            z = torch.cat([z, cond_emb], dim=1)

        out = self.fc(z).view(-1, self.nf0, self.s0_h, self.s0_w)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))

        return out
