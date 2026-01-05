"""
Encoder and Decoder for Simple VAE.

Adapted from ICTAI beta_vae ResNet architecture for (64, 32) images with 5 channels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def actvn(x):
    """Activation function."""
    return F.leaky_relu(x, 0.2)


class ResnetBlock(nn.Module):
    """ResNet block with dropout."""

    def __init__(self, fin, fout, fhidden=None, dropout_p=0.1):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhidden = fhidden if fhidden is not None else min(fin, fout)

        self.dropout = nn.Dropout2d(dropout_p)
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(actvn(x))
        dx = self.dropout(dx)
        dx = self.conv_1(actvn(dx))
        return x_s + 0.1 * dx


class Encoder(nn.Module):
    """
    VAE Encoder for (64, 32) images with 5 channels.

    Downsamples: 64x32 → 32x16 → 16x8 → 8x4
    """

    def __init__(self, channels=5, latent_dim=32, nf=32, nf_max=256, dropout_p=0.1):
        super().__init__()
        self.s0 = 4  # Final spatial size: 8x4
        self.nf = nf
        self.nf_max = nf_max

        # For 64x32: log2(64/8) = 3 downsampling steps
        nlayers = 3
        self.nf0 = min(nf_max, nf * 2**nlayers)  # Final channels

        # Initial conv
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

        # Final spatial: 8x4, flatten to self.nf0 * 8 * 4
        self.fc_mu = nn.Linear(self.nf0 * 8 * 4, latent_dim)
        self.fc_logvar = nn.Linear(self.nf0 * 8 * 4, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 5, 64, 32)

        Returns:
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(out.size(0), -1)  # Flatten

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        logvar = F.softplus(logvar)  # Ensure positive variance

        return mu, logvar


class Decoder(nn.Module):
    """
    VAE Decoder with conditioning support.

    Upsamples: 8x4 → 16x8 → 32x16 → 64x32
    """

    def __init__(self, latent_dim=32, channels=5, cond_dim=2, nf=32, nf_max=256, dropout_p=0.1):
        super().__init__()
        self.s0 = 4
        self.nf = nf
        self.nf_max = nf_max

        nlayers = 3
        self.nf0 = min(nf_max, nf * 2**nlayers)

        # Conditioning MLP
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, 128),
                nn.GELU(),
                nn.Linear(128, 256)
            )
            fc_input_dim = latent_dim + 256
        else:
            self.cond_mlp = None
            fc_input_dim = latent_dim

        # FC to initial spatial
        self.fc = nn.Linear(fc_input_dim, self.nf0 * 8 * 4)

        # ResNet blocks with upsampling
        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers - i), nf_max)
            nf1 = min(nf * 2**(nlayers - i - 1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1, dropout_p=dropout_p),
                nn.Upsample(scale_factor=2, mode='nearest'),
            ]

        blocks += [ResnetBlock(nf, nf, dropout_p=dropout_p)]

        self.resnet = nn.Sequential(*blocks)
        self.conv_out = nn.Conv2d(nf, channels, 3, padding=1)

    def forward(self, z, cond=None):
        """
        Args:
            z: (B, latent_dim)
            cond: (B, cond_dim) or None

        Returns:
            recon: (B, 5, 64, 32)
        """
        # Apply conditioning
        if self.cond_mlp is not None:
            if cond is not None:
                cond_emb = self.cond_mlp(cond)
            else:
                # Create zero embedding for unconditional sampling
                cond_emb = torch.zeros(z.size(0), 256, device=z.device, dtype=z.dtype)
            z = torch.cat([z, cond_emb], dim=1)

        # FC and reshape
        out = self.fc(z)
        out = out.view(out.size(0), self.nf0, 8, 4)

        # Upsample
        out = self.resnet(out)
        out = self.conv_out(actvn(out))

        return torch.sigmoid(out)  # Images in [0, 1]
