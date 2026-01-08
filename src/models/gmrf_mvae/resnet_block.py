"""
ResNet Block implementation for GMRF-MVAE.

This module provides the ResnetBlock class with skip connections,
matching the ICTAI original implementation exactly.
"""

import torch.nn as nn


def actvn(x):
    """
    Activation function: LeakyReLU with slope 0.2.

    This matches the ICTAI original implementation.

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    out = nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    """
    Residual block with skip connections.

    This implementation matches the ICTAI original exactly:
    - Two 3x3 convolutions
    - Skip connection (identity or learned 1x1 conv)
    - Output: x_s + 0.1*dx (0.1 scaling factor is critical!)

    Args:
        fin: Input channels
        fout: Output channels
        fhidden: Hidden channels (default: min(fin, fout))
        is_bias: Use bias in convolutions (default: True)
    """

    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout

        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Two 3x3 convolutions
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(
            self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias
        )

        # Learned shortcut if input/output channels differ
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(
                self.fin, self.fout, 1, stride=1, padding=0, bias=False
            )

    def forward(self, x):
        """
        Forward pass with residual connection.

        The output is: x_s + 0.1*dx
        where dx is the residual and x_s is the shortcut.

        The 0.1 scaling factor is CRITICAL for stability.
        """
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1 * dx  # CRITICAL: 0.1 scaling factor

        return out

    def _shortcut(self, x):
        """
        Compute shortcut connection.

        If fin != fout, use learned 1x1 conv.
        Otherwise, use identity.
        """
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s
