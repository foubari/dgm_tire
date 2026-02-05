"""
ResNet building blocks for GMRF MVAE.

Original architecture preserved from the source implementation.
"""

import torch.nn as nn


def actvn(x):
    """Leaky ReLU activation function."""
    out = nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    """
    Residual block with skip connections.

    Uses 0.1 scaling factor on residual path for training stability.
    """

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
