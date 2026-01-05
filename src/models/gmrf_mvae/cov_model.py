"""
Off-Diagonal Covariance Network for GMRF MVAE.

Adapted from ICTAI implementation.
"""

import torch
import torch.nn as nn


def compute_n_off_diag(modalities_dim):
    """Compute the number of lower off-diagonal elements."""
    n = len(modalities_dim)
    partial_sums = [modalities_dim[-1]]
    for i in range(n-2):
        partial_sums.append(partial_sums[-1] + modalities_dim[-i-2])
    partial_sums = partial_sums[::-1]
    res = 0
    for i in range(n-1):
        res += partial_sums[i] * modalities_dim[i]
    return res


class OffDiagonalCov(nn.Module):
    """
    MLP network that predicts off-diagonal covariance coefficients.

    Args:
        input_dims: List of embedding dimensions per modality
        encoded_dims: List of latent dimensions per modality
        hidden_dim: Hidden layer size
        n_layers: Number of hidden layers
    """

    def __init__(self, input_dims, encoded_dims, hidden_dim, n_layers):
        super().__init__()

        # Encoder layers
        total_input_size = sum(input_dims)
        layers = [nn.Linear(total_input_size, hidden_dim), nn.LeakyReLU(0.2)]

        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2)])

        self.encoder = nn.Sequential(*layers)

        # Output layer for off-diagonal blocks
        off_diagonal_elements = compute_n_off_diag(encoded_dims)
        self.off_diag_layer = nn.Linear(hidden_dim, off_diagonal_elements)

    def forward(self, *modalities):
        """
        Args:
            *modalities: List of embeddings (B, dim_i) per modality

        Returns:
            off_diag_coefficients: (B, n_off_diag)
        """
        concatenated = torch.cat(modalities, dim=1)
        encoded = self.encoder(concatenated)
        off_diag_coefficients = self.off_diag_layer(encoded)
        return off_diag_coefficients
