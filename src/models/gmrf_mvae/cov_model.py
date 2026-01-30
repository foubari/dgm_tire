"""
Off-Diagonal Covariance Network for GMRF MVAE.

Exact copy from ICTAI implementation.
"""

import torch
import torch.nn as nn


# Default activation function (matching ICTAI)
actvn = nn.LeakyReLU(2e-1)


def compute_n_off_diag(modalities_dim):
    """
    Compute the number of lower off-diagonal elements.

    For n modalities with dimensions [d1, d2, ..., dn], computes the total
    number of elements in the off-diagonal blocks of the covariance matrix.
    """
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

    Exact implementation from ICTAI.

    Args:
        input_dims: List of embedding dimensions per modality
        encoded_dims: List of latent dimensions per modality
        hidden_dim: Hidden layer size
        n_layers: Number of hidden layers
        activation: Activation function (default: LeakyReLU(0.2))
    """

    def __init__(self, input_dims, encoded_dims, hidden_dim, n_layers, activation=None):
        super().__init__()

        self.activation = actvn if activation is None else activation

        # Main encoder layers
        total_encoded_size = sum(input_dims)
        layers = [nn.Linear(total_encoded_size, hidden_dim), self.activation]
        layers.extend(
            layer for _ in range(n_layers - 1)
            for layer in [nn.Linear(hidden_dim, hidden_dim), self.activation]
        )
        self.encoder = nn.Sequential(*layers)

        # Output layer for the lower off-diagonal blocks
        off_diagonal_elements = compute_n_off_diag(encoded_dims)
        self.off_diag_layer = nn.Linear(hidden_dim, off_diagonal_elements)

    def forward(self, *modalities):
        """
        Args:
            *modalities: List of embeddings (B, dim_i) per modality

        Returns:
            off_diag_coefficients: (B, n_off_diag)
        """
        concatenated_modalities = torch.cat(modalities, dim=1)
        encoded = self.encoder(concatenated_modalities)

        # Compute the coefficients for the off-diagonal blocks
        off_diag_coefficients = self.off_diag_layer(encoded)

        return off_diag_coefficients
