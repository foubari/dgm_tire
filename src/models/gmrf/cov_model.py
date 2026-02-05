"""
Off-diagonal covariance model for GMRF MVAE.

This module computes off-diagonal elements for the covariance matrix assembly.
"""

import torch
import torch.nn as nn


actvn = nn.LeakyReLU(2e-1)


def compute_n_off_diag(modalities_dim):
    """
    Compute the number of lower off-diagonal elements for the covariance matrix.

    Args:
        modalities_dim: List of latent dimensions for each modality

    Returns:
        Total number of off-diagonal elements
    """
    n = len(modalities_dim)
    partial_sums = [modalities_dim[-1]]
    for i in range(n - 2):
        partial_sums.append(partial_sums[-1] + modalities_dim[-i - 2])
    partial_sums = partial_sums[::-1]
    res = 0
    for i in range(n - 1):
        res += partial_sums[i] * modalities_dim[i]
    return res


class OffDiagonalCov(nn.Module):
    """
    Neural network to predict off-diagonal covariance coefficients.

    Takes concatenated embeddings from all modality encoders and outputs
    coefficients for the off-diagonal blocks of the covariance matrix.
    """

    def __init__(self, input_dims, encoded_dims, hidden_dim, n_layers, activation=None):
        super(OffDiagonalCov, self).__init__()

        self.activation = actvn if activation is None else activation

        # Main encoder layers
        total_encoded_size = sum(input_dims)
        layers = [nn.Linear(total_encoded_size, hidden_dim), self.activation]
        layers.extend(
            layer for _ in range(n_layers - 1)
            for layer in [nn.Linear(hidden_dim, hidden_dim), self.activation]
        )
        self.encoder = nn.Sequential(*layers)

        # Output layers for the lower off-diagonal blocks
        off_diagonal_elements = compute_n_off_diag(encoded_dims)
        self.off_diag_layer = nn.Linear(hidden_dim, off_diagonal_elements)

    def forward(self, *modalities):
        """
        Forward pass.

        Args:
            *modalities: Variable number of tensors, each of shape [batch_size, latent_dim]

        Returns:
            off_diag_coefficients: Tensor of shape [batch_size, n_off_diag_elements]
        """
        concatenated_modalities = torch.cat(modalities, dim=1)
        encoded = self.encoder(concatenated_modalities)
        off_diag_coefficients = self.off_diag_layer(encoded)
        return off_diag_coefficients
