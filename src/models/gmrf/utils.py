"""
Utility functions and constants for GMRF MVAE.
"""

import math
import torch


class Constants:
    """Constants used throughout the GMRF model."""
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0
    relu_shift = 1
    exp_shift = 0
    exp_factor = 1


def assemble_covariance_matrix_corrected(mu_list, sigma_list, off_diag_coeffs, modalities_dim, epsilon=0.9, delta=1e-6):
    """
    Assembles the covariance matrix for a multimodal VAE, ensuring symmetry and positive definiteness.

    Parameters:
        mu_list: List of tensors, each of shape (batch_size, dim_k), mean vectors from modality encoders.
        sigma_list: List of tensors, each of shape (batch_size, dim_k, dim_k), diagonal covariance matrices.
        off_diag_coeffs: Tensor of shape (batch_size, num_off_diag_elements), output from the global encoder.
        modalities_dim: List of ints, dimensions of each modality.
        epsilon: Scalar or Tensor of shape (total_dim,), with values less than 1.
        delta: Small positive scalar to prevent division by zero.

    Returns:
        covariance_matrix: Tensor of shape (batch_size, total_dim, total_dim), the assembled covariance matrix.
    """
    total_dim = sum(modalities_dim)
    batch_size = mu_list[0].shape[0]
    device = mu_list[0].device

    # 1. Assemble Lambda, the big diagonal matrix from sigma_list
    sigma_diags = [torch.diagonal(sigma, dim1=-2, dim2=-1) for sigma in sigma_list]
    v = torch.cat(sigma_diags, dim=1)  # Shape: (batch_size, total_dim)

    # 2. Assemble M from off_diag_coeffs
    M = torch.zeros((batch_size, total_dim, total_dim), device=device)

    # Compute start and end indices for each modality
    start_indices = []
    end_indices = []
    start = 0
    for dim in modalities_dim:
        start_indices.append(start)
        end = start + dim
        end_indices.append(end)
        start = end

    # Prepare to fill M
    num_modalities = len(modalities_dim)
    off_diag_block_sizes = []
    modality_pairs = []
    for i in range(1, num_modalities):
        for j in range(i):
            block_size = modalities_dim[i] * modalities_dim[j]
            off_diag_block_sizes.append(block_size)
            modality_pairs.append((i, j))

    # Compute cumulative sum to get offsets
    off_diag_block_starts = [0]
    for size in off_diag_block_sizes:
        off_diag_block_starts.append(off_diag_block_starts[-1] + size)
    off_diag_block_starts = off_diag_block_starts[:-1]

    # Fill M with off-diagonal blocks
    for block_idx, (i, j) in enumerate(modality_pairs):
        start = off_diag_block_starts[block_idx]
        end = start + off_diag_block_sizes[block_idx]
        block_coeffs = off_diag_coeffs[:, start:end]
        block_coeffs = block_coeffs.view(batch_size, modalities_dim[i], modalities_dim[j])
        M[:, start_indices[i]:end_indices[i], start_indices[j]:end_indices[j]] = block_coeffs
        M[:, start_indices[j]:end_indices[j], start_indices[i]:end_indices[i]] = block_coeffs.transpose(1, 2)

    # 3. Compute s_i = sum_{j != i} |M_{ij}|
    s = torch.sum(torch.abs(M), dim=2) - torch.abs(torch.diagonal(M, dim1=1, dim2=2))
    s = s + delta

    # 4. Compute alpha_i
    if isinstance(epsilon, float) or isinstance(epsilon, int):
        epsilon = torch.full_like(v, epsilon)
    else:
        epsilon = epsilon.to(device)

    alpha = torch.minimum(torch.ones_like(s), (v * epsilon) / s)

    # 5. Compute alpha_{ij} = sqrt(alpha_i * alpha_j)
    alpha_i_sqrt = torch.sqrt(alpha)
    alpha_matrix = alpha_i_sqrt.unsqueeze(2) * alpha_i_sqrt.unsqueeze(1)

    # 6. Scale M symmetrically
    M_adjusted = M * alpha_matrix

    # 7. Construct the covariance matrix
    covariance_matrix = torch.diag_embed(v) + M_adjusted

    return covariance_matrix
