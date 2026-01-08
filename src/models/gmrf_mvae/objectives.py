"""
Objective functions for GMRF-MVAE training.

This module implements the ELBO computation with various reconstruction losses,
matching the ICTAI original implementation.
"""

import torch
import torch.nn.functional as F


def kl_divergence_gaussians(mu_q, Sigma_q, mu_p, Sigma_p):
    """
    Compute KL divergence between two multivariate Gaussians.

    KL(q||p) = 0.5 * [log|Σ_p|/|Σ_q| - d + tr(Σ_p^{-1}Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q)]

    Args:
        mu_q: Mean of q(z|x), shape [batch_size, latent_dim]
        Sigma_q: Covariance of q(z|x), shape [batch_size, latent_dim, latent_dim]
        mu_p: Mean of p(z), shape [latent_dim]
        Sigma_p: Covariance of p(z), shape [latent_dim, latent_dim]

    Returns:
        kl_div: KL divergence (scalar, averaged over batch)
    """
    batch_size, latent_dim = mu_q.shape

    # Inverse of Sigma_p: [latent_dim, latent_dim]
    Sigma_p_inv = torch.inverse(Sigma_p)

    # Expand to batch size
    Sigma_p_inv_expanded = Sigma_p_inv.unsqueeze(0).expand(
        batch_size, latent_dim, latent_dim
    )

    # Trace term: trace(Sigma_p_inv @ Sigma_q)
    trace_term = torch.einsum("bij,bij->b", Sigma_p_inv_expanded, Sigma_q)

    # Mahalanobis term: (mu_q - mu_p)^T @ Sigma_p_inv @ (mu_q - mu_p)
    diff = mu_q - mu_p.unsqueeze(0)  # [batch_size, latent_dim]
    mahalanobis_term = torch.einsum(
        "bi,bij,bj->b", diff, Sigma_p_inv_expanded, diff
    )

    # Log determinants
    sign_q, logdet_q = torch.slogdet(Sigma_q)  # [batch_size]
    sign_p, logdet_p = torch.slogdet(Sigma_p)  # scalar

    # Check positive definiteness
    if not torch.all(sign_q > 0):
        raise ValueError("Sigma_q is not positive definite")
    if sign_p <= 0:
        raise ValueError("Sigma_p is not positive definite")

    # KL divergence per sample
    kl_div = 0.5 * (
        logdet_p - logdet_q - latent_dim + trace_term + mahalanobis_term
    )

    return kl_div.mean()


def compute_elbo_dist(
    model, data, beta=1, loss_type="mse", alpha_mse=0.5, weights=None
):
    """
    Compute ELBO loss with weighted reconstruction.

    This matches the ICTAI original implementation exactly.

    Args:
        model: GMRF_MVAE model (must have mu_p, get_sigma_p(), muq, Sigmaq, recons)
        data: List/tuple of component images [img1, img2, ...]
        beta: KL weighting (default: 1)
        loss_type: Reconstruction loss type:
            - "mse": Mean Squared Error
            - "l1": L1 (MAE) loss
            - "l1_mse": Mixed L1-MSE with alpha_mse weighting
            - "split_l1_mse": ICTAI original! sum(w*MSE) + sum((1-w)*L1)
            - "bce": Binary Cross-Entropy
        alpha_mse: Weight for MSE in "l1_mse" mode (default: 0.5)
        weights: Per-component reconstruction weights (default: equal weights)

    Returns:
        elbo: ELBO loss to minimize (negative)
        recon_loss: Reconstruction loss (for logging)
        kl_divergence: KL divergence (for logging)
    """
    # Validate alpha_mse
    if not (0 <= alpha_mse <= 1):
        raise ValueError(f"alpha_mse must be in [0,1], got {alpha_mse}")

    # Default weights: equal weighting
    num_components = len(model.recons)
    if weights is None:
        weights = [1 / num_components] * num_components
    elif len(weights) != num_components:
        raise ValueError(
            f"recon_weights length ({len(weights)}) doesn't match "
            f"num_components ({num_components}). "
            f"Expected {num_components} weights."
        )

    # Get prior and posterior distributions
    mu_p = model.mu_p  # [total_latent_dim]
    Sigma_p = model.get_sigma_p()  # [total_latent_dim, total_latent_dim]
    mu_q = model.muq  # [batch_size, total_latent_dim]
    Sigma_q = model.Sigmaq  # [batch_size, total_latent_dim, total_latent_dim]

    # KL divergence: KL(q(z|x) || p(z))
    kl_divergence = kl_divergence_gaussians(mu_q, Sigma_q, mu_p, Sigma_p)

    # Reconstruction loss
    if loss_type == "mse":
        recon_loss = sum(
            w * F.mse_loss(mu, d) for w, mu, d in zip(weights, model.recons, data)
        )

    elif loss_type == "l1":
        recon_loss = sum(
            w * F.l1_loss(mu, d) for w, mu, d in zip(weights, model.recons, data)
        )

    elif loss_type == "l1_mse":
        # Mixed: alpha_mse * MSE + (1 - alpha_mse) * L1
        recon_loss = sum(
            w * (alpha_mse * F.mse_loss(mu, d) + (1 - alpha_mse) * F.l1_loss(mu, d))
            for w, mu, d in zip(weights, model.recons, data)
        )

    elif loss_type == "split_l1_mse":
        # ICTAI ORIGINAL FORMULA!
        # MSE gets weight w, L1 gets weight (1-w)
        # Total: sum(w*MSE) + sum((1-w)*L1)
        mse_term = sum(
            w * F.mse_loss(mu, d) for w, mu, d in zip(weights, model.recons, data)
        )
        l1_term = sum(
            (1 - w) * F.l1_loss(mu, d)
            for w, mu, d in zip(weights, model.recons, data)
        )
        recon_loss = mse_term + l1_term

    elif loss_type == "bce":
        recon_loss = sum(
            w * F.binary_cross_entropy_with_logits(mu, d)
            for w, mu, d in zip(weights, model.recons, data)
        )

    else:
        raise ValueError(
            f"Invalid loss_type: {loss_type}. "
            f"Choose 'mse', 'l1', 'l1_mse', 'split_l1_mse', or 'bce'."
        )

    # ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
    # We want to maximize ELBO, so minimize -ELBO
    elbo = -recon_loss - beta * kl_divergence

    return elbo, -recon_loss, kl_divergence
