"""
Objectives for MMVAE+ training.
"""

import torch
from numpy import prod

from .utils import log_mean_exp, is_multidata


def compute_microbatch_split(x, K):
    """Checks if batch needs to be broken down further to fit in memory."""
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


# MULTIMODAL OBJECTIVES
def _m_elbo(model, x, cond=None, K=1, test=False):
    """
    Internal ELBO computation with optional conditioning.
    
    Args:
        model: MMVAEplus model
        x: List of data tensors (one per modality)
        cond: List of condition tensors (one per modality) or None
        K: Number of samples for reparameterization
        test: If True, use self_and_cross_modal_generation_forward
    
    Returns:
        Stacked ELBO values
    """
    if test:
        qu_xs, px_us, uss = model.self_and_cross_modal_generation_forward(x, cond=cond, K=K)
    else:
        qu_xs, px_us, uss = model(x, cond=cond, K=K)
    
    qz_xs, qw_xs = [], []
    for r, qu_x in enumerate(qu_xs):
        qu_x_r_mean, qu_x_r_lv = model.vaes[r].qu_x_params
        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x_lv, qz_x_lv = torch.split(qu_x_r_lv, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x = model.vaes[r].qu_x(qw_x_mean, qw_x_lv)
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    
    lws = []
    for r, qu_x in enumerate(qu_xs):
        ws, zs = torch.split(uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
        lpw = model.pw(*model.pw_params).log_prob(ws).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(-1)
        lpx_u = [px_u.log_prob(x[d]).view(*px_u.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_u in enumerate(px_us[r])]

        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + model.params.beta * (lpz + lpw - lqz_x - lqw_x)
        lws.append(lw)
    
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_elbo(model, x, cond=None, K=1, test=False):
    """
    ELBO (IWAE) objective with optional conditioning.
    
    Args:
        model: MMVAEplus model
        x: List of data tensors (one per modality)
        cond: List of condition tensors (one per modality) or None
        K: Number of samples for reparameterization
        test: If True, use self_and_cross_modal_generation_forward
    
    Returns:
        Scalar ELBO loss
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    if cond is not None:
        cond_split = zip(*[_c.split(S) for _c in cond])
        lw = [_m_elbo(model, _x, cond=_c, K=K, test=test) for _x, _c in zip(x_split, cond_split)]
    else:
        lw = [_m_elbo(model, _x, cond=None, K=K, test=test) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _m_dreg(model, x, cond=None, K=1, test=False):
    """
    Internal DReG computation with optional conditioning.
    
    Args:
        model: MMVAEplus model
        x: List of data tensors (one per modality)
        cond: List of condition tensors (one per modality) or None
        K: Number of samples for reparameterization
        test: If True, use self_and_cross_modal_generation_forward
    
    Returns:
        Stacked DReG values and latent samples
    """
    if test:
        qu_xs, px_us, uss = model.self_and_cross_modal_generation_forward(x, cond=cond, K=K)
    else:
        qu_xs, px_us, uss = model(x, cond=cond, K=K)
    
    qu_xs_ = [vae.qu_x(*[p.detach() for p in vae.qu_x_params]) for vae in model.vaes]
    qz_xs, qw_xs = [], []
    
    for r, qu_x in enumerate(qu_xs_):
        qu_x_r_mean, qu_x_r_lv = model.vaes[r].qu_x_params
        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x_lv, qz_x_lv = torch.split(qu_x_r_lv, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x = model.vaes[r].qu_x(qw_x_mean, qw_x_lv)
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    
    lws = []
    for r, qu_x in enumerate(qu_xs_):
        ws, zs = torch.split(uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
        lpw = model.pw(*model.pw_params).log_prob(ws).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(-1)
        lpx_u = [px_u.log_prob(x[d]).view(*px_u.batch_shape[:2], -1)
                 .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_u in enumerate(px_us[r])]

        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + model.params.beta * (lpz + lpw - lqz_x - lqw_x)
        lws.append(lw)
    
    return torch.stack(lws), torch.stack(uss)


def m_dreg(model, x, cond=None, K=1, test=False):
    """
    DReG objective with optional conditioning.
    
    Args:
        model: MMVAEplus model
        x: List of data tensors (one per modality)
        cond: List of condition tensors (one per modality) or None
        K: Number of samples for reparameterization
        test: If True, use self_and_cross_modal_generation_forward
    
    Returns:
        Scalar DReG loss
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    if cond is not None:
        cond_split = zip(*[_c.split(S) for _c in cond])
        lw, uss = zip(*[_m_dreg(model, _x, cond=_c, K=K, test=test) for _x, _c in zip(x_split, cond_split)])
    else:
        lw, uss = zip(*[_m_dreg(model, _x, cond=None, K=K, test=test) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    uss = torch.cat(uss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if uss.requires_grad:
            uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum()

