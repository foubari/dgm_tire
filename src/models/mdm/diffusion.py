"""
Multinomial Diffusion Model with Classifier-Free Guidance

Based on: https://github.com/lucidrains/denoising-diffusion-pytorch
"""

import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from tqdm import tqdm

eps = 1e-8


def sum_except_batch(x, num_dims=1):
    """Sums all dimensions except the first."""
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    """Convert class indices to log-one-hot representation."""
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    """Convert log-one-hot representation to class indices."""
    return log_x.argmax(1)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    alphas = np.clip(alphas, a_min=0.001, a_max=1.)
    alphas = np.sqrt(alphas)
    return alphas


class MultinomialDiffusion(torch.nn.Module):
    """
    Multinomial Diffusion Model with Classifier-Free Guidance support.
    
    Args:
        num_classes: Number of classes in the categorical distribution
        shape: Shape of the data (e.g., (1, 64, 32) for segmentation masks)
        denoise_fn: Denoising network (UNet) that takes (t, x_t, cond) and returns logits
        timesteps: Number of diffusion timesteps
        cond_drop_prob: Probability of dropping conditioning during training (for CFG)
        loss_type: 'vb_stochastic' or 'vb_all'
        parametrization: 'x0' or 'direct'
    """
    
    def __init__(self, num_classes, shape, denoise_fn, timesteps=1000, 
                 cond_drop_prob=0.1, loss_type='vb_stochastic', parametrization='x0'):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')
            
        self.cond_drop_prob = cond_drop_prob
        self.num_classes = num_classes
        self._denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.shape = shape
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        alphas = cosine_beta_schedule(timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        # Convert to float32 and register buffers
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def multinomial_kl(self, log_prob1, log_prob2):
        """Compute KL divergence between two categorical distributions."""
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        """Predict q(x_{t-1} | x_t) for one timestep."""
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_pred(self, log_x_start, t):
        """Predict q(x_t | x_0)."""
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)
        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)
        )
        return log_probs

    def predict_start(self, log_x_t, t, cond=None):
        """Predict x_0 from x_t."""
        x_t = log_onehot_to_index(log_x_t)
        out = self._denoise_fn(t, x_t, cond)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out, dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        """Compute q(x_{t-1} | x_t, x_0)."""
        t_minus_1 = t - 1
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start)
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)
        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, log_x, t, cond=None):
        """Predict p(x_{t-1} | x_t) using the model."""
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t, cond=cond)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t, cond=cond)
        else:
            raise ValueError
        return log_model_pred

    @staticmethod
    def _cfg_mix(log_u, log_c, w):
        """
        Classifier-Free Guidance mixing in log-prob space.
        
        Args:
            log_u: log p(x_{t-1} | x_t, ∅) - unconditional
            log_c: log p(x_{t-1} | x_t, cond) - conditional
            w: guidance scale (float ≥ 0)
        
        Returns:
            Guided log-prob: log_u + w·(log_c – log_u)
        """
        return log_u + w * (log_c - log_u)

    @torch.no_grad()
    def p_sample(self, log_x, t, cond=None, guidance_scale=0.):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t) with optional classifier-free guidance.
        
        Args:
            log_x: Current x_t in log-one-hot format
            t: Current timestep
            cond: Conditioning tensor or None
            guidance_scale: Guidance scale (0 = unconditional, >0 = CFG)
        
        Returns:
            log_x_{t-1} in log-one-hot format
        """
        # Unconditional branch
        log_prob_u = self.p_pred(log_x, t, cond=None)
    
        # Either take unconditional only or mix with conditional branch
        if (cond is None) or (guidance_scale == 0):
            guided = log_prob_u
        else:
            log_prob_c = self.p_pred(log_x, t, cond=cond)
            guided = self._cfg_mix(log_prob_u, log_prob_c, guidance_scale)
    
        # Sample x_{t-1} ~ guided distribution
        return self.log_sample_categorical(guided)

    @torch.no_grad()
    def p_sample_loop(self, shape, *, cond=None, guidance_scale=0.):
        """
        Full reverse diffusion sampling loop.
        
        Args:
            shape: Shape tuple (batch_size, *self.shape)
            cond: Conditioning tensor or None
            guidance_scale: Guidance scale for CFG
        
        Returns:
            Sampled x_0 as integer class indices
        """
        device = self.log_alpha.device
        b = shape[0]
    
        log_x = self.log_sample_categorical(
            torch.zeros((b, self.num_classes) + self.shape, device=device)
        )
    
        for i in tqdm(reversed(range(self.num_timesteps))):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_x = self.p_sample(log_x, t, cond=cond, guidance_scale=guidance_scale)
        return log_onehot_to_index(log_x)

    def log_sample_categorical(self, logits):
        """Sample from categorical distribution using Gumbel-max trick."""
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        """Sample from q(x_t | x_0)."""
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    def kl_prior(self, log_x_start):
        """Compute KL divergence to prior."""
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()
        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, log_x_start, log_x_t, t, cond=None, detach_mean=False):
        """Compute loss term L_t."""
        log_true_prob = self.q_posterior(
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(log_x=log_x_t, t=t, cond=cond)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def sample_time(self, b, device, method='uniform'):
        """Sample timesteps for training."""
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, cond=None):
        """Compute training loss."""
        b, device = x.size(0), x.device

        if self.loss_type == 'vb_stochastic':
            x_start = x
            t, pt = self.sample_time(b, device, 'importance')
            log_x_start = index_to_log_onehot(x_start, self.num_classes)

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, cond=cond)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(log_x_start)
            vb_loss = kl / pt + kl_prior

            return -vb_loss

        elif self.loss_type == 'vb_all':
            return -self.nll(x)
        else:
            raise ValueError()

    def log_prob(self, x, cond=None):
        """Compute log probability (for training or evaluation)."""
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x, cond=cond)
        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)
            t, pt = self.sample_time(b, device, 'importance')
            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, cond=cond)
            kl_prior = self.kl_prior(log_x_start)
            loss = kl / pt + kl_prior
            return -loss

    def sample(self, num_samples, *, cond=None, guidance_scale=0.):
        """
        Sample from the model.
        
        Args:
            num_samples: Number of samples to generate
            cond: Conditioning tensor or None
            guidance_scale: Guidance scale for CFG
        
        Returns:
            Sampled x_0 as integer class indices
        """
        shape = (num_samples, *self.shape)
        return self.p_sample_loop(shape, cond=cond, guidance_scale=guidance_scale)

    def sample_chain(self, num_samples, *, cond=None, guidance_scale=0.):
        """Sample full reverse diffusion chain."""
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, cond=cond, guidance_scale=guidance_scale)
            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs

