"""
Gaussian Diffusion Model with Classifier-Free Guidance
"""

import math
import random
from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import Module
from tqdm.auto import tqdm

from einops import rearrange, reduce

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# Beta schedules
def linear_beta_schedule(timesteps):
    """Linear schedule, proposed in original ddpm paper"""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """Sigmoid schedule proposed in https://arxiv.org/abs/2212.11972 - Figure 8"""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(Module):
    """
    Gaussian Diffusion Model with Classifier-Free Guidance.
    
    Supports:
    - Unconditional sampling
    - Conditional sampling with guidance scale
    - Inpainting (spatial)
    """
    
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_v',
        beta_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        auto_normalize=True,
        offset_noise_strength=0.,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        cond_drop_prob=0.1,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        
        self.model = model
        self.cond_drop_prob = cond_drop_prob
        self._guidance_scale = None  # set temporarily at sampling
        self._sampling_cond = None
        
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2
        self.image_size = image_size
        
        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}
        
        # Beta schedule
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        # Sampling parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        
        # Register buffers
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        self.offset_noise_strength = offset_noise_strength
        
        # Loss weight (SNR)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)
        
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))
        
        # Normalization
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
    
    @property
    def device(self):
        return self.betas.device
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        if self._guidance_scale is not None and self._sampling_cond is not None:
            model_output = self.cfg_model_out(x, t, cond=self._sampling_cond, x_self_cond=x_self_cond)
        else:
            model_output = self.model(x, t, cond=self._sampling_cond, x_self_cond=x_self_cond)
        
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        return ModelPrediction(pred_noise, x_start)
    
    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start
        
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start
    
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device
        
        img = torch.randn(shape, device=device)
        imgs = [img]
        
        x_start = None
        
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)
        
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret
    
    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        img = torch.randn(shape, device=device)
        imgs = [img]
        
        x_start = None
        
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True)
            
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(img)
            
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)
        
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret
    
    @torch.inference_mode()
    def sample(self, *, batch_size=16, cond=None, guidance_scale=0., return_all_timesteps=False):
        (h, w), channels = self.image_size, self.channels
        self._guidance_scale = guidance_scale if (cond is not None) else None
        self._sampling_cond = cond
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        out = sample_fn((batch_size, channels, h, w), return_all_timesteps=return_all_timesteps)
        self._guidance_scale = None
        self._sampling_cond = None
        return out
    
    def cfg_model_out(self, x, t, *, cond, x_self_cond):
        """Classifier-free guidance: ε_cfg = ε₀ + s(ε_c − ε₀)"""
        eps_uncond = self.model(x, t, cond=None, x_self_cond=x_self_cond)
        eps_cond = self.model(x, t, cond=cond, x_self_cond=x_self_cond)
        s = self._guidance_scale
        return eps_uncond + s * (eps_cond - eps_uncond)
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def p_losses(self, x_start, t, *, cond=None, noise=None, offset_noise_strength=None):
        b, c, h, w = x_start.shape
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # Offset noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')
        
        # Noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Self-conditioning
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        
        # Conditional dropout
        if cond is not None and random() < self.cond_drop_prob:
            cond_used = None
        else:
            cond_used = cond
        
        model_out = self.model(x, t, cond=cond_used, x_self_cond=x_self_cond)
        
        # Target
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    def forward(self, img, *, cond=None, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        img = self.normalize(img)
        return self.p_losses(img, t, cond=cond, **kwargs)
    
    @torch.inference_mode()
    def inpaint(self, partial, mask, return_all_timesteps=False):
        """
        Inpaint the masked region(s) of `partial`.
        - `partial` and `mask` must have the same B,H,W.
        - If mask.shape[1] == 1, we broadcast across channels.
        - mask = 1 for known region, mask = 0 for unknown region to fill.
        Returns the inpainted image in [0..1].
        """
        device = self.device
        partial = partial.to(device)
        mask = mask.to(device)
        
        assert partial.shape[0] == mask.shape[0], "batch size mismatch"
        assert partial.shape[2:] == mask.shape[2:], "spatial size mismatch"
        
        if mask.shape[1] == 1 and partial.shape[1] > 1:
            mask = mask.expand(-1, partial.shape[1], -1, -1)
        
        assert partial.shape == mask.shape, f"After broadcasting, partial.shape={partial.shape} != mask.shape={mask.shape}"
        
        # Normalize
        partial_norm = self.normalize(partial)
        
        # Initialize: known region = partial, unknown region = random noise
        x = partial_norm * mask + torch.randn_like(partial_norm) * (1 - mask)
        
        all_steps = [x] if return_all_timesteps else None
        x_start = None
        
        timesteps_list = (
            reversed(range(self.num_timesteps))
            if not self.is_ddim_sampling
            else reversed(range(self.sampling_timesteps))
        )
        
        for t in timesteps_list:
            b_time = torch.full((partial.shape[0],), t, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            
            model_mean, _, model_log_variance, x_start_pred = self.p_mean_variance(
                x=x, t=b_time, x_self_cond=self_cond, clip_denoised=True
            )
            noise = torch.randn_like(x) if t > 0 else 0.0
            x = model_mean + (0.5 * model_log_variance).exp() * noise
            
            # Force known region to remain as partial
            x = partial_norm * mask + x * (1 - mask)
            
            x_start = x_start_pred
            
            if return_all_timesteps and all_steps is not None:
                all_steps.append(x)
        
        x_final = self.unnormalize(x)
        return x_final if not return_all_timesteps else torch.stack(all_steps, dim=1)

