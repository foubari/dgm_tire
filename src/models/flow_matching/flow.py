"""
Flow Matching with Classifier-Free Guidance
"""

import random
from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
from torch.nn import Module
from tqdm.auto import tqdm

from einops import reduce

# Helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class FlowMatching(Module):
    """
    Flow Matching with Classifier-Free Guidance.
    
    Forward process: x_t = (1-t) * x_0 + t * epsilon
    Velocity: v = epsilon - x_0 (target for training)
    
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
        timesteps=100,
        solver='euler',  # 'euler', 'heun', 'rk4'
        auto_normalize=True,
        cond_drop_prob=0.1,
    ):
        super().__init__()
        assert not (type(self) == FlowMatching and model.channels != model.out_dim)
        
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
        
        # Sampling parameters
        self.num_timesteps = int(timesteps)
        self.solver = solver
        assert solver in {'euler', 'heun', 'rk4'}
        
        # Normalization
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def q_sample(self, x_0, t, noise):
        """
        Forward process: x_t = (1-t) * x_0 + t * noise
        """
        # Ensure t is broadcastable
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=x_0.device, dtype=x_0.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            # (B,) -> (B, 1, 1, 1) for broadcasting
            t = t.view(-1, 1, 1, 1)
        
        return (1 - t) * x_0 + t * noise
    
    def velocity_target(self, x_0, noise):
        """
        Velocity target: v = noise - x_0
        """
        return noise - x_0
    
    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False):
        """
        Get model predictions (velocity).
        """
        if self._guidance_scale is not None and self._sampling_cond is not None:
            model_output = self.cfg_model_out(x, t, cond=self._sampling_cond, x_self_cond=x_self_cond)
        else:
            model_output = self.model(x, t, cond=self._sampling_cond, x_self_cond=x_self_cond)
        
        # Model predicts velocity v = noise - x_0
        # We can recover x_0 from x_t and v: x_0 = x_t - t * v
        velocity = model_output
        
        # Ensure t is broadcastable
        if isinstance(t, (int, float)):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.view(-1, 1, 1, 1)
        
        x_start = x - t * velocity
        
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = maybe_clip(x_start)
        
        return x_start, velocity
    
    def euler_step(self, x, t, dt, cond=None):
        """
        Euler integration step: x_{t-dt} = x_t - dt * v_t
        """
        # Get velocity prediction
        if cond is not None:
            self._sampling_cond = cond
        else:
            self._sampling_cond = None
        
        _, velocity = self.model_predictions(x, t)
        
        # Euler step: x_{t-dt} = x_t - dt * v_t
        x_next = x - dt * velocity
        
        return x_next
    
    def heun_step(self, x, t, dt, cond=None):
        """
        Heun's method (2nd order Runge-Kutta): more accurate than Euler
        """
        if cond is not None:
            self._sampling_cond = cond
        else:
            self._sampling_cond = None
        
        # First Euler step
        _, v1 = self.model_predictions(x, t)
        x_pred = x - dt * v1
        
        # Second step using predicted point
        _, v2 = self.model_predictions(x_pred, t - dt)
        
        # Average velocities
        v_avg = (v1 + v2) / 2
        x_next = x - dt * v_avg
        
        return x_next
    
    def rk4_step(self, x, t, dt, cond=None):
        """
        Runge-Kutta 4th order: most accurate but slower
        """
        if cond is not None:
            self._sampling_cond = cond
        else:
            self._sampling_cond = None
        
        # k1
        _, k1 = self.model_predictions(x, t)
        
        # k2
        x2 = x - (dt / 2) * k1
        _, k2 = self.model_predictions(x2, t - dt / 2)
        
        # k3
        x3 = x - (dt / 2) * k2
        _, k3 = self.model_predictions(x3, t - dt / 2)
        
        # k4
        x4 = x - dt * k3
        _, k4 = self.model_predictions(x4, t - dt)
        
        # Weighted average
        v_avg = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_next = x - dt * v_avg
        
        return x_next
    
    @torch.inference_mode()
    def sample(self, *, batch_size=16, cond=None, guidance_scale=0., return_all_timesteps=False):
        """
        Sample from the flow matching model.
        
        Args:
            batch_size: Number of samples to generate
            cond: Conditioning tensor (optional)
            guidance_scale: Classifier-free guidance scale (0 = unconditional)
            return_all_timesteps: Return all intermediate steps
        
        Returns:
            Generated samples in [0, 1] range
        """
        (h, w), channels = self.image_size, self.channels
        self._guidance_scale = guidance_scale if (cond is not None) else None
        self._sampling_cond = cond
        
        device = self.device
        
        # Initialize from noise: x_1 ~ N(0, I)
        x = torch.randn((batch_size, channels, h, w), device=device)
        
        if return_all_timesteps:
            imgs = [x]
        
        # Integration from t=1 to t=0
        dt = 1.0 / self.num_timesteps
        
        # Choose solver
        if self.solver == 'euler':
            step_fn = self.euler_step
        elif self.solver == 'heun':
            step_fn = self.heun_step
        elif self.solver == 'rk4':
            step_fn = self.rk4_step
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        for i in tqdm(range(self.num_timesteps), desc='sampling loop'):
            t = 1.0 - i * dt  # Start at t=1, go to t=0
            
            # Convert t to tensor
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            
            # Take step
            x = step_fn(x, t_tensor, dt, cond=cond)
            
            if return_all_timesteps:
                imgs.append(x)
        
        # Clean up
        self._guidance_scale = None
        self._sampling_cond = None
        
        ret = x if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret
    
    def cfg_model_out(self, x, t, *, cond, x_self_cond):
        """Classifier-free guidance: v_cfg = v_0 + s(v_c − v_0)"""
        v_uncond = self.model(x, t, cond=None, x_self_cond=x_self_cond)
        v_cond = self.model(x, t, cond=cond, x_self_cond=x_self_cond)
        s = self._guidance_scale
        return v_uncond + s * (v_cond - v_uncond)
    
    def p_losses(self, x_start, t, *, cond=None, noise=None):
        """
        Compute training loss.
        
        Args:
            x_start: Clean images x_0
            t: Time steps (uniformly sampled)
            cond: Conditioning (optional)
            noise: Noise samples (optional, sampled if None)
        
        Returns:
            Loss value
        """
        b, c, h, w = x_start.shape
        
        # Sample noise
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # Sample time uniformly in [0, 1]
        if t is None:
            t = torch.rand((b,), device=x_start.device, dtype=torch.float32)
        else:
            # Ensure t is in [0, 1]
            if isinstance(t, (int, float)):
                t = torch.tensor(t, device=x_start.device, dtype=torch.float32)
            if t.dim() == 0:
                t = t.expand(b)
            t = t.clamp(0, 1)
        
        # Forward process: x_t = (1-t) * x_0 + t * noise
        x_t = self.q_sample(x_start, t, noise)
        
        # Self-conditioning
        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond, _ = self.model_predictions(x_t, t)
                x_self_cond.detach_()
        
        # Conditional dropout
        if cond is not None and random.random() < self.cond_drop_prob:
            cond_used = None
        else:
            cond_used = cond
        
        # Predict velocity
        model_out = self.model(x_t, t, cond=cond_used, x_self_cond=x_self_cond)
        
        # Target velocity: v = noise - x_0
        target = self.velocity_target(x_start, noise)
        
        # MSE loss
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        return loss.mean()
    
    def forward(self, img, *, cond=None, **kwargs):
        """
        Forward pass for training.
        
        Args:
            img: Clean images x_0 in [0, 1]
            cond: Conditioning (optional)
        
        Returns:
            Loss value
        """
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        
        # Normalize to [-1, 1]
        img = self.normalize(img)
        
        # Sample time uniformly
        t = torch.rand((b,), device=device, dtype=torch.float32)
        
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
        
        # Integration from t=1 to t=0
        dt = 1.0 / self.num_timesteps
        
        # Choose solver
        if self.solver == 'euler':
            step_fn = self.euler_step
        elif self.solver == 'heun':
            step_fn = self.heun_step
        elif self.solver == 'rk4':
            step_fn = self.rk4_step
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        batch_size = partial.shape[0]
        
        for i in range(self.num_timesteps):
            t = 1.0 - i * dt  # Start at t=1, go to t=0
            
            # Convert t to tensor
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            
            # Take step
            x = step_fn(x, t_tensor, dt, cond=None)
            
            # Force known region to remain as partial
            x = partial_norm * mask + x * (1 - mask)
            
            if return_all_timesteps and all_steps is not None:
                all_steps.append(x)
        
        x_final = self.unnormalize(x)
        return x_final if not return_all_timesteps else torch.stack(all_steps, dim=1)
