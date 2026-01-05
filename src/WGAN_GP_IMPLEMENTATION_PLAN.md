# WGAN-GP Implementation Plan

## Overview

Implement a Wasserstein GAN with Gradient Penalty (WGAN-GP) model using a compressed latent space architecture for the EpureDGM project. The model will encode 5-component images (5 channels × 64×32 pixels) into a 20-dimensional latent vector (4 latents per component), then generate images from this compressed representation.

**Key Design Decisions:**
- Compressed latent space: 20 dimensions total (4 per component)
- UNet-based generator to match DDPM parameter count (~12M)
- Standard WGAN-GP (no self-conditioning)
- Classifier-free guidance via random condition dropout during training
- Three sampling modes: unconditional, conditional, inpainting

---

## Architecture Specifications

### 1. Encoder (Image → Latent)

**File:** `src_new/models/wgan_gp/encoder.py`

**Purpose:** Compress (B, 5, 64, 32) images to (B, 20) latent vectors

**Architecture:**
```
Input: (B, 5, 64, 32)
├── Conv2d(5 → 64, k=3, p=1) + GroupNorm + ReLU
├── Downsample Block 1: Conv2d(64 → 128, k=3, s=2, p=1) + GN + ReLU  → (B, 128, 32, 16)
├── Downsample Block 2: Conv2d(128 → 256, k=3, s=2, p=1) + GN + ReLU → (B, 256, 16, 8)
├── Downsample Block 3: Conv2d(256 → 512, k=3, s=2, p=1) + GN + ReLU → (B, 512, 8, 4)
├── AdaptiveAvgPool2d((1, 1)) → (B, 512)
├── Linear(512 → 256) + ReLU
└── Linear(256 → 20)
Output: (B, 20)
```

**Parameters:** ~1.3M

**Key Points:**
- Joint encoding of all components (captures inter-component correlations)
- Use GroupNorm instead of BatchNorm for stability
- Final projection to configurable latent_dim (default 20)

---

### 2. Generator (Latent → Image)

**File:** `src_new/models/wgan_gp/generator.py`

**Purpose:** Generate (B, 5, 64, 32) images from (B, 20) latent + optional conditions

**Architecture:**
```
Input: z (B, 20) + optional cond (B, cond_dim)
├── Condition Embedding: Linear(cond_dim → 128) → concat with z → (B, 148)
├── Linear(148 → 512×8×4) + Reshape → (B, 512, 8, 4)
├── GroupNorm + ReLU
│
├── Upsample Block 1:
│   ├── ConvTranspose2d(512 → 256, k=4, s=2, p=1) → (B, 256, 16, 8)
│   ├── GroupNorm + ReLU
│   └── ResBlock(256, 256) ×2
│
├── Upsample Block 2:
│   ├── ConvTranspose2d(256 → 128, k=4, s=2, p=1) → (B, 128, 32, 16)
│   ├── GroupNorm + ReLU
│   └── ResBlock(128, 128) ×2
│
├── Upsample Block 3:
│   ├── ConvTranspose2d(128 → 64, k=4, s=2, p=1) → (B, 64, 64, 32)
│   ├── GroupNorm + ReLU
│   └── ResBlock(64, 64) ×2
│
└── Conv2d(64 → 5, k=3, p=1) + Sigmoid
Output: (B, 5, 64, 32) in [0, 1]
```

**ResBlock Structure:**
```
ResBlock(dim, dim):
├── Conv2d(dim → dim, k=3, p=1) + GroupNorm + ReLU
├── Conv2d(dim → dim, k=3, p=1) + GroupNorm
└── Residual connection + ReLU
```

**Parameters:** Target ~10-12M (adjust by adding more ResBlocks or increasing dim if needed)

**Key Points:**
- Condition embedding is concatenated with latent vector
- Multiple ResBlocks per scale to increase capacity and match DDPM parameter count
- Use GroupNorm for stability with small batches
- Final Sigmoid to ensure [0, 1] output range

---

### 3. Critic/Discriminator (Image → Score)

**File:** `src_new/models/wgan_gp/critic.py`

**Purpose:** Estimate Wasserstein distance for (B, 5, 64, 32) images

**Architecture:**
```
Input: x (B, 5, 64, 32) + optional cond (B, cond_dim)
├── Condition Embedding: Linear(cond_dim → 64) → broadcast spatially → (B, 64, 64, 32)
├── Concat with image: (B, 5+64, 64, 32) if conditional
│
├── Conv2d(5/69 → 64, k=4, s=2, p=1) + LeakyReLU(0.2) → (B, 64, 32, 16)
├── Conv2d(64 → 128, k=4, s=2, p=1) + LayerNorm + LReLU → (B, 128, 16, 8)
├── Conv2d(128 → 256, k=4, s=2, p=1) + LayerNorm + LReLU → (B, 256, 8, 4)
├── Conv2d(256 → 512, k=4, s=2, p=1) + LayerNorm + LReLU → (B, 512, 4, 2)
├── AdaptiveAvgPool2d((1, 1)) → (B, 512)
└── Linear(512 → 1)
Output: (B, 1) - unbounded Wasserstein distance estimate
```

**Parameters:** ~3-5M

**Key Points:**
- PatchGAN-style with spatial downsampling
- Conditions are spatially broadcast and concatenated with image
- Use LayerNorm (not BatchNorm) for gradient penalty stability
- NO activation on final output (WGAN requires unbounded critic)
- LeakyReLU(0.2) for all hidden layers

---

### 4. WGAN-GP Wrapper

**File:** `src_new/models/wgan_gp/wgan.py`

**Purpose:** Main interface matching DDPM's API (forward, sample, inpaint)

**Class Structure:**
```python
class WGANGP(nn.Module):
    def __init__(
        self,
        encoder,
        generator,
        critic,
        *,
        image_size,
        latent_dim=20,
        lambda_gp=10,
        n_critic=5,
        cond_drop_prob=0.1,
    ):
        """
        Args:
            encoder: Encoder network (image → latent)
            generator: Generator network (latent → image)
            critic: Critic network (image → score)
            image_size: Tuple (H, W) = (64, 32)
            latent_dim: Total latent dimensions (default 20)
            lambda_gp: Gradient penalty weight (default 10)
            n_critic: Critic updates per generator update (default 5)
            cond_drop_prob: CFG dropout probability (default 0.1)
        """

    def forward(self, img, *, cond=None):
        """
        Training step - returns dict of losses.
        Call this method n_critic times for critic, then once for generator.

        Args:
            img: Real images (B, 5, 64, 32) in [0, 1]
            cond: Conditioning (B, cond_dim) or None

        Returns:
            Dict with keys:
            - 'critic_loss': Critic loss (when training critic)
            - 'generator_loss': Generator loss (when training generator)
            - 'wasserstein_distance': W-distance estimate
            - 'gradient_penalty': GP term
        """

    @torch.inference_mode()
    def sample(self, *, batch_size=16, cond=None, guidance_scale=0.):
        """
        Generate samples.

        Args:
            batch_size: Number of samples
            cond: Conditioning (B, cond_dim) or None
            guidance_scale: CFG scale (0=unconditional, >0=conditional)

        Returns:
            Generated images (B, 5, 64, 32) in [0, 1]
        """

    @torch.inference_mode()
    def inpaint(self, partial, mask, num_steps=100, lr=0.01):
        """
        Inpaint by optimizing latent code to match known regions.

        Args:
            partial: Partial image (B, 5, 64, 32) in [0, 1]
            mask: Binary mask (B, 5, 64, 32), 1=known, 0=unknown
            num_steps: Optimization iterations (default 100)
            lr: Latent optimization learning rate (default 0.01)

        Returns:
            Inpainted image (B, 5, 64, 32) in [0, 1]
        """

    def compute_gradient_penalty(self, real, fake, cond):
        """
        Compute WGAN-GP gradient penalty.

        GP = lambda_gp * E[(||∇D(x_hat)||_2 - 1)^2]
        where x_hat = epsilon * real + (1 - epsilon) * fake
        """
```

**Key Methods:**
- `forward()`: Returns losses dict (separate for critic/generator training)
- `sample()`: Generates with optional CFG
- `inpaint()`: Latent optimization for masked generation
- `compute_gradient_penalty()`: GP computation for WGAN training

---

## Training Algorithm

**File:** `src_new/models/wgan_gp/train.py`

### Training Loop Structure

```python
for epoch in range(num_epochs):
    for batch_idx, (real_images, cond) in enumerate(train_loader):

        # ===========================
        # Update Critic (n_critic=5 times)
        # ===========================
        for _ in range(n_critic):
            critic_optimizer.zero_grad()

            # Sample random latent for fake images
            z = torch.randn(batch_size, latent_dim, device=device)

            # Conditional dropout for CFG
            cond_for_gen = None if (cond is None or random.random() < cond_drop_prob) else cond

            # Generate fake images
            with torch.no_grad():
                fake = generator(z, cond=cond_for_gen)

            # Critic scores
            critic_real = critic(real_images, cond=cond)
            critic_fake = critic(fake.detach(), cond=cond)

            # Wasserstein loss
            wasserstein_distance = critic_real.mean() - critic_fake.mean()

            # Gradient penalty
            gp = compute_gradient_penalty(critic, real_images, fake, cond, lambda_gp)

            # Total critic loss: maximize W-distance, minimize GP
            critic_loss = -wasserstein_distance + gp

            critic_loss.backward()
            critic_optimizer.step()

        # ===========================
        # Update Generator (once)
        # ===========================
        generator_optimizer.zero_grad()

        # Sample random latent
        z = torch.randn(batch_size, latent_dim, device=device)

        # Conditional dropout
        cond_for_gen = None if (cond is None or random.random() < cond_drop_prob) else cond

        # Generate fake images
        fake = generator(z, cond=cond_for_gen)

        # Generator loss: maximize critic score on fake
        critic_fake = critic(fake, cond=cond)
        generator_loss = -critic_fake.mean()

        generator_loss.backward()
        generator_optimizer.step()

        # Update EMA generator
        ema_generator.update()
```

### Key Training Details

1. **Optimizers:**
   - Adam with lr=1e-4 for both generator and critic
   - Betas=(0.0, 0.9) for stability (WGAN-GP recommendation)

2. **EMA:**
   - Apply to generator only (not critic)
   - Decay=0.9999, update_every=10
   - Use EMA generator for sampling/evaluation

3. **Gradient Penalty:**
   - Interpolate between real and fake: `x_hat = ε·real + (1-ε)·fake`
   - Compute ∇D(x_hat) with respect to x_hat
   - Penalty: `λ_gp · E[(||∇D(x_hat)||_2 - 1)²]`
   - Default λ_gp=10

4. **Classifier-Free Guidance Training:**
   - Random condition dropout with probability `cond_drop_prob=0.1`
   - When dropped, pass `cond=None` to generator (unconditional)
   - Critic always sees true conditions (never dropped)

5. **Checkpointing:**
   - Save every `check_every` epochs
   - Save format: `{encoder_state_dict, generator_state_dict, ema_generator_state_dict, critic_state_dict, g_optimizer_state_dict, c_optimizer_state_dict, epoch, config}`

---

## Sampling Methods

**File:** `src_new/models/wgan_gp/sample.py`

### 1. Unconditional Sampling

```python
def sample_unconditional(model, num_samples, batch_sz, save_dir, ...):
    """Generate samples from random noise."""
    samples = model.sample(
        batch_size=batch_sz,
        cond=None,
        guidance_scale=0.0  # No guidance for unconditional
    )
    # Save using utils.io.save_component_images()
```

### 2. Conditional Sampling

```python
def sample_conditional(model, test_loader, guidance_scale, save_dir, ...):
    """Generate conditioned on test set conditions."""
    for images, cond in test_loader:
        samples = model.sample(
            batch_size=len(cond),
            cond=cond,
            guidance_scale=guidance_scale  # e.g., 2.0
        )
        # Save results
```

**Classifier-Free Guidance Implementation:**
```python
# In WGANGP.sample():
if guidance_scale > 0 and cond is not None:
    # Generate unconditional
    fake_uncond = generator(z, cond=None)
    # Generate conditional
    fake_cond = generator(z, cond=cond)
    # Apply guidance: x = x_uncond + scale * (x_cond - x_uncond)
    fake = fake_uncond + guidance_scale * (fake_cond - fake_uncond)
    fake = torch.clamp(fake, 0., 1.)
else:
    fake = generator(z, cond=cond)
```

### 3. Inpainting

```python
def sample_inpainting(model, test_loader, components_to_preserve, save_dir, ...):
    """Inpaint by preserving specified components."""
    for images, cond in test_loader:
        # Create mask: 1 for known, 0 for unknown
        mask = create_mask(components_to_preserve)  # Shape: (B, 5, 64, 32)
        partial = images * mask

        inpainted = model.inpaint(
            partial=partial,
            mask=mask,
            num_steps=100,  # Latent optimization steps
            lr=0.01
        )
        # Save results
```

**Inpainting via Latent Optimization:**
```python
# In WGANGP.inpaint():
# Initialize latent from encoding partial image
z = encoder(partial).clone().requires_grad_(True)
optimizer = torch.optim.Adam([z], lr=lr)

for step in range(num_steps):
    optimizer.zero_grad()
    generated = generator(z, cond=None)

    # Loss: MSE on known regions
    loss = F.mse_loss(generated * mask, partial * mask)
    loss.backward()
    optimizer.step()

# Final composite: known from partial, unknown from generated
with torch.no_grad():
    inpainted = generator(z, cond=None)
    result = partial * mask + inpainted * (1 - mask)
return result
```

---

## Configuration

**File:** `src_new/configs/wgan_gp_default.yaml`

```yaml
model:
  type: wgan_gp
  channels: null  # Auto-calculated from len(component_dirs)
  image_size: [64, 32]  # [H, W]

  # Latent space configuration
  latent_dim_per_component: 4  # NEW: Latent dims per component
  # total_latent_dim: null  # Auto-calculated as len(component_dirs) * latent_dim_per_component

  # Generator architecture
  dim: 64  # Base dimension for generator
  dim_mults: [1, 2, 4]  # Channel multipliers for upsampling blocks

  # WGAN-GP hyperparameters
  lambda_gp: 10  # Gradient penalty weight
  n_critic: 5  # Critic updates per generator update

  # Classifier-free guidance
  cond_drop_prob: 0.1  # Condition dropout probability for CFG
  cond_dim: null  # Auto-calculated from len(condition_columns)

data:
  root_dir: "data/"
  condition_csv: "data/dimensions.csv"
  component_dirs: ["group_nc", "group_km", "bt", "fpu", "tpc"]
  condition_columns: ["width_px", "height_px"]
  prefix_column: "matching"
  filename_pattern: "mdbk_{prefix}_{component}.png"
  split_column: "train"
  normalized: false

training:
  epochs: 1000
  batch_size: 64
  num_workers: 4
  lr_generator: 0.0001
  lr_critic: 0.0001
  beta1: 0.0  # Adam beta1 (WGAN-GP recommendation)
  beta2: 0.9  # Adam beta2
  optimizer: adam
  ema_decay: 0.9999
  ema_update_every: 10
  eval_every: 100
  check_every: 100
  seed: 0

paths:
  output_dir: "outputs/wgan_gp/"
  samples_dir: "samples/wgan_gp/"

sampling:
  num_samples: 16
  batch_sz: 8
  guidance_scale: 2.0
  mode: conditional  # unconditional, conditional, inpainting
  components: ["fpu", "group_nc", "group_km"]  # Components to preserve for inpainting

  # Inpainting parameters
  inpaint_steps: 100  # Latent optimization steps
  inpaint_lr: 0.01  # Learning rate for latent optimization
```

### Config Auto-completion Extension

**Modify:** `src_new/utils/config.py` → `auto_complete_config()`

Add WGAN-GP support:
```python
# After line 89, add:
elif model_type == 'wgan_gp':
    # Auto-calculate channels
    if 'channels' not in model_cfg or model_cfg['channels'] is None:
        component_dirs = data_cfg.get('component_dirs', [])
        if component_dirs:
            model_cfg['channels'] = len(component_dirs)
            print(f"Auto-calculated model.channels = {len(component_dirs)}")

    # Auto-calculate total_latent_dim from channels and latent_dim_per_component
    if 'total_latent_dim' not in model_cfg or model_cfg['total_latent_dim'] is None:
        channels = model_cfg.get('channels')
        latent_dim_per_component = model_cfg.get('latent_dim_per_component', 4)
        if channels:
            model_cfg['total_latent_dim'] = channels * latent_dim_per_component
            print(f"Auto-calculated model.total_latent_dim = {channels * latent_dim_per_component}")
```

### Config Validation Extension

**Modify:** `src_new/utils/config.py` → `validate_config()`

Add WGAN-GP validation after line 203:
```python
elif model_type == 'wgan_gp':
    # Requires component_dirs
    if 'component_dirs' not in data_cfg:
        raise ValueError("WGAN-GP requires data.component_dirs")

    # Validate channels
    if 'channels' in model_cfg and 'component_dirs' in data_cfg:
        if model_cfg['channels'] != len(data_cfg['component_dirs']):
            raise ValueError(
                f"model.channels ({model_cfg['channels']}) does not match "
                f"len(data.component_dirs) ({len(data_cfg['component_dirs'])})"
            )

    # Validate latent dim parameters
    if 'latent_dim_per_component' not in model_cfg:
        raise ValueError("WGAN-GP requires model.latent_dim_per_component")

    if 'lambda_gp' not in model_cfg:
        raise ValueError("WGAN-GP requires model.lambda_gp")

    if 'n_critic' not in model_cfg:
        raise ValueError("WGAN-GP requires model.n_critic")
```

---

## File Structure

```
src_new/models/wgan_gp/
├── __init__.py              # Export: WGANGP, Encoder, Generator, Critic
├── encoder.py               # Encoder: image → latent (20 dims)
├── generator.py             # Generator: latent → image
├── critic.py                # Critic: image → Wasserstein score
├── wgan.py                  # WGANGP wrapper (main interface)
├── train.py                 # Training script (GAN training loop)
└── sample.py                # Sampling script (3 modes)

src_new/configs/
└── wgan_gp_default.yaml     # Default configuration

src_new/scripts/
├── train.py                 # ADD: 'wgan_gp' to dispatcher
└── sample.py                # ADD: 'wgan_gp' to dispatcher
```

---

## Implementation Steps

### Phase 1: Core Architecture (Files 1-4)

1. **encoder.py** - Implement Encoder class
   - Joint encoding of 5 components
   - Downsample to 20-dim latent
   - ~1.3M parameters

2. **generator.py** - Implement Generator class
   - Upsample from 20-dim latent to (5, 64, 32)
   - ResBlocks for capacity (~10-12M params)
   - Condition embedding support

3. **critic.py** - Implement Critic class
   - PatchGAN-style discriminator
   - Condition embedding (spatial broadcast)
   - ~3-5M parameters

4. **wgan.py** - Implement WGANGP wrapper
   - Wrap encoder, generator, critic
   - forward() for training losses
   - sample() with CFG
   - inpaint() with latent optimization
   - compute_gradient_penalty()

### Phase 2: Training Infrastructure (File 5)

5. **train.py** - Training script
   - Data loading (use MultiComponentDataset)
   - WGAN-GP training loop (n_critic updates per G update)
   - EMA for generator
   - Checkpointing
   - Logging (Wasserstein distance, losses)
   - Follow structure of `src_new/models/ddpm/train.py`

### Phase 3: Sampling (File 6)

6. **sample.py** - Sampling script
   - load_model_from_checkpoint()
   - sample_unconditional()
   - sample_conditional() with CFG
   - sample_inpainting() with latent optimization
   - Follow structure of `src_new/models/ddpm/sample.py`

### Phase 4: Integration (Files 7-9)

7. **__init__.py** - Module exports
   ```python
   from .encoder import Encoder
   from .generator import Generator
   from .critic import Critic
   from .wgan import WGANGP
   ```

8. **wgan_gp_default.yaml** - Default config
   - All parameters with sensible defaults
   - Comments explaining each section

9. **Dispatcher integration**
   - Modify `src_new/scripts/train.py` to add 'wgan_gp' routing
   - Modify `src_new/scripts/sample.py` to add 'wgan_gp' routing

### Phase 5: Config Utilities

10. **Update config.py**
    - Extend `auto_complete_config()` for WGAN-GP
    - Extend `validate_config()` for WGAN-GP

---

## Critical Files Reference

### Architecture Reference
- [src_new/models/ddpm/unet.py](src_new/models/ddpm/unet.py) - UNet structure, ResBlocks, attention patterns
- [src_new/models/ddpm/diffusion.py](src_new/models/ddpm/diffusion.py:74-150) - Wrapper class structure, forward/sample/inpaint signatures

### Training Reference
- [src_new/models/ddpm/train.py](src_new/models/ddpm/train.py) - Training loop, EMA, checkpointing, data loading
- [src_new/datasets/base.py](src_new/datasets/base.py) - GenericDataset for multi-component loading

### Config Reference
- [src_new/configs/ddpm_default.yaml](src_new/configs/ddpm_default.yaml) - Config structure and format
- [src_new/utils/config.py](src_new/utils/config.py:58-109) - Auto-completion logic
- [src_new/utils/config.py](src_new/utils/config.py:112-238) - Validation logic

### Utilities Reference
- [src_new/utils/io.py](src_new/utils/io.py) - save_component_images() for output

---

## Parameter Count Target

- **Encoder:** ~1.3M parameters
- **Generator:** ~10-12M parameters (match DDPM UNet)
- **Critic:** ~3-5M parameters
- **Total:** ~15-18M parameters

**Adjustment Strategy:**
- Start with base architecture
- Count parameters: `sum(p.numel() for p in model.parameters())`
- If generator too small: add more ResBlocks or increase dim
- If generator too large: reduce dim_mults or remove ResBlocks

---

## Testing Checklist

1. **Unit Tests:**
   - [ ] Encoder forward pass with dummy input (B, 5, 64, 32) → (B, 20)
   - [ ] Generator forward pass with dummy input (B, 20) → (B, 5, 64, 32)
   - [ ] Critic forward pass with dummy input (B, 5, 64, 32) → (B, 1)
   - [ ] Gradient penalty computation

2. **Integration Tests:**
   - [ ] WGANGP.forward() returns proper loss dict
   - [ ] WGANGP.sample() generates images in [0, 1]
   - [ ] WGANGP.inpaint() respects mask regions
   - [ ] CFG produces different outputs at scale 0 vs 2.0

3. **Training Tests:**
   - [ ] Config loading and auto-completion
   - [ ] Data loader creates proper batches
   - [ ] Training loop runs for 10 epochs without errors
   - [ ] Checkpoint saving/loading works
   - [ ] EMA updates correctly

4. **Sampling Tests:**
   - [ ] Unconditional sampling produces diverse samples
   - [ ] Conditional sampling uses conditions correctly
   - [ ] Inpainting preserves known regions
   - [ ] Images save correctly with save_component_images()

---

## Known Challenges & Solutions

### Challenge 1: Mode Collapse
**Solution:** WGAN-GP is inherently more stable than vanilla GAN. Monitor Wasserstein distance - should stabilize over training. If collapse occurs, try:
- Increase n_critic (e.g., 10 instead of 5)
- Reduce generator learning rate
- Add spectral normalization to critic

### Challenge 2: Inpainting Quality
**Solution:** Latent optimization may struggle with sharp boundaries. Alternatives:
- Increase num_steps (e.g., 200-500)
- Use L1 loss instead of MSE for sharper boundaries
- Add perceptual loss if available
- Pre-train encoder with reconstruction objective

### Challenge 3: Classifier-Free Guidance Strength
**Solution:** CFG may be weaker than in diffusion models. Tuning:
- Try higher guidance_scale (3.0-5.0)
- Increase cond_drop_prob during training (0.2 instead of 0.1)
- Ensure conditions are normalized properly

### Challenge 4: Training Instability
**Solution:** WGAN-GP should be stable, but if issues occur:
- Verify gradient penalty implementation (should penalize ||∇D|| ≠ 1)
- Use LayerNorm in critic (not BatchNorm)
- Clip gradient norm if needed
- Monitor critic loss - should decrease then stabilize

---

## Success Metrics

1. **Training Stability:**
   - Wasserstein distance converges (not oscillating)
   - Gradient penalty stabilizes around 0-2
   - Generator loss decreases steadily

2. **Generation Quality:**
   - Visual inspection: sharp boundaries, correct shapes
   - FID score comparison with DDPM (target: comparable or better)
   - No mode collapse (diverse samples)

3. **Conditioning:**
   - Conditional samples respect input conditions
   - CFG increases conditioning strength (visual test)

4. **Inpainting:**
   - Known regions preserved exactly
   - Generated regions blend smoothly
   - No artifacts at mask boundaries

---

## Usage Examples

### Training
```bash
# Using global dispatcher
python src_new/scripts/train.py --model wgan_gp --config configs/wgan_gp_default.yaml

# Direct training script
python src_new/models/wgan_gp/train.py --config configs/wgan_gp_default.yaml
```

### Sampling
```bash
# Unconditional
python src_new/scripts/sample.py --model wgan_gp \
    --checkpoint outputs/wgan_gp/checkpoint_100.pt \
    --mode unconditional --num_samples 16

# Conditional with CFG
python src_new/scripts/sample.py --model wgan_gp \
    --checkpoint outputs/wgan_gp/checkpoint_100.pt \
    --mode conditional --guidance_scale 2.0

# Inpainting
python src_new/scripts/sample.py --model wgan_gp \
    --checkpoint outputs/wgan_gp/checkpoint_100.pt \
    --mode inpainting --components fpu group_nc
```

---

## Estimated Implementation Time

- **encoder.py:** 2-3 hours
- **generator.py:** 3-4 hours
- **critic.py:** 2-3 hours
- **wgan.py:** 3-4 hours
- **train.py:** 3-4 hours
- **sample.py:** 2-3 hours
- **Config & integration:** 1-2 hours
- **Testing & debugging:** 4-6 hours
- **Total:** 20-29 hours

---

## End of Plan
