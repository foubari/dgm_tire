# Plan: Implémentation des Modèles VAE pour EpureDGM

## Résumé Exécutif

**Objectifs:**
1. Implémenter 3 nouveaux modèles VAE : Simple VAE, GMRF MVAE, Meta VAE
2. Adapter tous les modèles VAE pour la génération conditionnelle
3. Homogénéiser les méthodes de masking avec VQVAE
4. Assurer la comparabilité en termes de nombre de paramètres
5. Implémenter scripts train.py et sample.py avec 3 modes de sampling

**Décisions clés:**
- Conditioning: MLP 2→128→256 (comme VQVAE/MM-VAE+)
- Masking: Suivre pattern VQVAE pour VAE et Meta VAE
- Paramètres: Respecter les implémentations ICTAI originales
- Structure: Même organisation que models/vqvae/ et models/mmvaeplus/
- Sampling modes: (1) Unconditional, (2) Conditional, (3) Inpainting/Cross-modal

---

## Partie 1: Simple VAE (Beta-VAE)

### 1.1 Architecture

**Source ICTAI:** `C:\Users\fouad\Desktop\phd\FOUAD\ICTAI\ICTAI\beta_vae\`

**Cible EpureDGM:** `src_new/models/vae/`

#### Structure du Modèle

```
VAE Architecture:
├── Encoder (ResNet-based)
│   ├── Input: (B, 5, 64, 32) [5 components stacked]
│   ├── Initial ResNet block (nf=32)
│   ├── 3× Progressive ResNet blocks with downsampling
│   │   └── Final: 256 filters, spatial 8×4
│   ├── Flatten: 256×8×4 = 8192
│   ├── FC: 8192 → latent_dim (mu)
│   └── FC: 8192 → latent_dim (logvar)
│
├── Decoder (Mirror architecture)
│   ├── Input: latent_dim [+ cond_emb(256)]
│   ├── FC: latent_dim+256 → 8192
│   ├── Reshape: (B, 256, 8, 4)
│   ├── 3× ResNet blocks with 2× upsampling
│   ├── Final Conv: nf → 5 channels
│   └── Output: (B, 5, 64, 32)
```

**Hyperparamètres:**
- `latent_dim`: 32 (default)
- `nf`: 32 (base filters)
- `nf_max`: 256 (max encoder/decoder filters)
- `beta`: 4.0 (KL weight)
- `dropout_p`: 0.1
- `cond_dim`: 2 (width_px, height_px)

**Paramètres estimés:** ~9.77M (comme ICTAI)

#### Conditioning

```python
# Encoder: No conditioning (encode images only)
mu, logvar = encoder(x)

# Decoder: Concatenate cond_emb with latent
cond_mlp = Sequential(
    Linear(2, 128), GELU(),
    Linear(128, 256)
)
cond_emb = cond_mlp(cond)  # (B, 256)
decoder_input = torch.cat([z, cond_emb], dim=1)  # (B, latent_dim+256)
recon = decoder(decoder_input)
```

#### Masking (Component-Level)

```python
def mask_components(x, p=0.5):
    """
    Component masking for inpainting.

    Args:
        x: (B, 5, H, W) - all 5 components
        p: Probability of masking

    Returns:
        Masked x with ONE random component kept, others zeroed
    """
    B = x.shape[0]
    masked_x = torch.zeros_like(x)

    for i in range(B):
        if random.random() < p:
            # Keep only ONE random component
            keep_idx = random.randint(0, 4)
            masked_x[i, keep_idx] = x[i, keep_idx]
        else:
            # Keep all components (no masking)
            masked_x[i] = x[i]

    return masked_x

# Training with progressive masking
mask_prob = min(1.0, (epoch / warmup_epochs) ** 0.5) * max_mask_prob
x_masked = mask_components(x, p=mask_prob)
```

---

### 1.2 Fichiers à Créer

```
src_new/models/vae/
├── __init__.py
├── model.py                # Main BetaVAE class
├── encoder.py             # ResNet encoder
├── decoder.py             # ResNet decoder with conditioning
├── train.py               # Training script
├── sample.py              # Sampling script (3 modes)
└── utils.py               # Masking utilities
```

#### 1. `model.py` (Main BetaVAE)

```python
class BetaVAE(nn.Module):
    def __init__(
        self,
        image_size=(64, 32),
        channels=5,
        cond_dim=2,
        latent_dim=32,
        nf=32,
        nf_max=256,
        beta=4.0,
        dropout_p=0.1
    ):
        super().__init__()
        self.encoder = Encoder(channels, latent_dim, nf, nf_max, dropout_p)
        self.decoder = Decoder(latent_dim, channels, cond_dim, nf, nf_max)
        self.beta = beta

    def forward(self, x, cond=None, mask_prob=0.0):
        # Apply masking if specified
        if mask_prob > 0:
            x = mask_components(x, p=mask_prob)

        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode with conditioning
        recon = self.decoder(z, cond)

        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss
        loss = recon_loss + self.beta * kl_loss

        return loss, recon_loss, kl_loss
```

#### 2. `train.py` (Training Script)

```python
def train(config):
    # Load dataset
    train_loader = get_dataloader(config, split='train')

    # Create model
    model = BetaVAE(**config['model']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        # Progressive masking schedule
        mask_prob = compute_mask_prob(epoch, config)

        for batch_idx, (x, cond) in enumerate(train_loader):
            x, cond = x.to(device), cond.to(device)

            # Forward pass with masking
            recon, mu, logvar = model(x, cond, mask_prob)

            # Compute loss
            loss, recon_loss, kl_loss = model.loss_function(recon, x, mu, logvar)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

#### 3. `sample.py` (Sampling Script)

```python
@torch.no_grad()
def sample_unconditional(model, num_samples, device):
    """Sample from prior N(0, I)."""
    z = torch.randn(num_samples, model.latent_dim).to(device)
    samples = model.decoder(z, cond=None)
    return samples

@torch.no_grad()
def sample_conditional(model, conditions, device):
    """Sample conditioned on specific attributes."""
    B = conditions.shape[0]
    z = torch.randn(B, model.latent_dim).to(device)
    samples = model.decoder(z, cond=conditions)
    return samples

@torch.no_grad()
def sample_inpainting(model, partial_x, mask, device):
    """
    Inpaint missing components.

    Args:
        partial_x: (B, 5, H, W) - some components are zero
        mask: (B, 5) - binary mask (1=observed, 0=missing)
    """
    # Encode partial observation
    mu, logvar = model.encoder(partial_x)

    # Sample multiple times from posterior
    num_samples = 10
    all_samples = []
    for _ in range(num_samples):
        z = model.reparameterize(mu, logvar)
        sample = model.decoder(z, cond=None)
        all_samples.append(sample)

    # Average over samples
    recon = torch.stack(all_samples).mean(dim=0)
    return recon
```

---

### 1.3 Configuration Files

**`configs/vae_epure_full.yaml`:**
```yaml
model:
  type: vae
  image_size: [64, 32]
  channels: 5
  cond_dim: 2
  latent_dim: 32
  nf: 32
  nf_max: 256
  beta: 4.0
  dropout_p: 0.1

data:
  root_dir: "data/epure"
  condition_csv: "data/epure/performances.csv"
  component_dirs: ["group_nc", "group_km", "bt", "fpu", "tpc"]
  condition_columns: ["d_cons_norm", "d_rigid_norm", "d_life_norm", "d_stab_norm"]
  prefix_column: "matching"
  filename_pattern: "{prefix}_{component}.png"
  split_column: "train"
  normalized: true

training:
  epochs: 100
  batch_size: 128
  lr: 0.0002
  optimizer: adamw
  weight_decay: 0.0001
  warmup_epochs: 20
  max_mask_prob: 0.7
  seed: 0

paths:
  output_dir: "outputs/vae/"
  samples_dir: "samples/vae/"

sampling:
  num_samples: 100
  batch_sz: 16
  mode: conditional  # or unconditional, inpainting
```

**`configs/vae_toy_test.yaml`:** (Quick test with 1 epoch)

---

## Partie 2: GMRF MVAE

### 2.1 Architecture

**Source ICTAI:** `C:\Users\fouad\Desktop\phd\FOUAD\ICTAI\ICTAI\gmrf_mvae\`

**Cible EpureDGM:** `src_new/models/gmrf_mvae/`

#### Structure du Modèle

```
GMRF_MVAE Architecture:
├── 5 Component VAEs (GMRF_VAE)
│   ├── Encoder (per component)
│   │   ├── Input: (B, 1, 64, 32) - single component
│   │   ├── ResNet blocks → latent_dim features
│   │   ├── Output: mu_z (B, latent_dim)
│   │   ├── Output: lambda_z (diagonal covariance) (B, latent_dim, latent_dim)
│   │   └── Output: cov_embedding (B, latent_dim) → for off-diagonal
│   │
│   └── Decoder (per component)
│       ├── Input: latent_dim [+ cond_emb(256)]
│       ├── FC → 256×8×4
│       ├── ResNet upsample blocks
│       └── Output: (B, 1, 64, 32)
│
├── OffDiagonalCov Network
│   ├── Input: Concatenate 5× cov_embeddings (B, 5×latent_dim)
│   ├── MLP: Linear(5×latent_dim, hidden_dim) → ReLU
│   ├── n_layers× Linear(hidden_dim, hidden_dim) → ReLU
│   └── Output: Lower triangular off-diagonal elements
│
└── GMRF Prior p(z)
    ├── mu_p: (5×latent_dim,) - learnable prior mean
    ├── diag_p: (5×latent_dim,) - learnable diagonal
    └── off_diag_p: Structured via OffDiagonalCov
```

**Hyperparamètres:**
- `latent_dim`: 16 (per component)
- `nf`: 64 (base filters)
- `nf_max_e`: 1024 (encoder max)
- `nf_max_d`: 512 (decoder max)
- `hidden_dim`: 256 (OffDiagonalCov)
- `n_layers`: 2 (OffDiagonalCov)
- `diagonal_transf`: 'softplus' (transformation for diagonal)
- `off_diag_scale`: 0.1 (initialization)
- `beta`: 1.0
- `cond_dim`: 2

**Paramètres estimés:** ~4.5-5.5M

#### Conditioning

```python
# Encoder: No conditioning (encode individual components)
for i, vae in enumerate(self.component_vaes):
    mu_i, lambda_i, cov_emb_i = vae.encoder(x[:, i:i+1])  # x[:, i] is component i

# Decoder: Concatenate cond_emb with latent
for i, vae in enumerate(self.component_vaes):
    cond_emb = cond_mlp(cond)  # (B, 256)
    decoder_input = torch.cat([z[:, i*latent_dim:(i+1)*latent_dim], cond_emb], dim=1)
    recon_i = vae.decoder(decoder_input)
```

#### Cross-Modal Generation (CRITIQUE!)

**Méthode clé:** Gaussian conditional distribution

```python
def conditional_generate(self, cond, idx_i, idx_cond, n_sample=1):
    """
    Generate component idx_i given observed component idx_cond.

    Args:
        cond: Observed component image (B, 1, 64, 32)
        idx_i: Index of target component to generate
        idx_cond: Index of conditioning component
        n_sample: Number of samples per observation

    Process:
        1. Encode conditioning component: mu_j, Sigma_j = encoder(cond)
        2. Sample from posterior: z_j ~ q(z_j|cond)
        3. Compute conditional distribution:
           - Extract blocks from prior: Sigma_ii, Sigma_jj, Sigma_ij
           - Conditional mean: mu_i|j = mu_i + Sigma_ij @ Sigma_jj^-1 @ (z_j - mu_j)
           - Conditional cov: Sigma_i|j = Sigma_ii - Sigma_ij @ Sigma_jj^-1 @ Sigma_ij^T
        4. Sample: z_i ~ N(mu_i|j, Sigma_i|j)
        5. Decode: x_i = decoder_i(z_i)
    """
    # Step 1: Encode conditioning component
    mu_j, lambda_j = self.component_vaes[idx_cond].encoder(cond)
    Sigma_j = torch.diag_embed(lambda_j)  # Diagonal covariance

    # Step 2: Sample from posterior
    dist_j = MultivariateNormal(mu_j, Sigma_j)
    z_j = dist_j.rsample([n_sample])  # (n_sample, B, latent_dim)

    # Step 3: Compute conditional distribution
    # Extract prior blocks
    mu_p = self.mu_p  # (5×latent_dim,)
    Sigma_p = self.assemble_prior_covariance()  # (5×latent_dim, 5×latent_dim)

    # Indices for slicing
    start_i = idx_i * self.latent_dim
    end_i = (idx_i + 1) * self.latent_dim
    start_j = idx_cond * self.latent_dim
    end_j = (idx_cond + 1) * self.latent_dim

    mu_i = mu_p[start_i:end_i]
    mu_j_prior = mu_p[start_j:end_j]
    Sigma_ii = Sigma_p[start_i:end_i, start_i:end_i]
    Sigma_jj = Sigma_p[start_j:end_j, start_j:end_j]
    Sigma_ij = Sigma_p[start_i:end_i, start_j:end_j]

    # Inverse Sigma_jj (add regularization for stability)
    Sigma_jj_inv = torch.linalg.solve(Sigma_jj + 1e-6 * torch.eye(self.latent_dim), torch.eye(self.latent_dim))

    # Conditional mean and covariance
    mu_cond = mu_i + (Sigma_ij @ Sigma_jj_inv) @ (z_j - mu_j_prior)
    Sigma_cond = Sigma_ii - (Sigma_ij @ Sigma_jj_inv @ Sigma_ij.T)

    # Step 4: Sample from conditional
    dist_i = MultivariateNormal(mu_cond, Sigma_cond + 1e-6 * torch.eye(self.latent_dim))
    z_i = dist_i.rsample()

    # Step 5: Decode
    recon_i = self.component_vaes[idx_i].decoder(z_i)

    return recon_i
```

#### Covariance Assembly (CRITICAL!)

```python
def assemble_covariance_matrix(self, sigma_list, off_diag_coeffs, epsilon=0.9):
    """
    Assemble full covariance matrix with diagonal dominance.

    Args:
        sigma_list: List of 5 diagonal covariances (each B, latent_dim, latent_dim)
        off_diag_coeffs: Off-diagonal coefficients from OffDiagonalCov
        epsilon: Diagonal dominance parameter (0.9)

    Returns:
        Sigma_q: (B, 5×latent_dim, 5×latent_dim) - positive definite
    """
    B = sigma_list[0].shape[0]
    total_dim = 5 * self.latent_dim

    # 1. Extract diagonal blocks
    v = torch.cat([torch.diagonal(sigma, dim1=1, dim2=2) for sigma in sigma_list], dim=1)  # (B, 5×latent_dim)

    # 2. Assemble off-diagonal matrix M
    M = torch.zeros(B, total_dim, total_dim).to(sigma_list[0].device)

    # Fill off-diagonal blocks from off_diag_coeffs
    # ... (complex indexing logic)

    # 3. Ensure positive definiteness via diagonal dominance
    # For each row i: s_i = sum_j |M_ij| - |M_ii|
    # Scale: alpha_i = min(1, v_i * epsilon / s_i)
    s = M.abs().sum(dim=2) - M.diagonal(dim1=1, dim2=2).abs()
    alpha = torch.min(torch.ones_like(v), (v * epsilon) / (s + 1e-8))

    # Symmetric scaling: alpha_matrix_ij = sqrt(alpha_i * alpha_j)
    alpha_sqrt = alpha.sqrt().unsqueeze(2)
    alpha_matrix = alpha_sqrt @ alpha_sqrt.transpose(1, 2)
    M_adjusted = M * alpha_matrix

    # 4. Final covariance: Sigma_q = diag(v) + M_adjusted
    Sigma_q = torch.diag_embed(v) + M_adjusted

    return Sigma_q
```

---

### 2.2 Fichiers à Créer

```
src_new/models/gmrf_mvae/
├── __init__.py
├── model.py               # Main GMRF_MVAE class
├── cov_model.py          # OffDiagonalCov network
├── encoder_decoder.py    # GMRF_VAE encoder/decoder (ResNet)
├── base_vae.py           # GMRF_VAE base class (single component VAE)
├── objectives.py         # ELBO, KL divergence, loss functions
├── utils.py              # assemble_covariance_matrix, etc.
├── train.py              # Training script
├── sample.py             # Sampling script (unconditional + cross-modal)
└── model_epure.py        # Epure_GMMVAE wrapper for 5 components
```

#### Key Implementation Details

**1. `cov_model.py` (OffDiagonalCov):**
```python
class OffDiagonalCov(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int, n_layers: int, output_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(sum(input_dims), hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, embeddings: List[torch.Tensor]):
        # Concatenate embeddings from all components
        x = torch.cat(embeddings, dim=1)
        x = F.relu(self.input_proj(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        out = self.output(x)
        return out
```

**2. `objectives.py` (ELBO with GMRF prior):**
```python
def compute_elbo(model, data, cond, beta=1.0, loss_type='mse', weights=None):
    """
    Compute ELBO with GMRF prior.

    Loss = Reconstruction + beta × KL(q(z|X) || p(z))

    where p(z) is multivariate Gaussian with GMRF structure.
    """
    # Encode all components
    mus, lambdas, cov_embs = [], [], []
    for i, vae in enumerate(model.component_vaes):
        mu_i, lambda_i, cov_emb_i = vae.encoder(data[:, i:i+1])
        mus.append(mu_i)
        lambdas.append(lambda_i)
        cov_embs.append(cov_emb_i)

    # Assemble posterior covariance
    sigma_list = [torch.diag_embed(lam) for lam in lambdas]
    off_diag = model.off_diag_cov(cov_embs)
    Sigma_q = model.assemble_covariance_matrix(sigma_list, off_diag)
    mu_q = torch.cat(mus, dim=1)

    # Sample from posterior
    dist_q = MultivariateNormal(mu_q, Sigma_q)
    z = dist_q.rsample()

    # Decode
    recons = []
    for i, vae in enumerate(model.component_vaes):
        z_i = z[:, i*model.latent_dim:(i+1)*model.latent_dim]
        recon_i = vae.decoder(z_i, cond)
        recons.append(recon_i)

    # Reconstruction loss
    if weights is None:
        weights = [1.0] * len(recons)

    if loss_type == 'mse':
        recon_loss = sum(w * F.mse_loss(r, d[:, i:i+1])
                        for i, (w, r, d) in enumerate(zip(weights, recons, data)))

    # KL divergence with GMRF prior
    mu_p = model.mu_p
    Sigma_p = model.assemble_prior_covariance()
    kl_loss = kl_divergence_gaussians(mu_q, Sigma_q, mu_p, Sigma_p)

    # ELBO
    elbo = -recon_loss - beta * kl_loss

    return elbo, recon_loss, kl_loss
```

---

### 2.3 Configuration Files

**`configs/gmrf_mvae_epure_full.yaml`:**
```yaml
model:
  type: gmrf_mvae
  image_size: [64, 32]
  channels: 1  # per component
  cond_dim: 2
  latent_dim: 16  # per component (total: 5×16=80)
  nf: 64
  nf_max_e: 1024
  nf_max_d: 512
  hidden_dim: 256  # OffDiagonalCov
  n_layers: 2      # OffDiagonalCov
  diagonal_transf: softplus
  off_diag_scale: 0.1
  beta: 1.0

data:
  root_dir: "data/epure"
  condition_csv: "data/epure/performances.csv"
  component_dirs: ["group_nc", "group_km", "bt", "fpu", "tpc"]
  condition_columns: ["d_cons_norm", "d_rigid_norm", "d_life_norm", "d_stab_norm"]
  prefix_column: "matching"
  filename_pattern: "{prefix}_{component}.png"
  split_column: "train"
  normalized: true

training:
  epochs: 100
  batch_size: 32
  lr: 0.0005
  optimizer: adam
  loss_type: mse
  recon_weights: [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal per component
  seed: 0

paths:
  output_dir: "outputs/gmrf_mvae/"
  samples_dir: "samples/gmrf_mvae/"

sampling:
  num_samples: 100
  batch_sz: 16
  mode: conditional  # or unconditional, cross_modal
  cross_modal:
    source_component: 0  # group_nc
    target_component: 1  # group_km
```

---

## Partie 3: Meta VAE

### 3.1 Architecture

**Source ICTAI:** `C:\Users\fouad\Desktop\phd\FOUAD\ICTAI\ICTAI\meta_vae\`

**Cible EpureDGM:** `src_new/models/meta_vae/`

#### Structure du Modèle

```
Meta-VAE Architecture:
├── Meta-Encoder (Hierarchical)
│   ├── Input: List of 5 component images (B, 3→1, 64, 32)
│   ├── Shared Backbone (per component)
│   │   ├── Conv2d(1→24, k=4, s=2, p=1) + BN + LeakyReLU  # 64×32 → 32×16
│   │   ├── Conv2d(24→48, k=4, s=2, p=1) + BN + LeakyReLU  # 32×16 → 16×8
│   │   ├── Conv2d(48→96, k=4, s=2, p=1) + BN + LeakyReLU  # 16×8 → 8×4
│   │   ├── Conv2d(96→128, k=4, s=2, p=1) + BN + LeakyReLU # 8×4 → 4×2
│   │   └── Flatten: 128×4×2 = 1024
│   ├── Component-Specific Processing (5 branches)
│   │   ├── Linear(1024→256) + BN + ReLU
│   │   ├── Linear(256→192) + BN + ReLU
│   │   └── Linear(192→192) → (B, 192) per component
│   ├── Combine: Concatenate 5×192 = 960
│   ├── Linear(960→512) + BN + ReLU
│   ├── Linear(512→384) + BN + ReLU
│   ├── fc_mu: Linear(384→latent_dim_meta)
│   └── fc_logvar: Linear(384→latent_dim_meta)
│
├── Meta-Decoder
│   ├── Input: z_meta (B, latent_dim_meta) [+ cond_emb(256)]
│   ├── Initial: Linear(latent_dim_meta+256→512) + ReLU
│   ├── 5 Parallel Decoders (one per component)
│   │   ├── Linear(512→256) + ReLU
│   │   ├── Linear(256→256) + ReLU
│   │   └── Linear(256→component_latent_dim)
│   └── Output: 5× component latent codes (B, component_latent_dim)
│
└── Frozen Marginal Decoders (5×)
    ├── Input: component_latent_dim
    ├── Linear(component_latent_dim→512)
    ├── ResNet upsample blocks
    └── Output: (B, 1, 64, 32) per component
```

**Hyperparamètres:**
- `latent_dim_meta`: 32 (meta-level latent)
- `component_latent_dim`: 4 (per marginal decoder)
- `channels`: 1 (per component, grayscale)
- `cond_dim`: 2
- **Frozen marginal decoders:** Pre-trained VAEs (total ~360-600K params)
- **Trainable:** Meta-encoder (~3.4M) + Meta-decoder (~300-500K)

**Paramètres estimés:** ~4-4.5M (trainable: ~3.7-4M)

#### Conditioning

```python
# Meta-Encoder: No conditioning (encode component images)
mu_meta, logvar_meta = meta_encoder(component_images)

# Meta-Decoder: Concatenate cond_emb with z_meta
cond_mlp = Sequential(Linear(2, 128), GELU(), Linear(128, 256))
cond_emb = cond_mlp(cond)  # (B, 256)
decoder_input = torch.cat([z_meta, cond_emb], dim=1)  # (B, latent_dim_meta+256)
component_latents = meta_decoder(decoder_input)  # 5× (B, component_latent_dim)

# Marginal Decoders: No further conditioning
for i, latent_i in enumerate(component_latents):
    recon_i = marginal_decoders[i](latent_i)
```

#### Masking (Binary Mask + Zeroing)

**Critical Feature:** Binary mask mechanism

```python
class MetaEncoder(nn.Module):
    def forward(self, component_images, mask=None):
        """
        Args:
            component_images: List of 5 tensors (B, 1, 64, 32)
            mask: (B, 5) - binary (1=observed, 0=missing)
        """
        component_features = []

        for i, img in enumerate(component_images):
            # Process through shared backbone
            feat = self.shared_backbone(img)  # (B, 1024)

            # Component-specific processing
            feat = self.component_specific[i](feat)  # (B, 192)

            # Apply mask: zero out if not observed
            if mask is not None:
                feat = feat * mask[:, i:i+1]  # Broadcasting

            component_features.append(feat)

        # Combine
        combined = torch.cat(component_features, dim=1)  # (B, 960)
        h = self.combine_layers(combined)  # (B, 384)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

def loss_function(recon_components, target_components, mu, logvar, mask=None):
    """
    Loss with optional component masking.

    Args:
        recon_components: List of 5 reconstructed (B, 1, 64, 32)
        target_components: List of 5 target (B, 1, 64, 32)
        mask: (B, 5) - weight components by mask
    """
    # Reconstruction loss (per component)
    recon_losses = []
    for i, (recon, target) in enumerate(zip(recon_components, target_components)):
        comp_loss = F.mse_loss(recon, target, reduction='none').sum(dim=[1,2,3])  # (B,)

        if mask is not None:
            comp_loss = comp_loss * mask[:, i]  # Weight by mask

        recon_losses.append(comp_loss)

    recon_loss = torch.stack(recon_losses).sum(dim=0).mean()

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    return recon_loss + kl_loss, recon_loss, kl_loss
```

#### Inpainting Mode

```python
@torch.no_grad()
def inpaint(model, partial_components, mask, num_samples=10):
    """
    Inpaint missing components.

    Args:
        partial_components: List of 5 tensors (some are zeros)
        mask: (B, 5) - 1=observed, 0=missing
        num_samples: Number of posterior samples

    Returns:
        Inpainted components (averaged over samples)
    """
    B = partial_components[0].shape[0]

    # Encode partial observation with mask
    mu, logvar = model.meta_encoder(partial_components, mask=mask)

    # Sample multiple times from posterior
    all_samples = []
    for _ in range(num_samples):
        z_meta = model.reparameterize(mu, logvar)
        component_latents = model.meta_decoder(z_meta, cond=None)

        # Decode each component
        recons = [model.marginal_decoders[i](latent)
                 for i, latent in enumerate(component_latents)]
        all_samples.append(torch.stack(recons))  # (5, B, 1, 64, 32)

    # Average over samples
    avg_recons = torch.stack(all_samples).mean(dim=0)  # (5, B, 1, 64, 32)

    # Replace observed components with originals
    final_recons = []
    for i in range(5):
        if mask[:, i].all():
            final_recons.append(partial_components[i])
        else:
            final_recons.append(avg_recons[i])

    return final_recons
```

---

### 3.2 Fichiers à Créer

```
src_new/models/meta_vae/
├── __init__.py
├── model.py               # Main MetaVAE class
├── encoder.py            # MetaEncoder (hierarchical)
├── decoder.py            # MetaDecoder (parallel branches)
├── marginal_vae.py       # Pre-trained marginal VAE
├── train.py              # Training script (2-stage)
├── sample.py             # Sampling script (unconditional + inpainting)
└── utils.py              # Masking utilities
```

#### Training Strategy (2-Stage)

**Stage 1: Train Marginal VAEs** (Freeze later)
```python
# Train 5 separate VAEs (one per component)
for i, component in enumerate(components):
    marginal_vae = MarginalVAE(
        latent_dim=4,
        image_size=(64, 32),
        channels=1
    )
    # Train on component_i data
    train_marginal_vae(marginal_vae, dataloader_i, epochs=50)
    # Save
    torch.save(marginal_vae.decoder.state_dict(), f"marginal_decoder_{i}.pt")
```

**Stage 2: Train Meta-VAE** (Freeze marginal decoders)
```python
# Load frozen marginal decoders
marginal_decoders = [
    MarginalVAE().decoder for _ in range(5)
]
for i, decoder in enumerate(marginal_decoders):
    decoder.load_state_dict(torch.load(f"marginal_decoder_{i}.pt"))
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False

# Train Meta-VAE
meta_vae = MetaVAE(
    marginal_decoders=marginal_decoders,
    latent_dim_meta=32,
    component_latent_dim=4
)

# Only meta-encoder and meta-decoder are trained
optimizer = optim.Adam([
    *meta_vae.meta_encoder.parameters(),
    *meta_vae.meta_decoder.parameters()
], lr=1e-4)
```

---

### 3.3 Configuration Files

**`configs/meta_vae_epure_full.yaml`:**
```yaml
model:
  type: meta_vae
  image_size: [64, 32]
  channels: 1  # per component
  cond_dim: 2
  latent_dim_meta: 32
  component_latent_dim: 4
  beta: 1.0
  marginal_vae_paths: [
    "pretrained/marginal_vae_group_nc.pt",
    "pretrained/marginal_vae_group_km.pt",
    "pretrained/marginal_vae_bt.pt",
    "pretrained/marginal_vae_fpu.pt",
    "pretrained/marginal_vae_tpc.pt"
  ]

data:
  root_dir: "data/epure"
  condition_csv: "data/epure/performances.csv"
  component_dirs: ["group_nc", "group_km", "bt", "fpu", "tpc"]
  condition_columns: ["d_cons_norm", "d_rigid_norm", "d_life_norm", "d_stab_norm"]
  prefix_column: "matching"
  filename_pattern: "{prefix}_{component}.png"
  split_column: "train"
  normalized: true

training:
  # Stage 1: Marginal VAEs (optional if already trained)
  stage1:
    epochs: 50
    batch_size: 64
    lr: 0.001

  # Stage 2: Meta-VAE
  stage2:
    epochs: 200
    batch_size: 16
    lr: 0.0001
    scheduler: ReduceLROnPlateau
    patience: 10
    factor: 0.5
    max_mask_prob: 0.5  # Progressive masking
    warmup_epochs: 50

  seed: 0

paths:
  output_dir: "outputs/meta_vae/"
  samples_dir: "samples/meta_vae/"
  pretrained_dir: "pretrained/"

sampling:
  num_samples: 100
  batch_sz: 16
  mode: conditional  # or unconditional, inpainting
  inpainting:
    num_posterior_samples: 10
```

---

## Partie 4: Homogénéisation et Comparabilité

### 4.1 Nombre de Paramètres - Vérification

**Objectif:** Assurer comparabilité entre tous les modèles

**Tableau de référence:**

| Modèle | Paramètres (M) | Source | Notes |
|--------|----------------|--------|-------|
| VQVAE | ~3.5 | EpureDGM | 1.2M VAE + 2.3M Prior |
| MM-VAE+ | ~7.0 | EpureDGM | 5× 1.4M per VAE |
| Simple VAE | ~9.77 | ICTAI | Suivre ICTAI exactement |
| GMRF MVAE | ~4.5-5.5 | ICTAI | 5× VAEs + OffDiagonalCov |
| Meta VAE | ~4-4.5 | ICTAI | 3.7-4M trainable + 0.6M frozen |
| DDPM | ~? | EpureDGM | À vérifier |
| MDM | ~? | EpureDGM | À vérifier |
| Flow Matching | ~? | EpureDGM | À vérifier |
| WGAN-GP | ~? | EpureDGM | À vérifier |

**Stratégie:**
- **Simple VAE:** Respecter config ICTAI (nf=32, latent_dim=32) → ~9.77M
- **GMRF MVAE:** Respecter config ICTAI (nf=64, latent_dim=16) → ~5M
- **Meta VAE:** Respecter config ICTAI (architecture équilibrée) → ~4M

**Script de vérification:**
```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Check all models
models = [vae, gmrf_mvae, meta_vae, vqvae, mmvaeplus]
for model in models:
    total, trainable = count_parameters(model)
    print(f"{model.__class__.__name__}: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
```

---

### 4.2 Scripts train.py et sample.py - Structure Unifiée

**Pattern général pour tous les modèles:**

```
models/{model_name}/
├── train.py     # Unified training script
└── sample.py    # Unified sampling script (3 modes)
```

#### train.py Structure

```python
#!/usr/bin/env python3
"""
Training script for {MODEL_NAME}.

Usage:
    python -m models.{model_name}.train --config configs/{model_name}_epure_full.yaml
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from models.{model_name}.model import {ModelClass}
from datasets.continuous import MultiComponentDataset
from utils.config import load_config

def train(config):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['training']['seed'])

    # Data
    train_dataset = MultiComponentDataset(config, split='train')
    train_loader = DataLoader(train_dataset, **config['training'])

    # Model
    model = {ModelClass}(**config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0

        for batch_idx, (x, cond) in enumerate(train_loader):
            x, cond = x.to(device), cond.to(device)

            # Forward
            loss, metrics = model.compute_loss(x, cond, epoch=epoch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Logging
        print(f"Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}")

        # Checkpointing
        if (epoch + 1) % config['training']['check_every'] == 0:
            save_checkpoint(model, optimizer, epoch, config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)

if __name__ == '__main__':
    main()
```

#### sample.py Structure (3 Modes)

```python
#!/usr/bin/env python3
"""
Sampling script for {MODEL_NAME}.

Supports 3 modes:
  1. Unconditional: Sample from prior
  2. Conditional: Sample given conditions
  3. Inpainting/Cross-modal: Generate missing components

Usage:
    # Unconditional
    python -m models.{model_name}.sample --checkpoint path/to/ckpt.pt --mode unconditional --num_samples 100

    # Conditional
    python -m models.{model_name}.sample --checkpoint path/to/ckpt.pt --mode conditional --conditions_csv data.csv

    # Inpainting (VAE, Meta VAE)
    python -m models.{model_name}.sample --checkpoint path/to/ckpt.pt --mode inpainting --mask_component 0

    # Cross-modal (GMRF MVAE)
    python -m models.{model_name}.sample --checkpoint path/to/ckpt.pt --mode cross_modal --source_comp 0 --target_comp 1
"""

import argparse
import torch
from pathlib import Path

from models.{model_name}.model import {ModelClass}
from utils.io import save_components

@torch.no_grad()
def sample_unconditional(model, num_samples, device):
    """Mode 1: Unconditional generation."""
    samples = model.sample(num_samples=num_samples, cond=None, device=device)
    return samples

@torch.no_grad()
def sample_conditional(model, conditions, device):
    """Mode 2: Conditional generation."""
    samples = model.sample(num_samples=len(conditions), cond=conditions, device=device)
    return samples

@torch.no_grad()
def sample_inpainting(model, partial_x, mask, device):
    """Mode 3a: Inpainting (for VAE, Meta VAE)."""
    samples = model.inpaint(partial_x, mask, device=device)
    return samples

@torch.no_grad()
def sample_cross_modal(model, observed_component, source_idx, target_idx, device):
    """Mode 3b: Cross-modal (for GMRF MVAE, MM-VAE+)."""
    samples = model.cross_modal_generate(observed_component, source_idx, target_idx, device=device)
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True,
                       choices=['unconditional', 'conditional', 'inpainting', 'cross_modal'])
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='samples/')

    # Mode-specific args
    parser.add_argument('--conditions_csv', type=str)
    parser.add_argument('--mask_component', type=int)
    parser.add_argument('--source_comp', type=int)
    parser.add_argument('--target_comp', type=int)

    args = parser.parse_args()

    # Load model
    config = load_config(args.config)
    model = {ModelClass}(**config['model'])
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Sample based on mode
    if args.mode == 'unconditional':
        samples = sample_unconditional(model, args.num_samples, device)

    elif args.mode == 'conditional':
        conditions = load_conditions(args.conditions_csv)
        samples = sample_conditional(model, conditions, device)

    elif args.mode == 'inpainting':
        # Load partial data with mask
        partial_x, mask = load_partial_data(args)
        samples = sample_inpainting(model, partial_x, mask, device)

    elif args.mode == 'cross_modal':
        # Load observed component
        observed = load_observed_component(args)
        samples = sample_cross_modal(model, observed, args.source_comp, args.target_comp, device)

    # Save
    save_components(samples, args.output_dir, config['data']['component_dirs'])
    print(f"Saved {len(samples)} samples to {args.output_dir}")

if __name__ == '__main__':
    main()
```

---

## Partie 5: Workflow de Mise en Œuvre

### 5.1 Ordre d'Implémentation

**Phase 1: Simple VAE** (1-2 jours)
1. ✅ Créer structure `models/vae/`
2. Implémenter encoder/decoder ResNet avec conditioning
3. Implémenter masking (pattern VQVAE)
4. Créer train.py et sample.py
5. Tester sur TOY dataset (quick test)
6. Configs: `vae_epure_full.yaml`, `vae_toy_test.yaml`

**Phase 2: GMRF MVAE** (2-3 jours)
1. ✅ Créer structure `models/gmrf_mvae/`
2. Porter OffDiagonalCov network
3. Porter assemble_covariance_matrix (critique!)
4. Implémenter conditional_generate (cross-modal)
5. Porter objectives.py (ELBO avec GMRF prior)
6. Créer train.py et sample.py
7. Tester sur TOY dataset
8. Configs: `gmrf_mvae_epure_full.yaml`, `gmrf_mvae_toy_test.yaml`

**Phase 3: Meta VAE** (2-3 jours)
1. ✅ Créer structure `models/meta_vae/`
2. Implémenter MetaEncoder hiérarchique avec masking
3. Implémenter MetaDecoder avec conditioning
4. Training 2-stage: marginal VAEs puis meta-VAE
5. Implémenter inpainting avec posterior sampling
6. Créer train.py et sample.py
7. Tester sur TOY dataset
8. Configs: `meta_vae_epure_full.yaml`, `meta_vae_toy_test.yaml`

**Phase 4: Vérification et Homogénéisation** (1 jour)
1. Vérifier nombre de paramètres (tous modèles)
2. Tester 3 modes de sampling pour chaque modèle
3. Vérifier conditioning uniforme
4. Tests end-to-end sur EPURE dataset

---

### 5.2 Fichiers Critiques à Créer

**Simple VAE (7 fichiers):**
1. `models/vae/__init__.py`
2. `models/vae/model.py` (~200 lignes)
3. `models/vae/encoder.py` (~150 lignes)
4. `models/vae/decoder.py` (~150 lignes)
5. `models/vae/train.py` (~200 lignes)
6. `models/vae/sample.py` (~250 lignes)
7. `models/vae/utils.py` (~100 lignes)

**GMRF MVAE (9 fichiers):**
1. `models/gmrf_mvae/__init__.py`
2. `models/gmrf_mvae/model.py` (~300 lignes)
3. `models/gmrf_mvae/cov_model.py` (~50 lignes)
4. `models/gmrf_mvae/encoder_decoder.py` (~200 lignes)
5. `models/gmrf_mvae/base_vae.py` (~150 lignes)
6. `models/gmrf_mvae/objectives.py` (~300 lignes, adapt ICTAI)
7. `models/gmrf_mvae/utils.py` (~250 lignes, assemble_covariance)
8. `models/gmrf_mvae/train.py` (~200 lignes)
9. `models/gmrf_mvae/sample.py` (~300 lignes)

**Meta VAE (8 fichiers):**
1. `models/meta_vae/__init__.py`
2. `models/meta_vae/model.py` (~250 lignes)
3. `models/meta_vae/encoder.py` (~200 lignes)
4. `models/meta_vae/decoder.py` (~150 lignes)
5. `models/meta_vae/marginal_vae.py` (~150 lignes)
6. `models/meta_vae/train.py` (~300 lignes, 2-stage)
7. `models/meta_vae/sample.py` (~300 lignes)
8. `models/meta_vae/utils.py` (~100 lignes)

**Configs (6 fichiers):**
1. `configs/vae_epure_full.yaml`
2. `configs/vae_toy_test.yaml`
3. `configs/gmrf_mvae_epure_full.yaml`
4. `configs/gmrf_mvae_toy_test.yaml`
5. `configs/meta_vae_epure_full.yaml`
6. `configs/meta_vae_toy_test.yaml`

**Total: 30 fichiers nouveaux**

---

## Partie 6: Points Critiques et Vérifications

### 6.1 Conditioning - Vérification

**Pattern uniforme à suivre:**
```python
# Tous les modèles VAE
cond_mlp = Sequential(
    Linear(2, 128),     # cond_dim → 128
    GELU(),
    Linear(128, 256)    # 128 → 256 (feature dim)
)

# Application au decoder
cond_emb = cond_mlp(cond) if cond is not None else torch.zeros(B, 256)
decoder_input = torch.cat([latent, cond_emb], dim=1)
```

**Vérifier:**
- ✅ Simple VAE: Decoder only
- ✅ GMRF MVAE: Decoder only (per component)
- ✅ Meta VAE: Meta-Decoder only
- ✅ VQVAE: Decoder only (déjà implémenté)
- ✅ MM-VAE+: Encoder + Decoder (déjà implémenté)

---

### 6.2 Masking - Homogénéisation

**Simple VAE et Meta VAE:** Suivre pattern VQVAE

```python
# VQVAE pattern (reference)
def mask_components(x, p=0.5):
    """Keep ONE random component, zero others."""
    masked_x = torch.zeros_like(x)
    for i in range(B):
        if random.random() < p:
            keep_idx = random.randint(0, num_components-1)
            masked_x[i, keep_idx] = x[i, keep_idx]
        else:
            masked_x[i] = x[i]
    return masked_x

# Training with progressive schedule
mask_prob = (epoch / warmup_epochs) ** 0.5 * max_mask_prob
```

**GMRF MVAE:** Pas de masking direct (covariance-based dependencies)

**Meta VAE:** Binary mask mechanism
```python
# Zero features for missing components
if mask is not None:
    feat = feat * mask[:, i:i+1]  # (B, feat_dim) * (B, 1)
```

---

### 6.3 Sampling Modes - Vérification

**Tableau de compatibilité:**

| Modèle | Unconditional | Conditional | Inpainting | Cross-Modal |
|--------|--------------|-------------|------------|-------------|
| Simple VAE | ✅ | ✅ | ✅ | ❌ |
| GMRF MVAE | ✅ | ✅ | ❌ | ✅ |
| Meta VAE | ✅ | ✅ | ✅ | ❌ |
| VQVAE | ✅ | ✅ | ✅ | ❌ |
| MM-VAE+ | ✅ | ✅ | ❌ | ✅ |

**Définitions:**
- **Unconditional:** Sample z ~ p(z), decode
- **Conditional:** Sample z ~ p(z), decode with cond
- **Inpainting:** Encode partial → sample posterior → decode (VAE, Meta VAE, VQVAE)
- **Cross-Modal:** Encode component_i → generate component_j (GMRF MVAE, MM-VAE+)

---

### 6.4 Tests de Non-Régression

**Après chaque implémentation:**

```python
# 1. Test parameter count
total, trainable = count_parameters(model)
assert total > 0, "Model has no parameters"
print(f"Parameters: {total/1e6:.2f}M")

# 2. Test forward pass
x = torch.randn(4, 5, 64, 32)  # Batch size 4
cond = torch.randn(4, 2)
output = model(x, cond)
assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"

# 3. Test loss computation
loss, recon_loss, kl_loss = model.loss_function(output, x, mu, logvar)
assert loss.requires_grad, "Loss does not require grad"
assert not torch.isnan(loss), "Loss is NaN"

# 4. Test sampling (unconditional)
samples = model.sample(num_samples=10, device='cpu')
assert samples.shape[0] == 10, f"Wrong number of samples: {samples.shape[0]}"

# 5. Test sampling (conditional)
cond = torch.randn(10, 2)
samples = model.sample(num_samples=10, cond=cond, device='cpu')
assert samples.shape == (10, 5, 64, 32), f"Wrong shape: {samples.shape}"
```

---

## Temps Estimé

- **Simple VAE:** 1-2 jours
  - Architecture: 4h
  - Training/Sampling: 3h
  - Tests: 2h

- **GMRF MVAE:** 2-3 jours
  - Architecture + OffDiagonalCov: 6h
  - Covariance assembly (critique!): 4h
  - Conditional generation: 3h
  - Training/Sampling: 4h
  - Tests: 3h

- **Meta VAE:** 2-3 jours
  - Architecture hiérarchique: 5h
  - 2-stage training: 4h
  - Inpainting avec masking: 3h
  - Training/Sampling: 4h
  - Tests: 3h

- **Vérification et homogénéisation:** 1 jour
  - Parameter checks: 2h
  - Sampling modes: 3h
  - Conditioning uniforme: 2h
  - Tests end-to-end: 3h

**Total: 6-9 jours**

---

## Résumé Final

**Livrable:**
- 3 nouveaux modèles VAE complets
- 30 fichiers Python (models + configs)
- Conditioning uniforme (MLP 2→128→256)
- Masking homogène (pattern VQVAE pour VAE/Meta VAE)
- 3 modes de sampling pour chaque modèle
- Comparabilité assurée (paramètres respectent ICTAI)
- Tests complets sur TOY et EPURE

**Prêt pour évaluation complète avec l'infrastructure déjà créée.**
