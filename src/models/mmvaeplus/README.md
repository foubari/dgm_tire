# MMVAE+ Model

Multimodal Variational Autoencoder Plus (MMVAE+) implementation for the Epure experiment.

## Overview

MMVAE+ is a multimodal generative model that learns shared and modality-specific latent representations across multiple components. This implementation supports:

- **5 components**: `['group_nc', 'group_km', 'bt', 'fpu', 'tpc']`
- **Image dimensions**: 64x32 (grayscale)
- **Conditioning**: width_px, height_px (2D scalar conditioning)
- **Unconditional generation**: Sample from prior
- **Conditional inpainting**: Generate all components conditioned on observed components

## Architecture

- **Encoder**: ResNet-based encoder that extracts private (w) and shared (z) latents
- **Decoder**: ResNet-based decoder that reconstructs images from combined latents (u = w + z)
- **5 VAEs**: One unimodal VAE per component
- **Conditioning**: MLP-based conditioning in both encoder and decoder

## Training

```bash
python src_new/scripts/train.py --model mmvaeplus --config src_new/configs/mmvaeplus_default.yaml --epochs 250 --batch_size 128
```

Or directly:
```bash
python src_new/models/mmvaeplus/train.py --config src_new/configs/mmvaeplus_default.yaml --epochs 250 --batch_size 128
```

## Sampling

### Unconditional Generation

```bash
python src_new/scripts/sample.py --model mmvaeplus --checkpoint outputs/mmvaeplus/2025-12-18_XX-XX-XX --config src_new/configs/mmvaeplus_default.yaml --mode unconditional --num_samples 100
```

### Conditional Inpainting

Generate all 5 components conditioned on some observed components:

```bash
python src_new/scripts/sample.py --model mmvaeplus --checkpoint outputs/mmvaeplus/2025-12-18_XX-XX-XX --config src_new/configs/mmvaeplus_default.yaml --mode inpainting --components group_nc bt
```

This will generate all 5 components conditioned on `group_nc` and `bt`, saving results in:
```
samples/mmvaeplus/2025-12-18_XX-XX-XX/
  group_nc/
    group_nc/  (self-reconstruction)
    group_km/  (cross-modal)
    bt/        (cross-modal)
    fpu/       (cross-modal)
    tpc/       (cross-modal)
  bt/
    group_nc/  (cross-modal)
    group_km/  (cross-modal)
    bt/        (self-reconstruction)
    ...
```

## Configuration

See `src_new/configs/mmvaeplus_default.yaml` for configuration options:

- **Model parameters**:
  - `latent_dim_w`: Private latent dimension (default: 16)
  - `latent_dim_z`: Shared latent dimension (default: 16)
  - `latent_dim_u`: Combined latent dimension (must equal w + z, default: 32)
  - `nf`: Number of filters in ResNet (default: 40)
  - `nf_max`: Maximum filters (default: 128)
  - `cond_dim`: Conditioning dimension (default: 2)
  - `priorposterior`: Distribution type ('Normal' or 'Laplace', default: 'Laplace')

- **Training parameters**:
  - `objective`: 'elbo' or 'dreg' (default: 'elbo')
  - `beta`: KL weight (default: 2.5)
  - `K`: Number of samples for reparameterization (default: 1)

## Dataset Format

The model expects data organized as:
```
data/
  train/
    group_nc/
      matching_XXX_group_nc.png
      ...
    group_km/
      matching_XXX_group_km.png
      ...
    ...
  test/
    ...
```

With a CSV file (`dimensions.csv`) containing:
- `matching`: Prefix/ID column
- `width_px`, `height_px`: Conditioning columns
- `train`: Boolean split column

## Key Features

1. **Cross-Modal Generation**: Generate any component conditioned on any other component(s)
2. **Shared Latent Space**: Learns shared representations across modalities
3. **Private Latent Space**: Captures modality-specific information
4. **Conditioning Support**: Incorporates scalar conditioning (width_px, height_px) in both encoder and decoder

## Notes

- The model uses `MultiComponentDataset` with `stacked=False` to return tuples instead of concatenated tensors
- Conditioning is replicated for all 5 modalities during training
- For inpainting, shared latents are extracted from observed modalities and private latents are resampled

