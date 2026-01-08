# EpureDGM - Deep Generative Models for Multi-Component Images

Framework for training and evaluating 9 generative models on multi-component image datasets.

## Models

**Diffusion Models**: DDPM, MDM, Flow Matching
**VAE Models**: VAE, GMRF-MVAE, Meta-VAE, VQVAE, MMVAEplus
**GAN Models**: WGAN-GP

> **Note**: MMVAEplus and Meta-VAE training incomplete (multiprocessing issues on Windows, marginal decoders already trained for seed 1)

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Full Training Pipeline (All Models)
```bash
python scripts/pipeline/run_pipeline.py --dataset epure
```

This runs training (100 epochs, 3 seeds), sampling (unconditional/conditional/inpainting), and evaluation (FID, IoU/Dice, CoM, RCE).

**Time**: ~25-35 hours
**Disk**: ~8 GB

### Specific Models Only
```bash
python scripts/pipeline/run_pipeline.py --dataset epure --models ddpm,vae,gmrf_mvae
```

### Custom Seeds
```bash
python scripts/pipeline/run_pipeline.py --dataset epure --seeds 42,123,456
```

### Quick Validation (1 epoch)
```bash
python scripts/pipeline/validate_models_complete.py --dataset test --models all
```

## Manual Training

```bash
# Single model
python src/models/gmrf_mvae/train.py --config src/configs/pipeline/epure/gmrf_mvae_epure_pipeline.yaml --seed 42
```

## Manual Sampling

```bash
# Unconditional
python src/models/ddpm/sample.py --checkpoint outputs/ddpm/run_seed0/check/checkpoint_best.pt --mode unconditional --num_samples 1000

# Conditional
python src/models/vae/sample.py --checkpoint outputs/vae/run_seed0/check/checkpoint_best.pt --mode conditional --num_samples 1000 --seed 42

# Inpainting
python src/models/gmrf_mvae/sample.py --checkpoint outputs/gmrf_mvae/run_seed0/check/checkpoint_best.pt --mode inpainting --components fpu --num_samples 1000 --seed 42
```

## Evaluation

```bash
python src/scripts/evaluate.py --model gmrf_mvae --dataset epure --seeds 0,1,2
```

## Monitoring GPU

```powershell
powershell .\scripts\monitor_gpu.ps1
```

## Project Structure

```
epure_dgm_new/
├── src/
│   ├── models/          # 9 model implementations
│   ├── evaluation/      # Metrics (FID, IoU/Dice, CoM, RCE)
│   ├── configs/         # YAML configs
│   └── scripts/         # Train/sample scripts
├── scripts/
│   └── pipeline/        # Automation scripts
├── data/
│   ├── epure/           # EPURE dataset (private)
│   └── toy_epure/       # TOY dataset (public)
└── outputs/             # Checkpoints & samples
```

## Key Features

- **GMRF-MVAE**: ICTAI-aligned (latent_dim=4, nf=32, beta=1.0, ~8-10M params)
- **3 Sampling Modes**: Unconditional, conditional, inpainting
- **4 Metrics**: FID, IoU/Dice, CoM, RCE
- **Multi-seed Support**: Statistical robustness

## Notes

- **EPURE dataset**: Private, not included in public repo
- **TOY dataset**: Included for testing
- Models save checkpoints every 25 epochs + best checkpoint
- Inpainting not supported by MDM and WGAN-GP
