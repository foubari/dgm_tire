#!/usr/bin/env python3
"""
Compare parameter counts across model implementations.

Usage:
    python scripts/compare_model_params.py

This script instantiates all models with their test configs and reports parameter counts.
"""

import sys
import yaml
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.gmrf_mvae.model import GMRF_MVAE
from models.vae.model import BetaVAE
from models.ddpm.diffusion import GaussianDiffusion
# Note: Other models omitted for simplicity - add as needed


def count_parameters(model, only_trainable=True):
    """Count parameters in a model."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def load_config(config_path):
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def instantiate_gmrf_mvae(config_path):
    """Instantiate GMRF-MVAE from config."""
    cfg = load_config(config_path)
    model_cfg = cfg['model']

    model = GMRF_MVAE(
        num_components=model_cfg['num_components'],
        latent_dim=model_cfg['latent_dim'],
        nf=model_cfg['nf'],
        nf_max=model_cfg['nf_max'],
        hidden_dim=model_cfg['hidden_dim'],
        n_layers=model_cfg['n_layers'],
        beta=model_cfg['beta'],
        cond_dim=model_cfg['cond_dim'],
        dropout_p=model_cfg.get('dropout_p', 0.1)
    )

    return model, cfg


def instantiate_vae(config_path):
    """Instantiate VAE from config."""
    cfg = load_config(config_path)
    model_cfg = cfg['model']

    model = BetaVAE(
        image_size=tuple(model_cfg['image_size']),
        channels=model_cfg['channels'],
        latent_dim=model_cfg['latent_dim'],
        nf=model_cfg['nf'],
        nf_max=model_cfg['nf_max'],
        beta=model_cfg['beta'],
        cond_dim=model_cfg.get('cond_dim', 0),
        dropout_p=model_cfg.get('dropout_p', 0.1)
    )

    return model, cfg


def instantiate_ddpm(config_path):
    """Instantiate DDPM from config."""
    cfg = load_config(config_path)
    model_cfg = cfg['model']

    model = GaussianDiffusion(
        image_size=tuple(model_cfg['image_size']),
        channels=model_cfg['channels'],
        dim=model_cfg['dim'],
        dim_mults=tuple(model_cfg['dim_mults']),
        timesteps=model_cfg['timesteps'],
        objective=model_cfg['objective'],
        beta_schedule=model_cfg['beta_schedule'],
        cond_dim=model_cfg.get('cond_dim', 0),
        cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1)
    )

    return model, cfg


def main():
    """Main comparison function."""
    print("=" * 80)
    print(" MODEL PARAMETER COUNT COMPARISON")
    print("=" * 80)
    print()

    # Define models to compare
    models = {
        'GMRF-MVAE (Test)': ('src/configs/pipeline/test/gmrf_mvae_pipeline_test.yaml', instantiate_gmrf_mvae),
        'GMRF-MVAE (Full)': ('src/configs/gmrf_mvae_epure_full.yaml', instantiate_gmrf_mvae),
        'VAE (Test)': ('src/configs/pipeline/test/vae_pipeline_test.yaml', instantiate_vae),
        'DDPM (Test)': ('src/configs/pipeline/test/ddpm_pipeline_test.yaml', instantiate_ddpm),
    }

    results = []

    for name, (config_path, instantiate_fn) in models.items():
        config_path = Path(config_path)

        if not config_path.exists():
            print(f"[SKIP] {name}: Config not found at {config_path}")
            continue

        try:
            model, cfg = instantiate_fn(config_path)
            param_count = count_parameters(model)
            param_count_total = count_parameters(model, only_trainable=False)

            results.append((name, param_count, param_count_total, cfg))

            print(f"[OK] {name}")
            print(f"     Config: {config_path}")
            print(f"     Trainable parameters: {param_count:,}")
            print(f"     Total parameters: {param_count_total:,}")

            # Print key architecture params for GMRF-MVAE
            if 'GMRF-MVAE' in name:
                m_cfg = cfg['model']
                print(f"     Architecture: latent_dim={m_cfg['latent_dim']}, "
                      f"nf={m_cfg['nf']}, nf_max={m_cfg['nf_max']}, "
                      f"hidden_dim={m_cfg['hidden_dim']}, beta={m_cfg['beta']}")

            print()

        except Exception as e:
            print(f"[ERROR] {name}: {str(e)}")
            print()

    # Summary table
    print("=" * 80)
    print(" SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<25} {'Trainable Params':>20} {'Total Params':>20}")
    print("-" * 80)

    for name, param_count, param_count_total, _ in results:
        print(f"{name:<25} {param_count:>20,} {param_count_total:>20,}")

    print("=" * 80)

    # GMRF-MVAE comparison
    gmrf_results = [(n, p, pt, c) for n, p, pt, c in results if 'GMRF-MVAE' in n]
    if len(gmrf_results) >= 2:
        print()
        print("GMRF-MVAE PARAMETER REDUCTION:")
        test_params = next(p for n, p, pt, c in gmrf_results if 'Test' in n)
        full_params = next(p for n, p, pt, c in gmrf_results if 'Full' in n)

        # Estimate old params (with old config: latent_dim=32, nf=64, hidden_dim=256, beta=4.0)
        # Approximation: ~34.4M params with old config
        old_estimate = 34_400_000
        reduction_pct = ((old_estimate - test_params) / old_estimate) * 100

        print(f"  Estimated OLD (latent_dim=32, nf=64, hidden_dim=256): ~{old_estimate/1e6:.1f}M params")
        print(f"  NEW (latent_dim=4, nf=32, hidden_dim=128):             {test_params/1e6:.1f}M params")
        print(f"  Reduction: ~{reduction_pct:.1f}% fewer parameters")
        print()


if __name__ == '__main__':
    main()
