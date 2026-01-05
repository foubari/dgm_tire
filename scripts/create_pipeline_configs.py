#!/usr/bin/env python3
"""
Generate pipeline configuration files for EpureDGM training.

This script creates standardized configs with:
- 100 epochs for EPURE and TOY datasets (pipeline configs)
- 1 epoch + 50 samples for test configs
- Optimized batch sizes per model
- Consistent checkpoint and evaluation frequencies

Usage:
    python scripts/create_pipeline_configs.py

Creates:
    - src/configs/pipeline/epure/{model}_epure_pipeline.yaml (9 files)
    - src/configs/pipeline/toy/{model}_toy_pipeline.yaml (9 files)
    - src/configs/pipeline/test/{model}_pipeline_test.yaml (9 files)
"""

import yaml
from pathlib import Path
import sys

# Model list
MODELS = [
    'ddpm', 'mdm', 'flow_matching',
    'vae', 'gmrf_mvae', 'meta_vae',
    'vqvae', 'wgan_gp', 'mmvaeplus'
]

# Optimized batch sizes (based on model complexity and GPU memory)
BATCH_SIZES = {
    'ddpm': 64,
    'mdm': 64,
    'flow_matching': 64,
    'vqvae': 64,
    'wgan_gp': 64,
    'mmvaeplus': 128,
    'vae': 128,
    'gmrf_mvae': 128,
    'meta_vae': 128,
}


def find_base_config(model: str, dataset: str) -> Path:
    """
    Find the base config file for a model/dataset combination.

    Args:
        model: Model name
        dataset: 'epure' or 'toy'

    Returns:
        Path to base config file
    """
    configs_dir = Path("src/configs")

    # VAE models have different naming
    if model in ['vae', 'gmrf_mvae', 'meta_vae']:
        if dataset == "epure":
            base = configs_dir / f"{model}_epure_full.yaml"
        else:
            # For TOY, VAE models use _toy_test.yaml instead of _toy_full.yaml
            base = configs_dir / f"{model}_toy_test.yaml"
    else:
        if dataset == "epure":
            base = configs_dir / f"{model}_default.yaml"
        else:
            base = configs_dir / f"{model}_toy_full.yaml"

    if not base.exists():
        raise FileNotFoundError(f"Base config not found: {base}")

    return base


def create_pipeline_config(model: str, dataset: str, epochs: int, num_samples: int) -> dict:
    """
    Create a pipeline config from base config.

    Args:
        model: Model name
        dataset: 'epure' or 'toy'
        epochs: Number of training epochs
        num_samples: Number of samples for sampling

    Returns:
        Config dictionary
    """
    base_config = find_base_config(model, dataset)

    with open(base_config) as f:
        config = yaml.safe_load(f)

    # Override training parameters
    config['training']['epochs'] = epochs
    config['training']['batch_size'] = BATCH_SIZES[model]

    # Adjust check_every and eval_every based on epochs
    if epochs == 1:
        config['training']['check_every'] = 1  # Save at end for 1 epoch
        config['training']['eval_every'] = 1
    else:
        config['training']['check_every'] = 25  # Save checkpoint every 25 epochs
        config['training']['eval_every'] = 10   # Evaluate every 10 epochs

    # Override sampling parameters
    if 'sampling' not in config:
        config['sampling'] = {}
    config['sampling']['num_samples'] = num_samples

    # Update output paths
    suffix = f"_{dataset}" if dataset == "toy" else ""
    config['paths']['output_dir'] = f"outputs/{model}{suffix}/"
    config['paths']['samples_dir'] = f"samples/{model}{suffix}/"

    # Add missing model parameters if needed
    if model == 'ddpm':
        if 'objective' not in config['model']:
            config['model']['objective'] = 'pred_v'
        if 'beta_schedule' not in config['model']:
            config['model']['beta_schedule'] = 'sigmoid'
    elif model == 'mdm':
        if 'objective' not in config['model']:
            config['model']['objective'] = 'pred_v'
        if 'beta_schedule' not in config['model']:
            config['model']['beta_schedule'] = 'sigmoid'
    elif model == 'flow_matching':
        if 'sigma_min' not in config['model']:
            config['model']['sigma_min'] = 0.001

    return config


def save_config(config: dict, output_path: Path):
    """Save config to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Created: {output_path}")


def main():
    """Generate all pipeline configs."""
    print("=" * 70)
    print("EpureDGM Pipeline Config Generator")
    print("=" * 70)
    print()

    configs_created = 0
    configs_failed = 0

    # Create pipeline configs (100 epochs, 1000 samples)
    print("Creating pipeline configs (100 epochs, 1000 samples)...")
    print()

    for dataset in ['epure', 'toy']:
        print(f"  Dataset: {dataset}")
        for model in MODELS:
            try:
                config = create_pipeline_config(model, dataset, epochs=100, num_samples=1000)
                output = Path(f"src/configs/pipeline/{dataset}/{model}_{dataset}_pipeline.yaml")
                save_config(config, output)
                configs_created += 1
            except Exception as e:
                print(f"✗ Failed: {model} ({dataset}) - {e}")
                configs_failed += 1
        print()

    # Create test configs (1 epoch, 50 samples) - EPURE
    print("Creating EPURE test configs (1 epoch, 50 samples)...")
    print()

    for model in MODELS:
        try:
            config = create_pipeline_config(model, 'epure', epochs=1, num_samples=50)
            output = Path(f"src/configs/pipeline/test/{model}_pipeline_test.yaml")
            save_config(config, output)
            configs_created += 1
        except Exception as e:
            print(f"✗ Failed: {model} (test epure) - {e}")
            configs_failed += 1

    print()

    # Create TOY test configs (1 epoch, 50 samples)
    print("Creating TOY test configs (1 epoch, 50 samples)...")
    print()

    test_toy_dir = Path("src/configs/pipeline/test_toy")
    test_toy_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        try:
            config = create_pipeline_config(model, 'toy', epochs=1, num_samples=50)
            output = test_toy_dir / f"{model}_pipeline_test.yaml"
            save_config(config, output)
            configs_created += 1
        except Exception as e:
            print(f"✗ Failed: {model} (test toy) - {e}")
            configs_failed += 1

    print()
    print("=" * 70)
    print(f"Summary: {configs_created} configs created, {configs_failed} failed")
    print("=" * 70)

    if configs_failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
