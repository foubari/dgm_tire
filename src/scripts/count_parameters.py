#!/usr/bin/env python3
"""
Count parameters for all models.

Usage:
    python count_parameters.py --dataset epure
    python count_parameters.py --dataset toy
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(n):
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_ddpm_params(dataset: str):
    """Get DDPM model parameters."""
    from models.ddpm import GaussianDiffusion, Unet

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'ddpm_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'ddpm_toy.yaml')

    model_cfg = config['model']

    unet = Unet(
        dim=model_cfg['dim'],
        channels=model_cfg['channels'],
        dim_mults=tuple(model_cfg['dim_mults']),
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    model = GaussianDiffusion(
        model=unet,
        image_size=tuple(model_cfg['image_size']),
        timesteps=model_cfg['timesteps'],
        objective=model_cfg.get('objective', 'pred_v'),
        beta_schedule=model_cfg.get('beta_schedule', 'sigmoid'),
        cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1)
    )

    return count_parameters(model)


def get_flow_matching_params(dataset: str):
    """Get Flow Matching model parameters."""
    from models.flow_matching import FlowMatching, FlowMatchingUnet

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'flow_matching_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'flow_matching_toy.yaml')

    model_cfg = config['model']

    unet = FlowMatchingUnet(
        dim=model_cfg['dim'],
        channels=model_cfg['channels'],
        dim_mults=tuple(model_cfg['dim_mults']),
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    model = FlowMatching(
        model=unet,
        image_size=tuple(model_cfg['image_size']),
        cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1)
    )

    return count_parameters(model)


def get_mdm_params(dataset: str):
    """Get MDM model parameters."""
    from models.mdm import MultinomialDiffusion, SegmentationUnet

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'mdm_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'mdm_toy.yaml')

    model_cfg = config['model']

    unet = SegmentationUnet(
        num_classes=model_cfg['num_classes'],
        dim=model_cfg['dim'],
        num_steps=model_cfg['timesteps'],
        dim_mults=tuple(model_cfg['dim_mults']),
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    resolution = tuple(config['data'].get('resolution', [64, 32]))

    model = MultinomialDiffusion(
        num_classes=model_cfg['num_classes'],
        shape=(1, *resolution),
        denoise_fn=unet,
        timesteps=model_cfg['timesteps'],
        cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1)
    )

    return count_parameters(model)


def get_vae_params(dataset: str):
    """Get VAE model parameters."""
    from models.vae import BetaVAE

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'vae_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'vae_toy.yaml')

    model_cfg = config['model']

    model = BetaVAE(
        in_channels=model_cfg['channels'],
        latent_dim=model_cfg['latent_dim'],
        hidden_dims=model_cfg.get('hidden_dims', [32, 64, 128, 256]),
        image_size=tuple(model_cfg['image_size']),
        beta=model_cfg.get('beta', 1.0),
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    return count_parameters(model)


def get_meta_vae_params(dataset: str):
    """Get Meta-VAE model parameters."""
    from models.meta_vae_wrong import MetaVAE

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'meta_vae_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'meta_vae_toy.yaml')

    model_cfg = config['model']

    model = MetaVAE(
        in_channels=model_cfg['channels'],
        latent_dim=model_cfg['latent_dim'],
        hidden_dims=model_cfg.get('hidden_dims', [32, 64, 128, 256]),
        image_size=tuple(model_cfg['image_size']),
        num_components=model_cfg['num_components'],
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    return count_parameters(model)


def get_gmrf_mvae_params(dataset: str):
    """Get GMRF-MVAE model parameters."""
    from models.gmrf_mvae_old import GMRF_MVAE

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'gmrf_mvae_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'gmrf_mvae_toy.yaml')

    model_cfg = config['model']

    model = GMRF_MVAE(
        in_channels=1,  # Per-component
        latent_dim=model_cfg['latent_dim'],
        hidden_dims=model_cfg.get('hidden_dims', [32, 64, 128]),
        image_size=tuple(model_cfg['image_size']),
        num_components=model_cfg['num_components'],
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    return count_parameters(model)


def get_mmvaeplus_params(dataset: str):
    """Get MMVAE+ model parameters."""
    from models.mmvaeplus import MMVAEPlus

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'mmvaeplus_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'mmvaeplus_toy.yaml')

    model_cfg = config['model']

    model = MMVAEPlus(
        in_channels=1,  # Per-component
        latent_dim=model_cfg['latent_dim'],
        hidden_dims=model_cfg.get('hidden_dims', [32, 64, 128]),
        image_size=tuple(model_cfg['image_size']),
        num_components=model_cfg['num_components'],
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    return count_parameters(model)


def get_wgan_gp_params(dataset: str):
    """Get WGAN-GP model parameters."""
    from models.wgan_gp import ConditionalEncoder, ConditionalGenerator, Critic

    if dataset == 'epure':
        config = load_config(_SRC_DIR / 'configs' / 'wgan_gp_default.yaml')
    else:
        config = load_config(_SRC_DIR / 'configs' / 'wgan_gp_toy.yaml')

    model_cfg = config['model']
    image_size = tuple(model_cfg['image_size'])

    encoder = ConditionalEncoder(
        in_channels=model_cfg['channels'],
        latent_dim=model_cfg['latent_dim'],
        image_size=image_size,
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    generator = ConditionalGenerator(
        latent_dim=model_cfg['latent_dim'],
        out_channels=model_cfg['channels'],
        image_size=image_size,
        cond_dim=model_cfg.get('cond_dim', 2)
    )

    critic = Critic(
        in_channels=model_cfg['channels'],
        image_size=image_size
    )

    enc_params = count_parameters(encoder)
    gen_params = count_parameters(generator)
    critic_params = count_parameters(critic)

    total = enc_params[0] + gen_params[0] + critic_params[0]
    trainable = enc_params[1] + gen_params[1] + critic_params[1]

    return total, trainable, {
        'encoder': enc_params,
        'generator': gen_params,
        'critic': critic_params
    }


def main():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('--dataset', type=str, default='epure', choices=['epure', 'toy'],
                       help='Dataset configuration to use')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Model Parameter Counts - {args.dataset.upper()} dataset")
    print(f"{'='*70}\n")

    models = [
        ('DDPM', get_ddpm_params),
        ('Flow Matching', get_flow_matching_params),
        ('MDM', get_mdm_params),
        ('VAE', get_vae_params),
        ('Meta-VAE', get_meta_vae_params),
        ('GMRF-MVAE', get_gmrf_mvae_params),
        ('MMVAE+', get_mmvaeplus_params),
        ('WGAN-GP', get_wgan_gp_params),
    ]

    results = []

    for name, get_params in models:
        try:
            result = get_params(args.dataset)
            if len(result) == 3:  # WGAN-GP returns extra info
                total, trainable, details = result
                results.append((name, total, trainable, details))
            else:
                total, trainable = result
                results.append((name, total, trainable, None))
            print(f"{name:<15} {format_params(total):>10} total  ({format_params(trainable)} trainable)")
            if len(result) == 3:
                for comp_name, (comp_total, comp_train) in details.items():
                    print(f"  - {comp_name:<12} {format_params(comp_total):>10}")
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")
            results.append((name, None, None, None))

    # Print summary table
    print(f"\n{'='*70}")
    print("Summary Table (for LaTeX/Paper)")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Parameters':>15}")
    print("-" * 30)
    for name, total, trainable, _ in results:
        if total:
            print(f"{name:<15} {total:>15,}")

    # Print as CSV
    print(f"\n{'='*70}")
    print("CSV Format")
    print(f"{'='*70}")
    print("model,total_params,trainable_params")
    for name, total, trainable, _ in results:
        if total:
            print(f"{name},{total},{trainable}")


if __name__ == '__main__':
    main()
