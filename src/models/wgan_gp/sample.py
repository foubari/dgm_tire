#!/usr/bin/env python3
"""
Sampling script for WGAN-GP - Modular version.

Supports two modes:
1. Unconditional: Generate samples without conditions
2. Conditional: Generate conditionally using test loader with guidance scale

NOTE: Inpainting is not yet implemented for WGAN-GP.

Usage:
    python sample.py --checkpoint outputs/.../checkpoint_100.pt --config configs/wgan_gp_default.yaml --mode unconditional
"""

import argparse
import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src_new to path
_THIS_FILE = Path(__file__).resolve()
_SRC_NEW_DIR = _THIS_FILE.parent.parent.parent
_PROJECT_ROOT = _SRC_NEW_DIR.parent

if str(_SRC_NEW_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_NEW_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.continuous import MultiComponentDataset
from models.wgan_gp import WGANGP, Encoder, Generator, Critic
from utils.config import load_config, auto_complete_config, validate_config, resolve_path
from utils.io import save_component_images


def load_model_from_checkpoint(checkpoint_path, config_path=None, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        config_path: Optional path to config YAML
        device: Device to load model on
    
    Returns:
        model, config, date_string
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Find config.yaml or args.pickle
    if checkpoint_path.is_file():
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = checkpoint_path
    
    # Try to load config
    config = None
    if config_path:
        config = load_config(config_path)
    else:
        # Try config.yaml in checkpoint directory
        config_yaml = checkpoint_dir / 'config.yaml'
        if config_yaml.exists():
            config = load_config(config_yaml)
        else:
            # Fallback to args.pickle
            args_path = checkpoint_dir / 'args.pickle'
            if not args_path.exists():
                args_path = checkpoint_dir.parent / 'args.pickle'
            
            if args_path.exists():
                with open(args_path, 'rb') as f:
                    args = pickle.load(f)
                # Convert args to config-like dict
                config = {
                    'model': {
                        'image_size': getattr(args, 'image_size', [64, 32]),
                        'dim': getattr(args, 'dim', 64),
                        'dim_mults': getattr(args, 'dim_mults', (1, 2, 4)),
                        'channels': getattr(args, 'channels', 5),
                        'cond_dim': getattr(args, 'cond_dim', 2),
                        'total_latent_dim': getattr(args, 'total_latent_dim', 20),
                        'lambda_gp': getattr(args, 'lambda_gp', 10),
                        'n_critic': getattr(args, 'n_critic', 5),
                        'cond_drop_prob': getattr(args, 'cond_drop_prob', 0.1),
                    },
                    'data': {
                        'component_dirs': getattr(args, 'component_dirs', ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']),
                        'condition_columns': getattr(args, 'condition_columns', ['width_px', 'height_px']),
                        'normalized': getattr(args, 'normalized', False),
                    }
                }
    
    if config is None:
        raise FileNotFoundError(
            f"Could not find config.yaml or args.pickle near {checkpoint_path}. "
            f"Please provide --config argument."
        )
    
    # Extract date from checkpoint directory path
    # If checkpoint is in check/ subdirectory, use parent directory name (timestamp)
    # Otherwise use checkpoint_dir name directly
    date_str = checkpoint_dir.parent.name if checkpoint_dir.name == 'check' else checkpoint_dir.name
    
    # Auto-complete config
    config = auto_complete_config(config)
    
    # Create models
    model_cfg = config['model']
    image_size = tuple(model_cfg['image_size'])
    channels = model_cfg['channels']
    cond_dim = model_cfg['cond_dim']
    latent_dim = model_cfg.get('total_latent_dim', 20)
    
    encoder = Encoder(in_channels=channels, latent_dim=latent_dim)
    generator = Generator(
        latent_dim=latent_dim,
        out_channels=channels,
        cond_dim=cond_dim,
        dim=model_cfg.get('dim', 64),
        dim_mults=tuple(model_cfg.get('dim_mults', [1, 2, 4]))
    )
    critic = Critic(in_channels=channels, cond_dim=cond_dim)
    
    # Create WGAN-GP wrapper
    model = WGANGP(
        encoder=encoder,
        generator=generator,
        critic=critic,
        image_size=image_size,
        latent_dim=latent_dim,
        lambda_gp=model_cfg.get('lambda_gp', 10),
        n_critic=model_cfg.get('n_critic', 5),
        cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1)
    )
    
    # Load checkpoint
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        check_dir = checkpoint_dir / 'check'
        if check_dir.exists():
            checkpoints = list(check_dir.glob('checkpoint_*.pt'))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
                checkpoint = torch.load(checkpoint_path, map_location=device)
            else:
                raise FileNotFoundError(f"No checkpoint found in {check_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")
    
    # Load state dicts (prefer EMA generator)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    if checkpoint.get('ema_generator_state_dict') is not None:
        generator.load_state_dict(checkpoint['ema_generator_state_dict'])
        print("Loaded EMA generator")
    else:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print("Loaded regular generator (no EMA found)")
    
    critic.load_state_dict(checkpoint['critic_state_dict'])
    
    # Update model with loaded components
    model.encoder = encoder.to(device)
    model.generator = generator.to(device)
    model.critic = critic.to(device)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Channels: {channels}")
    print(f"  Image size: {image_size}")
    print(f"  Latent dim: {latent_dim}")
    
    return model, config, date_str


def sample_unconditional(model, save_root, date_str, component_names, n_samples=1000, batch_sz=64, device='cuda'):
    """
    Unconditional sampling - save per component and full images.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
    
    model.to(device).eval()
    n_done = 0
    
    print(f"\nUnconditional sampling for model dated {date_str}")
    
    with torch.no_grad():
        while n_done < n_samples:
            b = min(batch_sz, n_samples - n_done)
            samples = model.sample(batch_size=b, cond=None, guidance_scale=0.0)  # (B, C, H, W) in [0, 1]
            
            # Ensure samples are in [0, 1] range
            samples = torch.clamp(samples, 0.0, 1.0)
            
            for n in range(b):
                prefix = f"img{n_done + n:04d}"
                save_component_images(samples[n:n+1], out_root, prefix, component_names)
            
            n_done += b
            print(f"  Generated {n_done}/{n_samples} samples", end='\r')
    
    print(f"\nDone – {n_done} samples saved under {out_root}")


def sample_conditional(model, test_loader, save_root, date_str, component_names, guidance_scale=2.0, device='cuda'):
    """
    Conditional sampling - generate conditioned on test set conditions.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
    
    model.to(device).eval()
    
    print(f"\nConditional sampling for model dated {date_str}")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Processing {len(test_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Sampling")):
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                images, cond = batch_data
                images = images.to(device)
                cond = cond.to(device).float() if cond is not None else None
            else:
                images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
                cond = None
            
            batch_size = images.size(0)
            
            # Generate samples
            samples = model.sample(
                batch_size=batch_size,
                cond=cond,
                guidance_scale=guidance_scale
            )  # (B, C, H, W) in [0, 1]
            
            # Ensure samples are in [0, 1] range
            samples = torch.clamp(samples, 0.0, 1.0)
            
            # Save samples
            for n in range(batch_size):
                prefix = f"img{batch_idx:04d}_{n:02d}"
                save_component_images(samples[n:n+1], out_root, prefix, component_names)
    
    print(f"\nDone – all samples saved under {out_root}")


def main():
    parser = argparse.ArgumentParser(description='Sample from WGAN-GP - Modular version')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file or directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file (optional if config.yaml exists near checkpoint)')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['unconditional', 'conditional'],
                       help='Sampling mode')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for unconditional mode')
    parser.add_argument('--batch_sz', type=int, default=64,
                       help='Batch size for sampling')
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                       help='Guidance scale for conditional mode (CFG)')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check for inpainting mode (not supported)
    if args.mode == 'inpainting':
        raise NotImplementedError(
            "Inpainting is not yet implemented for WGAN-GP. "
            "Please use 'unconditional' or 'conditional' sampling modes."
        )
    
    # Load model
    print("Loading models...")
    model, config, date_str = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    
    # Get component names from config
    component_names = config['data']['component_dirs']
    
    # Setup save directory
    save_root = resolve_path(config.get('paths', {}).get('samples_dir', 'samples/wgan_gp'))
    save_root.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'unconditional':
        save_root = save_root / 'unconditional'
        sample_unconditional(
            model, str(save_root), date_str, component_names,
            n_samples=args.num_samples, batch_sz=args.batch_sz, device=args.device
        )
    
    elif args.mode == 'conditional':
        save_root = save_root / 'conditional'
        
        # Create test loader
        data_cfg = config['data']
        root_dir = resolve_path(data_cfg['root_dir'])
        condition_csv = resolve_path(data_cfg['condition_csv'])
        
        # Add split to root_dir (data structure: root_dir/train/comp_dir and root_dir/test/comp_dir)
        test_root_dir = root_dir / 'test'
        
        test_dataset = MultiComponentDataset(
            root_dir=test_root_dir,
            condition_csv=condition_csv,
            component_dirs=data_cfg['component_dirs'],
            condition_columns=data_cfg['condition_columns'],
            prefix_column=data_cfg['prefix_column'],
            filename_pattern=data_cfg.get('filename_pattern', '{prefix}_{component}.png'),
            split='test',
            split_column=data_cfg.get('split_column', 'train'),
            normalized=data_cfg.get('normalized', False)
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        sample_conditional(
            model, test_loader, str(save_root), date_str, component_names,
            guidance_scale=args.guidance_scale, device=args.device
        )


if __name__ == '__main__':
    main()

