#!/usr/bin/env python3
"""
Sampling script for VQ-VAE with PixelCNN Prior - Modular version.

Supports three modes:
1. Unconditional: Generate samples without conditions
2. Conditional: Generate conditionally using test loader
3. Inpainting: Generate missing components while preserving known ones

Usage:
    python sample.py --checkpoint outputs/.../checkpoint_100.pt --config configs/vqvae_default.yaml --mode unconditional
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
from models.vqvae import VQVAE, PixelCNNPrior
from utils.config import load_config, auto_complete_config, validate_config, resolve_path
from utils.io import save_component_images


def load_model_from_checkpoint(checkpoint_path, config_path=None, device='cuda'):
    """
    Load VQ-VAE and Prior from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        config_path: Optional path to config YAML
        device: Device to load model on
    
    Returns:
        vqvae, prior, config, date_string
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
                        'latent_dim': getattr(args, 'latent_dim', 20),
                        'num_embeddings': getattr(args, 'num_embeddings', 512),
                        'commitment_cost': getattr(args, 'commitment_cost', 0.25),
                        'ema_decay': getattr(args, 'ema_decay', 0.99),
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
    
    vqvae = VQVAE(
        image_size=image_size,
        channels=model_cfg['channels'],
        cond_dim=model_cfg['cond_dim'],
        latent_dim=model_cfg['latent_dim'],
        num_embeddings=model_cfg['num_embeddings'],
        commitment_cost=model_cfg['commitment_cost'],
        ema_decay=model_cfg['ema_decay'],
        base_dim=model_cfg['dim'],
        dim_mults=tuple(model_cfg['dim_mults']),
    )
    
    # Create Prior if enabled
    prior = None
    if model_cfg.get('prior', {}).get('enabled', True):
        prior_cfg = model_cfg['prior']
        prior = PixelCNNPrior(
            num_embeddings=model_cfg['num_embeddings'],
            hidden_dim=prior_cfg['hidden_dim'],
            cond_dim=model_cfg['cond_dim'],
            num_layers=prior_cfg['num_layers'],
            kernel_size=prior_cfg.get('kernel_size', 3),
            cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1),
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
    
    # Load state dicts
    if 'vqvae_state_dict' in checkpoint and checkpoint['vqvae_state_dict'] is not None:
        state_dict = checkpoint['vqvae_state_dict']
        # Handle missing keys (e.g., new buffers like last_reset_usage)
        model_state_dict = vqvae.state_dict()
        missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint (will use defaults): {missing_keys}")
            # Add missing keys with default values
            for key in missing_keys:
                state_dict[key] = model_state_dict[key]
        vqvae.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("VQ-VAE state dict not found in checkpoint")
    
    if prior is not None:
        if 'prior_state_dict' in checkpoint and checkpoint['prior_state_dict'] is not None:
            prior_state_dict = checkpoint['prior_state_dict']
            # Handle missing keys
            model_state_dict = prior.state_dict()
            missing_keys = set(model_state_dict.keys()) - set(prior_state_dict.keys())
            if missing_keys:
                print(f"Warning: Missing keys in prior checkpoint (will use defaults): {missing_keys}")
                for key in missing_keys:
                    prior_state_dict[key] = model_state_dict[key]
            prior.load_state_dict(prior_state_dict, strict=False)
        else:
            print("WARNING: Prior state dict not found, using untrained prior")
    
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    if prior is not None:
        prior = prior.to(device)
        prior.eval()
    
    print(f"Models loaded from {checkpoint_path}")
    print(f"  VQ-VAE channels: {vqvae.channels}")
    print(f"  VQ-VAE image size: {vqvae.image_size}")
    print(f"  Prior enabled: {prior is not None}")
    
    return vqvae, prior, config, date_str


def sample_unconditional(vqvae, prior, save_root, date_str, component_names,
                        n_samples=1000, batch_sz=64, temperature=1.0, top_k=None, device='cuda'):
    """
    Unconditional sampling - generate without conditions.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
    
    vqvae.to(device).eval()
    if prior is not None:
        prior.to(device).eval()
    
    n_done = 0
    
    print(f"\nUnconditional sampling for model dated {date_str}")
    
    with torch.no_grad():
        while n_done < n_samples:
            b = min(batch_sz, n_samples - n_done)
            
            if prior is not None:
                # Sample from prior
                indices = prior.sample(
                    batch_size=b,
                    cond=None,
                    temperature=temperature,
                    top_k=top_k,
                    device=device
                )
            else:
                # Random indices if no prior
                indices = torch.randint(
                    0, vqvae.num_embeddings,
                    (b, 16, 8),
                    device=device
                )
            
            # Decode
            samples = vqvae.decode(indices, cond=None)

            # Ensure samples are in [0, 1] range
            samples = torch.clamp(samples, 0.0, 1.0)

            # Save each sample individually with proper index
            for i in range(b):
                save_component_images(samples[i:i+1], out_root, f"img{n_done+i:04d}", component_names)

            n_done += b
            print(f"\r[{date_str}] Saved {n_done}/{n_samples}", end="", flush=True)
    
    print(f"\nDone – unconditional samples saved under {out_root}")


def sample_conditional(vqvae, prior, test_loader, save_root, date_str, component_names,
                      guidance_scale=1.0, temperature=1.0, top_k=None, batch_size=None, device='cuda', verbose=True):
    """
    Conditional sampling using test loader.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
    
    vqvae.to(device).eval()
    if prior is not None:
        prior.to(device).eval()
    
    n_done = 0
    
    print(f"\nConditional sampling (guidance={guidance_scale}) for model dated {date_str}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_size is not None and n_done >= batch_size:
                break
            
            # Handle batch data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                images, cond = batch_data
                cond = cond.to(device).float()
                B = cond.shape[0]
            else:
                raise ValueError(f"Unexpected batch_data format: {type(batch_data)}")
            
            # Limit batch size if specified
            if batch_size is not None:
                remaining = batch_size - n_done
                if remaining <= 0:
                    break
                B = min(B, remaining)
                cond = cond[:B]
            
            # Sample from prior
            if prior is not None:
                if guidance_scale > 1.0:
                    # Classifier-free guidance
                    indices = prior.cfg_sample(
                        batch_size=B,
                        cond=cond,
                        guidance_scale=guidance_scale,
                        temperature=temperature,
                        top_k=top_k,
                        device=device
                    )
                else:
                    indices = prior.sample(
                        batch_size=B,
                        cond=cond,
                        temperature=temperature,
                        top_k=top_k,
                        device=device
                    )
            else:
                # Random indices if no prior
                indices = torch.randint(
                    0, vqvae.num_embeddings,
                    (B, 16, 8),
                    device=device
                )
            
            # Decode
            samples = vqvae.decode(indices, cond=cond)

            # Ensure samples are in [0, 1] range
            samples = torch.clamp(samples, 0.0, 1.0)

            # Save each sample individually with proper index
            for i in range(B):
                save_component_images(samples[i:i+1], out_root, f"img{n_done+i:04d}", component_names)

            n_done += B
            if verbose:
                print(f"\r[{date_str}] Saved {n_done} samples", end="", flush=True)
    
    print(f"\nDone – conditional samples saved under {out_root}")


def sample_inpainting(vqvae, prior, test_loader, save_root, date_str, component_names, components,
                     temperature=1.0, top_k=None, device='cuda', verbose=True):
    """
    Conditional inpainting - preserve specific components, generate the rest.

    Uses the ICTAI VAE masking strategy:
    1. Keep only one component (set others to 0)
    2. Directly reconstruct through VQ-VAE (encoder → quantizer → decoder)
    3. The model learned to do this during training with component masking

    Note: This version does NOT use the prior for inpainting.
    The VQ-VAE alone can inpaint because it was trained with masked inputs.
    """
    out_root = os.path.join(save_root, date_str)

    # Create component to index mapping
    component_to_idx = {name: idx for idx, name in enumerate(component_names)}
    component_indices = [component_to_idx[comp] for comp in components]

    # Make folders for each conditioning component
    for k_name in components:
        k_dir = os.path.join(out_root, k_name)
        os.makedirs(os.path.join(k_dir, "full"), exist_ok=True)
        for i_name in component_names:
            os.makedirs(os.path.join(k_dir, i_name), exist_ok=True)

    vqvae.to(device).eval()

    if verbose:
        print(f"\nConditional inpainting for model dated {date_str}")
        print(f"Processing {len(test_loader)} batches × {len(components)} components")

    # Track samples saved per component
    n_saved = {comp_name: 0 for comp_name in components}

    with torch.no_grad():
        total_batches = len(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                images, cond = batch_data
                images = images.to(device)
                cond = cond.to(device).float() if cond is not None else None
                B, C, H, W = images.shape
            else:
                images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
                B, C, H, W = images.shape
                cond = None

            for comp_name, comp_idx in zip(components, component_indices):
                if verbose:
                    print(f"  Batch {batch_idx+1}/{total_batches}, Component {comp_name} (idx={comp_idx})...", end='', flush=True)

                # Create masked image: keep only the known component, zero out the rest
                # (same strategy as ICTAI VAE inference)
                masked_images = torch.zeros_like(images)
                masked_images[:, comp_idx:comp_idx+1, :, :] = images[:, comp_idx:comp_idx+1, :, :]

                # Direct reconstruction through VQ-VAE
                # The model learned during training to reconstruct all components
                # from a single masked component
                inpainted = vqvae.reconstruct(masked_images, cond=cond)

                # Ensure output is in [0, 1]
                inpainted = torch.clamp(inpainted, 0.0, 1.0)

                if verbose:
                    print(" done")

                # Save results
                for n in range(B):
                    save_component_images(inpainted[n:n+1], os.path.join(out_root, comp_name), f"img{n_saved[comp_name]:04d}", component_names)
                    n_saved[comp_name] += 1

            if verbose:
                print(f"\r[{date_str}] Batch {batch_idx+1} done", end="", flush=True)

    if verbose:
        print(f"\nDone – all inpainting samples saved under {out_root}")


def create_test_loader(config, args):
    """Create test data loader from config."""
    data_cfg = config['data']
    
    # Resolve paths
    root_dir = resolve_path(data_cfg['root_dir'])
    condition_csv = resolve_path(data_cfg['condition_csv'])
    
    # Get component directories from config
    component_dirs = data_cfg['component_dirs']
    
    # Create test dataset
    test_dataset = MultiComponentDataset(
        root_dir=root_dir / "test",
        component_dirs=component_dirs,
        condition_csv=condition_csv,
        condition_columns=data_cfg['condition_columns'],
        prefix_column=data_cfg.get('prefix_column', 'matching'),
        filename_pattern=data_cfg.get('filename_pattern', '{prefix}_{component}.png'),
        split='test',
        split_column=data_cfg.get('split_column', 'train'),
        stacked=True,
        normalized=data_cfg.get('normalized', False),
    )
    
    # Create data loader
    batch_size = args.batch_sz if hasattr(args, 'batch_sz') and args.batch_sz else 64
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Always 0 for sampling
        pin_memory=torch.cuda.is_available(),
    )
    
    return test_loader


def main():
    parser = argparse.ArgumentParser(description='Sample from VQ-VAE model - Modular version')
    
    # Model args
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file or directory containing checkpoints')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (optional, will try to find in checkpoint dir)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sampling mode
    parser.add_argument('--mode', type=str, required=True,
                       choices=['unconditional', 'inpainting', 'conditional'],
                       help='Sampling mode')
    
    # Unconditional args
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for unconditional/conditional mode')
    parser.add_argument('--batch_sz', type=int, default=64,
                       help='Batch size for sampling')
    
    # Inpainting args
    parser.add_argument('--components', type=str, nargs='+',
                       default=None,
                       help='Components to condition on for inpainting mode')
    
    # Conditional args
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                       help='Guidance scale for conditional sampling (classifier-free guidance)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature for prior')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Top-k filtering for prior sampling')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base directory to save samples (overrides config)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load models and config
    print("Loading models...")
    vqvae, prior, config, date_str = load_model_from_checkpoint(args.checkpoint, args.config, device)
    
    # Get component names from config
    component_names = config['data']['component_dirs']
    
    # Determine save root based on mode
    if args.output_dir:
        save_root = resolve_path(args.output_dir)
    else:
        save_root = resolve_path(config['paths'].get('samples_dir', 'samples/vqvae'))
    
    if args.mode == 'unconditional':
        save_root = save_root / 'unconditional'
    elif args.mode == 'inpainting':
        save_root = save_root / 'inpainting'
    else:  # conditional
        save_root = save_root / 'conditional'
    
    # Create output directory
    os.makedirs(save_root, exist_ok=True)
    
    with torch.no_grad():
        if args.mode == 'unconditional':
            sample_unconditional(
                vqvae, prior, save_root, date_str, component_names,
                n_samples=args.num_samples,
                batch_sz=args.batch_sz,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
        
        elif args.mode == 'inpainting':
            # Get components from args or config
            if args.components is None:
                components = config.get('sampling', {}).get('components', component_names[:3])
            else:
                components = args.components
            
            # Validate components
            invalid = [c for c in components if c not in component_names]
            if invalid:
                raise ValueError(f"Invalid components: {invalid}. Valid: {component_names}")
            
            # Load test loader
            test_loader = create_test_loader(config, args)
            
            sample_inpainting(
                vqvae, prior, test_loader, save_root, date_str, component_names,
                components=components,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
        
        elif args.mode == 'conditional':
            # Load test loader
            test_loader = create_test_loader(config, args)
            
            sample_conditional(
                vqvae, prior, test_loader, save_root, date_str, component_names,
                guidance_scale=args.guidance_scale,
                temperature=args.temperature,
                top_k=args.top_k,
                batch_size=args.num_samples,
                device=device
            )
    
    print(f"\nSampling completed!")


if __name__ == '__main__':
    main()

