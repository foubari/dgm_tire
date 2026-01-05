#!/usr/bin/env python3
"""
Sampling script for Flow Matching - Modular version.

Supports three modes:
1. Unconditional: Generate samples without conditions
2. Conditional: Generate conditionally using test loader with guidance scale
3. Inpainting: Generate missing components while preserving known ones

Usage:
    python sample.py --checkpoint outputs/.../checkpoint_100.pt --config configs/flow_matching_default.yaml --mode unconditional
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
from models.flow_matching import FlowMatching, Unet
from utils.config import load_config, auto_complete_config, validate_config, resolve_path
from utils.io import save_component_images


def load_model_from_checkpoint(checkpoint_path, config_path=None, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., checkpoint_100.pt) or directory
        config_path: Optional path to config YAML (if None, tries to find config.yaml in checkpoint dir)
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
                        'timesteps': getattr(args, 'timesteps', 100),
                        'solver': getattr(args, 'solver', 'euler'),
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
    
    # Create model
    model_cfg = config['model']
    image_size = tuple(model_cfg['image_size'])
    
    unet = Unet(
        dim=model_cfg['dim'],
        dim_mults=tuple(model_cfg['dim_mults']),
        channels=model_cfg['channels'],
        cond_dim=model_cfg['cond_dim'],
        self_condition=False,
        flash_attn=False
    )
    
    model = FlowMatching(
        model=unet,
        image_size=image_size,
        timesteps=model_cfg['timesteps'],
        solver=model_cfg.get('solver', 'euler'),
        cond_drop_prob=model_cfg['cond_drop_prob']
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
    
    # Load state dict (check for EMA model)
    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        model.load_state_dict(checkpoint['ema_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Channels: {model.channels}")
    print(f"  Image size: {model.image_size}")
    print(f"  Timesteps: {model.num_timesteps}")
    print(f"  Solver: {model.solver}")
    
    # Check training configuration
    normalized = config['data'].get('normalized', False)
    print(f"  Normalized dimensions: {normalized}")
    
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
            
            save_component_images(samples, out_root, f"img{n_done:04d}", component_names)
            
            n_done += b
            print(f"\r[{date_str}] Saved {n_done}/{n_samples}", end="", flush=True)
    
    print(f"\nDone – unconditional samples saved under {out_root}")


def sample_conditional(model, test_loader, save_root, date_str, component_names,
                      guidance_scale=2.0, batch_size=None, device='cuda', verbose=True):
    """
    Conditional sampling using test loader with guidance scale.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
    
    model.to(device).eval()
    n_done = 0
    
    print(f"\nConditional sampling (guidance={guidance_scale}) for model dated {date_str}")
    if batch_size is not None:
        print(f"  Using batch_size={batch_size} for generation (test_loader batch_size may differ)")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            # Limit number of batches if batch_size is specified
            if batch_size is not None and n_done >= batch_size:
                break
            
            # Handle different batch data formats
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                images, cond = batch_data
                
                # Get batch size from images
                if isinstance(images, torch.Tensor):
                    B = images.shape[0]
                elif isinstance(images, (list, tuple)) and len(images) > 0:
                    B = len(images)
                else:
                    B = 1
                
                # Process condition tensor
                if isinstance(cond, torch.Tensor):
                    cond = cond.to(device).float()
                    if len(cond.shape) == 1:
                        if cond.shape[0] == 2:  # cond_dim = 2
                            cond = cond.unsqueeze(0).expand(B, -1)
                    elif len(cond.shape) == 2:
                        if cond.shape[0] != B:
                            if cond.shape[0] == 1:
                                cond = cond.expand(B, -1)
                            else:
                                B = cond.shape[0]
                else:
                    raise ValueError(f"Unexpected cond type: {type(cond)}")
                
                # Clamp condition values
                cond = torch.clamp(cond, -1000.0, 1000.0)
            else:
                raise ValueError(f"Unexpected batch_data format: {type(batch_data)}")
            
            # Limit batch size if specified
            if batch_size is not None:
                remaining = batch_size - n_done
                if remaining <= 0:
                    break
                B = min(B, remaining)
                if cond is not None:
                    cond = cond[:B]
            
            # Generate samples with conditions
            samples = model.sample(batch_size=B, cond=cond, guidance_scale=guidance_scale)
            
            # Ensure samples are in [0, 1] range
            samples = torch.clamp(samples, 0.0, 1.0)
            
            save_component_images(samples, out_root, f"img{batch_idx:04d}", component_names)
            
            n_done += B
            if verbose:
                print(f"\r[{date_str}] Saved {n_done} samples", end="", flush=True)
    
    print(f"\nDone – conditional samples saved under {out_root}")


def sample_inpainting(model, test_loader, save_root, date_str, component_names, components,
                     device='cuda', verbose=True):
    """
    Conditional inpainting - preserve specific components, generate the rest.
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
    
    model.to(device).eval()
    if verbose:
        print(f"\nConditional inpainting for model dated {date_str}")
        print(f"Processing {len(test_loader)} batches × {len(components)} components")
    
    with torch.no_grad():
        total_batches = len(test_loader)
        for batch_idx, batch_data in enumerate(test_loader):
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                images, cond = batch_data
                images = images.to(device)
                B, C, H, W = images.shape
            else:
                images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
                B, C, H, W = images.shape
            
            for comp_name, comp_idx in zip(components, component_indices):
                if verbose:
                    print(f"  Batch {batch_idx+1}/{total_batches}, Component {comp_name} (idx={comp_idx})...", end='', flush=True)
                
                # Create mask: preserve this component, inpaint others
                mask = torch.zeros((B, C, H, W), device=device)
                mask[:, comp_idx, :, :] = 1.0  # Preserve component comp_idx
                
                # Inpaint
                inpainted = model.inpaint(partial=images, mask=mask)
                
                if verbose:
                    print(" done")
                
                # Save results
                for n in range(B):
                    prefix = f"img{batch_idx:04d}_{n:02d}"
                    save_component_images(inpainted[n:n+1], os.path.join(out_root, comp_name), prefix, component_names)
            
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
    parser = argparse.ArgumentParser(description='Sample from Flow Matching model - Modular version')
    
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
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                       help='Guidance scale for conditional sampling')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base directory to save samples (overrides config)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model and config
    print("Loading model...")
    model, config, date_str = load_model_from_checkpoint(args.checkpoint, args.config, device)
    
    # Get component names from config
    component_names = config['data']['component_dirs']
    
    # Determine save root based on mode
    if args.output_dir:
        save_root = resolve_path(args.output_dir)
    else:
        save_root = resolve_path(config['paths'].get('samples_dir', 'samples/flow_matching'))
    
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
                model, save_root, date_str, component_names,
                n_samples=args.num_samples,
                batch_sz=args.batch_sz,
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
                model, test_loader, save_root, date_str, component_names,
                components=components,
                device=device
            )
        
        elif args.mode == 'conditional':
            # Load test loader
            test_loader = create_test_loader(config, args)
            
            sample_conditional(
                model, test_loader, save_root, date_str, component_names,
                guidance_scale=args.guidance_scale,
                batch_size=args.num_samples,
                device=device
            )
    
    print(f"\nSampling completed!")


if __name__ == '__main__':
    main()

