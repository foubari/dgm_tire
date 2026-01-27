#!/usr/bin/env python3
"""
Sampling script for MDM - Modular version.

Supports three modes:
1. Unconditional: Generate samples without conditions
2. Conditional: Generate conditionally using test loader with guidance scale
3. Inpainting: Generate missing components while preserving known ones

Usage:
    python sample.py --checkpoint outputs/.../checkpoint_100.pt --config configs/mdm_default.yaml --mode unconditional
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

from datasets.categorical import SegmentationDataset
from models.mdm import MultinomialDiffusion, SegmentationUnet
from utils.config import load_config, auto_complete_config, validate_config, resolve_path
# save_component_images not used - MDM uses save_mdm_sample instead
import imageio


def load_model_from_checkpoint(checkpoint_path, config_path=None, device='cuda'):
    """Load model from checkpoint."""
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
        config_yaml = checkpoint_dir / 'config.yaml'
        if config_yaml.exists():
            config = load_config(config_yaml)
        else:
            args_path = checkpoint_dir / 'args.pickle'
            if not args_path.exists():
                args_path = checkpoint_dir.parent / 'args.pickle'
            
            if args_path.exists():
                with open(args_path, 'rb') as f:
                    args = pickle.load(f)
                config = {
                    'model': {
                        'num_classes': getattr(args, 'num_classes', 6),
                        'dim': getattr(args, 'diffusion_dim', 64),
                        'dim_mults': getattr(args, 'dim_mults', (1, 2, 4)),
                        'timesteps': getattr(args, 'diffusion_steps', 500),
                        'cond_drop_prob': getattr(args, 'cond_drop_prob', 0.1),
                    },
                    'data': {
                        'resolution': getattr(args, 'resolution', [64, 32]),
                        'condition_columns': getattr(args, 'cond_cols', ['width_px', 'height_px']),
                    }
                }
    
    if config is None:
        raise FileNotFoundError(f"Could not find config.yaml or args.pickle near {checkpoint_path}")
    
    # Extract date from checkpoint directory path
    # If checkpoint is in check/ subdirectory, use parent directory name (timestamp)
    # Otherwise use checkpoint_dir name directly
    date_str = checkpoint_dir.parent.name if checkpoint_dir.name == 'check' else checkpoint_dir.name
    
    config = auto_complete_config(config)
    
    # Create model
    model_cfg = config['model']
    data_cfg = config['data']
    resolution = tuple(data_cfg.get('resolution', [64, 32]))
    
    unet = SegmentationUnet(
        num_classes=model_cfg['num_classes'],
        dim=model_cfg['dim'],
        num_steps=model_cfg['timesteps'],
        dim_mults=tuple(model_cfg['dim_mults']),
        cond_dim=model_cfg['cond_dim']
    )
    
    model = MultinomialDiffusion(
        num_classes=model_cfg['num_classes'],
        shape=(1, *resolution),
        denoise_fn=unet,
        timesteps=model_cfg['timesteps'],
        cond_drop_prob=model_cfg['cond_drop_prob'],
        loss_type=model_cfg.get('loss_type', 'elbo')
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Num classes: {model.num_classes}")
    print(f"  Shape: {model.shape}")
    print(f"  Diffusion steps: {model.num_timesteps}")
    
    return model, config, date_str


def save_mask(mask, path, scale_for_visualization=True, background_class=5):
    """
    Save segmentation mask as binary image (matches DDPM output format).

    Args:
        mask: Segmentation mask (categorical indices)
        path: Output path
        scale_for_visualization: If True, binarize (background=0, foreground=255)
                                 If False, save raw indices
        background_class: Class index to treat as background (default: 5 for EPURE)
    """
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    # Ensure 2D shape (H, W)
    if len(mask_np.shape) == 3:
        # (1, H, W) -> (H, W)
        mask_np = mask_np.squeeze(0)
    elif len(mask_np.shape) > 2:
        # (C, H, W) or other -> take first channel or squeeze
        mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np.squeeze()

    # Ensure 2D
    if len(mask_np.shape) != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask_np.shape}")

    if scale_for_visualization:
        # Binary output (matches DDPM format):
        # Background class -> 0 (black)
        # All other classes -> 255 (white)
        binary = np.where(mask_np == background_class, 0, 255).astype(np.uint8)
        mask_np = binary
    else:
        mask_np = mask_np.astype(np.uint8)

    imageio.imwrite(path, mask_np)


def save_mdm_sample(segmap, out_root, prefix, component_names, component_to_idx):
    """
    Save MDM segmentation map with proper folder structure.

    Creates:
    - full/: segmentation map (class indices visualized)
    - {component}/: binary mask for each component

    Args:
        segmap: (H, W) tensor with class indices
        out_root: output directory (should have full/, group_nc/, etc. subfolders)
        prefix: filename prefix (e.g., "img0000")
        component_names: list of component names ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']
        component_to_idx: dict mapping component name to class index
    """
    # Ensure segmap is on CPU and numpy
    if torch.is_tensor(segmap):
        segmap_np = segmap.cpu().numpy()
    else:
        segmap_np = segmap

    # Save full image (segmentation map with all classes)
    # Use save_mask with scale_for_visualization=True to show non-background as white
    full_path = os.path.join(out_root, "full", f"{prefix}_full.png")
    save_mask(segmap_np, full_path, scale_for_visualization=True, background_class=0)

    # Save individual component masks (binary: 0 or 255)
    for comp_name in component_names:
        comp_idx = component_to_idx[comp_name]
        # Create binary mask: 255 where this component exists, 0 elsewhere
        binary_mask = np.where(segmap_np == comp_idx, 255, 0).astype(np.uint8)
        comp_path = os.path.join(out_root, comp_name, f"{prefix}_{comp_name}.png")
        imageio.imwrite(comp_path, binary_mask)


def sample_unconditional(model, save_root, date_str, n_samples=1000, batch_sz=64, device='cuda'):
    """Unconditional sampling with proper folder structure."""
    out_root = os.path.join(save_root, date_str)

    # Component names and their MDM class indices
    # For epure: 0=background, 1=group_nc, 2=group_km, 3=bt, 4=fpu, 5=tpc
    component_names = ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']
    component_to_idx = {
        'group_nc': 1,
        'group_km': 2,
        'bt': 3,
        'fpu': 4,
        'tpc': 5
    }

    # Create folder structure like other models
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for comp_name in component_names:
        os.makedirs(os.path.join(out_root, comp_name), exist_ok=True)

    model.to(device).eval()
    n_done = 0

    print(f"\nUnconditional sampling for model dated {date_str}")

    with torch.no_grad():
        while n_done < n_samples:
            b = min(batch_sz, n_samples - n_done)
            samples = model.sample(num_samples=b, cond=None, guidance_scale=0.0)  # (B, H, W) or (B, 1, H, W)

            # Ensure samples is 3D (B, H, W)
            if len(samples.shape) == 4:
                # (B, 1, H, W) -> (B, H, W)
                samples = samples.squeeze(1)

            for i in range(b):
                prefix = f"img{n_done+i:04d}"
                save_mdm_sample(samples[i], out_root, prefix, component_names, component_to_idx)

            n_done += b
            print(f"\r[{date_str}] Saved {n_done}/{n_samples}", end="", flush=True)

    print(f"\nDone – unconditional samples saved under {out_root}")


def sample_conditional(model, test_loader, save_root, date_str, guidance_scale=2.0, batch_size=None, device='cuda'):
    """Conditional sampling with proper folder structure."""
    out_root = os.path.join(save_root, date_str)

    # Component names and their MDM class indices
    # For epure: 0=background, 1=group_nc, 2=group_km, 3=bt, 4=fpu, 5=tpc
    component_names = ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']
    component_to_idx = {
        'group_nc': 1,
        'group_km': 2,
        'bt': 3,
        'fpu': 4,
        'tpc': 5
    }

    # Create folder structure like other models
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for comp_name in component_names:
        os.makedirs(os.path.join(out_root, comp_name), exist_ok=True)

    model.to(device).eval()
    n_done = 0

    print(f"\nConditional sampling (guidance={guidance_scale}) for model dated {date_str}")

    with torch.no_grad():
        for batch_idx, (x, cond) in enumerate(test_loader):
            if batch_size is not None and n_done >= batch_size:
                break

            cond = cond.to(device) if cond is not None else None
            B = cond.shape[0] if cond is not None else 1

            if batch_size is not None:
                remaining = batch_size - n_done
                if remaining <= 0:
                    break
                B = min(B, remaining)
                if cond is not None:
                    cond = cond[:B]

            samples = model.sample(num_samples=B, cond=cond, guidance_scale=guidance_scale)

            # Ensure samples is 3D (B, H, W)
            if len(samples.shape) == 4:
                # (B, 1, H, W) -> (B, H, W)
                samples = samples.squeeze(1)

            for i in range(B):
                prefix = f"img{n_done+i:04d}"
                save_mdm_sample(samples[i], out_root, prefix, component_names, component_to_idx)

            n_done += B
            print(f"\r[{date_str}] Saved {n_done} samples", end="", flush=True)

    print(f"\nDone – conditional samples saved under {out_root}")


def segmap_to_multichannel(segmap, num_classes=5):
    """
    Convert segmentation map to multi-channel one-hot format.

    Args:
        segmap: (B, H, W) tensor with class indices in [0, num_classes-1]
        num_classes: Number of classes/components

    Returns:
        multichannel: (B, C, H, W) tensor with one-hot encoding
    """
    B, H, W = segmap.shape
    multichannel = torch.zeros(B, num_classes, H, W, device=segmap.device)
    for c in range(num_classes):
        multichannel[:, c] = (segmap == c).float()
    return multichannel


def sample_inpainting(model, test_loader, save_root, date_str, components_to_preserve, guidance_scale=2.0, batch_size=None, device='cuda'):
    """
    Inpainting: Generate missing components while preserving known ones.

    Saves results in component-specific subdirectories like DDPM.

    Args:
        model: MultinomialDiffusion model with inpaint() method
        test_loader: Test data loader
        save_root: Root directory for saving samples
        date_str: Timestamp string for organizing outputs
        components_to_preserve: List of component NAMES to preserve (e.g., ['group_nc', 'bt'])
        guidance_scale: Guidance scale for classifier-free guidance
        batch_size: Max number of samples to generate (None = all test set)
        device: Device to run on
    """
    out_root = os.path.join(save_root, date_str)

    # Component names and their MDM class indices
    # For epure: 0=background, 1=group_nc, 2=group_km, 3=bt, 4=fpu, 5=tpc
    component_names = ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']
    component_to_idx = {
        'group_nc': 1,
        'group_km': 2,
        'bt': 3,
        'fpu': 4,
        'tpc': 5
    }

    # Create component-specific subdirectories (like DDPM)
    for comp_name in components_to_preserve:
        comp_dir = os.path.join(out_root, comp_name)
        os.makedirs(os.path.join(comp_dir, "full"), exist_ok=True)
        for cname in component_names:
            os.makedirs(os.path.join(comp_dir, cname), exist_ok=True)

    # Track saved samples per component
    n_saved = {comp_name: 0 for comp_name in components_to_preserve}

    model.to(device).eval()
    n_done = 0

    print(f"\nInpainting (guidance={guidance_scale}) for model dated {date_str}")
    print(f"Preserving components: {components_to_preserve}")
    print(f"Processing {len(test_loader)} batches")

    with torch.no_grad():
        for batch_idx, (segmaps, cond) in enumerate(test_loader):
            if batch_size is not None and n_done >= batch_size:
                break

            segmaps = segmaps.to(device)
            cond = cond.to(device) if cond is not None else None
            B = segmaps.shape[0]

            if batch_size is not None:
                remaining = batch_size - n_done
                if remaining <= 0:
                    break
                B = min(B, remaining)
                segmaps = segmaps[:B]
                if cond is not None:
                    cond = cond[:B]

            # Iterate through each component to preserve
            for comp_name in components_to_preserve:
                comp_idx = component_to_idx[comp_name]

                # Call inpainting method
                inpainted = model.inpaint(
                    segmaps,
                    component_idx=comp_idx,
                    cond=cond,
                    guidance_scale=guidance_scale
                )

                # Ensure inpainted is 3D (B, H, W)
                if len(inpainted.shape) == 4:
                    inpainted = inpainted.squeeze(1)

                # Save using save_mdm_sample (preserves class information properly)
                comp_out_dir = os.path.join(out_root, comp_name)
                for i in range(B):
                    prefix = f"img{n_saved[comp_name]:04d}"
                    save_mdm_sample(inpainted[i], comp_out_dir, prefix, component_names, component_to_idx)
                    n_saved[comp_name] += 1

                print(f"\r[{date_str}] Component {comp_name}: Saved {n_saved[comp_name]} samples", end="", flush=True)

            print()  # New line after each batch

            n_done += B
            if n_done >= (batch_size or float('inf')):
                break

    print(f"\nDone – inpainting samples saved under {out_root}")
    for comp_name in components_to_preserve:
        print(f"  {comp_name}: {n_saved[comp_name]} samples in {out_root}/{comp_name}/")


def create_test_loader(config, args):
    """Create test data loader from config."""
    data_cfg = config['data']
    root_dir = resolve_path(data_cfg['root_dir'])
    condition_csv = resolve_path(data_cfg['condition_csv'])
    
    test_dataset = SegmentationDataset(
        root_dir=root_dir,
        condition_csv=condition_csv,
        condition_columns=data_cfg['condition_columns'],
        prefix_column=data_cfg.get('prefix_column', 'matching'),
        split='test',
        split_column=data_cfg.get('split_column', 'train'),
        resolution=tuple(data_cfg.get('resolution', [64, 32])),
        mask_format=data_cfg.get('mask_format', 'numpy'),
        mask_path=data_cfg.get('mask_path'),
        filename_pattern=data_cfg.get('filename_pattern', '{prefix}.png'),
        normalized=data_cfg.get('normalized', False),
    )
    
    batch_size = args.batch_sz if hasattr(args, 'batch_sz') and args.batch_sz else 64
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    return test_loader


def main():
    parser = argparse.ArgumentParser(description='Sample from MDM model - Modular version')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (optional)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--mode', type=str, required=True,
                       choices=['unconditional', 'conditional', 'inpainting'],
                       help='Sampling mode')

    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples')
    parser.add_argument('--batch_sz', type=int, default=64,
                       help='Batch size for sampling')

    parser.add_argument('--guidance_scale', type=float, default=2.0,
                       help='Guidance scale for conditional/inpainting sampling')

    parser.add_argument('--components', type=str, nargs='+',
                       help='Component names to preserve for inpainting (e.g., group_nc bt)')

    parser.add_argument('--output_dir', type=str, default=None,
                       help='Base directory to save samples (overrides config)')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading model...")
    model, config, date_str = load_model_from_checkpoint(args.checkpoint, args.config, device)

    if args.output_dir:
        save_root = resolve_path(args.output_dir)
    else:
        save_root = resolve_path(config['paths'].get('samples_dir', 'samples/mdm'))

    if args.mode == 'unconditional':
        save_root = save_root / 'unconditional'
    elif args.mode == 'conditional':
        save_root = save_root / 'conditional'
    else:
        save_root = save_root / 'inpainting'

    os.makedirs(save_root, exist_ok=True)

    with torch.no_grad():
        if args.mode == 'unconditional':
            sample_unconditional(
                model, save_root, date_str,
                n_samples=args.num_samples,
                batch_sz=args.batch_sz,
                device=device
            )
        elif args.mode == 'conditional':
            test_loader = create_test_loader(config, args)
            sample_conditional(
                model, test_loader, save_root, date_str,
                guidance_scale=args.guidance_scale,
                batch_size=args.num_samples,
                device=device
            )
        elif args.mode == 'inpainting':
            test_loader = create_test_loader(config, args)

            # Parse components to preserve (now expects component names, not indices)
            if args.components:
                components_to_preserve = args.components  # Already a list from nargs='+'
            else:
                # Default: preserve all components
                components_to_preserve = ['group_nc', 'group_km', 'bt', 'fpu', 'tpc']

            sample_inpainting(
                model, test_loader, save_root, date_str,
                components_to_preserve=components_to_preserve,
                guidance_scale=args.guidance_scale,
                batch_size=args.num_samples,
                device=device
            )

    print(f"\nSampling completed!")


if __name__ == '__main__':
    main()

