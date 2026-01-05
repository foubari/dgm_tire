#!/usr/bin/env python3
"""
Sampling script for MMVAE+ - Modular version.

Supports two modes:
1. Unconditional: Generate samples without conditions
2. Conditional/Inpainting: Generate all components conditioned on some observed components

Usage:
    python sample.py --checkpoint outputs/.../checkpoint_100.pt --config configs/mmvaeplus_default.yaml --mode unconditional
    python sample.py --checkpoint outputs/.../checkpoint_100.pt --config configs/mmvaeplus_default.yaml --mode inpainting --components group_nc bt
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
from models.mmvaeplus import MMVAEplusEpure
from models.mmvaeplus.utils import unpack_data, get_mean
from utils.config import load_config, auto_complete_config, validate_config, resolve_path
from utils.io import save_component_images


def load_model_from_checkpoint(checkpoint_path, config_path=None, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        config_path: Optional path to config YAML
        device: Device to load model on (string or torch.device)
    
    Returns:
        model, config, date_string
    """
    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device
    
    checkpoint_path = Path(checkpoint_path)
    
    # Handle directory vs file
    if checkpoint_path.is_dir():
        check_dir = checkpoint_path / 'check'
        if check_dir.exists():
            checkpoints = list(check_dir.glob('checkpoint_*.pt'))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
            else:
                raise FileNotFoundError(f"No checkpoint found in {check_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {check_dir}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    
    # Load config if provided
    if config_path:
        config = load_config(config_path)
        config = auto_complete_config(config)
    else:
        # Try to load from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Config path required if not in checkpoint")
    
    # Extract date string from checkpoint path
    date_str = checkpoint_path.parent.parent.name if checkpoint_path.parent.name == 'check' else checkpoint_path.parent.name
    
    # Create params object
    params_dict = checkpoint.get('params', {})
    if isinstance(params_dict, dict):
        class Params:
            latent_dim_w = params_dict.get('latent_dim_w', config['model']['latent_dim_w'])
            latent_dim_z = params_dict.get('latent_dim_z', config['model']['latent_dim_z'])
            latent_dim_u = params_dict.get('latent_dim_u', config['model']['latent_dim_u'])
            nf = params_dict.get('nf', config['model']['nf'])
            nf_max = params_dict.get('nf_max', config['model']['nf_max'])
            cond_dim = params_dict.get('cond_dim', config['model'].get('cond_dim', 0))
            priorposterior = params_dict.get('priorposterior', config['model'].get('priorposterior', 'Laplace'))
            datadir = params_dict.get('datadir', str(resolve_path(config['data']['root_dir'])))
            no_cuda = (device_obj.type == 'cpu')
            beta = params_dict.get('beta', config['training'].get('beta', 2.5))
            K = params_dict.get('K', config['training'].get('K', 1))
    else:
        Params = params_dict
    
    # Create model
    model = MMVAEplusEpure(Params()).to(device_obj)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Date: {date_str}")
    print(f"  Components: {len(model.vaes)}")
    print(f"  Cond dim: {Params.cond_dim}")
    
    return model, config, date_str


def sample_unconditional(model, n_samples, save_root, date_str, component_names, device='cuda'):
    """
    Unconditional sampling - generate samples without conditions.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
    
    model.to(device).eval()
    
    print(f"\nUnconditional sampling for model dated {date_str}")
    print(f"  Generating {n_samples} samples")
    
    # Generate samples
    with torch.no_grad():
        generations = model.generate_unconditional(N=n_samples, cond=None)
    
    # Save samples
    # generations is a list of 5 tensors, each of shape (n_samples, 1, 64, 32)
    for n in range(n_samples):
        prefix = f"img{n:04d}"
        # Stack components for saving: concatenate along channel dimension
        # Each gen[n] is (1, 64, 32), stack them to get (5, 64, 32), then add batch dim -> (1, 5, 64, 32)
        sample_tensor = torch.stack([gen[n].squeeze(0) for gen in generations], dim=0)  # (5, 64, 32)
        sample_tensor = sample_tensor.unsqueeze(0)  # (1, 5, 64, 32) - add batch dimension
        save_component_images(sample_tensor, out_root, prefix, component_names)
    
    print(f"\nDone – {n_samples} samples saved under {out_root}")


def sample_conditional(
    model,
    test_loader,
    save_root,
    date_str,
    component_names,
    device='cuda',
    max_batches=None,
    verbose=True
):
    """
    Conditional sampling - generate samples conditioned on dimensions from test set.
    
    Args:
        model: Trained MMVAEplus model
        test_loader: DataLoader returning (data_tuple, cond)
        save_root: Output directory
        date_str: Model checkpoint date
        component_names: List of all 5 component names
        device: Device
        max_batches: Maximum number of batches to process (None = all)
        verbose: Print progress
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)
    
    model.to(device).eval()
    
    if verbose:
        num_batches = min(len(test_loader), max_batches) if max_batches else len(test_loader)
        print(f"\nConditional sampling for model dated {date_str}")
        print(f"  Processing {num_batches} batches (out of {len(test_loader)} total)")
    
    n_done = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Sampling")):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Unpack batch: (data_tuple, cond)
            data_tuple, batch_cond = unpack_data(batch_data, device=device)
            
            # Replicate cond for all modalities (5 components)
            if batch_cond is not None:
                cond_list = [batch_cond] * len(data_tuple)
            else:
                cond_list = None
            
            B = batch_cond.shape[0] if batch_cond is not None else data_tuple[0].shape[0]
            
            # Generate samples conditioned on dimensions
            generations = model.generate_unconditional(N=B, cond=cond_list)
            
            # Save samples
            for n in range(B):
                prefix = f"img{n_done + n:04d}"
                # Stack components for saving
                sample_tensor = torch.stack([gen[n].squeeze(0) for gen in generations], dim=0)  # (5, 64, 32)
                sample_tensor = sample_tensor.unsqueeze(0)  # (1, 5, 64, 32)
                save_component_images(sample_tensor, out_root, prefix, component_names)
            
            n_done += B
    
    if verbose:
        print(f"\nDone – {n_done} samples saved under {out_root}")


def sample_conditional_inpainting(
    model,
    test_loader,
    save_root,
    date_str,
    component_names,
    components_to_condition,
    device='cuda',
    max_batches=None,
    verbose=True
):
    """
    Conditional inpainting: given K components, generate all 5 components.
    
    Uses self_and_cross_modal_generation_forward:
    - Shared latent (z) extracted from observed modality
    - Private latent (w) resampled for other modalities
    
    Args:
        model: Trained MMVAEplus model
        test_loader: DataLoader returning (data_tuple, cond)
        save_root: Output directory
        date_str: Model checkpoint date
        component_names: List of all 5 component names
        components_to_condition: List of components to use as input
        device: Device
        verbose: Print progress
    """
    out_root = os.path.join(save_root, date_str)
    
    # Create component to index mapping
    component_to_idx = {name: idx for idx, name in enumerate(component_names)}
    cond_indices = [component_to_idx[c] for c in components_to_condition]
    
    # Create output folders - same structure as DDPM inpainting
    for cond_comp in components_to_condition:
        cond_dir = os.path.join(out_root, cond_comp)
        # Create full/ folder for each conditioning component
        os.makedirs(os.path.join(cond_dir, "full"), exist_ok=True)
        # Create folder for each generated component
        for gen_comp in component_names:
            folder = os.path.join(cond_dir, gen_comp)
            os.makedirs(folder, exist_ok=True)
    
    model.to(device).eval()
    
    if verbose:
        print(f"\nConditional inpainting for model {date_str}")
        print(f"Conditioning on: {components_to_condition}")
        num_batches = min(len(test_loader), max_batches) if max_batches else len(test_loader)
        print(f"Processing {num_batches} batches (out of {len(test_loader)} total)")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Sampling")):
            if max_batches and batch_idx >= max_batches:
                break
            # Unpack batch: (data_tuple, cond)
            data_tuple, batch_cond = unpack_data(batch_data, device=device)
            
            # Move to device
            data = [d.to(device) for d in data_tuple]
            
            # Replicate cond for all modalities
            if batch_cond is not None:
                cond = [batch_cond] * len(data)
            else:
                cond = None
            
            # Cross-modal generation
            # px_us[i][j] = reconstruction of modality j conditioned on modality i
            _, px_us, _ = model.self_and_cross_modal_generation_forward(
                data,
                cond=cond
            )
            
            # Extract means - following notebook pattern exactly
            # px_us is a list (over cond) of lists (over gen) of distributions
            # px_us[cond][gen] is a Normal; its mean has shape [K, B, 1, H, W] (with K=1 typically)
            # get_mean(px_u) returns px_u.mean which has shape [K, B, 1, H, W]
            # Following notebook: get_mean(px_u).squeeze(1) removes channel dim -> [K, B, H, W]
            # But notebook says .squeeze(0), so maybe it's [B, 1, H, W] after K handling?
            
            # Let's check the actual shape and handle accordingly
            recons = []
            for cond_list in px_us:
                cond_recons = []
                for px_u in cond_list:
                    mean_tensor = get_mean(px_u)  # Could be [K, B, 1, H, W] or [B, 1, H, W]
                    
                    # Handle K dimension: if first dim is K=1, remove it
                    if mean_tensor.dim() == 5:
                        # Shape is [K, B, 1, H, W]
                        if mean_tensor.shape[0] == 1:
                            mean_tensor = mean_tensor[0]  # Take K=0 -> [B, 1, H, W]
                        else:
                            mean_tensor = mean_tensor[0]  # Take first K -> [B, 1, H, W]
                    
                    # Now mean_tensor should be [B, 1, H, W]
                    # Remove channel dimension (dim=1)
                    if mean_tensor.dim() == 4 and mean_tensor.shape[1] == 1:
                        mean_tensor = mean_tensor.squeeze(1)  # [B, H, W]
                    
                    cond_recons.append(mean_tensor)
                recons.append(cond_recons)
            
            # Now recons[cond_idx][gen_idx] should be [batch, H, W]
            B = recons[0][0].shape[0] if len(recons) > 0 and len(recons[0]) > 0 else data[0].shape[0]
            
            # Save: for each conditioning modality (only those requested), save all generated modalities
            for i in range(B):
                for cond_idx, cond_comp in zip(cond_indices, components_to_condition):
                    # Save individual components
                    for gen_idx, gen_comp in enumerate(component_names):
                        img_t = recons[cond_idx][gen_idx][i]  # Should be [H, W] tensor
                        
                        # Save image - following notebook pattern exactly
                        save_path = os.path.join(
                            out_root,
                            cond_comp,
                            gen_comp,
                            f"img{batch_idx:04d}_{i:02d}.png"
                        )
                        
                        # Use PIL to save - following notebook: img_t.squeeze(0).clamp(0,1)
                        from PIL import Image
                        if isinstance(img_t, torch.Tensor):
                            # img_t is [H, W], but notebook does squeeze(0) just in case
                            img_np = img_t.squeeze(0).clamp(0, 1).cpu().numpy()
                        else:
                            img_np = np.array(img_t)
                            img_np = np.clip(img_np, 0, 1)
                        
                        # Remove all singleton dimensions to ensure 2D
                        while img_np.ndim > 2:
                            img_np = img_np.squeeze()
                        
                        img_uint8 = (img_np * 255).astype(np.uint8)
                        Image.fromarray(img_uint8).save(save_path)
                    
                    # Save full superposition (sum of all components) - like DDPM
                    summed = sum(recons[cond_idx][g][i] for g in range(len(component_names)))
                    if isinstance(summed, torch.Tensor):
                        summed_np = summed.squeeze(0).clamp(0, 1).cpu().numpy()
                    else:
                        summed_np = np.array(summed)
                        summed_np = np.clip(summed_np, 0, 1)
                    
                    # Remove all singleton dimensions to ensure 2D
                    while summed_np.ndim > 2:
                        summed_np = summed_np.squeeze()
                    
                    summed_uint8 = (summed_np * 255).astype(np.uint8)
                    full_path = os.path.join(
                        out_root,
                        cond_comp,
                        'full',
                        f"img{batch_idx:04d}_{i:02d}.png"
                    )
                    Image.fromarray(summed_uint8).save(full_path)
            
            if verbose and batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}/{len(test_loader)}")
    
    if verbose:
        print(f"\nDone - samples saved to {out_root}")


def main():
    parser = argparse.ArgumentParser(description='MMVAE+ Sampling')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint file or directory')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['unconditional', 'conditional', 'inpainting'],
                      help='Sampling mode')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of samples (for unconditional)')
    parser.add_argument('--batch_sz', type=int, default=32,
                      help='Batch size for conditional sampling')
    parser.add_argument('--components', type=str, nargs='+', default=None,
                      help='Components to condition on (for inpainting)')
    parser.add_argument('--max_batches', type=int, default=None,
                      help='Maximum number of batches to process (for inpainting)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                      help='Guidance scale (for conditional mode, not used in MMVAE+)')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading models...")
    model, config, date_str = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    
    # Get component names
    component_names = config['data']['component_dirs']
    
    # Setup save directory
    save_root = resolve_path(config['paths'].get('samples_dir', 'samples/mmvaeplus'))
    
    # Add mode-specific subdirectory (like DDPM/MDM)
    if args.mode == 'unconditional':
        save_root = save_root / 'unconditional'
    elif args.mode == 'inpainting':
        save_root = save_root / 'inpainting'
    else:  # conditional
        save_root = save_root / 'conditional'
    
    save_root.mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'unconditional':
        sample_unconditional(
            model, args.num_samples, str(save_root), date_str,
            component_names, args.device
        )
    
    elif args.mode == 'inpainting':
        if args.components is None:
            raise ValueError("--components required for inpainting mode")
        
        # Validate components
        invalid = [c for c in args.components if c not in component_names]
        if invalid:
            raise ValueError(f"Invalid components: {invalid}. Valid: {component_names}")
        
        # Create test loader
        data_cfg = config['data']
        root_dir = resolve_path(data_cfg['root_dir'])
        condition_csv = resolve_path(data_cfg['condition_csv'])
        
        test_dataset = MultiComponentDataset(
            root_dir=root_dir / 'test',
            component_dirs=data_cfg['component_dirs'],
            condition_csv=condition_csv,
            condition_columns=data_cfg['condition_columns'],
            prefix_column=data_cfg['prefix_column'],
            filename_pattern=data_cfg.get('filename_pattern', '{prefix}_{component}.png'),
            split='test',
            split_column=data_cfg.get('split_column', 'train'),
            stacked=False,  # Return tuple
            normalized=data_cfg.get('normalized', False)
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        sample_conditional_inpainting(
            model, test_loader, str(save_root), date_str,
            component_names, args.components, args.device,
            max_batches=args.max_batches
        )
    
    elif args.mode == 'conditional':
        # Create test loader
        data_cfg = config['data']
        root_dir = resolve_path(data_cfg['root_dir'])
        condition_csv = resolve_path(data_cfg['condition_csv'])
        
        test_dataset = MultiComponentDataset(
            root_dir=root_dir / 'test',
            component_dirs=data_cfg['component_dirs'],
            condition_csv=condition_csv,
            condition_columns=data_cfg['condition_columns'],
            prefix_column=data_cfg['prefix_column'],
            filename_pattern=data_cfg.get('filename_pattern', '{prefix}_{component}.png'),
            split='test',
            split_column=data_cfg.get('split_column', 'train'),
            stacked=False,  # Return tuple
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
            model, test_loader, str(save_root), date_str,
            component_names, args.device,
            max_batches=args.max_batches
        )


if __name__ == '__main__':
    main()

