#!/usr/bin/env python3
"""
Sampling script for GMRF MVAE.

3 Sampling Modes:
1. Unconditional: Sample from prior p(z)
2. Conditional: Encode real data and decode from posterior
3. Inpainting/Cross-modal: Generate missing modalities using Gaussian conditional

Usage:
    # Unconditional sampling
    python sample.py --checkpoint outputs/gmrf/run_xxx/check/checkpoint_best.pt --mode unconditional --num_samples 100

    # Conditional sampling
    python sample.py --checkpoint outputs/gmrf/run_xxx/check/checkpoint_best.pt --mode conditional

    # Inpainting (preserve specific components, generate the rest)
    python sample.py --checkpoint outputs/gmrf/run_xxx/check/checkpoint_best.pt --mode inpainting --components group_nc group_km
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add src to path
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from torch.utils.data import DataLoader

from datasets.continuous import MultiComponentDataset
from models.gmrf import Epure_GMRF_MVAE
from utils.config import resolve_path


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_model(checkpoint_path, config, device):
    """Load trained GMRF MVAE model."""
    model_cfg = config['model']
    data_cfg = config['data']

    # Capture device string for class definition
    device_str = str(device)
    
    class Params:
        num_components = len(data_cfg['component_dirs'])
        latent_dim = model_cfg['latent_dim']
        diagonal_transf = model_cfg.get('diagonal_transf', 'softplus')
        hidden_dim = model_cfg['hidden_dim']
        n_layers = model_cfg['n_layers']
        nf = model_cfg['nf']
        nf_max_e = model_cfg.get('nf_max_e', 512)
        nf_max_d = model_cfg.get('nf_max_d', 256)
        cond_dim = model_cfg.get('cond_dim', 0)
        image_size = tuple(model_cfg.get('image_size', [64, 32]))
        reduced_diag = model_cfg.get('reduced_diag', False)
        device = device_str
        component_names = data_cfg['component_dirs']

    model = Epure_GMRF_MVAE(Params()).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def save_component_images(samples, out_root, prefix, component_names):
    """
    Save component images.

    Args:
        samples: [B, C, H, W] tensor or list of [B, 1, H, W] tensors
        out_root: Output root directory
        prefix: Filename prefix
        component_names: List of component names
    """
    # Handle list of tensors
    if isinstance(samples, list):
        samples = torch.cat(samples, dim=1)  # -> [B, C, H, W]

    B = samples.size(0)

    for i in range(B):
        sample = samples[i]  # [C, H, W]

        # Save full image (sum all components)
        full_img = torch.clamp(sample.sum(dim=0), 0, 1)
        full_img_np = full_img.cpu().numpy()
        full_img_uint8 = (full_img_np * 255).astype(np.uint8)
        full_path = os.path.join(out_root, "full", f"{prefix}_{i:04d}_full.png")
        Image.fromarray(full_img_uint8, mode='L').save(full_path)

        # Save individual components
        for j, comp_name in enumerate(component_names):
            comp_img = torch.clamp(sample[j], 0, 1)
            comp_img_np = comp_img.cpu().numpy()
            comp_img_uint8 = (comp_img_np * 255).astype(np.uint8)
            comp_path = os.path.join(out_root, comp_name, f"{prefix}_{i:04d}_{comp_name}.png")
            Image.fromarray(comp_img_uint8, mode='L').save(comp_path)


def create_test_loader(config, batch_size):
    """Create test data loader from config."""
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
        stacked=False,
        normalized=data_cfg.get('normalized', False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return test_loader


def sample_unconditional(model, num_samples, save_root, date_str, component_names, device, batch_size=64):
    """
    Mode 1: Unconditional sampling from GMRF prior.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)

    print(f"\nGenerating {num_samples} unconditional samples from GMRF prior...")

    n_done = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Sampling"):
            batch_sz = min(batch_size, num_samples - n_done)

            # Generate from prior
            generations = model.generate_for_calculating_unconditional_coherence(batch_sz)

            # Stack into [B, C, H, W]
            samples = torch.cat(generations, dim=1)

            # Save samples
            prefix = f"uncond"
            for j in range(batch_sz):
                save_component_images(
                    samples[j:j+1],
                    out_root,
                    f"uncond_{n_done + j:04d}",
                    component_names
                )

            n_done += batch_sz

    print(f"\nDone - {n_done} unconditional samples saved under {out_root}")


def sample_conditional(model, config, save_root, date_str, component_names, device, batch_size=64):
    """
    Mode 2: Conditional reconstruction - encode real data and decode.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)

    test_loader = create_test_loader(config, batch_size)
    actual_num_samples = len(test_loader.dataset)

    print(f"\nGenerating {actual_num_samples} conditional reconstructions...")

    n_done = 0

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Sampling"):
            data_tuple, cond = batch_data

            # Convert to list of tensors
            if isinstance(data_tuple, torch.Tensor):
                data = [data_tuple[:, i:i+1].to(device) for i in range(data_tuple.size(1))]
            else:
                data = [d.to(device) for d in data_tuple]

            cond = cond.to(device).float() if cond is not None and model.cond_dim > 0 else None

            B = data[0].size(0)

            # Forward pass
            model(data, cond=cond)

            # Get reconstructions
            recons = model.recons
            samples = torch.cat(recons, dim=1)

            # Save samples
            for j in range(B):
                save_component_images(
                    samples[j:j+1],
                    out_root,
                    f"cond_{n_done + j:04d}",
                    component_names
                )

            n_done += B

    print(f"\nDone - {n_done} conditional samples saved under {out_root}")


def sample_inpainting(model, config, save_root, date_str, component_names, components_to_preserve, device, batch_size=64):
    """
    Mode 3: Inpainting using Gaussian conditional.
    """
    out_root = os.path.join(save_root, date_str)

    component_to_idx = {name: idx for idx, name in enumerate(component_names)}
    preserve_indices = [component_to_idx[c] for c in components_to_preserve]

    # Create output folders
    for preserve_comp in components_to_preserve:
        preserve_dir = os.path.join(out_root, preserve_comp)
        os.makedirs(os.path.join(preserve_dir, "full"), exist_ok=True)
        for comp_name in component_names:
            os.makedirs(os.path.join(preserve_dir, comp_name), exist_ok=True)

    print(f"\nCross-modal generation using Gaussian conditional")
    print(f"Preserving components: {components_to_preserve}")

    test_loader = create_test_loader(config, batch_size)
    num_modalities = len(component_names)

    counters = {comp: 0 for comp in components_to_preserve}

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Inpainting"):
            data_tuple, cond = batch_data

            if isinstance(data_tuple, torch.Tensor):
                data = [data_tuple[:, i:i+1].to(device) for i in range(data_tuple.size(1))]
            else:
                data = [d.to(device) for d in data_tuple]

            cond = cond.to(device).float() if cond is not None and model.cond_dim > 0 else None

            B = data[0].size(0)

            for preserve_comp, preserve_idx in zip(components_to_preserve, preserve_indices):
                cond_image = data[preserve_idx]

                recons = []
                for target_idx in range(num_modalities):
                    recon_m = model.conditional_generate(
                        cond_image,
                        idx_i=target_idx,
                        idx_cond=preserve_idx,
                        cond=cond
                    )
                    if recon_m.dim() == 5:
                        recon_m = recon_m.squeeze(0)
                    recons.append(recon_m)

                recon = torch.cat(recons, dim=1)

                preserve_dir = os.path.join(out_root, preserve_comp)
                for n in range(B):
                    save_component_images(
                        recon[n:n+1],
                        preserve_dir,
                        f"inpaint_{counters[preserve_comp]:04d}",
                        component_names
                    )
                    counters[preserve_comp] += 1

    print(f"\nDone - inpainting samples saved under {out_root}")


def main():
    parser = argparse.ArgumentParser(description="Sample from GMRF MVAE")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, help='Path to config (if not in checkpoint dir)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['unconditional', 'conditional', 'inpainting'],
                       help='Sampling mode')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples (for unconditional mode)')
    parser.add_argument('--components', type=str, nargs='+', default=None,
                       help='Components to preserve (for inpainting mode)')
    parser.add_argument('--batch_sz', type=int, default=64, help='Batch size')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.is_dir():
        run_dir = checkpoint_path
        checkpoint_file = run_dir / 'check' / 'checkpoint_best.pt'
        if not checkpoint_file.exists():
            print(f"ERROR: Checkpoint not found at {checkpoint_file}")
            return
        checkpoint_path = checkpoint_file
    else:
        run_dir = checkpoint_path.parent.parent

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = run_dir / 'config.yaml'

    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    model = load_model(checkpoint_path, config, device)

    component_names = config['data']['component_dirs']
    checkpoint_dir = checkpoint_path.parent
    date_str = checkpoint_dir.parent.name if checkpoint_dir.name == 'check' else checkpoint_dir.name

    if args.output_dir:
        save_root = args.output_dir
    else:
        save_root = os.path.join(config['paths'].get('samples_dir', 'samples/gmrf'), args.mode)

    if args.mode == 'unconditional':
        sample_unconditional(
            model, args.num_samples, save_root, date_str,
            component_names, device, args.batch_sz
        )

    elif args.mode == 'conditional':
        sample_conditional(
            model, config, save_root, date_str,
            component_names, device, args.batch_sz
        )

    elif args.mode == 'inpainting':
        if args.components is None or len(args.components) == 0:
            print("ERROR: --components required for inpainting mode")
            print(f"Available components: {component_names}")
            return

        invalid = [c for c in args.components if c not in component_names]
        if invalid:
            print(f"ERROR: Invalid components: {invalid}")
            print(f"Available components: {component_names}")
            return

        sample_inpainting(
            model, config, save_root, date_str,
            component_names, args.components, device, args.batch_sz
        )

    print(f"\nDone! Samples saved to: {save_root}/{date_str}")


if __name__ == '__main__':
    main()
