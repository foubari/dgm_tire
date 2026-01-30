#!/usr/bin/env python3
"""
Sampling script for GMRF MVAE - ICTAI implementation.

3 Sampling Modes:
1. Unconditional: Sample from prior (no conditioning)
2. Conditional: Sample with specific conditions
3. Inpainting/Cross-modal: Generate missing components using Gaussian conditional

Usage:
    # Unconditional sampling
    python src/models/gmrf_mvae/sample.py --checkpoint outputs/gmrf_mvae/run_xxx/check/checkpoint_best.pt --mode unconditional --num_samples 100

    # Conditional sampling
    python src/models/gmrf_mvae/sample.py --checkpoint outputs/gmrf_mvae/run_xxx/check/checkpoint_best.pt --mode conditional --num_samples 100

    # Inpainting (preserve specific components, generate the rest)
    python src/models/gmrf_mvae/sample.py --checkpoint outputs/gmrf_mvae/run_xxx/check/checkpoint_best.pt --mode inpainting --components group_nc group_km
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.gmrf_mvae.model import Epure_GMMVAE
from datasets.continuous import MultiComponentDataset


class Params:
    """Parameter container matching ICTAI implementation."""

    def __init__(self, params_dict):
        self.latent_dim = params_dict.get('latent_dim', 4)
        self.diagonal_transf = params_dict.get('diagonal_transf', 'softplus')
        self.hidden_dim = params_dict.get('hidden_dim', 128)
        self.n_layers = params_dict.get('n_layers', 2)
        self.device = params_dict.get('device', 'cuda')
        self.reduced_diag = params_dict.get('reduced_diag', False)
        self.nf = params_dict.get('nf', 32)
        self.nf_max_e = params_dict.get('nf_max_e', 512)
        self.nf_max_d = params_dict.get('nf_max_d', 256)


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """
    Resolve a path string to an absolute Path.
    Handles relative paths and paths relative to project root.
    """
    path = Path(path_str)

    # If already absolute and exists, return it
    if path.is_absolute() and path.exists():
        return path

    # If absolute but doesn't exist, try to resolve from project root
    if path.is_absolute():
        # Remove leading slash for Unix-style paths
        path_str_clean = str(path).lstrip('/')
        if base_dir is None:
            _THIS_FILE = Path(__file__).resolve()
            base_dir = _THIS_FILE.parent.parent.parent.parent  # Go to project root
        potential_path = base_dir / path_str_clean
        if potential_path.exists():
            return potential_path.resolve()

    # Handle relative paths
    if base_dir is None:
        _THIS_FILE = Path(__file__).resolve()
        base_dir = _THIS_FILE.parent.parent.parent.parent  # Go to project root

    # Try relative to current working directory first
    if path.exists():
        return path.resolve()

    # Try relative to project root
    potential_path = base_dir / path
    if potential_path.exists():
        return potential_path.resolve()

    # If still not found, return the resolved path anyway (will raise error later)
    return (base_dir / path).resolve()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def load_model(checkpoint_path: Path, config: dict, device: torch.device) -> Epure_GMMVAE:
    """Load trained GMRF MVAE model (ICTAI implementation)."""
    # Create params object (ICTAI style)
    params_dict = {
        'latent_dim': config['model']['latent_dim'],
        'diagonal_transf': config['model'].get('diagonal_transf', 'softplus'),
        'hidden_dim': config['model']['hidden_dim'],
        'n_layers': config['model']['n_layers'],
        'device': str(device),
        'reduced_diag': config['model'].get('reduced_diag', False),
        'nf': config['model']['nf'],
        'nf_max_e': config['model'].get('nf_max_e', 512),
        'nf_max_d': config['model'].get('nf_max_d', 256),
    }
    params = Params(params_dict)

    # Create model (ICTAI style - 5 modalities for epure)
    model = Epure_GMMVAE(params).to(device)

    # Load checkpoint
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
    Save component images following DDPM structure.

    Args:
        samples: [B, C, H, W] tensor
        out_root: Output root directory
        prefix: Filename prefix
        component_names: List of component names
    """
    B = samples.size(0)

    for i in range(B):
        sample = samples[i]  # [C, H, W]

        # Save full image (sum all components)
        full_img = torch.clamp(sample.sum(dim=0), 0, 1)  # [H, W]
        full_img_np = full_img.cpu().numpy()
        if full_img_np.ndim > 2:
            full_img_np = full_img_np.squeeze()
        full_img_uint8 = (full_img_np * 255).astype(np.uint8)
        full_path = os.path.join(out_root, "full", f"{prefix}_{i:02d}_full.png")
        Image.fromarray(full_img_uint8, mode='L').save(full_path)

        # Save individual components
        for j, comp_name in enumerate(component_names):
            comp_img = torch.clamp(sample[j], 0, 1)  # [H, W]
            comp_img_np = comp_img.cpu().numpy()
            if comp_img_np.ndim > 2:
                comp_img_np = comp_img_np.squeeze()
            comp_img_uint8 = (comp_img_np * 255).astype(np.uint8)
            comp_path = os.path.join(out_root, comp_name, f"{prefix}_{i:02d}_{comp_name}.png")
            Image.fromarray(comp_img_uint8, mode='L').save(comp_path)


def sample_unconditional(
    model: Epure_GMMVAE,
    num_samples: int,
    save_root: str,
    date_str: str,
    component_names: list,
    device: torch.device,
    config: dict,
    batch_size: int = 64
):
    """
    Mode 1: Unconditional sampling from GMRF prior.

    Uses the ICTAI-style generate_for_calculating_unconditional_coherence method.
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

            # Use ICTAI-style unconditional generation
            # Returns list of [B, 1, H, W] tensors, one per modality
            generations = model.generate_for_calculating_unconditional_coherence(batch_sz)

            # Stack into [B, C, H, W]
            samples = torch.cat(generations, dim=1)

            # Save samples
            prefix = f"img{n_done:04d}"
            save_component_images(samples, out_root, prefix, component_names)

            n_done += batch_sz

    print(f"\nDone – {n_done} unconditional samples saved under {out_root}")


def sample_conditional(
    model: Epure_GMMVAE,
    config: dict,
    num_samples: int,
    save_root: str,
    date_str: str,
    component_names: list,
    device: torch.device,
    batch_size: int = 64
):
    """
    Mode 2: Conditional reconstruction - encode real data and decode.

    For ICTAI-style GMRF MVAE, this encodes the test images through the
    VAE encoders and decodes the samples from the posterior.
    """
    out_root = os.path.join(save_root, date_str)
    os.makedirs(os.path.join(out_root, "full"), exist_ok=True)
    for cname in component_names:
        os.makedirs(os.path.join(out_root, cname), exist_ok=True)

    # Create test loader to use real test set
    test_loader = create_test_loader(config, batch_size)
    actual_num_samples = len(test_loader.dataset)

    print(f"\nGenerating {actual_num_samples} conditional reconstructions (test set size)...")

    n_done = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Sampling")):
            x, cond = batch_data
            x = x.to(device)  # [B, C, H, W]

            B, C = x.size(0), x.size(1)

            # Convert stacked tensor to list of tensors (ICTAI format)
            data = [x[:, i:i+1] for i in range(C)]

            # Forward pass (ICTAI style - encodes and decodes)
            model(data)

            # Get reconstructions from model (stored after forward pass)
            # recons is a list of [B, 1, H, W] tensors
            recons = model.recons

            # Stack into [B, C, H, W]
            samples = torch.cat(recons, dim=1)

            # Save samples
            prefix = f"img{n_done:04d}"
            save_component_images(samples, out_root, prefix, component_names)

            n_done += B

    print(f"\nDone – {n_done} conditional samples saved under {out_root}")


def sample_inpainting(
    model: Epure_GMMVAE,
    test_loader,
    save_root: str,
    date_str: str,
    component_names: list,
    components_to_preserve: list,
    device: torch.device,
    verbose: bool = True
):
    """
    Mode 3: Inpainting using ICTAI-style Gaussian conditional.

    Uses the GMRF structure to sample missing modalities conditioned on
    observed modalities via Gaussian conditional distribution.

    Structure: samples/<model>/<date>/<preserved_component>/full/
                                                            /<all_components>/
    """
    out_root = os.path.join(save_root, date_str)

    # Create component to index mapping
    component_to_idx = {name: idx for idx, name in enumerate(component_names)}
    preserve_indices = [component_to_idx[c] for c in components_to_preserve]

    # Create output folders - same structure as DDPM/MM-VAE+
    for preserve_comp in components_to_preserve:
        preserve_dir = os.path.join(out_root, preserve_comp)
        os.makedirs(os.path.join(preserve_dir, "full"), exist_ok=True)
        for comp_name in component_names:
            os.makedirs(os.path.join(preserve_dir, comp_name), exist_ok=True)

    if verbose:
        print(f"\nCross-modal generation using Gaussian conditional (ICTAI style)")
        print(f"Preserving components: {components_to_preserve}")
        print(f"Processing {len(test_loader)} batches")

    # Global counter for each preserved component
    counters = {comp: 0 for comp in components_to_preserve}
    latent_dim = model.latent_dim
    num_modalities = len(component_names)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Inpainting")):
            x, cond = batch_data
            x = x.to(device)  # [B, C, H, W]

            B, C, H, W = x.shape

            # For each preserved component
            for preserve_comp, preserve_idx in zip(components_to_preserve, preserve_indices):
                # Convert to list format for ICTAI model
                data = [x[:, i:i+1] for i in range(C)]

                # Encode only the observed modality
                observed_data = [data[preserve_idx]]

                # Get observed modality VAE
                vae = model.modality_vaes[preserve_idx]
                mu, diag = vae.enc(observed_data[0])
                z_obs = mu  # Use mean for deterministic encoding

                # Build observed indices for latent space
                # Each modality has latent_dim dimensions
                observed_indices = list(range(
                    preserve_idx * latent_dim,
                    (preserve_idx + 1) * latent_dim
                ))

                # Use Gaussian conditional to generate missing modalities
                # conditional_generate returns samples for ALL dimensions
                z_full = model.conditional_generate(z_obs, observed_indices)  # [B, total_latent_dim]

                # Split z_full into per-modality latents and decode
                recons = []
                for m in range(num_modalities):
                    z_m = z_full[:, m * latent_dim:(m + 1) * latent_dim]
                    recon_m = model.modality_vaes[m].dec(z_m)
                    recons.append(recon_m)

                # Stack reconstructions: List of [B, 1, H, W] -> [B, C, H, W]
                recon = torch.cat(recons, dim=1)  # [B, C, H, W]

                # Save results in preserve_comp subfolder
                preserve_dir = os.path.join(out_root, preserve_comp)
                for n in range(B):
                    prefix = f"img{counters[preserve_comp]:04d}"
                    sample = recon[n:n+1]  # [1, C, H, W]
                    save_component_images(sample, preserve_dir, prefix, component_names)
                    counters[preserve_comp] += 1

    if verbose:
        print(f"\nDone – inpainting samples saved under {out_root}")


def create_test_loader(config, batch_size):
    """Create test data loader from config."""
    # Resolve paths
    root_dir = resolve_path(config['data']['root_dir'])
    condition_csv = resolve_path(config['data']['condition_csv'])

    # Add test subdirectory
    test_root_dir = root_dir / 'test'

    # Create test dataset
    test_dataset = MultiComponentDataset(
        root_dir=test_root_dir,
        condition_csv=condition_csv,
        component_dirs=config['data']['component_dirs'],
        condition_columns=config['data'].get('condition_columns', []),
        prefix_column=config['data']['prefix_column'],
        filename_pattern=config['data']['filename_pattern'],
        split_column=config['data']['split_column'],
        split='test',
        normalized=config['data'].get('normalized', False)
    )

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return test_loader


def main():
    parser = argparse.ArgumentParser(description="Sample from GMRF MVAE")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, help='Path to config (if not in checkpoint dir)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['unconditional', 'conditional', 'inpainting'],
                       help='Sampling mode')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate (for unconditional/conditional)')
    parser.add_argument('--components', type=str, nargs='+', default=None,
                       help='Components to preserve (for inpainting mode), e.g., --components group_nc group_km')
    parser.add_argument('--batch_sz', type=int, default=64,
                       help='Batch size for sampling')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (default: samples/<model>/<mode>)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    checkpoint_path = Path(args.checkpoint)

    # Handle both directory and .pt file paths
    if checkpoint_path.is_dir():
        # If directory provided, assume it's the run directory
        run_dir = checkpoint_path
        # Look for checkpoint_best.pt in check subdirectory
        checkpoint_file = run_dir / 'check' / 'checkpoint_best.pt'
        if not checkpoint_file.exists():
            print(f"ERROR: Checkpoint not found at {checkpoint_file}")
            return
        checkpoint_path = checkpoint_file
    else:
        # If .pt file provided, get run directory
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

    # Get component names
    component_names = config['data']['component_dirs']

    # Extract date string from checkpoint path
    # If checkpoint is in check/ subdirectory, use parent.parent (timestamp dir)
    # Otherwise use parent directly
    checkpoint_dir = checkpoint_path.parent
    date_str = checkpoint_dir.parent.name if checkpoint_dir.name == 'check' else checkpoint_dir.name

    # Determine output directory
    if args.output_dir:
        save_root = args.output_dir
    else:
        save_root = os.path.join(config['paths'].get('samples_dir', 'samples/gmrf_mvae'), args.mode)

    # Sample based on mode
    if args.mode == 'unconditional':
        sample_unconditional(
            model, args.num_samples, save_root, date_str,
            component_names, device, config, args.batch_sz
        )

    elif args.mode == 'conditional':
        if not config['data'].get('normalized', False):
            print("ERROR: Conditional sampling requires normalized conditions in config")
            return
        sample_conditional(
            model, config, args.num_samples, save_root, date_str,
            component_names, device, args.batch_sz
        )

    elif args.mode == 'inpainting':
        if args.components is None or len(args.components) == 0:
            print("ERROR: --components required for inpainting mode")
            print(f"Available components: {component_names}")
            return

        # Validate component names
        invalid = [c for c in args.components if c not in component_names]
        if invalid:
            print(f"ERROR: Invalid components: {invalid}")
            print(f"Available components: {component_names}")
            return

        # Create test loader
        test_loader = create_test_loader(config, args.batch_sz)

        # Run inpainting
        sample_inpainting(
            model, test_loader, save_root, date_str,
            component_names, args.components, device, verbose=True
        )

    print(f"\nDone! Samples saved to: {save_root}/{date_str}")


if __name__ == '__main__':
    main()
