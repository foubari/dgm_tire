#!/usr/bin/env python3
"""
Training script for GMRF MVAE.

Usage:
    python src_new/models/gmrf_mvae/train.py --config configs/gmrf_mvae_epure_full.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src_new to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.continuous import MultiComponentDataset
from models.gmrf_mvae.model import GMRF_MVAE


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


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


def train(
    model: GMRF_MVAE,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path
):
    """Train GMRF MVAE."""
    epochs = config['training']['epochs']
    lr = config['training']['lr']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Progressive masking
    warmup_epochs = config['training'].get('mask_warmup_epochs', 20)
    max_mask_prob = config['training'].get('max_mask_prob', 0.7)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        # Compute mask probability
        if epoch < warmup_epochs:
            mask_prob = ((epoch / warmup_epochs) ** 0.5) * max_mask_prob
        else:
            mask_prob = max_mask_prob

        total_loss = 0
        total_recon = 0
        total_kl = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x, cond = batch
            x = x.to(device)  # [B, C, 64, 32]
            if cond is not None:
                cond = cond.to(device)

            B, C = x.size(0), x.size(1)

            # Create random mask for component-level masking
            mask = torch.ones(B, C, device=device)
            if mask_prob > 0:
                for i in range(B):
                    if torch.rand(1).item() < mask_prob:
                        # Keep ONE random component, zero others
                        keep_idx = torch.randint(0, C, (1,)).item()
                        mask[i] = 0
                        mask[i, keep_idx] = 1

            # Apply mask to input
            masked_x = x * mask.view(B, C, 1, 1)

            # Forward pass
            recon, mu_list, logvar_list = model(masked_x, cond)

            # Loss
            loss, recon_loss, kl_loss = model.loss_function(recon, x, mu_list, logvar_list)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'mask_p': mask_prob
            })

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}), Mask prob: {mask_prob:.3f}")

        # Save checkpoint
        if (epoch + 1) % config['training'].get('check_every', 10) == 0:
            checkpoint_path = output_dir / 'check' / f'checkpoint_{epoch+1}.pt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / 'check' / 'checkpoint_best.pt'
            best_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, epoch, avg_loss, best_path)
            print(f"  Saved best model to {best_path}")

    print(f"\n{'='*60}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train GMRF MVAE")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    
    # Handle Unix-style absolute paths (e.g., /src_new/configs/...)
    if config_path.is_absolute() and not config_path.exists():
        # Try to resolve relative to src_new directory
        _THIS_FILE = Path(__file__).resolve()
        _SRC_NEW_DIR = _THIS_FILE.parent.parent.parent
        # Remove leading slash and try relative path
        relative_path = str(config_path).lstrip('/')
        potential_path = _SRC_NEW_DIR / relative_path
        if potential_path.exists():
            config_path = potential_path
        else:
            # Try just the filename in configs directory
            potential_path = _SRC_NEW_DIR / "configs" / config_path.name
            if potential_path.exists():
                config_path = potential_path
    
    # If still relative, try to resolve
    if not config_path.is_absolute():
        if config_path.exists():
            config_path = config_path.resolve()
        else:
            # Try relative to src_new directory
            _THIS_FILE = Path(__file__).resolve()
            _SRC_NEW_DIR = _THIS_FILE.parent.parent.parent
            potential_path = _SRC_NEW_DIR / config_path
            if potential_path.exists():
                config_path = potential_path.resolve()
            else:
                # Try configs subdirectory
                potential_path = _SRC_NEW_DIR / "configs" / config_path.name
                if potential_path.exists():
                    config_path = potential_path.resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config} (resolved to: {config_path})")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = args.seed if args.seed is not None else config['training'].get('seed', 42)
    set_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(config['paths']['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Resolve paths
    root_dir = resolve_path(config['data']['root_dir'])
    condition_csv = resolve_path(config['data']['condition_csv'])

    # Add train subdirectory to root_dir (data structure: root_dir/train/comp_dir)
    train_root_dir = root_dir / 'train'

    # Create dataset
    train_dataset = MultiComponentDataset(
        root_dir=train_root_dir,
        condition_csv=condition_csv,
        component_dirs=config['data']['component_dirs'],
        condition_columns=config['data'].get('condition_columns', []),
        prefix_column=config['data']['prefix_column'],
        filename_pattern=config['data']['filename_pattern'],
        split_column=config['data']['split_column'],
        split='train',
        normalized=config['data'].get('normalized', False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )

    print(f"Train dataset size: {len(train_dataset)}")

    # Create model
    num_components = len(config['data']['component_dirs'])
    use_conditioning = config['data'].get('normalized', False)
    cond_dim = len(config['data'].get('condition_columns', [])) if use_conditioning else 0

    model = GMRF_MVAE(
        num_components=num_components,
        latent_dim=config['model']['latent_dim'],
        nf=config['model']['nf'],
        nf_max=config['model']['nf_max'],
        hidden_dim=config['model']['hidden_dim'],
        n_layers=config['model']['n_layers'],
        beta=config['model']['beta'],
        cond_dim=cond_dim,
        dropout_p=config['model']['dropout_p']
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    train(model, train_loader, config, device, output_dir)

    print(f"\nAll done! Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
