#!/usr/bin/env python3
"""
Training script for Beta-VAE.

Usage:
    python -m models.vae.train --config configs/vae_epure_full.yaml
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.vae.model import BetaVAE
from datasets.continuous import MultiComponentDataset
from utils.config import load_config


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """Resolve a path string to an absolute Path."""
    path = Path(path_str)

    if path.is_absolute() and path.exists():
        return path

    # Try relative to base_dir if provided
    if base_dir:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate

    # Try relative to current working directory
    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate.resolve()

    # Try relative to project root (3 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    candidate = project_root / path
    if candidate.exists():
        return candidate.resolve()

    # Return as-is if nothing works
    return path.resolve()


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


def compute_mask_prob(epoch, warmup_epochs=20, max_mask_prob=0.7):
    """Progressive masking schedule."""
    if epoch < warmup_epochs:
        return ((epoch / warmup_epochs) ** 0.5) * max_mask_prob
    return max_mask_prob


def train_epoch(model, loader, optimizer, epoch, config, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    mask_prob = compute_mask_prob(
        epoch,
        config['training'].get('warmup_epochs', 20),
        config['training'].get('max_mask_prob', 0.7)
    )

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, cond) in enumerate(pbar):
        x, cond = x.to(device), cond.to(device)

        # Forward pass with masking
        recon, mu, logvar = model(x, cond, mask_prob=mask_prob)

        # Compute loss
        loss, recon_loss, kl_loss = model.loss_function(recon, x, mu, logvar)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        if batch_idx % 50 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'mask_p': f'{mask_prob:.2f}'
            })

    n = len(loader)
    return total_loss/n, total_recon/n, total_kl/n


@torch.no_grad()
def eval_epoch(model, loader, device):
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for x, cond in loader:
        x, cond = x.to(device), cond.to(device)

        recon, mu, logvar = model(x, cond, mask_prob=0.0)
        loss, recon_loss, kl_loss = model.loss_function(recon, x, mu, logvar)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(loader)
    return total_loss/n, total_recon/n, total_kl/n


def train(config):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(config['training']['seed'])

    # Data - Extract config and create datasets properly
    data_cfg = config['data']
    root_dir = resolve_path(data_cfg['root_dir'])
    condition_csv = resolve_path(data_cfg['condition_csv'])
    component_dirs = data_cfg['component_dirs']

    train_dataset = MultiComponentDataset(
        root_dir=root_dir / "train",
        component_dirs=component_dirs,
        condition_csv=condition_csv,
        condition_columns=data_cfg['condition_columns'],
        prefix_column=data_cfg.get('prefix_column', 'matching'),
        filename_pattern=data_cfg.get('filename_pattern', '{prefix}_{component}.png'),
        split='train',
        split_column=data_cfg.get('split_column', 'train'),
        stacked=True,
        normalized=data_cfg.get('normalized', False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        drop_last=True
    )

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )

    # Model - Filter out 'type' from config (used by pipeline, not by model)
    model_cfg = {k: v for k, v in config['model'].items() if k != 'type'}
    model = BetaVAE(**model_cfg).to(device)

    # Count parameters and add to config
    num_params = sum(p.numel() for p in model.parameters())
    config['model']['num_parameters'] = num_params
    print(f"Model parameters: {num_params/1e6:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )

    # Training loop - Create timestamped output directory
    base_output_dir = Path(config['paths']['output_dir'])
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint subdirectory
    checkpoint_dir = output_dir / "check"
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config with num_parameters
    import yaml
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"Output directory: {output_dir}")

    best_loss = float('inf')

    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, epoch, config, device
        )

        # Eval
        if epoch % config['training'].get('eval_every', 10) == 0:
            test_loss, test_recon, test_kl = eval_epoch(model, test_loader, device)
            print(f"\nEpoch {epoch} - Test Loss: {test_loss:.4f} (Recon: {test_recon:.4f}, KL: {test_kl:.4f})")

            # Save best
            if test_loss < best_loss:
                best_loss = test_loss
                save_checkpoint(
                    model, optimizer, epoch, test_loss,
                    checkpoint_dir / 'checkpoint_best.pt'
                )
                print(f"Saved best model (loss: {best_loss:.4f})")

        # Checkpoint
        if epoch % config['training'].get('check_every', 50) == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                checkpoint_dir / f'checkpoint_{epoch}.pt'
            )

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Override seed if provided
    if args.seed is not None:
        config['training']['seed'] = args.seed

    train(config)


if __name__ == '__main__':
    main()
