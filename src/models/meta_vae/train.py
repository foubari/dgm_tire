#!/usr/bin/env python3
"""
Training script for Meta VAE.

2-Stage Training:
1. Stage 1: Train marginal VAEs for each component (unconditional)
2. Stage 2: Train Meta-VAE with frozen marginal decoders

Usage:
    python src_new/models/meta_vae/train.py --config configs/meta_vae_epure_full.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src_new to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.continuous import MultiComponentDataset
from models.meta_vae.model import MetaVAE, MetaEncoder, MetaDecoder, MarginalDecoder


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """Resolve a path string to an absolute Path."""
    path = Path(path_str)

    if path.is_absolute() and path.exists():
        return path

    if path.is_absolute():
        path_str_clean = str(path).lstrip('/')
        if base_dir is None:
            _THIS_FILE = Path(__file__).resolve()
            base_dir = _THIS_FILE.parent.parent.parent.parent
        potential_path = base_dir / path_str_clean
        if potential_path.exists():
            return potential_path.resolve()

    if base_dir is None:
        _THIS_FILE = Path(__file__).resolve()
        base_dir = _THIS_FILE.parent.parent.parent.parent

    if path.exists():
        return path.resolve()

    potential_path = base_dir / path
    if potential_path.exists():
        return potential_path.resolve()

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


def train_marginal_vae(
    component_idx: int,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path
) -> MarginalDecoder:
    """
    Stage 1: Train a marginal VAE for a single component.

    Returns the trained decoder.
    """
    latent_dim = config['model']['component_latent_dim']
    epochs = config['training'].get('marginal_epochs', 50)
    lr = config['training'].get('marginal_lr', 0.001)
    beta = config['model'].get('marginal_beta', 1.0)

    print(f"\n{'='*60}")
    print(f"Training Marginal VAE for Component {component_idx}")
    print(f"{'='*60}\n")

    # Simple encoder for single component
    class MarginalEncoder(nn.Module):
        def __init__(self, latent_dim: int = 4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=4, stride=2, padding=1),  # 64x32 → 32x16
                nn.BatchNorm2d(24),
                nn.LeakyReLU(0.2),
                nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),  # 32x16 → 16x8
                nn.BatchNorm2d(48),
                nn.LeakyReLU(0.2),
                nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1),  # 16x8 → 8x4
                nn.BatchNorm2d(96),
                nn.LeakyReLU(0.2),
                nn.Conv2d(96, 128, kernel_size=4, stride=2, padding=1),  # 8x4 → 4x2
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Flatten()
            )
            self.fc_mu = nn.Linear(128 * 4 * 2, latent_dim)
            self.fc_logvar = nn.Linear(128 * 4 * 2, latent_dim)

        def forward(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

    encoder = MarginalEncoder(latent_dim).to(device)
    decoder = MarginalDecoder(latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr
    )

    best_loss = float('inf')

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x, cond = batch
            x = x.to(device)  # [B, C, 64, 32]
            component = x[:, component_idx:component_idx+1]  # [B, 1, 64, 32]

            # Forward pass
            mu, logvar = encoder(component)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            recon = decoder(z)

            # Loss
            recon_loss = nn.functional.mse_loss(recon, component, reduction='sum') / x.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + beta * kl_loss

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
                'kl': kl_loss.item()
            })

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = output_dir / f"marginal_decoder_{component_idx}.pt"
            torch.save(decoder.state_dict(), checkpoint_path)
            print(f"  Saved best decoder to {checkpoint_path}")

    # Load best decoder
    decoder.load_state_dict(torch.load(output_dir / f"marginal_decoder_{component_idx}.pt"))

    # Free GPU memory (encoder and optimizer no longer needed)
    del encoder
    del optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return decoder


def train_meta_vae(
    marginal_decoders: list,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path
):
    """
    Stage 2: Train Meta-VAE with frozen marginal decoders.
    """
    latent_dim = config['model']['latent_dim']
    component_latent_dim = config['model']['component_latent_dim']
    beta = config['model']['beta']
    epochs = config['training']['epochs']
    lr = config['training']['lr']

    # Conditioning setup
    use_conditioning = config['data'].get('normalized', False)
    cond_dim = len(config['data'].get('condition_columns', [])) if use_conditioning else 0

    print(f"\n{'='*60}")
    print(f"Training Meta-VAE (Stage 2)")
    print(f"  Meta latent dim: {latent_dim}")
    print(f"  Component latent dim: {component_latent_dim}")
    print(f"  Conditioning dim: {cond_dim}")
    print(f"  Beta: {beta}")
    print(f"{'='*60}\n")

    # Create Meta-VAE
    model = MetaVAE(
        marginal_decoders=marginal_decoders,
        latent_dim=latent_dim,
        component_latent_dim=component_latent_dim,
        beta=beta,
        cond_dim=cond_dim
    ).to(device)

    # Count parameters and add to config
    num_params = sum(p.numel() for p in model.parameters())
    config['model']['num_parameters'] = num_params
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Progressive masking schedule
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

            # Create random mask
            mask = torch.ones(B, C, device=device)
            if mask_prob > 0:
                for i in range(B):
                    if torch.rand(1).item() < mask_prob:
                        # Keep ONE random component
                        keep_idx = torch.randint(0, C, (1,)).item()
                        mask[i] = 0
                        mask[i, keep_idx] = 1

            # Apply mask to input
            masked_x = x * mask.view(B, C, 1, 1)

            # Forward pass
            recon, mu, logvar, z_meta, z_components = model(masked_x, mask, cond)

            # Loss
            loss, loss_dict = model.loss_function(recon, x, mu, logvar, mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss_dict['total']
            total_recon += loss_dict['reconstruction']
            total_kl += loss_dict['kl']

            pbar.set_postfix({
                'loss': loss_dict['total'],
                'recon': loss_dict['reconstruction'],
                'kl': loss_dict['kl'],
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
    parser = argparse.ArgumentParser(description="Train Meta VAE")
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
    output_dir = Path(config['paths']['output_dir']) / f'{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Resolve paths
    root_dir = resolve_path(config['data']['root_dir'])
    condition_csv = resolve_path(config['data']['condition_csv'])

    # Add train subdirectory
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

    # Stage 1: Train marginal VAEs
    print("\n" + "="*60)
    print("STAGE 1: Training Marginal VAEs")
    print("="*60)

    marginal_decoders = []
    num_components = len(config['data']['component_dirs'])

    for i in range(num_components):
        decoder = train_marginal_vae(i, train_loader, config, device, output_dir)
        marginal_decoders.append(decoder)

    # Stage 2: Train Meta-VAE
    print("\n" + "="*60)
    print("STAGE 2: Training Meta-VAE")
    print("="*60)

    train_meta_vae(marginal_decoders, train_loader, config, device, output_dir)

    print(f"\nAll done! Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
