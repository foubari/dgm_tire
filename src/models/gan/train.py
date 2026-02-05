#!/usr/bin/env python3
"""
Training script for Standard GAN (vanilla GAN with BCE loss).

Usage:
    python train.py --config configs/gan_default.yaml
    python train.py --config configs/gan_default.yaml --epochs 100 --batch_size 32
"""

import os
import sys
import platform
import time
import pickle
import argparse
from pathlib import Path

# Add src_new to path
_THIS_FILE = Path(__file__).resolve()
_SRC_NEW_DIR = _THIS_FILE.parent.parent.parent
_PROJECT_ROOT = _SRC_NEW_DIR.parent

if str(_SRC_NEW_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_NEW_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_IS_WINDOWS = platform.system() == 'Windows'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ema_pytorch import EMA

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from datasets.continuous import MultiComponentDataset
from models.gan import GAN, Encoder, Generator, Discriminator
from utils.config import load_config, auto_complete_config, validate_config, resolve_path


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_data_loaders(config, args):
    """Create train and test data loaders."""
    data_cfg = config['data']
    training_cfg = config['training']

    root_dir = resolve_path(data_cfg['root_dir'])
    condition_csv = resolve_path(data_cfg['condition_csv'])

    # Add split to root_dir (data structure: root_dir/train/comp_dir and root_dir/test/comp_dir)
    train_root_dir = root_dir / 'train'
    test_root_dir = root_dir / 'test'

    train_dataset = MultiComponentDataset(
        root_dir=train_root_dir,
        condition_csv=condition_csv,
        component_dirs=data_cfg['component_dirs'],
        condition_columns=data_cfg['condition_columns'],
        prefix_column=data_cfg['prefix_column'],
        filename_pattern=data_cfg.get('filename_pattern', '{prefix}_{component}.png'),
        split='train',
        split_column=data_cfg.get('split_column', 'train'),
        normalized=data_cfg.get('normalized', False)
    )

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

    batch_size = training_cfg['batch_size']
    num_workers = training_cfg.get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset, test_dataset


def train_epoch(model, train_loader, optimizer_g, optimizer_d, device):
    """Train Standard GAN for one epoch. 1:1 D:G update ratio."""
    model.train()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(train_loader):
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            images, cond = batch_data
            images = images.to(device)
            cond = cond.to(device).float() if cond is not None else None
        else:
            images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
            cond = None

        # ===========================
        # Update Discriminator (once)
        # ===========================
        optimizer_d.zero_grad()

        loss_dict = model(images, cond=cond, train_discriminator=True)
        d_loss = loss_dict['discriminator_loss']

        d_loss.backward()
        optimizer_d.step()

        total_d_loss += loss_dict['discriminator_loss'].item()
        total_d_real += loss_dict['d_real_mean']
        total_d_fake += loss_dict['d_fake_mean']

        # ===========================
        # Update Generator (once)
        # ===========================
        optimizer_g.zero_grad()

        loss_dict = model(images, cond=cond, train_discriminator=False)
        g_loss = loss_dict['generator_loss']

        g_loss.backward()
        optimizer_g.step()

        total_g_loss += loss_dict['generator_loss'].item()
        num_batches += 1

    return {
        'generator_loss': total_g_loss / num_batches if num_batches > 0 else 0.0,
        'discriminator_loss': total_d_loss / num_batches if num_batches > 0 else 0.0,
        'd_real_mean': total_d_real / num_batches if num_batches > 0 else 0.0,
        'd_fake_mean': total_d_fake / num_batches if num_batches > 0 else 0.0,
    }


def eval_model(model, test_loader, device):
    """Evaluate model on test set using discriminator scores."""
    model.eval()

    total_d_real = 0.0
    total_d_fake = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_data in test_loader:
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                images, cond = batch_data
                images = images.to(device)
                cond = cond.to(device).float() if cond is not None else None
            else:
                images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
                cond = None

            # Sample random latent
            batch_size = images.size(0)
            z = torch.randn(batch_size, model.latent_dim, device=device)

            # Generate fake images
            fake = model.generator(z, cond=cond)

            # Discriminator scores
            d_real = model.discriminator(images, cond=cond).mean().item()
            d_fake = model.discriminator(fake, cond=cond).mean().item()

            total_d_real += d_real
            total_d_fake += d_fake
            num_batches += 1

    avg_d_real = total_d_real / num_batches if num_batches > 0 else 0.0
    avg_d_fake = total_d_fake / num_batches if num_batches > 0 else 0.0
    # Return separation metric: higher means D can still distinguish real/fake
    return avg_d_real - avg_d_fake


def save_checkpoint(encoder, generator, ema_generator, discriminator, optimizer_g, optimizer_d, epoch, loss, check_dir):
    """Save model checkpoint."""
    check_dir = Path(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'ema_generator_state_dict': ema_generator.ema_model.state_dict() if ema_generator.ema_model is not None else None,
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }

    checkpoint_path = check_dir / f'checkpoint_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Standard GAN')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')

    # Allow overriding config values
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr_generator', type=float, default=None,
                       help='Generator learning rate (overrides config)')
    parser.add_argument('--lr_discriminator', type=float, default=None,
                       help='Discriminator learning rate (overrides config)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of data loader workers (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (default: timestamp)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu, overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')

    args = parser.parse_args()

    # Load and auto-complete config (auto-completion before validation)
    config = load_config(args.config)
    config = auto_complete_config(config)
    validate_config(config, model_type='gan')

    # Override config with command-line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr_generator is not None:
        config['training']['lr_generator'] = args.lr_generator
    if args.lr_discriminator is not None:
        config['training']['lr_discriminator'] = args.lr_discriminator
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    if args.output_dir is not None:
        config['paths']['output_dir'] = args.output_dir
    if args.device is not None:
        device_str = args.device
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        config['training']['seed'] = args.seed

    # Set seed
    seed = config['training'].get('seed', 42)
    set_seed(seed)

    # Create output directory
    output_dir = resolve_path(config['paths']['output_dir'])
    if args.name is None:
        name = time.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        name = args.name
    output_dir = output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config and args
    with open(output_dir / 'config.yaml', 'w') as f:
        import yaml
        yaml.dump(config, f)
    with open(output_dir / 'args.pickle', 'wb') as f:
        pickle.dump(args, f)

    # Setup device
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading data...")
    train_loader, test_loader, train_dataset, test_dataset = create_data_loaders(config, args)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Create models
    print("Creating Standard GAN model...")
    model_cfg = config['model']
    data_cfg = config['data']

    image_size = tuple(model_cfg['image_size'])
    channels = model_cfg['channels']
    cond_dim = model_cfg['cond_dim']
    latent_dim = model_cfg.get('total_latent_dim', 20)

    print(f"  Image size: {image_size}")
    print(f"  Channels: {channels} (components: {data_cfg['component_dirs']})")
    print(f"  Cond dim: {cond_dim} (columns: {data_cfg['condition_columns']})")
    print(f"  Latent dim: {latent_dim}")

    # Create encoder, generator, discriminator
    encoder = Encoder(in_channels=channels, latent_dim=latent_dim)
    generator = Generator(
        latent_dim=latent_dim,
        out_channels=channels,
        cond_dim=cond_dim,
        dim=model_cfg.get('dim', 64),
        dim_mults=tuple(model_cfg.get('dim_mults', [1, 2, 4]))
    )
    discriminator = Discriminator(
        in_channels=channels,
        cond_dim=cond_dim,
        use_spectral_norm=model_cfg.get('use_spectral_norm', True)
    )

    # Create GAN wrapper
    model = GAN(
        encoder=encoder,
        generator=generator,
        discriminator=discriminator,
        image_size=image_size,
        latent_dim=latent_dim,
        label_smoothing=model_cfg.get('label_smoothing', 0.1),
        cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1)
    )

    encoder = encoder.to(device)
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    num_params_encoder = count_parameters(encoder)
    num_params_generator = count_parameters(generator)
    num_params_discriminator = count_parameters(discriminator)
    total_params = num_params_encoder + num_params_generator + num_params_discriminator

    # Add num_parameters to config
    config['model']['num_parameters'] = {
        'encoder': num_params_encoder,
        'generator': num_params_generator,
        'discriminator': num_params_discriminator,
        'total': total_params
    }

    print(f"Encoder parameters: {num_params_encoder:,}")
    print(f"Generator parameters: {num_params_generator:,}")
    print(f"Discriminator parameters: {num_params_discriminator:,}")
    print(f"Total parameters: {total_params:,}")

    # Create optimizers
    training_cfg = config['training']
    optimizer_name = training_cfg.get('optimizer', 'adam')
    lr_generator = training_cfg.get('lr_generator', 0.0002)
    lr_discriminator = training_cfg.get('lr_discriminator', 0.0002)
    beta1 = training_cfg.get('beta1', 0.5)
    beta2 = training_cfg.get('beta2', 0.999)

    if optimizer_name == 'adam':
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_generator, betas=(beta1, beta2))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_discriminator, betas=(beta1, beta2))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Create EMA for generator only
    ema_decay = training_cfg.get('ema_decay', 0.9999)
    ema_update_every = training_cfg.get('ema_update_every', 10)
    ema_generator = EMA(generator, beta=ema_decay, update_every=ema_update_every)
    ema_generator.to(device)

    # TensorBoard writer (optional)
    writer = SummaryWriter(output_dir / 'tb') if HAS_TENSORBOARD else None

    # Training loop
    epochs = training_cfg['epochs']
    eval_every = training_cfg.get('eval_every', 100)
    check_every = training_cfg.get('check_every', 100)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Update ratio: 1:1 (1 discriminator step per generator step)")
    print(f"  Label smoothing: {model_cfg.get('label_smoothing', 0.1)}")
    print(f"Output directory: {output_dir}\n")

    best_val_sep = -float('inf')

    for epoch in range(1, epochs + 1):
        # Train
        loss_dict = train_epoch(model, train_loader, optimizer_g, optimizer_d, device)

        # Update EMA
        ema_generator.update()

        # Log (with try/except to avoid crash if tensorboard file is deleted)
        if writer:
            try:
                writer.add_scalar('Loss/Generator', loss_dict['generator_loss'], epoch)
                writer.add_scalar('Loss/Discriminator', loss_dict['discriminator_loss'], epoch)
                writer.add_scalar('D_Real_Mean', loss_dict['d_real_mean'], epoch)
                writer.add_scalar('D_Fake_Mean', loss_dict['d_fake_mean'], epoch)
            except Exception as e:
                print(f"Warning: TensorBoard logging failed: {e}")

        print(f"Epoch {epoch}/{epochs} - G Loss: {loss_dict['generator_loss']:.4f}, "
              f"D Loss: {loss_dict['discriminator_loss']:.4f}, "
              f"D(real): {loss_dict['d_real_mean']:.4f}, "
              f"D(fake): {loss_dict['d_fake_mean']:.4f}")

        # Evaluate
        if epoch % eval_every == 0:
            # Use EMA generator for evaluation
            model.generator = ema_generator.ema_model
            val_sep = eval_model(model, test_loader, device)
            model.generator = generator  # Restore original generator

            if writer:
                try:
                    writer.add_scalar('Val_D_Separation', val_sep, epoch)
                except Exception:
                    pass  # Ignore tensorboard errors
            print(f"  Val D separation (D(real)-D(fake)): {val_sep:.4f}")

            if val_sep > best_val_sep:
                best_val_sep = val_sep
                save_checkpoint(encoder, generator, ema_generator, discriminator, optimizer_g, optimizer_d,
                              epoch, val_sep, output_dir / 'check')

        # Save checkpoint
        if epoch % check_every == 0:
            save_checkpoint(encoder, generator, ema_generator, discriminator, optimizer_g, optimizer_d,
                          epoch, loss_dict['generator_loss'], output_dir / 'check')

    print(f"\nTraining completed! Best val D separation: {best_val_sep:.4f}")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()
