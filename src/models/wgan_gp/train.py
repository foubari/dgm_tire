#!/usr/bin/env python3
"""
Training script for WGAN-GP (Wasserstein GAN with Gradient Penalty) - Modular version.

Usage:
    python train.py --config configs/wgan_gp_default.yaml
    python train.py --config configs/wgan_gp_default.yaml --epochs 100 --batch_size 32
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
from models.wgan_gp import WGANGP, Encoder, Generator, Critic
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


def train_epoch(model, train_loader, optimizer_g, optimizer_c, device, n_critic):
    """Train WGAN-GP for one epoch."""
    model.train()
    
    total_g_loss = 0.0
    total_c_loss = 0.0
    total_wd = 0.0
    total_gp = 0.0
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
        # Update Critic (n_critic times)
        # ===========================
        for _ in range(n_critic):
            optimizer_c.zero_grad()
            
            loss_dict = model(images, cond=cond, train_critic=True)
            critic_loss = loss_dict['critic_loss']
            
            critic_loss.backward()
            optimizer_c.step()
            
            total_c_loss += loss_dict['critic_loss'].item()
            total_wd += loss_dict['wasserstein_distance']
            total_gp += loss_dict['gradient_penalty']
        
        # ===========================
        # Update Generator (once)
        # ===========================
        optimizer_g.zero_grad()
        
        loss_dict = model(images, cond=cond, train_critic=False)
        generator_loss = loss_dict['generator_loss']
        
        generator_loss.backward()
        optimizer_g.step()
        
        total_g_loss += loss_dict['generator_loss'].item()
        num_batches += 1
    
    return {
        'generator_loss': total_g_loss / num_batches if num_batches > 0 else 0.0,
        'critic_loss': total_c_loss / (num_batches * n_critic) if num_batches > 0 else 0.0,
        'wasserstein_distance': total_wd / (num_batches * n_critic) if num_batches > 0 else 0.0,
        'gradient_penalty': total_gp / (num_batches * n_critic) if num_batches > 0 else 0.0,
    }


def eval_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    
    total_wd = 0.0
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
            
            # Critic scores
            critic_real = model.critic(images, cond=cond)
            critic_fake = model.critic(fake, cond=cond)
            
            # Wasserstein distance
            wasserstein_distance = (critic_real.mean() - critic_fake.mean()).item()
            total_wd += wasserstein_distance
            num_batches += 1
    
    return total_wd / num_batches if num_batches > 0 else 0.0


def save_checkpoint(encoder, generator, ema_generator, critic, optimizer_g, optimizer_c, epoch, loss, check_dir):
    """Save model checkpoint."""
    check_dir = Path(check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'ema_generator_state_dict': ema_generator.ema_model.state_dict() if ema_generator.ema_model is not None else None,
        'critic_state_dict': critic.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_c_state_dict': optimizer_c.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    checkpoint_path = check_dir / f'checkpoint_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train WGAN-GP - Modular version')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    # Allow overriding config values
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr_generator', type=float, default=None,
                       help='Generator learning rate (overrides config)')
    parser.add_argument('--lr_critic', type=float, default=None,
                       help='Critic learning rate (overrides config)')
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
    validate_config(config, model_type='wgan_gp')
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr_generator is not None:
        config['training']['lr_generator'] = args.lr_generator
    if args.lr_critic is not None:
        config['training']['lr_critic'] = args.lr_critic
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
    print("Creating WGAN-GP model...")
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
    
    # Create encoder, generator, critic
    encoder = Encoder(in_channels=channels, latent_dim=latent_dim)
    generator = Generator(
        latent_dim=latent_dim,
        out_channels=channels,
        cond_dim=cond_dim,
        dim=model_cfg.get('dim', 64),
        dim_mults=tuple(model_cfg.get('dim_mults', [1, 2, 4]))
    )
    critic = Critic(in_channels=channels, cond_dim=cond_dim)
    
    # Create WGAN-GP wrapper
    model = WGANGP(
        encoder=encoder,
        generator=generator,
        critic=critic,
        image_size=image_size,
        latent_dim=latent_dim,
        lambda_gp=model_cfg.get('lambda_gp', 10),
        n_critic=model_cfg.get('n_critic', 5),
        cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1)
    )
    
    encoder = encoder.to(device)
    generator = generator.to(device)
    critic = critic.to(device)
    
    num_params_encoder = count_parameters(encoder)
    num_params_generator = count_parameters(generator)
    num_params_critic = count_parameters(critic)
    total_params = num_params_encoder + num_params_generator + num_params_critic

    # Add num_parameters to config
    config['model']['num_parameters'] = {
        'encoder': num_params_encoder,
        'generator': num_params_generator,
        'critic': num_params_critic,
        'total': total_params
    }

    print(f"Encoder parameters: {num_params_encoder:,}")
    print(f"Generator parameters: {num_params_generator:,}")
    print(f"Critic parameters: {num_params_critic:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizers
    training_cfg = config['training']
    optimizer_name = training_cfg.get('optimizer', 'adam')
    lr_generator = training_cfg.get('lr_generator', 0.0001)
    lr_critic = training_cfg.get('lr_critic', 0.0001)
    beta1 = training_cfg.get('beta1', 0.0)
    beta2 = training_cfg.get('beta2', 0.9)
    
    if optimizer_name == 'adam':
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_generator, betas=(beta1, beta2))
        optimizer_c = torch.optim.Adam(critic.parameters(), lr=lr_critic, betas=(beta1, beta2))
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
    n_critic = model_cfg.get('n_critic', 5)
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  n_critic: {n_critic} (critic updates per generator update)")
    print(f"Output directory: {output_dir}\n")
    
    best_val_wd = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        loss_dict = train_epoch(model, train_loader, optimizer_g, optimizer_c, device, n_critic)
        
        # Update EMA
        ema_generator.update()
        
        # Log (with try/except to avoid crash if tensorboard file is deleted)
        if writer:
            try:
                writer.add_scalar('Loss/Generator', loss_dict['generator_loss'], epoch)
                writer.add_scalar('Loss/Critic', loss_dict['critic_loss'], epoch)
                writer.add_scalar('Wasserstein_Distance', loss_dict['wasserstein_distance'], epoch)
                writer.add_scalar('Gradient_Penalty', loss_dict['gradient_penalty'], epoch)
            except Exception as e:
                print(f"Warning: TensorBoard logging failed: {e}")
        
        print(f"Epoch {epoch}/{epochs} - G Loss: {loss_dict['generator_loss']:.4f}, "
              f"C Loss: {loss_dict['critic_loss']:.4f}, "
              f"WD: {loss_dict['wasserstein_distance']:.4f}, "
              f"GP: {loss_dict['gradient_penalty']:.4f}")
        
        # Evaluate
        if epoch % eval_every == 0:
            # Use EMA generator for evaluation
            model.generator = ema_generator.ema_model
            val_wd = eval_model(model, test_loader, device)
            model.generator = generator  # Restore original generator
            
            if writer:
                try:
                    writer.add_scalar('Val_Wasserstein_Distance', val_wd, epoch)
                except Exception:
                    pass  # Ignore tensorboard errors
            print(f"  Val WD: {val_wd:.4f}")
            
            if val_wd < best_val_wd:
                best_val_wd = val_wd
                save_checkpoint(encoder, generator, ema_generator, critic, optimizer_g, optimizer_c,
                              epoch, val_wd, output_dir / 'check')
        
        # Save checkpoint
        if epoch % check_every == 0:
            save_checkpoint(encoder, generator, ema_generator, critic, optimizer_g, optimizer_c,
                          epoch, loss_dict['generator_loss'], output_dir / 'check')
    
    print(f"\nTraining completed! Best val WD: {best_val_wd:.4f}")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()

