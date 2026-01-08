#!/usr/bin/env python3
"""
Training script for VQ-VAE with PixelCNN Prior - Modular version.

Two-phase training:
1. Phase 1: Train VQ-VAE (encoder + quantizer + decoder)
2. Phase 2: Freeze VQ-VAE, train PixelCNN Prior

Usage:
    python train.py --config configs/vqvae_default.yaml
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from datasets.continuous import MultiComponentDataset
from models.vqvae import VQVAE, PixelCNNPrior, mask_components
from utils.config import load_config, auto_complete_config, validate_config, resolve_path


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def count_parameters(model, only_trainable=True):
    """Count the number of parameters in a model."""
    filt = lambda p: p.requires_grad if only_trainable else True
    return sum(p.numel() for p in model.parameters() if filt(p))


def train_epoch_vqvae(model, train_loader, optimizer, device, epoch=1, mask_prob_max=0.5, warmup_epochs=100, max_grad_norm=1.0):
    """
    Train VQ-VAE for one epoch with component masking.

    Args:
        model: VQ-VAE model
        train_loader: DataLoader
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number (for progressive masking)
        mask_prob_max: Maximum masking probability (reduced from 0.7 to 0.5 for stability)
        warmup_epochs: Number of epochs to warm up masking (increased from 20 to 100)
        max_grad_norm: Maximum gradient norm for clipping (default 1.0)
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_commit = 0.0
    num_batches = 0

    # Progressive masking schedule (like ICTAI VAE)
    # Use smoother schedule: square root for slower increase
    progress = min(1.0, epoch / warmup_epochs)
    mask_prob = (progress ** 0.5) * mask_prob_max  # Slower increase with sqrt schedule

    for batch_idx, batch_data in enumerate(train_loader):
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            images, cond = batch_data
            images = images.to(device)
            cond = cond.to(device).float() if cond is not None else None
        else:
            images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
            cond = None

        # Apply component masking
        masked_images = mask_components(images, p=mask_prob)

        optimizer.zero_grad()

        # Forward pass: input=masked, target=original (full image)
        loss, loss_dict = model(masked_images, cond=cond, target=images)

        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosion
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        total_loss += loss_dict['total']
        total_recon += loss_dict['recon']
        total_vq += loss_dict['vq']
        total_commit += loss_dict['commit']
        num_batches += 1
        
        # Warn if VQ loss is exploding (per batch check)
        if loss_dict['vq'] > 1e6:
            print(f"\nWARNING: VQ loss is very high ({loss_dict['vq']:.2e}) at epoch {epoch}, batch {batch_idx}")
            print(f"  Mask prob: {mask_prob:.3f}")
            print(f"  Consider reducing mask_prob_max or increasing warmup_epochs")

    avg_vq = total_vq / num_batches if num_batches > 0 else 0.0
    
    return {
        'total': total_loss / num_batches if num_batches > 0 else 0.0,
        'recon': total_recon / num_batches if num_batches > 0 else 0.0,
        'vq': avg_vq,
        'commit': total_commit / num_batches if num_batches > 0 else 0.0,
        'mask_prob': mask_prob,
    }


def train_epoch_prior(prior, vqvae, train_loader, optimizer, device):
    """Train PixelCNN Prior (VQ-VAE is frozen)."""
    prior.train()
    vqvae.eval()  # Freeze VQ-VAE
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_data in train_loader:
        if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
            images, cond = batch_data
            images = images.to(device)
            cond = cond.to(device).float() if cond is not None else None
        else:
            images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
            cond = None
        
        # Encode images to latent indices (no grad)
        with torch.no_grad():
            indices = vqvae.encode(images)  # (B, H, W)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = prior(indices, cond)  # (B, num_embeddings, H, W)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, indices)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def eval_model_vqvae(model, eval_loader, device):
    """Evaluate VQ-VAE on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_vq = 0.0
    total_commit = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in eval_loader:
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                images, cond = batch_data
                images = images.to(device)
                cond = cond.to(device).float() if cond is not None else None
            else:
                images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
                cond = None
            
            loss, loss_dict = model(images, cond=cond)
            total_loss += loss_dict['total']
            total_recon += loss_dict['recon']
            total_vq += loss_dict['vq']
            total_commit += loss_dict['commit']
            num_batches += 1
    
    return {
        'total': total_loss / num_batches if num_batches > 0 else 0.0,
        'recon': total_recon / num_batches if num_batches > 0 else 0.0,
        'vq': total_vq / num_batches if num_batches > 0 else 0.0,
        'commit': total_commit / num_batches if num_batches > 0 else 0.0,
    }


def eval_model_prior(prior, vqvae, eval_loader, device):
    """Evaluate PixelCNN Prior."""
    prior.eval()
    vqvae.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in eval_loader:
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                images, cond = batch_data
                images = images.to(device)
                cond = cond.to(device).float() if cond is not None else None
            else:
                images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
                cond = None
            
            indices = vqvae.encode(images)
            logits = prior(indices, cond)
            loss = F.cross_entropy(logits, indices)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(vqvae, prior, optimizer_vqvae, optimizer_prior, epoch, loss, checkpoint_dir, phase='vqvae'):
    """Save checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')
    
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'vqvae_state_dict': vqvae.state_dict() if vqvae is not None else None,
        'prior_state_dict': prior.state_dict() if prior is not None else None,
        'optimizer_vqvae_state_dict': optimizer_vqvae.state_dict() if optimizer_vqvae is not None else None,
        'optimizer_prior_state_dict': optimizer_prior.state_dict() if optimizer_prior is not None else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")


def create_data_loaders(config, args):
    """Create train and test data loaders from config."""
    data_cfg = config['data']
    
    # Resolve paths
    root_dir = resolve_path(data_cfg['root_dir'])
    condition_csv = resolve_path(data_cfg['condition_csv'])
    
    # Get component directories from config
    component_dirs = data_cfg['component_dirs']
    
    # Create train dataset
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
    
    # Create data loaders
    training_cfg = config['training']
    batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size else training_cfg['batch_size']
    num_workers = args.num_workers if hasattr(args, 'num_workers') and args.num_workers is not None else training_cfg.get('num_workers', 4)
    
    # Force num_workers=0 on Windows
    if _IS_WINDOWS and num_workers > 0:
        print("WARNING: Forcing num_workers=0 on Windows (multiprocessing compatibility)")
        num_workers = 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, test_loader, train_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE - Modular version')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    # Allow overriding config values
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
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
    
    # Load and validate config
    config = load_config(args.config)
    validate_config(config, model_type='vqvae')
    config = auto_complete_config(config)
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr_vqvae'] = args.lr
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
    
    # Create VQ-VAE model
    print("Creating VQ-VAE model...")
    model_cfg = config['model']
    data_cfg = config['data']
    
    image_size = tuple(model_cfg['image_size'])
    channels = model_cfg['channels']
    cond_dim = model_cfg['cond_dim']
    
    print(f"  Image size: {image_size}")
    print(f"  Channels: {channels} (components: {data_cfg['component_dirs']})")
    print(f"  Cond dim: {cond_dim} (columns: {data_cfg['condition_columns']})")
    
    vqvae = VQVAE(
        image_size=image_size,
        channels=channels,
        cond_dim=cond_dim,
        latent_dim=model_cfg['latent_dim'],
        num_embeddings=model_cfg['num_embeddings'],
        commitment_cost=model_cfg['commitment_cost'],
        ema_decay=model_cfg['ema_decay'],
        base_dim=model_cfg['dim'],
        dim_mults=tuple(model_cfg['dim_mults']),
    )
    
    vqvae = vqvae.to(device)
    
    num_params_vqvae = count_parameters(vqvae)
    print(f"VQ-VAE parameters: {num_params_vqvae:,}")
    
    # Create Prior model (if enabled)
    prior = None
    optimizer_prior = None
    if model_cfg.get('prior', {}).get('enabled', True):
        print("Creating PixelCNN Prior...")
        prior_cfg = model_cfg['prior']
        
        prior = PixelCNNPrior(
            num_embeddings=model_cfg['num_embeddings'],
            hidden_dim=prior_cfg['hidden_dim'],
            cond_dim=cond_dim,
            num_layers=prior_cfg['num_layers'],
            kernel_size=prior_cfg.get('kernel_size', 3),
            cond_drop_prob=model_cfg.get('cond_drop_prob', 0.1),
        )
        
        prior = prior.to(device)
        num_params_prior = count_parameters(prior)
        print(f"Prior parameters: {num_params_prior:,}")

    # Add num_parameters to config
    total_params = num_params_vqvae + (num_params_prior if prior else 0)
    config['model']['num_parameters'] = {
        'vqvae': num_params_vqvae,
        'prior': num_params_prior if prior else 0,
        'total': total_params
    }
    
    # Create optimizers
    training_cfg = config['training']
    
    optimizer_name = training_cfg.get('optimizer_vqvae', 'adam')
    lr_vqvae = training_cfg.get('lr_vqvae', 0.0002)
    
    if optimizer_name == 'adam':
        optimizer_vqvae = torch.optim.Adam(vqvae.parameters(), lr=lr_vqvae)
    elif optimizer_name == 'adamax':
        optimizer_vqvae = torch.optim.Adamax(vqvae.parameters(), lr=lr_vqvae)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    if prior is not None:
        optimizer_name_prior = training_cfg.get('optimizer_prior', 'adam')
        lr_prior = training_cfg.get('lr_prior', 0.0001)
        
        if optimizer_name_prior == 'adam':
            optimizer_prior = torch.optim.Adam(prior.parameters(), lr=lr_prior)
        elif optimizer_name_prior == 'adamax':
            optimizer_prior = torch.optim.Adamax(prior.parameters(), lr=lr_prior)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name_prior}")
    
    # TensorBoard writer (optional)
    writer = SummaryWriter(output_dir / 'tb') if HAS_TENSORBOARD else None
    
    # Training loop
    epochs = training_cfg['epochs']
    train_prior_after_epoch = training_cfg.get('train_prior_after_epoch', 500)
    eval_every = training_cfg.get('eval_every', 100)
    check_every = training_cfg.get('check_every', 100)
    codebook_reset_interval = model_cfg.get('codebook_reset_interval', 100)
    
    # Print model summary before training
    print(f"\n{'='*80}")
    print(f"MODEL SUMMARY")
    print(f"{'='*80}")
    print(f"VQ-VAE parameters: {num_params_vqvae:,}")
    if prior is not None:
        print(f"Prior parameters: {num_params_prior:,}")
        total_params = num_params_vqvae + num_params_prior
        print(f"Total parameters: {total_params:,}")
    print(f"{'='*80}\n")
    
    print(f"Starting training for {epochs} epochs...")
    print(f"  Phase 1 (VQ-VAE): epochs 0-{train_prior_after_epoch}")
    if prior is not None:
        print(f"  Phase 2 (Prior): epochs {train_prior_after_epoch}-{epochs}")
    print(f"Output directory: {output_dir}\n")
    
    best_val_loss = float('inf')
    best_val_loss_prior = float('inf')
    
    for epoch in range(1, epochs + 1):
        if epoch <= train_prior_after_epoch:
            # Phase 1: Train VQ-VAE with component masking
            # Use more conservative masking schedule to prevent loss explosion
            loss_dict = train_epoch_vqvae(
                vqvae, train_loader, optimizer_vqvae, device, 
                epoch=epoch,
                mask_prob_max=0.5,  # Reduced from default 0.7 for stability
                warmup_epochs=100,  # Increased from default 20 for slower increase
                max_grad_norm=1.0   # Gradient clipping to prevent explosion
            )

            # Log
            if writer:
                writer.add_scalar('Loss_VQVAE/Total', loss_dict['total'], epoch)
                writer.add_scalar('Loss_VQVAE/Recon', loss_dict['recon'], epoch)
                writer.add_scalar('Loss_VQVAE/VQ', loss_dict['vq'], epoch)
                writer.add_scalar('Loss_VQVAE/Commit', loss_dict['commit'], epoch)
                writer.add_scalar('Loss_VQVAE/MaskProb', loss_dict['mask_prob'], epoch)

            print(f"Epoch {epoch}/{epochs} [VQ-VAE] (mask_p={loss_dict['mask_prob']:.2f}) - "
                  f"Total: {loss_dict['total']:.4f}, "
                  f"Recon: {loss_dict['recon']:.4f}, "
                  f"VQ: {loss_dict['vq']:.4f}, "
                  f"Commit: {loss_dict['commit']:.4f}")
            
            # Evaluate
            if epoch % eval_every == 0:
                val_loss_dict = eval_model_vqvae(vqvae, test_loader, device)
                if writer:
                    writer.add_scalar('Val_Loss_VQVAE/Total', val_loss_dict['total'], epoch)
                    writer.add_scalar('Val_Loss_VQVAE/Recon', val_loss_dict['recon'], epoch)
                
                print(f"  Val Loss: {val_loss_dict['total']:.4f} "
                      f"(Recon: {val_loss_dict['recon']:.4f})")
                
                if val_loss_dict['total'] < best_val_loss:
                    best_val_loss = val_loss_dict['total']
                    save_checkpoint(vqvae, None, optimizer_vqvae, None, epoch, 
                                  val_loss_dict['total'], output_dir / 'check', phase='vqvae')
            
            # Codebook reset
            if epoch % codebook_reset_interval == 0 and epoch > 0:
                vqvae.quantizer.reset_unused_codes()
                usage = vqvae.quantizer.get_codebook_usage()
                print(f"  Codebook usage: {usage['used_codes']}/{usage['total_codes']} "
                      f"({usage['usage_rate']*100:.1f}%)")
            
            # Save checkpoint
            if epoch % check_every == 0:
                save_checkpoint(vqvae, None, optimizer_vqvae, None, epoch, 
                              loss_dict['total'], output_dir / 'check', phase='vqvae')
        
        else:
            # Phase 2: Train Prior
            if prior is None:
                print(f"Epoch {epoch}: Prior not enabled, skipping...")
                continue
            
            loss_prior = train_epoch_prior(prior, vqvae, train_loader, optimizer_prior, device)
            
            # Log
            if writer:
                writer.add_scalar('Loss_Prior/Total', loss_prior, epoch)
            
            print(f"Epoch {epoch}/{epochs} [Prior] - Loss: {loss_prior:.4f}")
            
            # Evaluate
            if epoch % eval_every == 0:
                val_loss_prior = eval_model_prior(prior, vqvae, test_loader, device)
                if writer:
                    writer.add_scalar('Val_Loss_Prior/Total', val_loss_prior, epoch)
                
                print(f"  Val Loss: {val_loss_prior:.4f}")
                
                if val_loss_prior < best_val_loss_prior:
                    best_val_loss_prior = val_loss_prior
                    save_checkpoint(vqvae, prior, optimizer_vqvae, optimizer_prior, epoch,
                                  val_loss_prior, output_dir / 'check', phase='prior')
            
            # Save checkpoint
            if epoch % check_every == 0:
                save_checkpoint(vqvae, prior, optimizer_vqvae, optimizer_prior, epoch,
                              loss_prior, output_dir / 'check', phase='prior')
    
    print(f"\nTraining completed!")
    print(f"  Best VQ-VAE val loss: {best_val_loss:.4f}")
    if prior is not None:
        print(f"  Best Prior val loss: {best_val_loss_prior:.4f}")
    
    if writer:
        writer.close()


if __name__ == '__main__':
    main()

