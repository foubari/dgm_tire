#!/usr/bin/env python3
"""
Training script for MDM (Multinomial Diffusion Model) - Modular version.

Usage:
    python train.py --config configs/mdm_default.yaml
"""

import os
import sys
import platform
import time
import pickle
import argparse
from pathlib import Path
from random import random

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

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from datasets.categorical import SegmentationDataset
from models.mdm import MultinomialDiffusion, SegmentationUnet
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


def train_epoch(model, train_loader, optimizer, device, cond_drop_prob):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, cond) in enumerate(train_loader):
        x = x.to(device).squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        cond = cond.to(device) if cond is not None else None
        
        # Classifier-free guidance: randomly drop conditioning
        if cond is not None and random() < cond_drop_prob:
            cond = None
        
        # Compute loss
        loss = -model.log_prob(x, cond=cond).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def eval_model(model, eval_loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, cond in eval_loader:
            x = x.to(device).squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            cond = cond.to(device) if cond is not None else None
            loss = -model.log_prob(x, cond=cond).mean()
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def create_data_loaders(config, args):
    """Create train and test data loaders from config."""
    data_cfg = config['data']
    
    # Resolve paths
    root_dir = resolve_path(data_cfg['root_dir'])
    condition_csv = resolve_path(data_cfg['condition_csv'])
    
    # Create train dataset
    train_dataset = SegmentationDataset(
        root_dir=root_dir,
        condition_csv=condition_csv,
        condition_columns=data_cfg['condition_columns'],
        prefix_column=data_cfg.get('prefix_column', 'matching'),
        split='train',
        split_column=data_cfg.get('split_column', 'train'),
        resolution=tuple(data_cfg.get('resolution', [64, 32])),
        mask_format=data_cfg.get('mask_format', 'numpy'),
        mask_path=data_cfg.get('mask_path'),
        filename_pattern=data_cfg.get('filename_pattern', '{prefix}.png'),
        normalized=data_cfg.get('normalized', False),
    )
    
    # Create test dataset
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
    parser = argparse.ArgumentParser(description='Train MDM - Modular version')
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
    validate_config(config, model_type='mdm')
    config = auto_complete_config(config)
    
    # Override config with command-line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
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
    
    # Create model
    print("Creating model...")
    model_cfg = config['model']
    data_cfg = config['data']
    
    resolution = tuple(data_cfg.get('resolution', [64, 32]))
    num_classes = model_cfg['num_classes']
    cond_dim = model_cfg['cond_dim']
    
    print(f"  Resolution: {resolution}")
    print(f"  Num classes: {num_classes}")
    print(f"  Cond dim: {cond_dim} (columns: {data_cfg['condition_columns']})")
    
    unet = SegmentationUnet(
        num_classes=num_classes,
        dim=model_cfg['dim'],
        num_steps=model_cfg['timesteps'],
        dim_mults=tuple(model_cfg['dim_mults']),
        cond_dim=cond_dim
    )
    
    model = MultinomialDiffusion(
        num_classes=num_classes,
        shape=(1, *resolution),  # (1, H, W)
        denoise_fn=unet,
        timesteps=model_cfg['timesteps'],
        cond_drop_prob=model_cfg['cond_drop_prob'],
        loss_type=model_cfg.get('loss_type', 'elbo')
    )
    
    model = model.to(device)

    num_params = count_parameters(model)
    config['model']['num_parameters'] = num_params
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    training_cfg = config['training']
    lr = training_cfg['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # TensorBoard writer (optional)
    writer = SummaryWriter(output_dir / 'tb') if HAS_TENSORBOARD else None
    
    # Training loop
    epochs = training_cfg['epochs']
    eval_every = training_cfg.get('eval_every', 100)
    check_every = training_cfg.get('check_every', 100)
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Output directory: {output_dir}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, model_cfg['cond_drop_prob'])
        
        # Log
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % eval_every == 0:
            val_loss = eval_model(model, test_loader, device)
            if writer:
                writer.add_scalar('Loss/Val', val_loss, epoch)
            print(f"  Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss,
                              output_dir / 'check')
        
        # Save checkpoint
        if epoch % check_every == 0:
            save_checkpoint(model, optimizer, epoch, train_loss,
                          output_dir / 'check')
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()

