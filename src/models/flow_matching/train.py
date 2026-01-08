#!/usr/bin/env python3
"""
Training script for Flow Matching - Modular version.

Usage:
    python train.py --config configs/flow_matching_default.yaml
    python train.py --config configs/flow_matching_default.yaml --epochs 100 --batch_size 32
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
from models.flow_matching import FlowMatching, Unet
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
    
    for batch_idx, batch_data in enumerate(train_loader):
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            images, cond = batch_data
            images = images.to(device)
            cond = cond.to(device).float()
        else:
            images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
            cond = None
        
        optimizer.zero_grad()
        
        # Forward pass
        loss = model(images, cond=cond)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def eval_model(model, eval_loader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in eval_loader:
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                images, cond = batch_data
                images = images.to(device)
                cond = cond.to(device).float()
            else:
                images = batch_data.to(device) if isinstance(batch_data, torch.Tensor) else batch_data[0].to(device)
                cond = None
            
            loss = model(images, cond=cond)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, ema_model, optimizer, epoch, loss, checkpoint_dir):
    """Save checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.ema_model.state_dict() if ema_model else None,
        'optimizer_state_dict': optimizer.state_dict(),
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
    parser = argparse.ArgumentParser(description='Train Flow Matching - Modular version')
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
    validate_config(config, model_type='flow_matching')
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
    
    image_size = tuple(model_cfg['image_size'])
    channels = model_cfg['channels']
    cond_dim = model_cfg['cond_dim']
    
    print(f"  Image size: {image_size}")
    print(f"  Channels: {channels} (components: {data_cfg['component_dirs']})")
    print(f"  Cond dim: {cond_dim} (columns: {data_cfg['condition_columns']})")
    
    unet = Unet(
        dim=model_cfg['dim'],
        dim_mults=tuple(model_cfg['dim_mults']),
        channels=channels,
        cond_dim=cond_dim,
        self_condition=False,
        flash_attn=False
    )
    
    model = FlowMatching(
        model=unet,
        image_size=image_size,
        timesteps=model_cfg['timesteps'],
        solver=model_cfg.get('solver', 'euler'),
        cond_drop_prob=model_cfg['cond_drop_prob']
    )
    
    model = model.to(device)

    num_params = count_parameters(model)
    config['model']['num_parameters'] = num_params
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    training_cfg = config['training']
    optimizer_name = training_cfg.get('optimizer', 'adam')
    lr = training_cfg['lr']
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create EMA model
    ema_decay = training_cfg.get('ema_decay', 0.9999)
    ema_update_every = training_cfg.get('ema_update_every', 10)
    ema_model = EMA(model, beta=ema_decay, update_every=ema_update_every)
    ema_model.to(device)
    
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
        
        # Update EMA
        ema_model.update()
        
        # Log
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % eval_every == 0:
            val_loss = eval_model(ema_model.ema_model, test_loader, device)
            if writer:
                writer.add_scalar('Loss/Val', val_loss, epoch)
            print(f"  Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ema_model.ema_model, ema_model, optimizer, epoch, val_loss,
                              output_dir / 'check')
        
        # Save checkpoint
        if epoch % check_every == 0:
            save_checkpoint(ema_model.ema_model, ema_model, optimizer, epoch, train_loss,
                          output_dir / 'check')
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()

