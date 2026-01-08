#!/usr/bin/env python3
"""
Training script for MMVAE+ - Modular version.

Usage:
    python train.py --config configs/mmvaeplus_default.yaml
    python train.py --config configs/mmvaeplus_default.yaml --epochs 100 --batch_size 32
"""

import os
import sys
import platform
import time
import pickle
import argparse
from pathlib import Path
from datetime import datetime

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

from datasets.continuous import MultiComponentDataset
from models.mmvaeplus import MMVAEplusEpure
from models.mmvaeplus.objectives import m_elbo, m_dreg
from models.mmvaeplus.utils import unpack_data
from utils.config import load_config, auto_complete_config, validate_config, resolve_path


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(model, train_loader, optimizer, objective_fn, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Unpack data: (data_tuple, cond) where data_tuple is tuple of 5 tensors
        data, cond = unpack_data(batch, device=device)
        
        # Replicate cond for each modality (5 components)
        if cond is not None:
            cond_list = [cond] * len(data)  # List of 5 condition tensors
        else:
            cond_list = None
        
        optimizer.zero_grad()
        
        # Compute loss (negative ELBO or DReG)
        loss = -objective_fn(model, data, cond=cond_list, K=model.params.K)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def eval_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            data, cond = unpack_data(batch, device=device)
            
            if cond is not None:
                cond_list = [cond] * len(data)
            else:
                cond_list = None
            
            loss = -m_elbo(model, data, cond=cond_list, K=model.params.K, test=True)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, loss, output_dir):
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{epoch}.pt"
    
    # Convert params object to dict for serialization
    params_dict = {}
    if hasattr(model.params, '__dict__'):
        params_dict = model.params.__dict__.copy()
    else:
        # If params is a class, get all attributes
        params_dict = {
            'latent_dim_w': model.params.latent_dim_w,
            'latent_dim_z': model.params.latent_dim_z,
            'latent_dim_u': model.params.latent_dim_u,
            'nf': model.params.nf,
            'nf_max': model.params.nf_max,
            'cond_dim': getattr(model.params, 'cond_dim', 0),
            'priorposterior': model.params.priorposterior,
            'datadir': getattr(model.params, 'datadir', None),
            'no_cuda': getattr(model.params, 'no_cuda', False),
            'beta': getattr(model.params, 'beta', 2.5),
            'K': getattr(model.params, 'K', 1),
        }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'params': params_dict,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='MMVAE+ Epure Training')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device (cuda/cpu, overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load and auto-complete config
    config = load_config(args.config)
    config = auto_complete_config(config)
    validate_config(config, model_type='mmvaeplus')
    
    # Override with command-line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Setup device
    device = torch.device(args.device if args.device else config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Set seed
    seed = args.seed if args.seed is not None else config['training'].get('seed', 42)
    set_seed(seed)
    
    # Create output directory
    output_dir = resolve_path(config['paths']['output_dir'])
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_dir / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    check_dir = output_dir / 'check'
    check_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Save config with num_parameters (will be added later)
    import yaml
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load data
    print("Loading data...")
    data_cfg = config['data']
    root_dir = resolve_path(data_cfg['root_dir'])
    condition_csv = resolve_path(data_cfg['condition_csv'])
    
    train_dataset = MultiComponentDataset(
        root_dir=root_dir / 'train',
        component_dirs=data_cfg['component_dirs'],
        condition_csv=condition_csv,
        condition_columns=data_cfg['condition_columns'],
        prefix_column=data_cfg['prefix_column'],
        filename_pattern=data_cfg.get('filename_pattern', '{prefix}_{component}.png'),
        split='train',
        split_column=data_cfg.get('split_column', 'train'),
        stacked=False,  # Return tuple instead of concatenated tensor
        normalized=data_cfg.get('normalized', False)
    )
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating MMVAE+ model...")
    model_cfg = config['model']
    
    # Create params object
    class Params:
        latent_dim_w = model_cfg['latent_dim_w']
        latent_dim_z = model_cfg['latent_dim_z']
        latent_dim_u = model_cfg['latent_dim_u']
        nf = model_cfg['nf']
        nf_max = model_cfg['nf_max']
        cond_dim = model_cfg.get('cond_dim', 0)
        priorposterior = model_cfg.get('priorposterior', 'Laplace')
        datadir = str(root_dir)
        no_cuda = (device.type == 'cpu')
        beta = config['training'].get('beta', 2.5)
        K = config['training'].get('K', 1)
    
    model = MMVAEplusEpure(Params()).to(device)

    # Count parameters and add to config
    num_params = sum(p.numel() for p in model.parameters())
    config['model']['num_parameters'] = num_params

    print(f"Model: {model.modelName}")
    print(f"  Components: {len(model.vaes)}")
    print(f"  Latent dims: w={Params.latent_dim_w}, z={Params.latent_dim_z}, u={Params.latent_dim_u}")
    print(f"  Cond dim: {Params.cond_dim}")
    print(f"  Total parameters: {num_params:,}")
    
    # Optimizer
    training_cfg = config['training']
    optimizer_name = training_cfg.get('optimizer', 'adam').lower()
    lr = training_cfg['lr']
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            amsgrad=training_cfg.get('amsgrad', True)
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=training_cfg.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Objective function
    objective_name = training_cfg.get('objective', 'elbo')
    if objective_name == 'elbo':
        objective_fn = m_elbo
    elif objective_name == 'dreg':
        objective_fn = m_dreg
    else:
        raise ValueError(f"Unknown objective: {objective_name}")
    
    print(f"Objective: {objective_name}")
    
    # TensorBoard writer (optional)
    writer = SummaryWriter(output_dir / 'tb') if HAS_TENSORBOARD else None
    
    # Training loop
    epochs = training_cfg['epochs']
    eval_every = training_cfg.get('eval_every', 50)
    checkpoint_epochs = training_cfg.get('checkpoint_epochs', [50, 100, 150, 250])
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Objective: {objective_name}")
    print(f"  K: {Params.K}")
    print(f"  Beta: {Params.beta}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        loss = train_epoch(model, train_loader, optimizer, objective_fn, device, epoch)
        print(f"  Train loss: {loss:.4f}")
        
        # Log
        if writer:
            writer.add_scalar('Loss/Train', loss, epoch)
        
        # Evaluate
        if epoch % eval_every == 0:
            val_loss = eval_model(model, test_loader, device)
            print(f"  Val loss: {val_loss:.4f}")
            
            if writer:
                writer.add_scalar('Loss/Val', val_loss, epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, check_dir)
        
        # Save checkpoint at specified epochs
        if epoch in checkpoint_epochs:
            save_checkpoint(model, optimizer, epoch, loss, check_dir)
    
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    if writer:
        writer.close()


if __name__ == '__main__':
    main()

