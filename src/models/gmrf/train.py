#!/usr/bin/env python3
"""
Training script for GMRF MVAE with conditioning support.

Usage:
    python train.py --config configs/gmrf_default.yaml
    python train.py --config configs/gmrf_default.yaml --epochs 100 --batch_size 32
"""

import os
import sys
import platform
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_IS_WINDOWS = platform.system() == 'Windows'

import torch
from torch.utils.data import DataLoader

# Optional tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

import yaml

from datasets.continuous import MultiComponentDataset
from models.gmrf import Epure_GMRF_MVAE
from models.gmrf.objectives import compute_elbo_dist
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


def unpack_data(batch, device='cuda'):
    """
    Unpack batch from MultiComponentDataset.

    Returns:
        data: List of tensors on device (one per component)
        cond: (B, cond_dim) tensor on device or None
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        data_tuple, cond = batch

        # Handle stacked tensor or tuple
        if isinstance(data_tuple, torch.Tensor):
            # Stacked format: (B, C, H, W) -> list of (B, 1, H, W)
            data = [data_tuple[:, i:i+1].to(device) for i in range(data_tuple.size(1))]
        else:
            # Already tuple/list format
            data = [d.to(device) for d in data_tuple]

        cond = cond.to(device).float() if cond is not None else None
        return data, cond
    else:
        # Fallback for old format (no conditioning)
        if isinstance(batch, torch.Tensor):
            data = [batch[:, i:i+1].to(device) for i in range(batch.size(1))]
        elif isinstance(batch, (tuple, list)):
            data = [d.to(device) for d in batch]
        else:
            data = [batch.to(device)]
        return data, None


def train_epoch(model, train_loader, optimizer, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    training_cfg = config['training']
    model_cfg = config['model']

    loss_type = training_cfg.get('recon_loss', 'mse')
    alpha_mse = training_cfg.get('alpha_mse', 0.5)
    recon_weights = training_cfg.get('recon_weights', None)
    beta = model_cfg.get('beta', 1.0)

    for batch_idx, batch in enumerate(train_loader):
        data, cond = unpack_data(batch, device=device)

        optimizer.zero_grad()

        # Forward pass
        model(data, cond=cond)

        # Compute ELBO
        elbo, recon_loss, kl_div = compute_elbo_dist(
            model, data, beta=beta,
            loss_type=loss_type, alpha_mse=alpha_mse, weights=recon_weights
        )

        loss = -elbo
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += (-recon_loss.item())
        total_kl += kl_div.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}, recon={-recon_loss.item():.4f}, kl={kl_div.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon = total_recon / num_batches if num_batches > 0 else 0.0
    avg_kl = total_kl / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_recon, avg_kl


def eval_model(model, test_loader, config, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    training_cfg = config['training']
    model_cfg = config['model']

    loss_type = training_cfg.get('recon_loss', 'mse')
    alpha_mse = training_cfg.get('alpha_mse', 0.5)
    recon_weights = training_cfg.get('recon_weights', None)
    beta = model_cfg.get('beta', 1.0)

    with torch.no_grad():
        for batch in test_loader:
            data, cond = unpack_data(batch, device=device)

            model(data, cond=cond)

            elbo, _, _ = compute_elbo_dist(
                model, data, beta=beta,
                loss_type=loss_type, alpha_mse=alpha_mse, weights=recon_weights
            )

            total_loss += (-elbo.item())
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, loss, output_dir, is_best=False):
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_best:
        checkpoint_path = output_dir / "checkpoint_best.pt"
    else:
        checkpoint_path = output_dir / f"checkpoint_{epoch}.pt"

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='GMRF MVAE Training')
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
    validate_config(config, model_type='gmrf')

    # Override with command-line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    # Setup device
    device = torch.device(
        args.device if args.device else
        config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )
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
        stacked=False,
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

    num_workers = 0 if _IS_WINDOWS else config['training'].get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    print("Creating GMRF MVAE model...")
    model_cfg = config['model']

    # Capture device string for class definition
    device_str = str(device)
    
    # Create params object
    class Params:
        num_components = len(data_cfg['component_dirs'])
        latent_dim = model_cfg['latent_dim']
        diagonal_transf = model_cfg.get('diagonal_transf', 'softplus')
        hidden_dim = model_cfg['hidden_dim']
        n_layers = model_cfg['n_layers']
        nf = model_cfg['nf']
        nf_max_e = model_cfg.get('nf_max_e', 512)
        nf_max_d = model_cfg.get('nf_max_d', 256)
        cond_dim = model_cfg.get('cond_dim', 0)
        image_size = tuple(model_cfg.get('image_size', [64, 32]))
        reduced_diag = model_cfg.get('reduced_diag', False)
        device = device_str
        component_names = data_cfg['component_dirs']

    model = Epure_GMRF_MVAE(Params()).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    config['model']['num_parameters'] = num_params

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"Model: {model.modelName}")
    print(f"  Components: {Params.num_components} ({Params.component_names})")
    print(f"  Latent dim: {Params.latent_dim} (total: {Params.num_components * Params.latent_dim})")
    print(f"  Cond dim: {Params.cond_dim}")
    print(f"  Image size: {Params.image_size}")
    print(f"  Total parameters: {num_params:,}")

    # Optimizer
    training_cfg = config['training']
    optimizer_name = training_cfg.get('optimizer', 'adam').lower()
    lr = training_cfg['lr']

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            amsgrad=training_cfg.get('amsgrad', False)
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=training_cfg.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # TensorBoard writer (optional)
    writer = SummaryWriter(output_dir / 'tb') if HAS_TENSORBOARD else None

    # Training loop
    epochs = training_cfg['epochs']
    eval_every = training_cfg.get('eval_every', 10)
    check_every = training_cfg.get('check_every', 50)

    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Loss type: {training_cfg.get('recon_loss', 'mse')}")
    print(f"  Beta: {model_cfg.get('beta', 1.0)}")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_recon': [], 'train_kl': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        loss, recon, kl = train_epoch(model, train_loader, optimizer, config, device, epoch)
        print(f"  Train loss: {loss:.4f}, recon: {recon:.4f}, kl: {kl:.4f}")

        history['train_loss'].append(loss)
        history['train_recon'].append(recon)
        history['train_kl'].append(kl)

        # Log
        if writer:
            writer.add_scalar('Loss/Train', loss, epoch)
            writer.add_scalar('Recon/Train', recon, epoch)
            writer.add_scalar('KL/Train', kl, epoch)

        # Evaluate
        if epoch % eval_every == 0:
            val_loss = eval_model(model, test_loader, config, device)
            print(f"  Val loss: {val_loss:.4f}")

            history['val_loss'].append(val_loss)

            if writer:
                writer.add_scalar('Loss/Val', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, check_dir, is_best=True)

        # Save checkpoint at intervals
        if epoch % check_every == 0:
            save_checkpoint(model, optimizer, epoch, loss, check_dir)

    # Save final checkpoint
    save_checkpoint(model, optimizer, epochs, loss, check_dir)

    # Save training history
    import pickle
    with open(output_dir / 'history.pkl', 'wb') as f:
        pickle.dump(history, f)

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
