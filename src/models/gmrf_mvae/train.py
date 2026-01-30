#!/usr/bin/env python3
"""
Training script for GMRF MVAE - ICTAI implementation.

Usage:
    python src/models/gmrf_mvae/train.py --config configs/gmrf_mvae_epure_full.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.continuous import MultiComponentDataset
from models.gmrf_mvae.model import Epure_GMMVAE
from models.gmrf_mvae.objectives import compute_elbo_dist


class Params:
    """Parameter container matching ICTAI implementation."""

    def __init__(self, params_dict):
        self.latent_dim = params_dict.get('latent_dim', 4)
        self.diagonal_transf = params_dict.get('diagonal_transf', 'softplus')
        self.hidden_dim = params_dict.get('hidden_dim', 128)
        self.n_layers = params_dict.get('n_layers', 2)
        self.device = params_dict.get('device', 'cuda')
        self.reduced_diag = params_dict.get('reduced_diag', False)
        self.nf = params_dict.get('nf', 32)
        self.nf_max_e = params_dict.get('nf_max_e', 512)
        self.nf_max_d = params_dict.get('nf_max_d', 256)


def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    """Resolve a path string to an absolute Path."""
    path = Path(path_str)

    if path.is_absolute() and path.exists():
        return path

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


def train(
    model: Epure_GMMVAE,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path
):
    """
    Train GMRF MVAE - ICTAI style.

    Uses compute_elbo_dist for loss computation with full covariance matrix.
    """
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    beta = config['model'].get('beta', 1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss configuration (ICTAI alignment)
    recon_loss_type = config['training'].get('recon_loss', 'split_l1_mse')
    alpha_mse = config['training'].get('alpha_mse', 0.5)
    recon_weights = config['training'].get('recon_weights', None)

    print(f"\nTraining configuration:")
    print(f"  - epochs: {epochs}")
    print(f"  - lr: {lr}")
    print(f"  - beta: {beta}")
    print(f"  - recon_loss: {recon_loss_type}")
    print(f"  - alpha_mse: {alpha_mse}")
    print(f"  - recon_weights: {recon_weights}")

    best_loss = float('inf')
    mean_losses, mean_log_px_zs, mean_kls = [], [], []

    for epoch in range(epochs):
        model.train()

        batch_losses, batch_log_px_zs, batch_kls = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x, cond = batch
            x = x.to(device)  # [B, C, 64, 32]

            B, C = x.size(0), x.size(1)

            # Convert stacked tensor to list of tensors (ICTAI format)
            data = [x[:, i:i+1] for i in range(C)]

            # Forward pass (ICTAI style - stores results in model)
            model(data)

            # Compute ELBO (ICTAI style)
            elbo, log_px_z, kl_div = compute_elbo_dist(
                model, data, beta=beta,
                loss_type=recon_loss_type,
                alpha_mse=alpha_mse,
                weights=recon_weights
            )

            loss = -elbo

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_log_px_zs.append(log_px_z.item())
            batch_kls.append(kl_div.item())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{log_px_z.item():.4f}',
                'kl': f'{kl_div.item():.4f}',
            })

        # Compute epoch averages
        mean_loss = sum(batch_losses) / len(batch_losses)
        mean_log_px_z = sum(batch_log_px_zs) / len(batch_log_px_zs)
        mean_kl = sum(batch_kls) / len(batch_kls)

        mean_losses.append(mean_loss)
        mean_log_px_zs.append(mean_log_px_z)
        mean_kls.append(mean_kl)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {mean_loss:.4f} (log p(x|z): {mean_log_px_z:.4f}, KL: {mean_kl:.4f})")

        # Save generations at intervals (ICTAI style)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            save_generations(model, data, epoch, output_dir, recon_loss_type)

        # Save checkpoint at intervals
        if epoch != 0 and ((epoch + 1) % max(1, epochs // 2) == 0 or epoch == epochs - 1):
            checkpoint_path = output_dir / f'model_state_dict_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved model to {checkpoint_path}")

        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_path = output_dir / 'model_best.pt'
            torch.save(model.state_dict(), best_path)

    # Save training curves
    save_training_curves(mean_losses, mean_log_px_zs, mean_kls, output_dir / 'losses.png')

    print(f"\n{'='*60}")
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"{'='*60}\n")


def save_generations(model, data, epoch, output_dir, recon_loss_type):
    """Save cross-modal and unconditional generations."""
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid, save_image

    model.eval()
    with torch.no_grad():
        # Cross-modal generations
        try:
            recons_mat = model.self_and_cross_modal_generation([d[:1] for d in data])

            # Apply sigmoid if using BCE
            if recon_loss_type == 'bce':
                recons_mat = [[torch.sigmoid(r) for r in row] for row in recons_mat]

            # Save cross-modal grid
            save_cross_generations(recons_mat, output_dir / f'cross_gen_epoch_{epoch}.png')
        except Exception as e:
            print(f"  Warning: Could not save cross-modal generations: {e}")

        # Unconditional generations
        try:
            generation = model.generate_for_calculating_unconditional_coherence(20)
            if recon_loss_type == 'bce':
                generation = [torch.sigmoid(g) for g in generation]

            save_unconditional_generations(generation, output_dir / f'uncond_gen_{epoch}.png')
        except Exception as e:
            print(f"  Warning: Could not save unconditional generations: {e}")

    model.train()


def save_cross_generations(cross_generations, save_path):
    """Save cross-modal generation matrix."""
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    num_modalities = len(cross_generations)

    grid = []
    for row in cross_generations:
        row_images = torch.cat([img.squeeze(0).cpu() for img in row], dim=2)
        grid.append(row_images)

    full_grid = torch.cat(grid, dim=1)
    grid_image = make_grid(full_grid.unsqueeze(0), nrow=1, padding=2, normalize=False).permute(1, 2, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_image, cmap='gray')
    plt.axis("off")
    plt.title("Cross-Generation Results")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_unconditional_generations(generations, save_path):
    """Save unconditional generation grid."""
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    generations = [gen.cpu() for gen in generations]

    # Superposition
    superposition = torch.stack(generations, dim=0).sum(dim=0)
    all_generations = generations + [superposition]

    grids = []
    for tensor in all_generations:
        tensor = tensor.view(-1, tensor.shape[1], tensor.shape[-2], tensor.shape[-1])
        grid = make_grid(tensor, nrow=1, padding=2, normalize=False)
        grids.append(grid)

    full_grid = torch.cat(grids, dim=2)
    full_grid_image = full_grid.permute(1, 2, 0).numpy()
    full_grid_image = full_grid_image[::-1]  # Flip vertically

    plt.figure(figsize=(8, len(generations[0]) * 2))
    plt.imshow(full_grid_image, vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Unconditional Generations")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_training_curves(mean_losses, mean_log_px_zs, mean_kls, save_path):
    """Save training loss curves."""
    import matplotlib.pyplot as plt

    epochs = len(mean_losses)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), mean_losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), mean_log_px_zs, label='log p(x|z)')
    plt.xlabel('Epochs')
    plt.ylabel('log p(x|z)')
    plt.title('log p(x|z)')

    plt.subplot(1, 3, 3)
    plt.plot(range(epochs), mean_kls, label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train GMRF MVAE (ICTAI implementation)")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    # Resolve config path
    config_path = resolve_path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    seed = args.seed if args.seed is not None else config['training'].get('seed', 42)
    set_seed(seed)

    # Device
    device_str = config['model'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = Path(config['paths']['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Resolve paths
    root_dir = resolve_path(config['data']['root_dir'])
    condition_csv = resolve_path(config['data']['condition_csv'])
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

    # Create params object (ICTAI style)
    params_dict = {
        'latent_dim': config['model']['latent_dim'],
        'diagonal_transf': config['model'].get('diagonal_transf', 'softplus'),
        'hidden_dim': config['model']['hidden_dim'],
        'n_layers': config['model']['n_layers'],
        'device': str(device),
        'reduced_diag': config['model'].get('reduced_diag', False),
        'nf': config['model']['nf'],
        'nf_max_e': config['model'].get('nf_max_e', 512),
        'nf_max_d': config['model'].get('nf_max_d', 256),
    }
    params = Params(params_dict)

    # Create model (ICTAI style)
    model = Epure_GMMVAE(params).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Count params by component (like ICTAI)
    print(f"  - cov params: {sum(p.numel() for p in model.off_diag_cov.parameters()):,}")
    for i, vae in enumerate(model.modality_vaes):
        if i == 0:
            print(f"  - vae params (each): {sum(p.numel() for p in vae.parameters()):,}")
            print(f"    - enc params: {sum(p.numel() for p in vae.enc.parameters()):,}")
            print(f"    - dec params: {sum(p.numel() for p in vae.dec.parameters()):,}")

    # Train
    train(model, train_loader, config, device, output_dir)

    print(f"\nAll done! Model saved to: {output_dir}")


if __name__ == '__main__':
    main()
