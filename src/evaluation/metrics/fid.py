"""
FID (Fréchet Inception Distance) computation for EpureDGM.

Adapted from ICTAI fid_utils.py and fid_score.py.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Union, Optional
from PIL import Image
from tqdm import tqdm
from scipy import linalg

# Import from local copies (no ICTAI dependency)
from .inception import InceptionV3
from .fid_score import calculate_frechet_distance


def pil_to_rgb_np(path: Path) -> np.ndarray:
    """Convert PIL image to RGB numpy array normalized to [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0


@torch.no_grad()
def get_activations_from_paths(
    file_paths: List[Path],
    model: InceptionV3,
    device: torch.device,
    batch_size: int = 64,
    dims: int = 2048,
) -> np.ndarray:
    """Extract Inception features from image paths."""
    n_batch = (len(file_paths) + batch_size - 1) // batch_size
    feats = np.empty((len(file_paths), dims), dtype=np.float32)
    ptr = 0

    for bi in tqdm(range(n_batch), desc="Extracting features", leave=False):
        paths = file_paths[bi * batch_size:(bi + 1) * batch_size]
        imgs = [pil_to_rgb_np(p) for p in paths]
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)  # (B,3,H,W)
        batch = torch.from_numpy(imgs).to(device).float()

        pred = model(batch)[0]  # pool3
        if pred.shape[-1] != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))

        feats[ptr:ptr + pred.size(0)] = pred.squeeze(-1).squeeze(-1).cpu().numpy()
        ptr += pred.size(0)

    return feats


def fid_from_activations(real_acts: np.ndarray, gen_acts: np.ndarray) -> float:
    """Compute FID from pre-computed activations."""
    mu_real = real_acts.mean(axis=0)
    mu_gen = gen_acts.mean(axis=0)
    sigma_real = np.cov(real_acts, rowvar=False)
    sigma_gen = np.cov(gen_acts, rowvar=False)

    return float(calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen))


def compute_fid(
    real_dir: Union[str, Path],
    gen_dir: Union[str, Path],
    inception_path: Union[str, Path] = "data/pt_inception-2015-12-05-6726825d.pth",
    batch_size: int = 64,
    device: Union[str, torch.device] = 'cuda',
    cache_dir: Optional[Path] = None,
) -> float:
    """
    Compute FID between real and generated images.

    Args:
        real_dir: Directory with real images
        gen_dir: Directory with generated images
        inception_path: Path to Inception model weights
        batch_size: Batch size for feature extraction
        device: Device to use (cuda or cpu)
        cache_dir: Optional directory to cache activations

    Returns:
        FID score
    """
    real_dir = Path(real_dir)
    gen_dir = Path(gen_dir)
    inception_path = Path(inception_path)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load Inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], path_state_dict=str(inception_path)).to(device)
    model.eval()

    # List image files
    def list_imgs(folder):
        f = Path(folder)
        return sorted([*f.glob('*.png'), *f.glob('*.jpg'), *f.glob('*.jpeg')])

    real_files = list_imgs(real_dir)
    gen_files = list_imgs(gen_dir)

    if len(real_files) == 0:
        raise ValueError(f"No images found in real directory: {real_dir}")
    if len(gen_files) == 0:
        raise ValueError(f"No images found in generated directory: {gen_dir}")

    # Try to load from cache
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        real_cache = cache_dir / f"real_{real_dir.name}.npy"
        gen_cache = cache_dir / f"gen_{gen_dir.name}.npy"

        if real_cache.exists():
            real_acts = np.load(real_cache)
        else:
            real_acts = get_activations_from_paths(real_files, model, device, batch_size)
            np.save(real_cache, real_acts)

        if gen_cache.exists():
            gen_acts = np.load(gen_cache)
        else:
            gen_acts = get_activations_from_paths(gen_files, model, device, batch_size)
            np.save(gen_cache, gen_acts)
    else:
        # No caching
        real_acts = get_activations_from_paths(real_files, model, device, batch_size)
        gen_acts = get_activations_from_paths(gen_files, model, device, batch_size)

    # Compute FID
    fid_score = fid_from_activations(real_acts, gen_acts)

    return fid_score


def compute_fid_with_shared_cache(
    real_dir: Union[str, Path],
    gen_dir: Union[str, Path],
    model_name: str,
    dataset_name: str,
    inception_path: Union[str, Path] = "data/pt_inception-2015-12-05-6726825d.pth",
    batch_size: int = 64,
    device: Union[str, torch.device] = 'cuda',
    cache_root: Optional[Path] = None,
) -> float:
    """
    Compute FID with two-tier caching strategy (ICTAI-style):
    - Real data features: Shared across all models (fid_cache/{dataset}/real_{component}.npy)
    - Generated features: Model-specific (fid_cache/{dataset}/{model}/gen_{component}.npy)
    
    Args:
        real_dir: Directory with real images
        gen_dir: Directory with generated images
        model_name: Model identifier for cache organization
        dataset_name: Dataset name (toy/epure) for cache organization
        inception_path: Path to Inception model weights
        batch_size: Batch size for feature extraction
        device: Device to use
        cache_root: Root directory for cache (default: ./fid_cache)
    
    Returns:
        FID score
    """
    real_dir = Path(real_dir)
    gen_dir = Path(gen_dir)
    inception_path = Path(inception_path)
    
    # Setup cache directories
    if cache_root is None:
        cache_root = Path("fid_cache")
    
    dataset_cache = cache_root / dataset_name  # Shared real data cache
    model_cache = dataset_cache / model_name   # Model-specific generated cache
    
    dataset_cache.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load Inception model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], path_state_dict=str(inception_path)).to(device)
    model.eval()
    
    # List image files
    def list_imgs(folder):
        f = Path(folder)
        return sorted([*f.glob('*.png'), *f.glob('*.jpg'), *f.glob('*.jpeg')])
    
    real_files = list_imgs(real_dir)
    gen_files = list_imgs(gen_dir)
    
    if len(real_files) == 0:
        raise ValueError(f"No images found in real directory: {real_dir}")
    if len(gen_files) == 0:
        raise ValueError(f"No images found in generated directory: {gen_dir}")
    
    # Cache real data (shared across models)
    real_cache_name = f"real_{real_dir.name}.npy"
    real_cache = dataset_cache / real_cache_name
    
    if real_cache.exists():
        print(f"  Loading cached real features: {real_cache}")
        real_acts = np.load(real_cache)
    else:
        print(f"  Computing real features (will be cached)...")
        real_acts = get_activations_from_paths(real_files, model, device, batch_size)
        np.save(real_cache, real_acts)
        print(f"  Cached real features: {real_cache}")
    
    # Cache generated data (model-specific)
    gen_cache_name = f"gen_{gen_dir.name}.npy"
    gen_cache = model_cache / gen_cache_name
    
    if gen_cache.exists():
        print(f"  Loading cached generated features: {gen_cache}")
        gen_acts = np.load(gen_cache)
    else:
        print(f"  Computing generated features...")
        gen_acts = get_activations_from_paths(gen_files, model, device, batch_size)
        np.save(gen_cache, gen_acts)
        print(f"  Cached generated features: {gen_cache}")
    
    # Compute FID
    fid_score = fid_from_activations(real_acts, gen_acts)

    return fid_score


# ============================================================================
# Cacheable Metric Interface
# ============================================================================

from .base_metric import CacheableMetric
from typing import Dict, Any


class FIDMetric(CacheableMetric):
    """FID metric with automatic caching via CacheableMetric interface."""

    def __init__(self, cache_root: Path, dataset_name: str, inception_path: str, device: str = 'cuda'):
        """
        Initialize FID metric.

        Args:
            cache_root: Root directory for evaluation caches
            dataset_name: Dataset name (toy/epure)
            inception_path: Path to Inception-V3 model weights
            device: Device for computation (cuda/cpu)
        """
        super().__init__(cache_root, dataset_name)
        self.inception_path = Path(inception_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.inception_model = None  # Lazy load

    @property
    def metric_name(self) -> str:
        return "fid"

    def _get_inception_model(self):
        """Lazy load Inception model."""
        if self.inception_model is None:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception_model = InceptionV3(
                [block_idx],
                path_state_dict=str(self.inception_path)
            ).to(self.device)
            self.inception_model.eval()
        return self.inception_model

    def _list_image_files(self, directory: Path) -> List[Path]:
        """List all image files in directory."""
        return sorted([
            *directory.glob('*.png'),
            *directory.glob('*.jpg'),
            *directory.glob('*.jpeg')
        ])

    def compute_real_data(self, real_dirs: Dict[str, Path], **kwargs) -> np.ndarray:
        """
        Extract Inception features from real images.

        Args:
            real_dirs: {'full': Path} to real images directory
            **kwargs: Can include batch_size

        Returns:
            Numpy array of shape (N, 2048) with Inception features
        """
        real_dir = real_dirs.get('full')
        if real_dir is None:
            # Try to get 'full' from component dirs
            first_comp = list(real_dirs.keys())[0]
            real_dir = real_dirs[first_comp].parent / 'full'

        batch_size = kwargs.get('batch_size', 64)
        model = self._get_inception_model()

        real_files = self._list_image_files(Path(real_dir))
        if len(real_files) == 0:
            raise ValueError(f"No images found in {real_dir}")

        activations = get_activations_from_paths(
            file_paths=real_files,
            model=model,
            batch_size=batch_size,
            device=self.device
        )
        return activations  # Shape: (N, 2048)

    def compute_generated_data(self, gen_dirs: Dict[str, Path], **kwargs) -> np.ndarray:
        """
        Extract Inception features from generated images.

        Args:
            gen_dirs: {'full': Path} to generated images directory
            **kwargs: Can include batch_size

        Returns:
            Numpy array of shape (M, 2048) with Inception features
        """
        gen_dir = gen_dirs.get('full')
        if gen_dir is None:
            # Try to get 'full' from component dirs
            first_comp = list(gen_dirs.keys())[0]
            gen_dir = gen_dirs[first_comp].parent / 'full'

        batch_size = kwargs.get('batch_size', 64)
        model = self._get_inception_model()

        gen_files = self._list_image_files(Path(gen_dir))
        if len(gen_files) == 0:
            raise ValueError(f"No images found in {gen_dir}")

        activations = get_activations_from_paths(
            file_paths=gen_files,
            model=model,
            batch_size=batch_size,
            device=self.device
        )
        return activations  # Shape: (M, 2048)

    def compute_distance(self, real_data: np.ndarray, gen_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Compute FID from cached Inception features.

        Args:
            real_data: Real activations (N, 2048)
            gen_data: Generated activations (M, 2048)
            **kwargs: Unused

        Returns:
            {'fid': float}
        """
        fid_score = fid_from_activations(real_data, gen_data)
        return {'fid': float(fid_score)}
