# ══════════════════════════════════════════════════════════════
#  METRIC TOOLKIT  ▸  FID / KID  (+ optional bootstrap)
#                   *model-aware caching*
# ══════════════════════════════════════════════════════════════
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Callable, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .inception import InceptionV3
from .fid_score import calculate_frechet_distance
# --------------------------------------------------------------

# add a default set only if the caller does not pass its own
COMPONENTS = ['group_nc', 'group_km', 'bt', 'gi', 'fpu', 'tpc']


# ────────── activation utilities ──────────────────────────────
def pil_to_rgb_np(path: Path) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0


@torch.no_grad()
def get_activations_paths(
        file_paths: List[Path],
        model: InceptionV3,
        device: torch.device,
        batch_size: int = 64,
        dims: int = 2048,
) -> np.ndarray:
    n_batch = (len(file_paths) + batch_size - 1) // batch_size
    feats = np.empty((len(file_paths), dims), dtype=np.float32)
    ptr = 0
    for bi in tqdm(range(n_batch), leave=False):
        paths = file_paths[bi * batch_size:(bi + 1) * batch_size]
        imgs = [pil_to_rgb_np(p) for p in paths]
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)   # (B,3,H,W)
        batch = torch.from_numpy(imgs).to(device).float()

        pred = model(batch)[0]                        # pool3
        if pred.shape[-1] != 1:
            pred = torch.nn.functional.adaptive_avg_pool2d(pred, (1, 1))
        feats[ptr:ptr + pred.size(0)] = pred.squeeze(-1).squeeze(-1).cpu().numpy()
        ptr += pred.size(0)
    return feats


# ────────── distance functions ────────────────────────────────
def fid_from_activations(a: np.ndarray, b: np.ndarray) -> float:
    mu1, mu2 = a.mean(0), b.mean(0)
    sigma1, sigma2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


def kid_from_activations(a: np.ndarray,
                         b: np.ndarray,
                         degree: int = 3,
                         gamma: float = None,
                         coef0: float = 1.0) -> float:
    if gamma is None:
        gamma = 1.0 / a.shape[1]

    def _poly(X, Y):
        return (gamma * X.dot(Y.T) + coef0) ** degree

    m, n = len(a), len(b)
    XX = _poly(a, a); YY = _poly(b, b); XY = _poly(a, b)
    mmd = (XX.sum() - np.trace(XX)) / (m * (m - 1))
    mmd += (YY.sum() - np.trace(YY)) / (n * (n - 1))
    mmd -= 2 * XY.mean()
    return float(mmd)


def bootstrap_metric(
    a: np.ndarray,
    b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    sample_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[float, float]:
    """Return mean ± std of boot-strapped metric."""
    rng = rng or np.random.default_rng()
    m = sample_size or len(a)
    n = sample_size or len(b)
    vals = []
    for _ in tqdm(range(n_boot), leave=False):
        ia = rng.choice(len(a), m, replace=True)
        ib = rng.choice(len(b), n, replace=True)
        vals.append(metric_fn(a[ia], b[ib]))
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


# ────────── main entry point ──────────────────────────────────
def compute_metrics_all(
    real_root: Union[str, Path],
    uncond_root: Union[str, Path],
    cond_root: Union[str, Path],
    inception_state_dict: Union[str, Path],
    *,
    model_name: str,                          # ← NEW: identifier for this run
    components: Sequence[str] = COMPONENTS,
    cond_components = None, 
    device: Union[str, torch.device] = 'cuda',
    batch_size: int = 64,
    do_fid: bool = True,
    do_kid: bool = True,
    do_fid_boot: bool = True,
    do_kid_boot: bool = True,
    n_boot: int = 1000,
    sample_size: Optional[int] = None,
    conditional = True,
    unconditional = True,
    cache_dir: Union[str, Path] = './fid_cache',
) -> Dict[str, dict]:
    """
    Returns a nested dict, e.g.
        results['fid']['uncond']['bt']          = 19.7
        results['kid_boot']['cond']['gi']['bt'] = (0.0031, 0.0004)
    """
    if cond_components is None:
        cond_components = components
    device = torch.device(device)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], path_state_dict=inception_state_dict).to(device)

    cache_dir = Path(cache_dir)
    data_cache = cache_dir                     # shared for real activations
    run_cache  = cache_dir / model_name        # model-specific activations
    run_cache.mkdir(parents=True, exist_ok=True)

    # ----- helper that respects shared vs. model-specific cache -------------
    def cached(tag: str, files: List[Path], *, shared: bool = False) -> np.ndarray:
        folder = data_cache if shared else run_cache
        folder.mkdir(parents=True, exist_ok=True)
        p = folder / f'{tag}.npy'
        print(f'[Saving activations] → {p}')  # ← Add this line
        if p.exists():
            return np.load(p)
        feats = get_activations_paths(files, model, device, batch_size)
        np.save(p, feats)
        return feats

    def list_imgs(folder):
        f = Path(folder)
        return sorted([*f.glob('*.png'), *f.glob('*.jpg'), *f.glob('*.jpeg')])

    # ── collect / load real activations once (shared) ───────────────────────
    real_act = {
        comp: cached(f'real_{comp}', list_imgs(Path(real_root) / comp), shared=True)
        for comp in [*components, 'full']
    }

    # container for everything
    results: Dict[str, dict] = {
        'fid':        {'uncond': {}, 'cond': {}},
        'kid':        {'uncond': {}, 'cond': {}},
        'fid_boot':   {'uncond': {}, 'cond': {}},
        'kid_boot':   {'uncond': {}, 'cond': {}},
    }

    want = {
        'fid': do_fid,
        'kid': do_kid,
        'fid_boot': do_fid_boot,
        'kid_boot': do_kid_boot,
    }

    # ── unconditional metrics ───────────────────────────────────────────────
    if unconditional:
        for comp in tqdm([*components, 'full'], desc='Unconditional'):
            gen_files = list_imgs(Path(uncond_root) / comp)
            gen_act = cached(f'uncond_{comp}', gen_files)          # model-specific
    
            if want['fid']:
                results['fid']['uncond'][comp] = fid_from_activations(real_act[comp], gen_act)
            if want['kid']:
                results['kid']['uncond'][comp] = kid_from_activations(real_act[comp], gen_act)
            if want['fid_boot']:
                results['fid_boot']['uncond'][comp] = bootstrap_metric(
                    real_act[comp], gen_act, fid_from_activations, n_boot, sample_size)
            if want['kid_boot']:
                results['kid_boot']['uncond'][comp] = bootstrap_metric(
                    real_act[comp], gen_act, kid_from_activations, n_boot, sample_size)

    # ── conditional metrics ─────────────────────────────────────────────────
    if conditional:
        cond_root = Path(cond_root)
        for cond in tqdm(cond_components, desc='Conditional'):
            results['fid']['cond'][cond] = {}
            results['kid']['cond'][cond] = {}
            results['fid_boot']['cond'][cond] = {}
            results['kid_boot']['cond'][cond] = {}
    
            for tgt in [*components, 'full']:
                folder = cond_root / cond / tgt
                if not folder.is_dir():
                    continue
                gen_files = list_imgs(folder)
                gen_act = cached(f'cond_{cond}_{tgt}', gen_files)  # model-specific
    
                if want['fid']:
                    results['fid']['cond'][cond][tgt] = fid_from_activations(real_act[tgt], gen_act)
                if want['kid']:
                    results['kid']['cond'][cond][tgt] = kid_from_activations(real_act[tgt], gen_act)
                if want['fid_boot']:
                    results['fid_boot']['cond'][cond][tgt] = bootstrap_metric(
                        real_act[tgt], gen_act, fid_from_activations, n_boot, sample_size)
                if want['kid_boot']:
                    results['kid_boot']['cond'][cond][tgt] = bootstrap_metric(
                        real_act[tgt], gen_act, kid_from_activations, n_boot, sample_size)

    # prune disabled metric branches
    for k in list(results):
        if not want[k]:
            del results[k]

    return results
