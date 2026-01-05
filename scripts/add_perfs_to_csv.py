"""
Script to calculate performance metrics from binary mask images and add them to CSV.

Reads dimensions.csv, loads component mask images (fpu, group_km, full),
calculates raw features (masses, roundness) and derived performance metrics
relative to a baseline sample.

Outputs performances.csv with all original columns plus 16 new performance columns.
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from scipy.spatial.distance import euclidean


def load_and_binarize_image(img_path: Path, threshold: float = 0.5) -> Optional[np.ndarray]:
    """
    Load an image and convert it to binary mask.

    Args:
        img_path: Path to the image file
        threshold: Threshold for binarization (0.5 for float, 127 for uint8)

    Returns:
        Binary mask as numpy array, or None if file doesn't exist
    """
    if not img_path.exists():
        return None

    try:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        arr = np.array(img, dtype=np.float32)

        # Normalize to [0, 1] if needed
        if arr.max() > 1.0:
            arr = arr / 255.0
            threshold_val = threshold if threshold <= 1.0 else threshold / 255.0
        else:
            threshold_val = threshold

        # Binarize
        binary_mask = (arr > threshold_val).astype(np.uint8)
        return binary_mask

    except Exception as e:
        warnings.warn(f"Error loading image {img_path}: {e}")
        return None


def find_image_file(images_root: Path, prefix: str, component: str,
                   split: str) -> Optional[Path]:
    """
    Find image file with fallback logic.

    Tries:
    1. ROOT/split/component/mdbk_{prefix}_{component}.png
    2. ROOT/split/component/{prefix}.png

    Args:
        images_root: Root directory for images
        prefix: Sample prefix (without mdbk_)
        component: Component name (fpu, group_km, full)
        split: 'train' or 'test'

    Returns:
        Path to image file, or None if not found
    """
    comp_dir = images_root / split / component

    # Try with full naming convention
    candidate1 = comp_dir / f"mdbk_{prefix}_{component}.png"
    if candidate1.exists():
        return candidate1

    # Fallback: try without mdbk prefix and without component suffix
    candidate2 = comp_dir / f"{prefix}.png"
    if candidate2.exists():
        return candidate2

    # Try other extensions
    for ext in ['.jpg', '.jpeg', '.bmp']:
        candidate3 = comp_dir / f"mdbk_{prefix}_{component}{ext}"
        if candidate3.exists():
            return candidate3

    return None


def calculate_roundness(mask: np.ndarray, top_frac: float = 0.20,
                        mirror_full: bool = False) -> float:
    """
    Calculate roundness of the top arc using sagitta/chord method.

    Args:
        mask: Binary mask
        top_frac: Fraction of image height to consider as "top"
        mirror_full: If True, reconstruct full epure by mirroring

    Returns:
        Roundness value (sagitta/chord ratio), or NaN if calculation fails
    """
    try:
        # Optional mirroring
        if mirror_full:
            # Mirror horizontally: [flipped | original]
            mask = np.concatenate([np.fliplr(mask), mask], axis=1)

        # Find contours
        contours = measure.find_contours(mask, 0.5)
        if len(contours) == 0:
            return np.nan

        # Take the longest contour
        contour = max(contours, key=len)

        # Define "top" zone based on y-coordinate
        y_coords = contour[:, 0]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_threshold = y_min + top_frac * (y_max - y_min)

        # Filter points in top zone
        top_mask = y_coords <= y_threshold
        if top_mask.sum() < 2:
            return np.nan

        top_points = contour[top_mask]

        # Find leftmost and rightmost points
        x_coords = top_points[:, 1]
        left_idx = x_coords.argmin()
        right_idx = x_coords.argmax()

        P_L = top_points[left_idx]
        P_R = top_points[right_idx]

        # Calculate chord length
        chord = euclidean(P_L, P_R)
        if chord == 0:
            return np.nan

        # Calculate sagitta (max perpendicular distance to chord)
        # Line from P_L to P_R: we need distance from each point to this line
        # Vector along chord
        chord_vec = P_R - P_L
        chord_unit = chord_vec / chord

        # Perpendicular distances
        distances = []
        for point in top_points:
            vec = point - P_L
            # Project onto chord
            proj_length = np.dot(vec, chord_unit)
            proj_point = P_L + proj_length * chord_unit
            # Perpendicular distance
            perp_dist = euclidean(point, proj_point)
            distances.append(perp_dist)

        sagitta = max(distances) if distances else 0.0

        # Roundness = sagitta / chord
        roundness = sagitta / chord
        return roundness

    except Exception as e:
        warnings.warn(f"Error calculating roundness: {e}")
        return np.nan


def calculate_features_for_sample(
    images_root: Path,
    prefix: str,
    split: str,
    comp_side: str,
    comp_top: str,
    comp_full: str,
    rho_side: float,
    rho_top: float,
    rho_total: float,
    roundness_source: str,
    top_frac: float,
    mirror_full: bool
) -> Dict[str, float]:
    """
    Calculate all raw features for a single sample.

    Returns:
        Dictionary with keys: m_top, m_side, m_total, round_top
        Values are NaN if images are missing
    """
    features = {
        'm_top': np.nan,
        'm_side': np.nan,
        'm_total': np.nan,
        'round_top': np.nan
    }

    # Load masks
    mask_side = None
    mask_top = None
    mask_full = None

    # Load side component
    img_path = find_image_file(images_root, prefix, comp_side, split)
    if img_path:
        mask_side = load_and_binarize_image(img_path)

    # Load top component
    img_path = find_image_file(images_root, prefix, comp_top, split)
    if img_path:
        mask_top = load_and_binarize_image(img_path)

    # Load full component
    img_path = find_image_file(images_root, prefix, comp_full, split)
    if img_path:
        mask_full = load_and_binarize_image(img_path)

    # Calculate masses
    if mask_top is not None:
        features['m_top'] = mask_top.sum() * rho_top

    if mask_side is not None:
        features['m_side'] = mask_side.sum() * rho_side

    if mask_full is not None:
        features['m_total'] = mask_full.sum() * rho_total

    # Calculate roundness
    roundness_mask = None
    if roundness_source == comp_full and mask_full is not None:
        roundness_mask = mask_full
    elif roundness_source == comp_top and mask_top is not None:
        roundness_mask = mask_top
    elif roundness_source == comp_side and mask_side is not None:
        roundness_mask = mask_side

    if roundness_mask is not None:
        features['round_top'] = calculate_roundness(roundness_mask, top_frac, mirror_full)

    return features


def main():
    parser = argparse.ArgumentParser(
        description='Add performance metrics to CSV from mask images'
    )

    # I/O paths
    parser.add_argument('--csv_in', type=str, default='data/epure/dimensions.csv',
                       help='Input CSV path')
    parser.add_argument('--csv_out', type=str, default='data/epure/performances.csv',
                       help='Output CSV path')
    parser.add_argument('--images_root', type=str, default='data/epure',
                       help='Root directory for images')

    # Component names
    parser.add_argument('--comp_side', type=str, default='fpu',
                       help='Side component directory name')
    parser.add_argument('--comp_top', type=str, default='group_km',
                       help='Top component directory name')
    parser.add_argument('--comp_full', type=str, default='full',
                       help='Full assembly component directory name')

    # Baseline
    parser.add_argument('--baseline_prefix', type=str, default='binc_3b3_709014_6',
                       help='Baseline sample prefix (without mdbk_)')

    # Density coefficients
    parser.add_argument('--rho_top', type=float, default=1.0,
                       help='Density coefficient for top component')
    parser.add_argument('--rho_side', type=float, default=1.0,
                       help='Density coefficient for side component')
    parser.add_argument('--rho_total', type=float, default=1.0,
                       help='Density coefficient for total assembly')

    # Roundness parameters
    parser.add_argument('--roundness_source', type=str, default='full',
                       choices=['full', 'fpu', 'group_km'],
                       help='Which mask to use for roundness calculation')
    parser.add_argument('--top_frac', type=float, default=0.20,
                       help='Fraction of height to consider as top arc')
    parser.add_argument('--mirror_full', type=int, default=0,
                       help='Mirror the mask horizontally before roundness calc (0 or 1)')

    # Performance metric coefficients
    parser.add_argument('--alpha_cons', type=float, default=1.0,
                       help='Coefficient for consumption metric')
    parser.add_argument('--alpha_rigid', type=float, default=0.5,
                       help='Coefficient for rigidity metric')
    parser.add_argument('--w_top', type=float, default=0.6,
                       help='Weight for top mass in rigidity')
    parser.add_argument('--w_side', type=float, default=0.4,
                       help='Weight for side mass in rigidity')
    parser.add_argument('--alpha_life', type=float, default=0.8,
                       help='Coefficient for lifetime metric')
    parser.add_argument('--alpha_stab', type=float, default=-1.0,
                       help='Coefficient for stability metric (negative because rounder = less stable)')

    # Debug options
    parser.add_argument('--debug_prefix', type=str, default=None,
                       help='Optional: save debug visualization for this prefix')

    args = parser.parse_args()

    # Convert paths
    csv_in = Path(args.csv_in)
    csv_out = Path(args.csv_out)
    images_root = Path(args.images_root)
    mirror_full = bool(args.mirror_full)

    print("=" * 80)
    print("ADD PERFORMANCE METRICS TO CSV")
    print("=" * 80)
    print(f"Input CSV:      {csv_in}")
    print(f"Output CSV:     {csv_out}")
    print(f"Images root:    {images_root}")
    print(f"Baseline:       {args.baseline_prefix}")
    print(f"Components:     side={args.comp_side}, top={args.comp_top}, full={args.comp_full}")
    print(f"Roundness src:  {args.roundness_source} (top_frac={args.top_frac}, mirror={mirror_full})")
    print()

    # Load CSV
    if not csv_in.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_in}")

    df = pd.read_csv(csv_in)
    print(f"✓ Loaded CSV: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print()

    # Check required columns
    if 'matching' not in df.columns:
        raise ValueError("CSV must contain 'matching' column")
    if 'train' not in df.columns or 'test' not in df.columns:
        raise ValueError("CSV must contain 'train' and 'test' columns")

    # Find baseline row
    baseline_rows = df[df['matching'] == args.baseline_prefix]
    if len(baseline_rows) == 0:
        raise ValueError(f"Baseline prefix '{args.baseline_prefix}' not found in CSV")
    if len(baseline_rows) > 1:
        warnings.warn(f"Multiple rows found for baseline prefix, using first one")

    baseline_row = baseline_rows.iloc[0]
    baseline_split = 'train' if baseline_row['train'] else 'test'

    print(f"✓ Found baseline: {args.baseline_prefix} (split={baseline_split})")

    # Calculate baseline features
    print("Calculating baseline features...")
    baseline_features = calculate_features_for_sample(
        images_root, args.baseline_prefix, baseline_split,
        args.comp_side, args.comp_top, args.comp_full,
        args.rho_side, args.rho_top, args.rho_total,
        args.roundness_source, args.top_frac, mirror_full
    )

    # Check baseline features are valid
    if any(np.isnan(v) for v in baseline_features.values()):
        raise ValueError(
            f"Baseline images missing or invalid! Features: {baseline_features}\n"
            f"Check that images exist in: {images_root}/{baseline_split}/"
        )

    m_top0 = baseline_features['m_top']
    m_side0 = baseline_features['m_side']
    m_total0 = baseline_features['m_total']
    round_top0 = baseline_features['round_top']

    print(f"  Baseline features:")
    print(f"    m_top0    = {m_top0:.4f}")
    print(f"    m_side0   = {m_side0:.4f}")
    print(f"    m_total0  = {m_total0:.4f}")
    print(f"    round_top0 = {round_top0:.6f}")
    print()

    # Initialize new columns
    new_cols = {
        'm_top': [],
        'm_side': [],
        'm_total': [],
        'round_top': [],
        'dm_top': [],
        'dm_side': [],
        'dm_total': [],
        'dround_top': [],
        'd_cons': [],
        'd_rigid': [],
        'd_life': [],
        'd_stab': []
    }

    # Process each row
    print("Processing samples...")
    missing_count = 0

    for idx, row in df.iterrows():
        prefix = row['matching']
        split = 'train' if row['train'] else 'test'

        # Calculate features
        features = calculate_features_for_sample(
            images_root, prefix, split,
            args.comp_side, args.comp_top, args.comp_full,
            args.rho_side, args.rho_top, args.rho_total,
            args.roundness_source, args.top_frac, mirror_full
        )

        # Store raw features
        new_cols['m_top'].append(features['m_top'])
        new_cols['m_side'].append(features['m_side'])
        new_cols['m_total'].append(features['m_total'])
        new_cols['round_top'].append(features['round_top'])

        # Calculate deltas
        if prefix == args.baseline_prefix:
            # Baseline gets zeros
            dm_top = 0.0
            dm_side = 0.0
            dm_total = 0.0
            dround_top = 0.0
        elif any(np.isnan(v) for v in features.values()):
            # Missing images -> NaN
            dm_top = np.nan
            dm_side = np.nan
            dm_total = np.nan
            dround_top = np.nan
            missing_count += 1
            warnings.warn(f"Missing images for {prefix} (split={split})")
        else:
            # Calculate relative deltas
            dm_top = (features['m_top'] - m_top0) / m_top0
            dm_side = (features['m_side'] - m_side0) / m_side0
            dm_total = (features['m_total'] - m_total0) / m_total0
            dround_top = (features['round_top'] - round_top0) / round_top0

        new_cols['dm_top'].append(dm_top)
        new_cols['dm_side'].append(dm_side)
        new_cols['dm_total'].append(dm_total)
        new_cols['dround_top'].append(dround_top)

        # Calculate performance metrics
        if np.isnan(dm_total):
            d_cons = np.nan
            d_rigid = np.nan
            d_life = np.nan
            d_stab = np.nan
        else:
            d_cons = args.alpha_cons * dm_total
            d_rigid = args.alpha_rigid * (args.w_top * dm_top + args.w_side * dm_side)
            d_life = args.alpha_life * dm_top
            d_stab = args.alpha_stab * dround_top

        new_cols['d_cons'].append(d_cons)
        new_cols['d_rigid'].append(d_rigid)
        new_cols['d_life'].append(d_life)
        new_cols['d_stab'].append(d_stab)

    print(f"✓ Processed {len(df)} samples ({missing_count} with missing images)")
    print()

    # Add new columns to dataframe
    for col_name, col_data in new_cols.items():
        df[col_name] = col_data

    # Print summary statistics
    print("Summary statistics:")
    print("-" * 80)
    for col in ['m_top', 'm_side', 'm_total', 'round_top',
                'dm_top', 'dm_side', 'dm_total', 'dround_top',
                'd_cons', 'd_rigid', 'd_life', 'd_stab']:
        values = df[col].dropna()
        if len(values) > 0:
            print(f"  {col:12s}: min={values.min():8.4f}, mean={values.mean():8.4f}, max={values.max():8.4f}")
        else:
            print(f"  {col:12s}: all NaN")
    print()

    # Save output CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"✓ Saved output CSV: {csv_out}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Total rows: {len(df)}")
    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
