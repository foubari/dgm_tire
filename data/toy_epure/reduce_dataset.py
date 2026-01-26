#!/usr/bin/env python3
"""
Reduce toy dataset by randomly selecting K samples.

This script creates a reduced version of performances.csv by randomly sampling
K entries while maintaining the specified train/test ratio.

Usage:
    # Basic: Create performances_1000.csv with 800 train + 200 test
    python reduce_dataset.py --k 1000

    # Custom ratio: 90% train, 10% test
    python reduce_dataset.py --k 1000 --ratio 0.9

    # Different sizes for experimentation
    python reduce_dataset.py --k 500 --seed 42
    python reduce_dataset.py --k 2000 --seed 42

    # Also copy images to a new directory (optional)
    python reduce_dataset.py --k 1000 --copy-images

After running, update your config to use the new CSV:
    condition_csv: data/toy_epure/performances_1000.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def reduce_dataset(
    input_csv: Path,
    k: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    copy_images: bool = False
):
    """
    Reduce dataset to K samples with specified train/test ratio.

    Args:
        input_csv: Path to original performances.csv
        k: Total number of samples to keep
        train_ratio: Ratio of train samples (default: 0.8)
        seed: Random seed for reproducibility
        copy_images: Whether to copy images to new directory
    """
    np.random.seed(seed)

    # Load CSV
    df = pd.read_csv(input_csv)
    total_samples = len(df)

    print(f"Original dataset: {total_samples} samples")
    print(f"  Train: {df['train'].sum()}")
    print(f"  Test: {df['test'].sum()}")

    # Calculate split
    k_train = int(k * train_ratio)
    k_test = k - k_train

    print(f"\nReducing to {k} samples (train={k_train}, test={k_test})")

    # Sample separately from train and test
    train_df = df[df['train'] == True]
    test_df = df[df['test'] == True]

    # Check we have enough samples
    if k_train > len(train_df):
        print(f"WARNING: Requested {k_train} train samples but only {len(train_df)} available")
        k_train = len(train_df)
    if k_test > len(test_df):
        print(f"WARNING: Requested {k_test} test samples but only {len(test_df)} available")
        k_test = len(test_df)

    # Random sampling
    sampled_train = train_df.sample(n=k_train, random_state=seed)
    sampled_test = test_df.sample(n=k_test, random_state=seed)

    # Combine and sort by matching column for consistency
    reduced_df = pd.concat([sampled_train, sampled_test], ignore_index=True)
    reduced_df = reduced_df.sort_values('matching').reset_index(drop=True)

    # Save CSV
    output_csv = input_csv.parent / f"performances_{k}.csv"
    reduced_df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")
    print(f"  Train: {reduced_df['train'].sum()}")
    print(f"  Test: {reduced_df['test'].sum()}")

    # Optionally copy images
    if copy_images:
        copy_images_to_new_dir(reduced_df, input_csv.parent, k)

    print(f"\n" + "="*50)
    print(f"Done! To use this reduced dataset:")
    print(f"  1. Edit your config YAML")
    print(f"  2. Change: condition_csv: data/toy_epure/{output_csv.name}")
    print(f"="*50)

    return output_csv


def copy_images_to_new_dir(df: pd.DataFrame, base_dir: Path, k: int):
    """Copy selected images to new directory."""
    output_dir = base_dir.parent / f"toy_epure_{k}"
    output_dir.mkdir(exist_ok=True)

    print(f"\nCopying images to {output_dir}...")

    # Component directories for toy dataset
    component_dirs = ['group_nc', 'group_km', 'fpu']

    for split in ['train', 'test']:
        split_df = df[df[split] == True]

        if len(split_df) == 0:
            continue

        # Create directories
        for comp in component_dirs:
            (output_dir / split / comp).mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'full').mkdir(parents=True, exist_ok=True)

        # Copy files
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split}"):
            matching = row['matching']

            # Copy component images
            for comp in component_dirs:
                src = base_dir / split / comp / f"{matching}_{comp}.png"
                dst = output_dir / split / comp / f"{matching}_{comp}.png"
                if src.exists():
                    shutil.copy2(src, dst)

            # Copy full image
            src_full = base_dir / split / 'full' / f"{matching}_full.png"
            dst_full = output_dir / split / 'full' / f"{matching}_full.png"
            if src_full.exists():
                shutil.copy2(src_full, dst_full)

    # Also copy the reduced CSV to the new directory
    csv_src = base_dir / f"performances_{k}.csv"
    csv_dst = output_dir / f"performances_{k}.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, csv_dst)

    print(f"Images copied to: {output_dir}")
    print(f"CSV also copied to: {csv_dst}")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce toy dataset to K samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python reduce_dataset.py --k 1000              # 800 train + 200 test
    python reduce_dataset.py --k 500 --ratio 0.9   # 450 train + 50 test
    python reduce_dataset.py --k 2000 --seed 123   # Different random selection
    python reduce_dataset.py --k 1000 --copy-images  # Also copy images
        """
    )
    parser.add_argument('--k', type=int, default=1000,
                        help='Total number of samples (default: 1000)')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Train ratio, e.g., 0.8 = 80%% train (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--copy-images', action='store_true',
                        help='Copy images to new directory data/toy_epure_{K}/')
    parser.add_argument('--input-csv', type=str, default=None,
                        help='Input CSV path (default: performances.csv in same directory)')

    args = parser.parse_args()

    # Determine input CSV path
    if args.input_csv:
        input_csv = Path(args.input_csv)
    else:
        # Default: performances.csv in same directory as this script
        script_dir = Path(__file__).parent
        input_csv = script_dir / 'performances.csv'

    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found")
        print("Make sure you're running from the correct directory or use --input-csv")
        return

    reduce_dataset(
        input_csv=input_csv,
        k=args.k,
        train_ratio=args.ratio,
        seed=args.seed,
        copy_images=args.copy_images
    )


if __name__ == '__main__':
    main()