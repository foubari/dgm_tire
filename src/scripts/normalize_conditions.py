#!/usr/bin/env python3
"""
Script de normalisation des colonnes de conditioning dans les CSV.

Normalise les colonnes d_cons, d_rigid, d_life, d_stab en utilisant Min-Max [-1, 1]
pour compatibilité avec le toy dataset et les réseaux utilisant tanh.

Usage:
    python src_new/scripts/normalize_conditions.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def normalize_csv(csv_path: Path, ranges: dict, dry_run: bool = False, no_interactive: bool = False) -> None:
    """
    Ajoute colonnes normalisées au CSV.

    Args:
        csv_path: Chemin vers performances.csv
        ranges: Dict {col: (min, max)} pour normalisation
        dry_run: Si True, n'écrit pas le fichier (juste affiche les stats)
        no_interactive: Si True, ne demande pas de confirmation
    """
    if not csv_path.exists():
        print(f"⚠️  File not found: {csv_path}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {csv_path}")
    print(f"{'='*60}")

    # Lire CSV
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    # Vérifier si déjà normalisé
    norm_cols = [f"{col}_norm" for col in ['d_cons', 'd_rigid', 'd_life', 'd_stab']]
    if any(col in df.columns for col in norm_cols):
        print(f"⚠️  Already contains normalized columns: {[c for c in norm_cols if c in df.columns]}")
        if no_interactive:
            print("Overwriting existing normalized columns...")
            user_input = 'y'
        else:
            user_input = input("Continue and overwrite? (y/n): ")
        if user_input.lower() != 'y':
            print("Skipped.")
            return

    # Normaliser chaque colonne
    for col in ['d_cons', 'd_rigid', 'd_life', 'd_stab']:
        if col not in df.columns:
            print(f"⚠️  Column {col} not found, skipping...")
            continue

        col_min, col_max = ranges[col]
        col_norm = f"{col}_norm"

        # Min-Max [-1, 1]: normalized = 2 * (x - min) / (max - min) - 1
        df[col_norm] = 2 * (df[col] - col_min) / (col_max - col_min) - 1

        print(f"\n  {col} -> {col_norm}:")
        print(f"    Original range: [{df[col].min():.3f}, {df[col].max():.3f}]")
        print(f"    Normalized range: [{df[col_norm].min():.3f}, {df[col_norm].max():.3f}]")
        print(f"    Mean: {df[col_norm].mean():.3f}, Std: {df[col_norm].std():.3f}")

        # Vérifier NaN
        if df[col_norm].isna().any():
            print(f"    ⚠️  WARNING: {df[col_norm].isna().sum()} NaN values introduced!")

    if dry_run:
        print("\n[DRY RUN] Would have saved file with new columns.")
        return

    # Sauvegarder avec nouvelles colonnes
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved: {csv_path}")
    print(f"   New shape: {df.shape}")
    print(f"   Added columns: {[col for col in norm_cols if col in df.columns]}")


def compute_global_ranges(epure_csv: Path) -> dict:
    """
    Calcule les ranges globaux depuis le dataset EPURE.

    Args:
        epure_csv: Chemin vers data/epure/performances.csv

    Returns:
        Dict {col: (min, max)}
    """
    if not epure_csv.exists():
        raise FileNotFoundError(f"EPURE CSV not found: {epure_csv}")

    df_epure = pd.read_csv(epure_csv)

    ranges = {}
    for col in ['d_cons', 'd_rigid', 'd_life', 'd_stab']:
        if col not in df_epure.columns:
            raise ValueError(f"Column {col} not found in EPURE CSV")

        ranges[col] = (df_epure[col].min(), df_epure[col].max())

    return ranges


def main():
    """Fonction principale."""
    # Check for --no-interactive flag
    no_interactive = '--no-interactive' in sys.argv or '-y' in sys.argv

    print("\n" + "="*60)
    print("CSV Normalization Script - EpureDGM")
    print("="*60)
    print("Normalizes d_cons, d_rigid, d_life, d_stab to [-1, 1]")
    print()

    # Chemins des fichiers
    project_root = Path(__file__).parent.parent.parent
    epure_csv = project_root / "data" / "epure" / "performances.csv"
    epure_preprocessed_csv = project_root / "data" / "epure" / "preprocessed" / "performances.csv"
    toy_csv = project_root / "data" / "toy_epure" / "performances.csv"

    # 1. Calculer ranges globaux depuis EPURE dataset
    print("Step 1: Computing global ranges from EPURE dataset...")
    try:
        ranges = compute_global_ranges(epure_csv)
    except Exception as e:
        print(f"❌ Error computing ranges: {e}")
        sys.exit(1)

    print("\nGlobal ranges (from EPURE):")
    for col, (vmin, vmax) in ranges.items():
        print(f"  {col:10s}: [{vmin:7.3f}, {vmax:7.3f}]  (range: {vmax-vmin:.3f})")

    # 2. Demander confirmation
    print("\n" + "="*60)
    print("Files to normalize:")
    print(f"  1. {epure_csv}")
    print(f"  2. {epure_preprocessed_csv}")
    print(f"  3. {toy_csv}")
    print("="*60)

    if no_interactive:
        print("\n[Auto-confirmed with --no-interactive flag]")
        user_input = 'y'
    else:
        user_input = input("\nProceed with normalization? (y/n): ")

    if user_input.lower() != 'y':
        print("Cancelled.")
        return

    # 3. Normaliser tous les CSV
    print("\nStep 2: Normalizing CSV files...")

    # EPURE main
    normalize_csv(epure_csv, ranges, no_interactive=no_interactive)

    # EPURE preprocessed
    normalize_csv(epure_preprocessed_csv, ranges, no_interactive=no_interactive)

    # TOY
    normalize_csv(toy_csv, ranges, no_interactive=no_interactive)

    # 4. Vérifications finales
    print("\n" + "="*60)
    print("✅ Normalization complete!")
    print("="*60)

    print("\nVerification:")
    for csv_path in [epure_csv, epure_preprocessed_csv, toy_csv]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            norm_cols = [c for c in df.columns if c.endswith('_norm')]
            if norm_cols:
                print(f"\n  {csv_path.name}:")
                print(f"    Shape: {df.shape}")
                print(f"    Normalized columns: {norm_cols}")

                # Vérifier range
                for col in ['d_cons_norm', 'd_rigid_norm', 'd_life_norm', 'd_stab_norm']:
                    if col in df.columns:
                        vmin, vmax = df[col].min(), df[col].max()
                        if vmin < -1.01 or vmax > 1.01:
                            print(f"    ⚠️  {col}: [{vmin:.3f}, {vmax:.3f}] - OUT OF RANGE!")
                        else:
                            print(f"    ✓ {col}: [{vmin:.3f}, {vmax:.3f}]")

    print("\n✅ All done! You can now use 'normalized: true' in your configs.")


if __name__ == '__main__':
    main()
