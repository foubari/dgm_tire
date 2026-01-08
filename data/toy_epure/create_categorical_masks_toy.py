"""
Script pour créer les masques catégoriels à partir des images ToyTire (toy dataset).

Convertit les images binaires (64x32) de chaque composant en masques catégoriels
au format attendu par MDM (N, 1, 64, 32) avec valeurs 0-3 (3 composants + background).

Composants : group_nc, group_km, fpu
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple

# Configuration
SOURCE_DIR = Path(__file__).parent  # Chemin relatif au script
TARGET_DIR = SOURCE_DIR
PREPROCESSED_DIR = TARGET_DIR / "preprocessed"

# Composants (ordre de priorité pour la superposition)
COMPONENTS = [
    "group_nc",  # 0 (carcass)
    "group_km",  # 1 (crown)
    "fpu"        # 2 (flanks)
]
BACKGROUND_LBL = 3  # Background = 3 (4 classes au total: 0-2 composants + 3 background)

# Taille des images
IMG_SIZE = (64, 32)  # 64x32 pour ToyTire (cropped from symmetric 64x64)

# CSV source pour les conditions
CONDITIONS_CSV_SOURCE = SOURCE_DIR / "performances.csv"


def extract_prefix(filename: str) -> str:
    """
    Extrait le préfixe depuis un nom de fichier.
    Format: toy_00001_group_nc.png -> toy_00001
    """
    base = os.path.splitext(filename)[0]
    # Enlever le nom du composant à la fin
    for comp in COMPONENTS + ['full']:
        if base.endswith(f"_{comp}"):
            return base[:-len(f"_{comp}")]
    # Fallback: prendre tout sauf la dernière partie
    parts = base.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    return base


def collect_prefixes(component_dir: Path) -> List[str]:
    """
    Collecte tous les préfixes uniques depuis un dossier de composant.
    """
    if not component_dir.exists():
        return []

    png_files = sorted(component_dir.glob("*.png"))
    prefixes = sorted({extract_prefix(f.name) for f in png_files})
    return prefixes


def build_split_masks(split_root: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Construit les masques catégoriels pour un split (train ou test).

    Args:
        split_root: Chemin vers le dossier train/ ou test/

    Returns:
        masks: Array (N, 1, H, W) avec dtype uint8
        idx_map: Mapping prefix -> index numérique
    """
    component_dirs = [split_root / comp for comp in COMPONENTS]

    # Collecter tous les préfixes depuis le premier composant
    prefixes = collect_prefixes(component_dirs[0])

    if not prefixes:
        print(f"⚠️  No prefixes found in {component_dirs[0]}")
        return np.array([]).reshape(0, 1, *IMG_SIZE), {}

    masks: List[np.ndarray] = []
    idx_map: Dict[str, int] = {}

    for idx, prefix in tqdm(enumerate(prefixes), desc=f"Building {split_root.name} masks", total=len(prefixes)):
        # Initialiser le masque avec background
        seg_mask = np.full(IMG_SIZE, BACKGROUND_LBL, dtype=np.uint8)

        # Superposer les composants dans l'ordre de priorité
        for comp_idx, comp_name in enumerate(COMPONENTS):
            comp_dir = component_dirs[comp_idx]
            comp_file = comp_dir / f"{prefix}_{comp_name}.png"

            if not comp_file.exists():
                continue

            # Charger et binariser l'image
            try:
                img = Image.open(comp_file).convert('L')
                arr = np.array(img)

                # Binariser: pixel > 128 -> 1, sinon 0
                bin_mask = arr > 128

                # Assigner la classe seulement si le pixel est encore background
                seg_mask[(bin_mask) & (seg_mask == BACKGROUND_LBL)] = comp_idx

            except Exception as e:
                print(f"\n⚠️  Error processing {comp_file}: {e}")
                continue

        # Pas de flip vertical (images déjà correctes)
        masks.append(seg_mask)
        idx_map[prefix] = idx

    # Stack en (N, H, W) puis ajouter la dimension channel: (N, 1, H, W)
    masks_np = np.stack(masks, axis=0)  # (N, H, W)
    masks_np = masks_np[:, None, ...]    # (N, 1, H, W)

    return masks_np, idx_map


def create_conditions_csv(split: str, idx_map: Dict[str, int], output_path: Path):
    """
    Crée un CSV avec les conditions et le mapping condition_matching.

    Args:
        split: 'train' ou 'test'
        idx_map: Mapping prefix -> index
        output_path: Chemin de sortie pour le CSV
    """
    # Essayer de charger le CSV source
    if CONDITIONS_CSV_SOURCE.exists():
        try:
            df = pd.read_csv(CONDITIONS_CSV_SOURCE)
            print(f"✅ Loaded conditions from {CONDITIONS_CSV_SOURCE}")
        except Exception as e:
            print(f"⚠️  Error loading CSV: {e}")
            df = None
    else:
        print(f"⚠️  Conditions CSV not found at {CONDITIONS_CSV_SOURCE}")
        df = None

    if df is None:
        # Créer un CSV minimal avec juste les préfixes et indices
        data = {
            'matching': list(idx_map.keys()),
            'condition_matching': list(idx_map.values()),
            'width_px': [0] * len(idx_map),
            'height_px': [0] * len(idx_map),
            'train': [split == 'train'] * len(idx_map)
        }
        df = pd.DataFrame(data)
    else:
        # Ajouter le mapping condition_matching
        df['matching_norm'] = df.get('matching', pd.Series()).str.lower().str.strip()
        df['condition_matching'] = df['matching_norm'].map(idx_map)

        # Filtrer selon le split si la colonne 'train' existe
        if 'train' in df.columns:
            if split == 'train':
                df = df[df['train'] == True].reset_index(drop=True)
            else:
                df = df[df['train'] == False].reset_index(drop=True)

    # Sauvegarder
    df.to_csv(output_path, index=False)
    print(f"✅ Saved conditions CSV to {output_path}")
    return df


def main():
    """Fonction principale."""
    print("="*60)
    print("Création des masques catégoriels pour MDM (ToyTire Dataset)")
    print("="*60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print(f"Components: {', '.join(COMPONENTS)}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Background label: {BACKGROUND_LBL}")
    print()

    # Créer le dossier preprocessed
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Traiter train et test
    all_idx_maps = {}

    for split in ['train', 'test']:
        split_dir = SOURCE_DIR / split

        if not split_dir.exists():
            print(f"❌ Split directory not found: {split_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split.upper()}")
        print(f"{'='*60}")

        # Construire les masques
        masks, idx_map = build_split_masks(split_dir)

        if len(masks) == 0:
            print(f"⚠️  No masks created for {split}")
            continue

        print(f"✅ Created {len(masks)} masks with shape {masks.shape}")

        # Sauvegarder en .npy
        # Format principal: 64x32
        npy_filename_64x32 = f"{split}_64x32.npy"

        # Sauvegarder avec le nom 64x32 (format principal)
        npy_path_64x32 = PREPROCESSED_DIR / npy_filename_64x32
        np.save(npy_path_64x32, masks.astype('uint8'), allow_pickle=True, fix_imports=True)
        print(f"✅ Saved masks to {npy_path_64x32}")

        # Créer le CSV avec les conditions
        csv_filename = "dims_cond.csv" if split == 'train' else None
        if csv_filename:
            csv_path = PREPROCESSED_DIR / csv_filename
            create_conditions_csv(split, idx_map, csv_path)

        all_idx_maps[split] = idx_map

    print("\n" + "="*60)
    print("✅ Conversion complete!")
    print("="*60)
    print(f"\nFichiers créés dans: {PREPROCESSED_DIR}")
    print("\nFichiers .npy:")
    for split in ['train', 'test']:
        npy_file = PREPROCESSED_DIR / f"{split}_64x32.npy"
        if npy_file.exists():
            masks = np.load(npy_file)
            print(f"  - {npy_file.name}: {masks.shape}, values: {np.unique(masks)}")

    csv_file = PREPROCESSED_DIR / "dims_cond.csv"
    if csv_file.exists():
        print(f"\nCSV créé: {csv_file}")


if __name__ == '__main__':
    main()
