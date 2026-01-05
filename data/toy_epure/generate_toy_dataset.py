"""
Script de génération de dataset ToyTire adapté pour EpureDGM.

Basé sur ToyTire_v7_Generator.ipynb - Génère des pneus en coupe avec 3 composants:
- group_nc (carcass dans le notebook) : Corps du pneu (forme d'ampoule avec extrémités arrondies)
- group_km (crown dans le notebook) : Couronne enveloppante au sommet
- fpu (flanks dans le notebook) : Flancs triangulaires

Les images sont symétriques, donc on ne garde que la moitié gauche (64×32).
Génère 20,000 échantillons au format compatible EpureDGM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm

# Configuration
CONFIG = {
    "n_samples": 20000,
    "resolution": 64,  # 64x64 pixels
    "seed": 42,
    "train_ratio": 0.8,
    "test_ratio": 0.2,

    # Géométrie
    "y_top_range": [6, 12],
    "y_bottom_range": [52, 58],

    # Largeurs
    "w_belly_range": [20, 28],
    "w_bottom_range": [10, 18],

    # Position du ventre
    "belly_position_range": [0.30, 0.50],

    # Épaisseurs (carcass <= crown, enforced)
    "thickness_carcass_range": [3, 5],
    "thickness_crown_range": [4, 7],
    "thickness_flanks_bottom_range": [1, 3],

    # Arrondi des extrémités
    "lip_rounding_range": [2, 4],

    # Sculptures (tread) - désactivé par défaut
    "add_tread": False,
    "tread_height_range": [2, 4],
    "tread_width_range": [3, 5],
    "tread_spacing_range": [2, 4],
    "n_tread_blocks_range": [4, 8],
}


def generate_components(resolution, y_top, y_bottom,
                        w_belly, w_bottom,
                        belly_position,
                        thickness_carcass, thickness_crown,
                        thickness_flanks_bottom,
                        lip_rounding=3,
                        add_tread=False,
                        tread_height=3,
                        tread_width=4,
                        tread_spacing=3,
                        n_tread_blocks=6):
    """
    Génère les 3 composants (copié depuis ToyTire_v7_Generator.ipynb cellule 7).

    Returns:
        dict avec clés: carcass, crown, flanks, full, y_belly, y_top, y_bottom, has_tread
    """
    cx = resolution // 2

    # Enforce constraint: thickness_carcass <= thickness_crown
    thickness_carcass = min(thickness_carcass, thickness_crown)

    y_belly = int(y_top + belly_position * (y_bottom - y_top))
    arc_height = y_belly - y_top
    flank_height = y_bottom - y_belly

    thickness_flanks_top = thickness_crown

    carcass = np.zeros((resolution, resolution), dtype=np.uint8)
    crown = np.zeros((resolution, resolution), dtype=np.uint8)
    flanks = np.zeros((resolution, resolution), dtype=np.uint8)

    # Paramètres ellipse
    ellipse_center_y = y_belly
    ellipse_rx_outer = w_belly
    ellipse_ry_outer = arc_height
    ellipse_rx_inner = max(0, w_belly - thickness_carcass)
    ellipse_ry_inner = max(0, arc_height - thickness_carcass)

    # Position où commence l'arrondi des lèvres
    y_lip_start = y_bottom - lip_rounding

    # ==================== CARCASSE ====================
    for y in range(resolution):
        if y < y_top or y > y_bottom:
            continue

        if y <= y_belly:
            # ===== PARTIE HAUTE : DEMI-ELLIPSE =====
            dy = y - ellipse_center_y

            if ellipse_ry_outer > 0:
                ratio_outer = (dy / ellipse_ry_outer) ** 2
                if ratio_outer <= 1:
                    w_outer = ellipse_rx_outer * np.sqrt(1 - ratio_outer)
                else:
                    w_outer = 0
            else:
                w_outer = ellipse_rx_outer

            if ellipse_ry_inner > 0 and ellipse_rx_inner > 0:
                ratio_inner = (dy / ellipse_ry_inner) ** 2
                if ratio_inner <= 1:
                    w_inner = ellipse_rx_inner * np.sqrt(1 - ratio_inner)
                else:
                    w_inner = 0
            else:
                w_inner = 0

        else:
            # ===== PARTIE BASSE : FLANCS + ARRONDI =====
            t = (y - y_belly) / flank_height if flank_height > 0 else 0
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))

            w_outer_base = w_belly + t_smooth * (w_bottom - w_belly)
            w_inner_base = max(0, w_outer_base - thickness_carcass)

            # Arrondi des extrémités (lèvres)
            if y >= y_lip_start and lip_rounding > 0:
                dy_lip = y - y_lip_start
                if dy_lip <= lip_rounding:
                    lip_curve = lip_rounding * (1 - np.sqrt(1 - (dy_lip / lip_rounding) ** 2))
                    w_outer = w_outer_base - lip_curve * 0.5
                    w_inner = w_inner_base + lip_curve * 0.5
                else:
                    w_outer = w_outer_base
                    w_inner = w_inner_base
            else:
                w_outer = w_outer_base
                w_inner = w_inner_base

        # Remplir la carcasse
        for x in range(resolution):
            dist = abs(x - cx)
            if w_inner <= dist <= w_outer:
                carcass[y, x] = 1

    # ==================== COURONNE ====================
    crown_rx_outer = ellipse_rx_outer + thickness_crown
    crown_ry_outer = ellipse_ry_outer + thickness_crown
    crown_center_y = y_belly

    for y in range(resolution):
        if y > y_belly:
            continue

        dy = y - crown_center_y

        if crown_ry_outer > 0:
            ratio_crown = (dy / crown_ry_outer) ** 2
            if ratio_crown <= 1:
                w_crown_outer = crown_rx_outer * np.sqrt(1 - ratio_crown)
            else:
                continue
        else:
            w_crown_outer = crown_rx_outer

        if ellipse_ry_outer > 0:
            ratio_outer = (dy / ellipse_ry_outer) ** 2
            if ratio_outer <= 1:
                w_carcass_outer = ellipse_rx_outer * np.sqrt(1 - ratio_outer)
            else:
                w_carcass_outer = 0
        else:
            w_carcass_outer = ellipse_rx_outer

        for x in range(resolution):
            dist = abs(x - cx)
            if w_carcass_outer < dist <= w_crown_outer:
                crown[y, x] = 1
            elif dist <= w_crown_outer and carcass[y, x] == 0:
                if y < y_top + thickness_crown:
                    crown[y, x] = 1

    crown = crown & (~carcass).astype(np.uint8)

    # ==================== SCULPTURES (TREAD) ====================
    if add_tread:
        crown_rows = np.where(crown.any(axis=1))[0]
        if len(crown_rows) > 0:
            y_crown_top = crown_rows.min()
            crown_cols_top = np.where(crown[y_crown_top, :] > 0)[0]
            if len(crown_cols_top) > 0:
                x_left = crown_cols_top.min()
                x_right = crown_cols_top.max()
                tread_zone_width = x_right - x_left

                total_block_width = n_tread_blocks * tread_width + (n_tread_blocks - 1) * tread_spacing
                start_x = cx - total_block_width // 2

                for i in range(n_tread_blocks):
                    block_x = start_x + i * (tread_width + tread_spacing)

                    for dy in range(tread_height):
                        y_tread = y_crown_top - tread_height + dy
                        if y_tread < 0:
                            continue
                        for dx in range(tread_width):
                            x_tread = block_x + dx
                            if 0 <= x_tread < resolution:
                                crown[y_tread, x_tread] = 1

    # ==================== FLANCS ====================
    for y in range(resolution):
        if y <= y_belly or y > y_bottom:
            continue

        t = (y - y_belly) / flank_height if flank_height > 0 else 0
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))

        w_carcass_outer_base = w_belly + t_smooth * (w_bottom - w_belly)

        # Appliquer l'arrondi des lèvres
        if y >= y_lip_start and lip_rounding > 0:
            dy_lip = y - y_lip_start
            if dy_lip <= lip_rounding:
                lip_curve = lip_rounding * (1 - np.sqrt(1 - (dy_lip / lip_rounding) ** 2))
                w_carcass_outer = w_carcass_outer_base - lip_curve * 0.5
            else:
                w_carcass_outer = w_carcass_outer_base
        else:
            w_carcass_outer = w_carcass_outer_base

        current_thickness = thickness_flanks_top + t * (thickness_flanks_bottom - thickness_flanks_top)
        w_flanks_outer = w_carcass_outer + current_thickness

        for x in range(resolution):
            dist = abs(x - cx)
            if w_carcass_outer < dist <= w_flanks_outer:
                flanks[y, x] = 1

    flanks = flanks & (~carcass).astype(np.uint8) & (~crown).astype(np.uint8)

    full = (carcass | crown | flanks).astype(np.uint8)

    return {
        'carcass': carcass,
        'crown': crown,
        'flanks': flanks,
        'full': full,
        'y_belly': y_belly,
        'y_top': y_top,
        'y_bottom': y_bottom,
        'has_tread': add_tread
    }


def validate_components(carcass, crown, flanks):
    """
    Valide les composants générés (copié depuis cellule 9).

    Returns:
        (valid: bool, errors: list)
    """
    errors = []

    if (carcass & crown).sum() > 0:
        errors.append(f"Overlap carcass-crown: {(carcass & crown).sum()} px")
    if (carcass & flanks).sum() > 0:
        errors.append(f"Overlap carcass-flanks: {(carcass & flanks).sum()} px")
    if (crown & flanks).sum() > 0:
        errors.append(f"Overlap crown-flanks: {(crown & flanks).sum()} px")

    full = carcass | crown | flanks
    labeled, n_components = ndimage.label(full)
    if n_components != 1:
        errors.append(f"Full not connected: {n_components} components")

    labeled_crown, n_crown = ndimage.label(crown)
    if n_crown != 1:
        errors.append(f"Crown not connected: {n_crown} components")

    min_area = 50
    if carcass.sum() < min_area:
        errors.append(f"Carcass too small: {carcass.sum()} px")
    if crown.sum() < min_area:
        errors.append(f"Crown too small: {crown.sum()} px")
    if flanks.sum() < min_area:
        errors.append(f"Flanks too small: {flanks.sum()} px")

    return len(errors) == 0, errors


def compute_metrics(carcass, crown, flanks):
    """
    Calcule les métriques géométriques (copié depuis cellule 13).

    Returns:
        dict avec height_px, width_px, area_*, ratios
    """
    full = carcass | crown | flanks

    rows = np.where(full.any(axis=1))[0]
    cols = np.where(full.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    height_px = int(rows.max() - rows.min() + 1)
    width_px = int(cols.max() - cols.min() + 1)
    aspect_ratio_px = height_px / width_px if width_px > 0 else 0

    area_full = int(full.sum())
    area_carcass = int(carcass.sum())
    area_crown = int(crown.sum())
    area_flanks = int(flanks.sum())

    return {
        'height_px': height_px,
        'width_px': width_px,
        'aspect_ratio_px': aspect_ratio_px,
        'area_full': area_full,
        'area_carcass': area_carcass,
        'area_crown': area_crown,
        'area_flanks': area_flanks,
        'carcass_to_full': area_carcass / area_full if area_full > 0 else 0,
        'crown_to_full': area_crown / area_full if area_full > 0 else 0,
        'flanks_to_full': area_flanks / area_full if area_full > 0 else 0,
    }


def sample_parameters(config, rng):
    """
    Sample des paramètres aléatoires (copié depuis cellule 24).
    """
    def uniform(r):
        return rng.uniform(r[0], r[1])
    def uniform_int(r):
        return rng.integers(r[0], r[1] + 1)

    # Contrainte: thickness_carcass <= thickness_crown
    thickness_crown = uniform_int(config['thickness_crown_range'])
    thickness_carcass = uniform_int([
        config['thickness_carcass_range'][0],
        min(config['thickness_carcass_range'][1], thickness_crown)
    ])

    params = {
        'y_top': uniform_int(config['y_top_range']),
        'y_bottom': uniform_int(config['y_bottom_range']),
        'w_belly': uniform_int(config['w_belly_range']),
        'w_bottom': uniform_int(config['w_bottom_range']),
        'belly_position': uniform(config['belly_position_range']),
        'thickness_carcass': thickness_carcass,
        'thickness_crown': thickness_crown,
        'thickness_flanks_bottom': uniform_int(config['thickness_flanks_bottom_range']),
        'lip_rounding': uniform_int(config['lip_rounding_range']),
        'add_tread': config['add_tread'],
    }

    if config['add_tread']:
        params.update({
            'tread_height': uniform_int(config['tread_height_range']),
            'tread_width': uniform_int(config['tread_width_range']),
            'tread_spacing': uniform_int(config['tread_spacing_range']),
            'n_tread_blocks': uniform_int(config['n_tread_blocks_range']),
        })

    return params


def save_sample_epure_style(sample_id, components, split, output_dir):
    """
    Sauvegarde au format EpureDGM :
    - Nom: toy_{sample_id:05d}_{component}.png
    - Dossiers: {split}/{component}/
    - Renommer: carcass → group_nc, crown → group_km, flanks → fpu
    - Crop to 64×32 (left half due to symmetry)
    """
    split_dir = output_dir / split

    # Mapper vers noms EpureDGM
    component_map = {
        'carcass': 'group_nc',  # Renommer carcass → group_nc
        'crown': 'group_km',    # Renommer crown → group_km
        'flanks': 'fpu',        # Renommer flanks → fpu
        'full': 'full'
    }

    for orig_name, new_name in component_map.items():
        img = components[orig_name]

        # Crop to left half (64×32) since images are symmetric
        # Original: 64×64, crop to: 64×32 (keep left half)
        img_cropped = img[:, :32]  # Take left 32 columns

        filename = f"toy_{sample_id:05d}_{new_name}.png"
        filepath = split_dir / new_name / filename
        plt.imsave(str(filepath), img_cropped, cmap='gray', vmin=0, vmax=1)


def create_performances_csv(all_records, output_dir):
    """
    Crée CSV au format EpureDGM (20 colonnes).

    Colonnes :
    - matching, train, test, width_px, height_px, filepath,
    - width_px_norm, height_px_norm,
    - m_top, m_side, m_total, round_top,
    - dm_top, dm_side, dm_total, dround_top,
    - d_cons, d_rigid, d_life, d_stab
    """
    rows = []

    for rec in all_records:
        metrics = rec['metrics']

        # Normaliser les aires pour scale [0, 300]
        m_top = min(300, metrics['area_crown'] * 0.5)
        m_side = min(300, metrics['area_flanks'] * 0.5)
        m_total = min(600, metrics['area_full'] * 0.3)
        round_top = metrics.get('crown_to_full', 0.5)

        # Delta metrics normalisés [-1, 1]
        dm_top = np.clip((metrics['crown_to_full'] - 0.3) / 0.3, -1, 1)
        dm_side = np.clip((metrics['flanks_to_full'] - 0.2) / 0.2, -1, 1)
        dm_total = np.clip((m_total - 300) / 300, -1, 1)
        dround_top = np.clip((round_top - 0.5) * 2, -1, 1)

        # Performance metrics [-1, 1]
        d_cons = np.clip(np.random.randn() * 0.2, -1, 1)
        d_rigid = np.clip(np.random.randn() * 0.2, -1, 1)
        d_life = np.clip(metrics['carcass_to_full'] - 0.5, -1, 1)
        d_stab = np.clip(metrics.get('aspect_ratio_px', 1.0) - 1.0, -1, 1)

        row = {
            'matching': f"toy_{rec['sample_id']:05d}",
            'train': rec['split'] == 'train',
            'test': rec['split'] == 'test',
            'width_px': metrics['width_px'],
            'height_px': metrics['height_px'],
            'filepath': f"data/toy_epure/{rec['split']}/full/toy_{rec['sample_id']:05d}_full.png",
            'width_px_norm': metrics['width_px'] / 64.0,
            'height_px_norm': metrics['height_px'] / 64.0,
            'm_top': m_top,
            'm_side': m_side,
            'm_total': m_total,
            'round_top': round_top,
            'dm_top': dm_top,
            'dm_side': dm_side,
            'dm_total': dm_total,
            'dround_top': dround_top,
            'd_cons': d_cons,
            'd_rigid': d_rigid,
            'd_life': d_life,
            'd_stab': d_stab,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ordre exact des colonnes
    columns = [
        'matching', 'train', 'test', 'width_px', 'height_px', 'filepath',
        'width_px_norm', 'height_px_norm',
        'm_top', 'm_side', 'm_total', 'round_top',
        'dm_top', 'dm_side', 'dm_total', 'dround_top',
        'd_cons', 'd_rigid', 'd_life', 'd_stab'
    ]

    df = df[columns]
    df.to_csv(output_dir / 'performances.csv', index=False)
    return df


def main():
    """Génère le dataset complet."""
    output_dir = Path(__file__).parent

    print("=" * 60)
    print("Génération du Toy Dataset ToyTire pour EpureDGM")
    print("=" * 60)
    print(f"Nombre d'échantillons : {CONFIG['n_samples']}")
    print(f"Résolution : {CONFIG['resolution']}×{CONFIG['resolution']} → crop to 64×32")
    print(f"Composants : group_nc, group_km, fpu")
    print(f"Train/Test split : {CONFIG['train_ratio']:.0%} / {CONFIG['test_ratio']:.0%}")
    print()

    # Créer structure avec noms EpureDGM
    for split in ['train', 'test']:
        for comp in ['group_nc', 'group_km', 'fpu', 'full']:
            (output_dir / split / comp).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(CONFIG['seed'])
    n_samples = CONFIG['n_samples']
    n_train = int(n_samples * CONFIG['train_ratio'])

    splits = ['train'] * n_train + ['test'] * (n_samples - n_train)
    rng.shuffle(splits)

    all_records = []
    sample_id = 0
    max_retries = 10

    pbar = tqdm(total=n_samples, desc="Generating")

    while sample_id < n_samples:
        # Générer paramètres
        for _ in range(max_retries):
            params = sample_parameters(CONFIG, rng)

            # Générer composants
            components = generate_components(
                resolution=CONFIG['resolution'],
                **params
            )

            # Valider
            valid, errors = validate_components(
                components['carcass'],
                components['crown'],
                components['flanks']
            )

            if valid:
                break

        if not valid:
            continue

        # Sauvegarder
        split = splits[sample_id]
        save_sample_epure_style(sample_id, components, split, output_dir)

        # Métriques
        metrics = compute_metrics(
            components['carcass'],
            components['crown'],
            components['flanks']
        )

        if metrics is None:
            continue

        all_records.append({
            'sample_id': sample_id,
            'split': split,
            'metrics': metrics,
            'params': params
        })

        sample_id += 1
        pbar.update(1)

    pbar.close()

    # Créer CSV
    print("\nCréation du CSV de performances...")
    df = create_performances_csv(all_records, output_dir)
    print(f"✅ CSV sauvegardé : {output_dir / 'performances.csv'}")
    print(f"   Shape : {df.shape}")

    print("\n" + "=" * 60)
    print("✅ Génération terminée!")
    print("=" * 60)
    print(f"Train samples: {df['train'].sum()}")
    print(f"Test samples: {df['test'].sum()}")
    print(f"\nDimensions (px):")
    print(f"  Width : {df['width_px'].min():.0f} - {df['width_px'].max():.0f}")
    print(f"  Height: {df['height_px'].min():.0f} - {df['height_px'].max():.0f}")
    print(f"\nProchaine étape : Exécuter create_categorical_masks_toy.py")


if __name__ == '__main__':
    main()
