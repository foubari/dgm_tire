# =============================================================
# CELLULES À AJOUTER AU NOTEBOOK ToyTire_v7_Generator.ipynb
# =============================================================

# Cellule 1: Markdown - Section 14
"""
## 14. Mécanique du Pneu et Déformation

### Modèle simplifié de raideur verticale

Basé sur la littérature en mécanique du pneu, on modélise l'épure comme un ensemble de **ressorts verticaux en parallèle** :

```
    ┌───────────────────┐
    │   F (force)       │
    │       ↓           │
    ├───┬───┬───┬───┬───┤
    │ k₁│ k₂│ k₃│ k₄│ k₅│  ← ressorts par colonne
    │   │   │   │   │   │
    └───┴───┴───┴───┴───┘
```

Pour chaque colonne $x$ :
$$k_x = \\frac{E_x \\cdot A_x}{L_x}$$

où :
- $E_x$ = module effectif (moyenne pondérée des composants)
- $A_x$ = aire de la section (pixels)
- $L_x$ = hauteur de la colonne

Raideur totale (ressorts en parallèle) :
$$K_{vert} = \\sum_x k_x$$

### Indices de performance

| Indice | Formule | Interprétation |
|--------|---------|----------------|
| `K_vert` | $\\sum k_x$ | Raideur verticale totale |
| `delta_rel` | $F / (K_{vert} \\cdot R)$ | Déflexion relative |
| `mass_index` | $\\sum \\rho_c \\cdot \\text{area}_c$ | Masse fictive |
| `performance_index` | $K_{vert} / \\text{mass}$ | Portance par unité de masse |

→ Un pneu "performant" = forte raideur pour une masse faible
"""

# Cellule 2: Code - Classe TireMechanics
# --------------------------------------

from dataclasses import dataclass, field
from typing import Dict, Tuple
from scipy.ndimage import map_coordinates


@dataclass
class MaterialProperties:
    """Propriétés matériaux fictives pour chaque composant."""
    E_carcass: float = 1.0      # Module carcasse (rigide)
    E_crown: float = 0.8        # Module couronne
    E_flanks: float = 0.5       # Module flancs (souple)
    rho_carcass: float = 1.0    # Densité carcasse
    rho_crown: float = 1.2      # Densité couronne
    rho_flanks: float = 0.8     # Densité flancs


@dataclass 
class TireMechanics:
    """
    Calcule les propriétés mécaniques d'une épure et simule la déformation.
    
    Usage:
    ------
    >>> mechanics = TireMechanics()
    >>> props = mechanics.compute_properties(carcass, crown, flanks)
    >>> deformed = mechanics.apply_vertical_load(carcass, crown, flanks, force=1.0)
    """
    
    materials: MaterialProperties = field(default_factory=MaterialProperties)
    nominal_force: float = 1.0
    max_strain: float = 0.3
    
    def compute_properties(self, carcass: np.ndarray, crown: np.ndarray, 
                          flanks: np.ndarray) -> Dict[str, float]:
        """Calcule K_vert, masse, déflexion et indices de performance."""
        carcass = (carcass > 0.5).astype(np.float32)
        crown = (crown > 0.5).astype(np.float32)
        flanks = (flanks > 0.5).astype(np.float32)
        full = (carcass + crown + flanks) > 0
        
        if full.sum() == 0:
            return {'K_vert': 0, 'mass_index': 0, 'delta_rel': 0, 
                    'stiffness_index': 0, 'performance_index': 0,
                    'height_px': 0, 'width_px': 0, 'R_out': 0, 'k_per_column': np.array([])}
        
        rows = np.where(full.any(axis=1))[0]
        cols = np.where(full.any(axis=0))[0]
        y_top, y_bottom = rows.min(), rows.max()
        x_left, x_right = cols.min(), cols.max()
        
        height_px = y_bottom - y_top + 1
        width_px = x_right - x_left + 1
        R_out = height_px / 2
        
        # Raideur par colonne
        K_vert = 0.0
        k_per_column = np.zeros(full.shape[1])
        
        for x in range(x_left, x_right + 1):
            col_full = full[:, x]
            if col_full.sum() == 0:
                continue
            
            A_x = col_full.sum()
            active_rows = np.where(col_full)[0]
            L_x = active_rows.max() - active_rows.min() + 1
            if L_x == 0:
                continue
            
            # Module effectif
            E_eff = (
                carcass[:, x].sum() * self.materials.E_carcass +
                crown[:, x].sum() * self.materials.E_crown +
                flanks[:, x].sum() * self.materials.E_flanks
            ) / max(A_x, 1)
            
            k_x = E_eff * A_x / L_x
            k_per_column[x] = k_x
            K_vert += k_x
        
        # Masse
        mass_index = (
            carcass.sum() * self.materials.rho_carcass +
            crown.sum() * self.materials.rho_crown +
            flanks.sum() * self.materials.rho_flanks
        )
        
        # Déflexion
        delta = self.nominal_force / K_vert if K_vert > 0 else 0
        delta_rel = delta / R_out if R_out > 0 else 0
        
        # Indices
        stiffness_index = 1.0 / delta_rel if delta_rel > 0 else 0
        performance_index = stiffness_index / mass_index if mass_index > 0 else 0
        
        return {
            'K_vert': K_vert, 'mass_index': mass_index,
            'delta': delta, 'delta_rel': delta_rel,
            'stiffness_index': stiffness_index, 'performance_index': performance_index,
            'height_px': height_px, 'width_px': width_px, 'R_out': R_out,
            'k_per_column': k_per_column
        }
    
    def apply_vertical_load(self, carcass: np.ndarray, crown: np.ndarray,
                           flanks: np.ndarray, force: float = 1.0,
                           return_displacement: bool = False) -> Dict[str, np.ndarray]:
        """
        Applique une charge verticale et retourne les composants déformés.
        """
        carcass = (carcass > 0.5).astype(np.float32)
        crown = (crown > 0.5).astype(np.float32)
        flanks = (flanks > 0.5).astype(np.float32)
        
        resolution = carcass.shape[0]
        full = (carcass + crown + flanks) > 0
        
        if full.sum() == 0:
            return {'carcass': carcass, 'crown': crown, 'flanks': flanks, 
                    'full': full.astype(np.float32)}
        
        props = self.compute_properties(carcass, crown, flanks)
        K_vert = props['K_vert']
        k_per_column = props['k_per_column']
        
        if K_vert == 0:
            return {'carcass': carcass, 'crown': crown, 'flanks': flanks,
                    'full': full.astype(np.float32)}
        
        # Déplacement total (limité)
        delta_total = force / K_vert
        rows = np.where(full.any(axis=1))[0]
        height = rows.max() - rows.min() + 1
        delta_total = min(delta_total, height * self.max_strain)
        
        # Champ de déplacement
        displacement_y = np.zeros((resolution, resolution), dtype=np.float32)
        cols = np.where(full.any(axis=0))[0]
        
        for x in cols:
            k_x = k_per_column[x]
            if k_x == 0:
                continue
            
            compliance_x = (1.0 / k_x) / (1.0 / K_vert)
            delta_x = delta_total * compliance_x
            
            col_mask = full[:, x]
            active_rows = np.where(col_mask)[0]
            if len(active_rows) == 0:
                continue
            
            col_top = active_rows.min()
            col_height = active_rows.max() - col_top + 1
            
            for y in active_rows:
                t = (y - col_top) / col_height if col_height > 1 else 0
                displacement_y[y, x] = -delta_x * t
        
        # Appliquer la déformation
        y_coords, x_coords = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='ij')
        new_y = y_coords - displacement_y
        
        def warp_image(img):
            warped = map_coordinates(img, [new_y, x_coords.astype(np.float32)], 
                                    order=1, mode='constant', cval=0)
            return (warped > 0.5).astype(np.float32)
        
        result = {
            'carcass': warp_image(carcass),
            'crown': warp_image(crown),
            'flanks': warp_image(flanks),
        }
        result['full'] = ((result['carcass'] + result['crown'] + result['flanks']) > 0).astype(np.float32)
        
        if return_displacement:
            result['displacement_field'] = displacement_y
            result['delta_total'] = delta_total
        
        return result
    
    def visualize_deformation(self, carcass, crown, flanks, force=1.0, figsize=(14, 6)):
        """Visualise avant/après déformation."""
        props = self.compute_properties(carcass, crown, flanks)
        deformed = self.apply_vertical_load(carcass, crown, flanks, force, return_displacement=True)
        
        colors = {'carcass': [0.9, 0.2, 0.2], 'crown': [0.2, 0.5, 0.9], 'flanks': [0.2, 0.8, 0.3]}
        
        def make_overlay(c, cr, f):
            resolution = c.shape[0]
            overlay = np.zeros((resolution, resolution, 3))
            for name, img, color in [('carcass', c, colors['carcass']),
                                     ('crown', cr, colors['crown']),
                                     ('flanks', f, colors['flanks'])]:
                mask = img > 0.5
                for ch in range(3):
                    overlay[:, :, ch] += mask * color[ch]
            return np.clip(overlay, 0, 1)
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        axes[0].imshow(make_overlay(carcass, crown, flanks), origin='upper')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(make_overlay(deformed['carcass'], deformed['crown'], deformed['flanks']), origin='upper')
        axes[1].set_title(f'Déformé (F={force:.1f})')
        axes[1].axis('off')
        
        blend = 0.3 * make_overlay(carcass, crown, flanks) + 0.7 * make_overlay(
            deformed['carcass'], deformed['crown'], deformed['flanks'])
        axes[2].imshow(np.clip(blend, 0, 1), origin='upper')
        axes[2].set_title('Superposition')
        axes[2].axis('off')
        
        if 'displacement_field' in deformed:
            disp = deformed['displacement_field']
            im = axes[3].imshow(disp, origin='upper', cmap='RdBu_r', vmin=-abs(disp).max(), vmax=abs(disp).max())
            plt.colorbar(im, ax=axes[3], label='Déplacement (px)')
        axes[3].set_title('Champ de déplacement')
        axes[3].axis('off')
        
        patches = [mpatches.Patch(color=colors[name], label=name) for name in colors]
        fig.legend(handles=patches, loc='lower center', ncol=3)
        
        props_text = f"K_vert: {props['K_vert']:.1f}  |  Mass: {props['mass_index']:.0f}  |  Perf: {props['performance_index']:.4f}"
        fig.suptitle(props_text, fontsize=11)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        return fig

# Cellule 3: Code - Test avec un pneu généré
# ------------------------------------------

# Créer un pneu test
test_tire = generate_components(
    resolution=64, y_top=8, y_bottom=56,
    w_belly=24, w_bottom=14, belly_position=0.40,
    thickness_carcass=4, thickness_crown=5,
    thickness_flanks_bottom=2, lip_rounding=3
)

# Calculer et visualiser
mechanics = TireMechanics()
fig = mechanics.visualize_deformation(
    test_tire['carcass'], test_tire['crown'], test_tire['flanks'],
    force=2.0
)
plt.show()


# Cellule 4: Code - Comparaison de plusieurs forces
# -------------------------------------------------

def show_force_progression(components, forces=[0.5, 1.0, 2.0, 5.0]):
    """Montre la déformation pour différentes forces."""
    mechanics = TireMechanics()
    props = mechanics.compute_properties(components['carcass'], components['crown'], components['flanks'])
    
    colors = {'carcass': [0.9, 0.2, 0.2], 'crown': [0.2, 0.5, 0.9], 'flanks': [0.2, 0.8, 0.3]}
    
    def make_overlay(c, cr, f):
        overlay = np.zeros((64, 64, 3))
        for name, img, color in [('carcass', c, colors['carcass']),
                                 ('crown', cr, colors['crown']),
                                 ('flanks', f, colors['flanks'])]:
            for ch in range(3):
                overlay[:, :, ch] += (img > 0.5) * color[ch]
        return np.clip(overlay, 0, 1)
    
    fig, axes = plt.subplots(2, len(forces), figsize=(4*len(forces), 8))
    
    original = make_overlay(components['carcass'], components['crown'], components['flanks'])
    orig_h = np.where(components['full'].any(axis=1))[0]
    orig_height = len(orig_h) if len(orig_h) > 0 else 0
    
    for idx, force in enumerate(forces):
        deformed = mechanics.apply_vertical_load(
            components['carcass'], components['crown'], components['flanks'], force
        )
        
        deformed_overlay = make_overlay(deformed['carcass'], deformed['crown'], deformed['flanks'])
        def_h = np.where(deformed['full'].any(axis=1))[0]
        delta_h = orig_height - (len(def_h) if len(def_h) > 0 else 0)
        
        axes[0, idx].imshow(original, origin='upper')
        axes[0, idx].set_title(f'Original')
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(deformed_overlay, origin='upper')
        axes[1, idx].set_title(f'F={force:.1f}\nΔh={delta_h}px')
        axes[1, idx].axis('off')
    
    patches = [mpatches.Patch(color=colors[name], label=name) for name in colors]
    fig.legend(handles=patches, loc='lower center', ncol=3)
    fig.suptitle(f'Progression de la déformation\nK_vert={props["K_vert"]:.1f}, Perf={props["performance_index"]:.4f}', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

show_force_progression(test_tire)


# Cellule 5: Code - Comparaison de géométries
# -------------------------------------------

def compare_geometries():
    """Compare les performances de différentes géométries."""
    mechanics = TireMechanics()
    
    configs = [
        {"name": "Fin", "thickness_carcass": 3, "thickness_crown": 4, "w_belly": 20},
        {"name": "Moyen", "thickness_carcass": 4, "thickness_crown": 5, "w_belly": 24},
        {"name": "Épais", "thickness_carcass": 5, "thickness_crown": 7, "w_belly": 28},
        {"name": "Large", "thickness_carcass": 4, "thickness_crown": 5, "w_belly": 28},
    ]
    
    colors = {'carcass': [0.9, 0.2, 0.2], 'crown': [0.2, 0.5, 0.9], 'flanks': [0.2, 0.8, 0.3]}
    
    def make_overlay(c, cr, f):
        overlay = np.zeros((64, 64, 3))
        for name, img, color in [('carcass', c, colors['carcass']),
                                 ('crown', cr, colors['crown']),
                                 ('flanks', f, colors['flanks'])]:
            for ch in range(3):
                overlay[:, :, ch] += (img > 0.5) * color[ch]
        return np.clip(overlay, 0, 1)
    
    fig, axes = plt.subplots(2, len(configs), figsize=(4*len(configs), 8))
    results = []
    
    for idx, cfg in enumerate(configs):
        comp = generate_components(
            resolution=64, y_top=8, y_bottom=56,
            w_belly=cfg["w_belly"], w_bottom=14,
            belly_position=0.40,
            thickness_carcass=cfg["thickness_carcass"],
            thickness_crown=cfg["thickness_crown"],
            thickness_flanks_bottom=2, lip_rounding=3
        )
        
        props = mechanics.compute_properties(comp['carcass'], comp['crown'], comp['flanks'])
        deformed = mechanics.apply_vertical_load(comp['carcass'], comp['crown'], comp['flanks'], force=2.0)
        
        results.append({
            'name': cfg['name'], 'K_vert': props['K_vert'],
            'mass': props['mass_index'], 'perf': props['performance_index']
        })
        
        axes[0, idx].imshow(make_overlay(comp['carcass'], comp['crown'], comp['flanks']), origin='upper')
        axes[0, idx].set_title(f"{cfg['name']}\nK={props['K_vert']:.1f}")
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(make_overlay(deformed['carcass'], deformed['crown'], deformed['flanks']), origin='upper')
        axes[1, idx].set_title(f"Perf={props['performance_index']:.4f}")
        axes[1, idx].axis('off')
    
    patches = [mpatches.Patch(color=colors[name], label=name) for name in colors]
    fig.legend(handles=patches, loc='lower center', ncol=3)
    fig.suptitle("Comparaison de géométries (F=2.0)\nHaut: Original, Bas: Déformé", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()
    
    # Tableau récapitulatif
    print(f"\n{'Nom':<10} {'K_vert':>10} {'Masse':>10} {'Performance':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['name']:<10} {r['K_vert']:>10.2f} {r['mass']:>10.0f} {r['perf']:>12.5f}")
    
    # Conclusion
    best = max(results, key=lambda x: x['perf'])
    print(f"\n→ Meilleure performance : {best['name']} (ratio portance/masse optimal)")

compare_geometries()


# Cellule 6: Markdown - Section 15
"""
## 15. Intégration dans le Dataset

Les propriétés mécaniques peuvent être ajoutées au CSV pour chaque échantillon.
Cela permet d'entraîner des modèles génératifs conditionnés sur la performance.

### Colonnes ajoutées :
- `mech_K_vert` : Raideur verticale
- `mech_mass_index` : Indice de masse
- `mech_delta_rel` : Déflexion relative
- `mech_performance_index` : Performance globale

### Pour les modèles génératifs :
1. **Génération conditionnelle** : Conditionner sur `performance_index` pour générer des pneus performants
2. **Évaluation** : Comparer déformation prédite vs déformation théorique
3. **Dataset augmenté** : Inclure les images déformées comme targets
"""


# Cellule 7: Code - Ajout des métriques au dataset
# ------------------------------------------------

def add_mechanics_to_dataframe(df, data_dir):
    """Ajoute les propriétés mécaniques au DataFrame."""
    mechanics = TireMechanics()
    data_dir = Path(data_dir)
    
    new_cols = {k: [] for k in ['mech_K_vert', 'mech_mass_index', 'mech_delta_rel', 
                                 'mech_stiffness_index', 'mech_performance_index']}
    
    for idx in tqdm(range(len(df)), desc="Calcul mécanique"):
        row = df.iloc[idx]
        
        carcass = plt.imread(str(data_dir / row['img_carcass']))
        crown = plt.imread(str(data_dir / row['img_crown']))
        flanks = plt.imread(str(data_dir / row['img_flanks']))
        
        if len(carcass.shape) == 3:
            carcass, crown, flanks = carcass[:,:,0], crown[:,:,0], flanks[:,:,0]
        
        props = mechanics.compute_properties(carcass, crown, flanks)
        
        new_cols['mech_K_vert'].append(props['K_vert'])
        new_cols['mech_mass_index'].append(props['mass_index'])
        new_cols['mech_delta_rel'].append(props['delta_rel'])
        new_cols['mech_stiffness_index'].append(props['stiffness_index'])
        new_cols['mech_performance_index'].append(props['performance_index'])
    
    for col, values in new_cols.items():
        df[col] = values
    
    return df

# Exemple d'utilisation (si dataset déjà généré):
# df = add_mechanics_to_dataframe(df, output_dir)
# df.to_csv(Path(output_dir) / 'metadata_with_mechanics.csv', index=False)


# Cellule 8: Code - Génération de paires (original, déformé)
# ----------------------------------------------------------

def generate_deformation_pairs(df, data_dir, output_dir_deformed, force=2.0):
    """
    Génère les images déformées pour tout le dataset.
    
    Utile pour entraîner un modèle qui prédit l'image déformée
    à partir de l'image originale.
    """
    mechanics = TireMechanics()
    data_dir = Path(data_dir)
    output_dir_deformed = Path(output_dir_deformed)
    
    # Créer les dossiers
    for comp in ['carcass', 'crown', 'flanks', 'full']:
        (output_dir_deformed / comp).mkdir(parents=True, exist_ok=True)
    
    delta_totals = []
    
    for idx in tqdm(range(len(df)), desc="Génération déformées"):
        row = df.iloc[idx]
        sample_id = row['sample_id']
        
        # Charger
        carcass = plt.imread(str(data_dir / row['img_carcass']))
        crown = plt.imread(str(data_dir / row['img_crown']))
        flanks = plt.imread(str(data_dir / row['img_flanks']))
        
        if len(carcass.shape) == 3:
            carcass, crown, flanks = carcass[:,:,0], crown[:,:,0], flanks[:,:,0]
        
        # Déformer
        deformed = mechanics.apply_vertical_load(carcass, crown, flanks, force, 
                                                 return_displacement=True)
        
        delta_totals.append(deformed.get('delta_total', 0))
        
        # Sauvegarder
        for comp_name in ['carcass', 'crown', 'flanks', 'full']:
            img_path = output_dir_deformed / comp_name / f"{sample_id:05d}.png"
            plt.imsave(str(img_path), deformed[comp_name], cmap='gray')
    
    # Ajouter les chemins et delta au DataFrame
    df_copy = df.copy()
    df_copy['img_carcass_deformed'] = df_copy['sample_id'].apply(lambda x: f"carcass/{x:05d}.png")
    df_copy['img_crown_deformed'] = df_copy['sample_id'].apply(lambda x: f"crown/{x:05d}.png")
    df_copy['img_flanks_deformed'] = df_copy['sample_id'].apply(lambda x: f"flanks/{x:05d}.png")
    df_copy['applied_force'] = force
    df_copy['delta_total_px'] = delta_totals
    
    df_copy.to_csv(output_dir_deformed / 'metadata_deformed.csv', index=False)
    
    print(f"\n✅ Images déformées générées dans {output_dir_deformed}")
    print(f"   Force appliquée: {force}")
    print(f"   Delta moyen: {np.mean(delta_totals):.2f} px")
    
    return df_copy

# Exemple :
# df_deformed = generate_deformation_pairs(df, output_dir, './toytire_v7_deformed', force=2.0)


# Cellule 9: Markdown - Évaluation pour modèles génératifs
"""
## 16. Évaluation des Modèles Génératifs

### Classe DeformationEvaluator

Pour évaluer si un modèle génératif prédit correctement la déformation :

```python
evaluator = DeformationEvaluator()

scores = evaluator.evaluate_deformation(
    original_carcass, original_crown, original_flanks,    # Entrée
    predicted_carcass, predicted_crown, predicted_flanks,  # Sortie du modèle
    force=2.0
)

print(scores['global_score'])  # 0-1, 1 = parfait
```

### Métriques d'évaluation :
| Métrique | Description |
|----------|-------------|
| `volume_conservation` | Le volume total est-il conservé ? |
| `component_ratio_preservation` | Les ratios carcasse/crown/flanks sont-ils préservés ? |
| `shape_similarity` | IoU avec la déformation théorique |
| `height_change_accuracy` | Le changement de hauteur est-il correct ? |
| `global_score` | Score pondéré global |
"""


# Cellule 10: Code - DeformationEvaluator
# ---------------------------------------

@dataclass
class DeformationEvaluator:
    """Évalue la qualité d'une déformation prédite par un modèle."""
    
    mechanics: TireMechanics = field(default_factory=TireMechanics)
    
    def evaluate_deformation(self, 
                            orig_carcass, orig_crown, orig_flanks,
                            pred_carcass, pred_crown, pred_flanks,
                            force: float = 1.0) -> Dict[str, float]:
        """Compare déformation prédite vs théorique."""
        scores = {}
        
        # Binariser
        orig_c = (orig_carcass > 0.5).astype(np.float32)
        orig_cr = (orig_crown > 0.5).astype(np.float32)
        orig_f = (orig_flanks > 0.5).astype(np.float32)
        pred_c = (pred_carcass > 0.5).astype(np.float32)
        pred_cr = (pred_crown > 0.5).astype(np.float32)
        pred_f = (pred_flanks > 0.5).astype(np.float32)
        
        # Déformation théorique
        theoretical = self.mechanics.apply_vertical_load(orig_c, orig_cr, orig_f, force)
        
        # 1. Conservation du volume
        orig_vol = orig_c.sum() + orig_cr.sum() + orig_f.sum()
        pred_vol = pred_c.sum() + pred_cr.sum() + pred_f.sum()
        if orig_vol > 0:
            scores['volume_conservation'] = np.exp(-5 * abs(pred_vol/orig_vol - 1))
        else:
            scores['volume_conservation'] = 0.0
        
        # 2. Conservation des ratios
        if orig_vol > 0 and pred_vol > 0:
            orig_ratios = np.array([orig_c.sum(), orig_cr.sum(), orig_f.sum()]) / orig_vol
            pred_ratios = np.array([pred_c.sum(), pred_cr.sum(), pred_f.sum()]) / pred_vol
            scores['component_ratio_preservation'] = np.exp(-10 * np.abs(orig_ratios - pred_ratios).mean())
        else:
            scores['component_ratio_preservation'] = 0.0
        
        # 3. Similarité de forme (IoU avec théorique)
        def iou(a, b):
            inter = ((a > 0.5) & (b > 0.5)).sum()
            union = ((a > 0.5) | (b > 0.5)).sum()
            return inter / union if union > 0 else 0
        
        scores['shape_similarity'] = np.mean([
            iou(pred_c, theoretical['carcass']),
            iou(pred_cr, theoretical['crown']),
            iou(pred_f, theoretical['flanks'])
        ])
        
        # 4. Précision du changement de hauteur
        def get_h(img):
            rows = np.where(img.any(axis=1))[0]
            return len(rows) if len(rows) > 0 else 0
        
        orig_h = get_h(orig_c + orig_cr + orig_f > 0)
        pred_h = get_h(pred_c + pred_cr + pred_f > 0)
        theo_h = get_h(theoretical['full'])
        
        expected_dh = orig_h - theo_h
        actual_dh = orig_h - pred_h
        if abs(expected_dh) > 0:
            scores['height_change_accuracy'] = max(0, 1 - abs(actual_dh - expected_dh) / abs(expected_dh))
        else:
            scores['height_change_accuracy'] = 1.0 if abs(actual_dh) < 2 else 0.0
        
        # Score global
        weights = {'volume_conservation': 0.25, 'component_ratio_preservation': 0.20,
                   'shape_similarity': 0.35, 'height_change_accuracy': 0.20}
        scores['global_score'] = sum(weights[k] * scores[k] for k in weights)
        
        return scores
    
    def summary(self, scores: Dict[str, float]) -> str:
        """Résumé textuel des scores."""
        lines = ["=" * 50, "DEFORMATION EVALUATION", "=" * 50]
        for k, v in scores.items():
            if k != 'global_score':
                status = "✓" if v >= 0.8 else "⚠" if v >= 0.5 else "✗"
                lines.append(f"{status} {k:35s}: {v:.3f}")
        lines.extend(["-" * 50, f"{'GLOBAL SCORE':35s}: {scores['global_score']:.3f}", "=" * 50])
        return "\n".join(lines)

# Test
evaluator = DeformationEvaluator()

# Simuler une "prédiction parfaite" (= déformation théorique)
theoretical = mechanics.apply_vertical_load(
    test_tire['carcass'], test_tire['crown'], test_tire['flanks'], force=2.0
)

perfect_scores = evaluator.evaluate_deformation(
    test_tire['carcass'], test_tire['crown'], test_tire['flanks'],
    theoretical['carcass'], theoretical['crown'], theoretical['flanks'],
    force=2.0
)
print("Prédiction parfaite (= déformation théorique):")
print(evaluator.summary(perfect_scores))

# Simuler une "mauvaise prédiction" (= original non déformé)
bad_scores = evaluator.evaluate_deformation(
    test_tire['carcass'], test_tire['crown'], test_tire['flanks'],
    test_tire['carcass'], test_tire['crown'], test_tire['flanks'],
    force=2.0
)
print("\nMauvaise prédiction (= original sans déformation):")
print(evaluator.summary(bad_scores))
