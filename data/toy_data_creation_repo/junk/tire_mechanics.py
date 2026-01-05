"""
TireMechanics - Calcul de propriétés mécaniques et déformation pour épures de pneu

Basé sur un modèle simplifié de ressorts en parallèle :
- Chaque colonne = un ressort vertical
- k_x = E_x * A_x / L_x
- K_total = sum(k_x)

Inspiré des modèles de raideur radiale en mécanique du pneu.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


@dataclass
class MaterialProperties:
    """Propriétés matériaux fictives pour chaque composant."""
    # Module d'Young relatif (unités arbitraires)
    E_carcass: float = 1.0      # Carcasse : rigide (structure principale)
    E_crown: float = 0.8        # Couronne : légèrement plus souple
    E_flanks: float = 0.5       # Flancs : plus souples
    
    # Densité relative (pour calcul de masse)
    rho_carcass: float = 1.0
    rho_crown: float = 1.2      # Couronne un peu plus dense (renforts)
    rho_flanks: float = 0.8     # Flancs plus légers


@dataclass 
class TireMechanics:
    """
    Calcule les propriétés mécaniques d'une épure de pneu et simule la déformation.
    
    Modèle simplifié :
    - Ressorts verticaux en parallèle
    - Déformation sous charge verticale
    - Conservation de volume (approximée)
    
    Usage:
    ------
    >>> mechanics = TireMechanics()
    >>> props = mechanics.compute_properties(carcass, crown, flanks)
    >>> deformed = mechanics.apply_vertical_load(carcass, crown, flanks, force=1.0)
    """
    
    materials: MaterialProperties = field(default_factory=MaterialProperties)
    
    # Paramètres de simulation
    nominal_force: float = 1.0  # Force nominale pour calcul de déflexion
    max_strain: float = 0.3     # Déformation max (30%) pour éviter les artefacts
    
    def compute_properties(self, carcass: np.ndarray, crown: np.ndarray, 
                          flanks: np.ndarray) -> Dict[str, float]:
        """
        Calcule les propriétés mécaniques à partir des images de composants.
        
        Parameters:
        -----------
        carcass, crown, flanks : np.ndarray
            Images binaires des composants (H, W)
            
        Returns:
        --------
        Dict avec :
            - K_vert : raideur verticale totale
            - mass_index : indice de masse
            - delta_rel : déflexion relative sous charge nominale
            - stiffness_index : indice de rigidité (1/delta_rel)
            - performance_index : stiffness_index / mass_index
            - height_px, width_px : dimensions
        """
        # Binariser si nécessaire
        carcass = (carcass > 0.5).astype(np.float32)
        crown = (crown > 0.5).astype(np.float32)
        flanks = (flanks > 0.5).astype(np.float32)
        
        full = (carcass + crown + flanks) > 0
        
        if full.sum() == 0:
            return self._empty_properties()
        
        # Dimensions
        rows = np.where(full.any(axis=1))[0]
        cols = np.where(full.any(axis=0))[0]
        
        y_top, y_bottom = rows.min(), rows.max()
        x_left, x_right = cols.min(), cols.max()
        
        height_px = y_bottom - y_top + 1
        width_px = x_right - x_left + 1
        R_out = height_px / 2  # Rayon externe approximé
        
        # ====== RAIDEUR VERTICALE ======
        # Pour chaque colonne : k_x = E_eff * A_x / L_x
        K_vert = 0.0
        k_per_column = np.zeros(full.shape[1])
        
        for x in range(x_left, x_right + 1):
            col_carcass = carcass[:, x]
            col_crown = crown[:, x]
            col_flanks = flanks[:, x]
            col_full = full[:, x]
            
            if col_full.sum() == 0:
                continue
            
            # Aire de la section (pixels actifs)
            A_x = col_full.sum()
            
            # Hauteur de la colonne
            active_rows = np.where(col_full)[0]
            L_x = active_rows.max() - active_rows.min() + 1
            
            if L_x == 0:
                continue
            
            # Module effectif (moyenne pondérée par aire)
            E_eff = (
                col_carcass.sum() * self.materials.E_carcass +
                col_crown.sum() * self.materials.E_crown +
                col_flanks.sum() * self.materials.E_flanks
            ) / max(A_x, 1)
            
            # Raideur de cette colonne
            k_x = E_eff * A_x / L_x
            k_per_column[x] = k_x
            K_vert += k_x
        
        # ====== MASSE ======
        mass_index = (
            carcass.sum() * self.materials.rho_carcass +
            crown.sum() * self.materials.rho_crown +
            flanks.sum() * self.materials.rho_flanks
        )
        
        # ====== DÉFLEXION ======
        if K_vert > 0:
            delta = self.nominal_force / K_vert
            delta_rel = delta / R_out if R_out > 0 else 0
        else:
            delta = 0
            delta_rel = 0
        
        # ====== INDICES DE PERFORMANCE ======
        stiffness_index = 1.0 / delta_rel if delta_rel > 0 else 0
        performance_index = stiffness_index / mass_index if mass_index > 0 else 0
        
        return {
            'K_vert': K_vert,
            'mass_index': mass_index,
            'delta': delta,
            'delta_rel': delta_rel,
            'stiffness_index': stiffness_index,
            'performance_index': performance_index,
            'height_px': height_px,
            'width_px': width_px,
            'R_out': R_out,
            'k_per_column': k_per_column,  # Utile pour la déformation
        }
    
    def _empty_properties(self) -> Dict[str, float]:
        """Retourne des propriétés vides pour une image vide."""
        return {
            'K_vert': 0, 'mass_index': 0, 'delta': 0, 'delta_rel': 0,
            'stiffness_index': 0, 'performance_index': 0,
            'height_px': 0, 'width_px': 0, 'R_out': 0,
            'k_per_column': np.array([]),
        }
    
    def apply_vertical_load(self, carcass: np.ndarray, crown: np.ndarray,
                           flanks: np.ndarray, force: float = 1.0,
                           return_displacement: bool = False) -> Dict[str, np.ndarray]:
        """
        Applique une charge verticale et retourne les composants déformés.
        
        Le modèle :
        - Chaque colonne se comprime proportionnellement à sa compliance (1/k)
        - La déformation est appliquée depuis le bas (contact sol)
        - Conservation de volume approximée (étalement latéral)
        
        Parameters:
        -----------
        carcass, crown, flanks : np.ndarray
            Images binaires des composants
        force : float
            Force appliquée (unités arbitraires)
        return_displacement : bool
            Si True, retourne aussi le champ de déplacement
            
        Returns:
        --------
        Dict avec :
            - 'carcass', 'crown', 'flanks', 'full' : images déformées
            - 'displacement_field' : (optionnel) champ de déplacement vertical
        """
        # Binariser
        carcass = (carcass > 0.5).astype(np.float32)
        crown = (crown > 0.5).astype(np.float32)
        flanks = (flanks > 0.5).astype(np.float32)
        
        resolution = carcass.shape[0]
        full = (carcass + crown + flanks) > 0
        
        if full.sum() == 0:
            return {
                'carcass': carcass, 'crown': crown, 
                'flanks': flanks, 'full': full.astype(np.float32)
            }
        
        # Calculer les propriétés
        props = self.compute_properties(carcass, crown, flanks)
        K_vert = props['K_vert']
        k_per_column = props['k_per_column']
        
        if K_vert == 0:
            return {
                'carcass': carcass, 'crown': crown,
                'flanks': flanks, 'full': full.astype(np.float32)
            }
        
        # Déplacement total
        delta_total = force / K_vert
        
        # Limiter la déformation max
        rows = np.where(full.any(axis=1))[0]
        height = rows.max() - rows.min() + 1
        max_delta = height * self.max_strain
        delta_total = min(delta_total, max_delta)
        
        # ====== CHAMP DE DÉPLACEMENT ======
        # Créer un champ de déplacement vertical pour chaque pixel
        displacement_y = np.zeros((resolution, resolution), dtype=np.float32)
        
        # Pour chaque colonne, calculer le déplacement
        cols = np.where(full.any(axis=0))[0]
        if len(cols) == 0:
            return {
                'carcass': carcass, 'crown': crown,
                'flanks': flanks, 'full': full.astype(np.float32)
            }
        
        y_bottom = rows.max()
        y_top = rows.min()
        
        for x in cols:
            k_x = k_per_column[x]
            if k_x == 0:
                continue
            
            # Compliance relative de cette colonne
            compliance_x = (1.0 / k_x) / (1.0 / K_vert) if K_vert > 0 else 1.0
            
            # Déplacement de cette colonne (plus compliant = plus de déplacement)
            # Normaliser pour que la moyenne soit delta_total
            delta_x = delta_total * compliance_x
            
            # Le déplacement varie linéairement du haut (0) vers le bas (delta_x)
            col_mask = full[:, x]
            active_rows = np.where(col_mask)[0]
            
            if len(active_rows) == 0:
                continue
            
            col_top = active_rows.min()
            col_bottom = active_rows.max()
            col_height = col_bottom - col_top + 1
            
            for y in active_rows:
                # Interpolation linéaire : 0 en haut, delta_x en bas
                t = (y - col_top) / col_height if col_height > 1 else 0
                displacement_y[y, x] = -delta_x * t  # Négatif = vers le haut (compression)
        
        # ====== APPLIQUER LA DÉFORMATION ======
        # Utiliser l'interpolation pour déformer les images
        y_coords, x_coords = np.meshgrid(np.arange(resolution), np.arange(resolution), indexing='ij')
        
        # Nouvelles coordonnées (déplacées)
        new_y = y_coords - displacement_y  # Soustraire car on "remonte" les pixels
        new_x = x_coords.astype(np.float32)  # Pas de déplacement horizontal pour l'instant
        
        # Appliquer à chaque composant
        def warp_image(img):
            # Utiliser map_coordinates pour l'interpolation
            warped = map_coordinates(img, [new_y, new_x], order=1, mode='constant', cval=0)
            return (warped > 0.5).astype(np.float32)
        
        deformed_carcass = warp_image(carcass)
        deformed_crown = warp_image(crown)
        deformed_flanks = warp_image(flanks)
        deformed_full = (deformed_carcass + deformed_crown + deformed_flanks) > 0
        
        result = {
            'carcass': deformed_carcass,
            'crown': deformed_crown,
            'flanks': deformed_flanks,
            'full': deformed_full.astype(np.float32),
        }
        
        if return_displacement:
            result['displacement_field'] = displacement_y
            result['delta_total'] = delta_total
        
        return result
    
    def visualize_deformation(self, carcass: np.ndarray, crown: np.ndarray,
                             flanks: np.ndarray, force: float = 1.0,
                             figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Visualise l'épure avant et après déformation.
        
        Parameters:
        -----------
        carcass, crown, flanks : np.ndarray
            Images binaires des composants
        force : float
            Force appliquée
        figsize : tuple
            Taille de la figure
            
        Returns:
        --------
        matplotlib.Figure
        """
        # Calculer propriétés et déformation
        props = self.compute_properties(carcass, crown, flanks)
        deformed = self.apply_vertical_load(carcass, crown, flanks, force, 
                                           return_displacement=True)
        
        # Couleurs
        colors = {
            'carcass': [0.9, 0.2, 0.2],
            'crown': [0.2, 0.5, 0.9],
            'flanks': [0.2, 0.8, 0.3]
        }
        
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
        
        # 1. Original
        original_overlay = make_overlay(carcass, crown, flanks)
        axes[0].imshow(original_overlay, origin='upper')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # 2. Déformé
        deformed_overlay = make_overlay(deformed['carcass'], deformed['crown'], 
                                       deformed['flanks'])
        axes[1].imshow(deformed_overlay, origin='upper')
        axes[1].set_title(f'Déformé (F={force:.1f})')
        axes[1].axis('off')
        
        # 3. Superposition (original en transparence + déformé)
        blend = 0.3 * original_overlay + 0.7 * deformed_overlay
        axes[2].imshow(np.clip(blend, 0, 1), origin='upper')
        axes[2].set_title('Superposition')
        axes[2].axis('off')
        
        # 4. Champ de déplacement
        if 'displacement_field' in deformed:
            disp = deformed['displacement_field']
            im = axes[3].imshow(disp, origin='upper', cmap='RdBu_r', 
                               vmin=-disp.max(), vmax=disp.max())
            plt.colorbar(im, ax=axes[3], label='Déplacement (px)')
            axes[3].set_title('Champ de déplacement')
        axes[3].axis('off')
        
        # Légende et infos
        patches = [mpatches.Patch(color=colors[name], label=name) for name in colors]
        fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10)
        
        # Texte avec propriétés
        props_text = (
            f"K_vert: {props['K_vert']:.2f}\n"
            f"Mass: {props['mass_index']:.1f}\n"
            f"δ_rel: {props['delta_rel']:.4f}\n"
            f"Perf: {props['performance_index']:.4f}"
        )
        fig.text(0.02, 0.98, props_text, fontsize=9, verticalalignment='top',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        
        return fig


@dataclass
class DeformationEvaluator:
    """
    Évalue si une déformation générée par un modèle est physiquement cohérente.
    
    Pour un modèle génératif qui prédit image_déformée à partir de image_originale :
    - Compare la déformation prédite avec la déformation théorique
    - Vérifie la conservation de volume
    - Vérifie la cohérence des ratios de composants
    """
    
    mechanics: TireMechanics = field(default_factory=TireMechanics)
    
    def evaluate_deformation(self, 
                            original_carcass: np.ndarray, original_crown: np.ndarray, 
                            original_flanks: np.ndarray,
                            predicted_carcass: np.ndarray, predicted_crown: np.ndarray,
                            predicted_flanks: np.ndarray,
                            force: float = 1.0) -> Dict[str, float]:
        """
        Évalue la qualité d'une déformation prédite.
        
        Parameters:
        -----------
        original_* : np.ndarray
            Images originales (avant déformation)
        predicted_* : np.ndarray
            Images prédites par le modèle (après déformation)
        force : float
            Force appliquée
            
        Returns:
        --------
        Dict avec scores :
            - volume_conservation : conservation du volume total
            - component_ratio_preservation : conservation des ratios de composants
            - shape_similarity : similarité avec déformation théorique
            - height_change_accuracy : précision du changement de hauteur
            - global_score : score global
        """
        scores = {}
        
        # Binariser
        orig_c = (original_carcass > 0.5).astype(np.float32)
        orig_cr = (original_crown > 0.5).astype(np.float32)
        orig_f = (original_flanks > 0.5).astype(np.float32)
        
        pred_c = (predicted_carcass > 0.5).astype(np.float32)
        pred_cr = (predicted_crown > 0.5).astype(np.float32)
        pred_f = (predicted_flanks > 0.5).astype(np.float32)
        
        # Calculer déformation théorique
        theoretical = self.mechanics.apply_vertical_load(
            orig_c, orig_cr, orig_f, force, return_displacement=True
        )
        
        # 1. Conservation du volume
        orig_volume = orig_c.sum() + orig_cr.sum() + orig_f.sum()
        pred_volume = pred_c.sum() + pred_cr.sum() + pred_f.sum()
        
        if orig_volume > 0:
            volume_ratio = pred_volume / orig_volume
            # Score max quand ratio = 1, décroît sinon
            scores['volume_conservation'] = np.exp(-5 * abs(volume_ratio - 1))
        else:
            scores['volume_conservation'] = 0.0
        
        # 2. Conservation des ratios de composants
        if orig_volume > 0 and pred_volume > 0:
            orig_ratios = np.array([orig_c.sum(), orig_cr.sum(), orig_f.sum()]) / orig_volume
            pred_ratios = np.array([pred_c.sum(), pred_cr.sum(), pred_f.sum()]) / pred_volume
            ratio_diff = np.abs(orig_ratios - pred_ratios).mean()
            scores['component_ratio_preservation'] = np.exp(-10 * ratio_diff)
        else:
            scores['component_ratio_preservation'] = 0.0
        
        # 3. Similarité avec déformation théorique (IoU)
        def iou(a, b):
            intersection = ((a > 0.5) & (b > 0.5)).sum()
            union = ((a > 0.5) | (b > 0.5)).sum()
            return intersection / union if union > 0 else 0
        
        iou_carcass = iou(pred_c, theoretical['carcass'])
        iou_crown = iou(pred_cr, theoretical['crown'])
        iou_flanks = iou(pred_f, theoretical['flanks'])
        scores['shape_similarity'] = (iou_carcass + iou_crown + iou_flanks) / 3
        
        # 4. Précision du changement de hauteur
        def get_height(img):
            rows = np.where(img.any(axis=1))[0]
            return rows.max() - rows.min() + 1 if len(rows) > 0 else 0
        
        orig_h = get_height(orig_c + orig_cr + orig_f > 0)
        pred_h = get_height(pred_c + pred_cr + pred_f > 0)
        theo_h = get_height(theoretical['full'])
        
        if orig_h > 0 and theo_h > 0:
            expected_change = orig_h - theo_h
            actual_change = orig_h - pred_h
            if abs(expected_change) > 0:
                height_accuracy = 1 - min(1, abs(actual_change - expected_change) / abs(expected_change))
            else:
                height_accuracy = 1.0 if abs(actual_change) < 2 else 0.0
            scores['height_change_accuracy'] = height_accuracy
        else:
            scores['height_change_accuracy'] = 0.0
        
        # Score global
        weights = {
            'volume_conservation': 0.25,
            'component_ratio_preservation': 0.20,
            'shape_similarity': 0.35,
            'height_change_accuracy': 0.20
        }
        scores['global_score'] = sum(weights[k] * scores[k] for k in weights)
        
        return scores
    
    def summary(self, scores: Dict[str, float]) -> str:
        """Génère un résumé textuel des scores."""
        lines = ["=" * 50]
        lines.append("DEFORMATION EVALUATION")
        lines.append("=" * 50)
        
        for key, value in scores.items():
            if key == 'global_score':
                continue
            status = "✓" if value >= 0.8 else "⚠" if value >= 0.5 else "✗"
            lines.append(f"{status} {key:35s}: {value:.3f}")
        
        lines.append("-" * 50)
        gs = scores['global_score']
        status = "✓" if gs >= 0.8 else "⚠" if gs >= 0.5 else "✗"
        lines.append(f"{status} {'GLOBAL SCORE':35s}: {gs:.3f}")
        lines.append("=" * 50)
        
        return "\n".join(lines)


# ============================================================
# FONCTION POUR AJOUTER AU DATASET
# ============================================================

def add_mechanics_to_dataset(df, data_dir, mechanics=None):
    """
    Ajoute les propriétés mécaniques au DataFrame du dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec les métadonnées du dataset
    data_dir : str or Path
        Répertoire contenant les images
    mechanics : TireMechanics, optional
        Instance de TireMechanics (créée si non fournie)
        
    Returns:
    --------
    pd.DataFrame avec colonnes additionnelles :
        - mech_K_vert, mech_mass_index, mech_delta_rel, etc.
    """
    from pathlib import Path
    
    if mechanics is None:
        mechanics = TireMechanics()
    
    data_dir = Path(data_dir)
    
    new_columns = {
        'mech_K_vert': [],
        'mech_mass_index': [],
        'mech_delta_rel': [],
        'mech_stiffness_index': [],
        'mech_performance_index': [],
    }
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Charger les images
        carcass = plt.imread(str(data_dir / row['img_carcass']))
        crown = plt.imread(str(data_dir / row['img_crown']))
        flanks = plt.imread(str(data_dir / row['img_flanks']))
        
        if len(carcass.shape) == 3:
            carcass = carcass[:, :, 0]
            crown = crown[:, :, 0]
            flanks = flanks[:, :, 0]
        
        # Calculer propriétés
        props = mechanics.compute_properties(carcass, crown, flanks)
        
        new_columns['mech_K_vert'].append(props['K_vert'])
        new_columns['mech_mass_index'].append(props['mass_index'])
        new_columns['mech_delta_rel'].append(props['delta_rel'])
        new_columns['mech_stiffness_index'].append(props['stiffness_index'])
        new_columns['mech_performance_index'].append(props['performance_index'])
    
    for col, values in new_columns.items():
        df[col] = values
    
    return df


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    # Créer un pneu test simple
    resolution = 64
    cx = resolution // 2
    
    # Forme simple
    carcass = np.zeros((resolution, resolution), dtype=np.float32)
    crown = np.zeros((resolution, resolution), dtype=np.float32)
    flanks = np.zeros((resolution, resolution), dtype=np.float32)
    
    # Carcasse : forme de U
    for y in range(10, 55):
        for x in range(resolution):
            dist = abs(x - cx)
            if y < 30:  # Partie haute
                if 15 < dist < 20:
                    carcass[y, x] = 1
            else:  # Partie basse
                if 10 < dist < 15:
                    carcass[y, x] = 1
    
    # Couronne : sur le dessus
    for y in range(5, 15):
        for x in range(resolution):
            dist = abs(x - cx)
            if dist < 22:
                crown[y, x] = 1
    crown = crown * (1 - carcass)  # Pas de chevauchement
    
    # Flancs : sur les côtés en bas
    for y in range(30, 55):
        for x in range(resolution):
            dist = abs(x - cx)
            if 15 < dist < 20:
                flanks[y, x] = 1
    flanks = flanks * (1 - carcass) * (1 - crown)
    
    # Test
    mechanics = TireMechanics()
    props = mechanics.compute_properties(carcass, crown, flanks)
    
    print("=" * 50)
    print("PROPRIÉTÉS MÉCANIQUES")
    print("=" * 50)
    for k, v in props.items():
        if k != 'k_per_column':
            print(f"{k:25s}: {v}")
    
    # Visualiser
    fig = mechanics.visualize_deformation(carcass, crown, flanks, force=2.0)
    plt.savefig('deformation_test.png', dpi=150, bbox_inches='tight')
    print("\nFigure sauvegardée: deformation_test.png")