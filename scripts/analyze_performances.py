"""
Script d'analyse et de visualisation du CSV de performances.

Analyse le fichier performances.csv créé par add_perfs_to_csv.py et génère:
- Statistiques descriptives
- Distributions des colonnes de performance
- Matrice de corrélation
- Comparaisons train/test
- Graphiques de visualisation
"""

import argparse
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration des styles
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


# Colonnes de performance à analyser
RAW_FEATURES = ['m_top', 'm_side', 'm_total', 'round_top']
DELTA_FEATURES = ['dm_top', 'dm_side', 'dm_total', 'dround_top']
PERF_METRICS = ['d_cons', 'd_rigid', 'd_life', 'd_stab']
ALL_PERF_COLS = RAW_FEATURES + DELTA_FEATURES + PERF_METRICS


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV de performances."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ CSV chargé: {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def check_missing_values(df: pd.DataFrame) -> None:
    """Vérifie et affiche les valeurs manquantes."""
    print("\n" + "=" * 80)
    print("VALEURS MANQUANTES")
    print("=" * 80)
    
    missing = df[ALL_PERF_COLS].isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Colonne': missing.index,
        'Manquantes': missing.values,
        'Pourcentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Manquantes'] > 0].sort_values('Manquantes', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
        print(f"\n⚠️  {len(missing_df)} colonnes avec des valeurs manquantes")
    else:
        print("✓ Aucune valeur manquante dans les colonnes de performance")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Affiche les statistiques descriptives."""
    print("\n" + "=" * 80)
    print("STATISTIQUES DESCRIPTIVES")
    print("=" * 80)
    
    # Filtrer les colonnes qui existent
    available_cols = [col for col in ALL_PERF_COLS if col in df.columns]
    
    if not available_cols:
        print("⚠️  Aucune colonne de performance trouvée dans le CSV")
        return
    
    stats = df[available_cols].describe()
    print(stats.round(4))
    
    # Statistiques supplémentaires
    print("\n" + "-" * 80)
    print("STATISTIQUES PAR SPLIT")
    print("-" * 80)
    
    if 'train' in df.columns:
        for split_name, split_df in [('Train', df[df['train'] == True]), ('Test', df[df['train'] == False])]:
            print(f"\n{split_name} ({len(split_df)} échantillons):")
            split_stats = split_df[available_cols].describe()
            print(split_stats.round(4))


def plot_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Génère des graphiques de distribution pour chaque colonne de performance."""
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES GRAPHIQUES DE DISTRIBUTION")
    print("=" * 80)
    
    available_cols = [col for col in ALL_PERF_COLS if col in df.columns]
    
    if not available_cols:
        print("⚠️  Aucune colonne de performance trouvée")
        return
    
    # Créer les sous-graphiques
    n_cols = 4
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(available_cols):
        ax = axes[idx]
        
        # Filtrer les NaN
        data = df[col].dropna()
        
        if len(data) > 0:
            # Histogramme
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {data.mean():.4f}')
            ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Médiane: {data.median():.4f}')
            ax.set_title(f'{col}\n(n={len(data)}, σ={data.std():.4f})', fontsize=10, fontweight='bold')
            ax.set_xlabel('Valeur')
            ax.set_ylabel('Fréquence')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{col}\n(Aucune donnée)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(col, fontsize=10)
    
    # Masquer les axes inutilisés
    for idx in range(len(available_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_path}")
    plt.close()


def plot_boxplots_by_split(df: pd.DataFrame, output_dir: Path) -> None:
    """Génère des boxplots comparant train et test."""
    if 'train' not in df.columns:
        print("⚠️  Colonne 'train' introuvable, impossible de comparer train/test")
        return
    
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES BOXPLOTS PAR SPLIT")
    print("=" * 80)
    
    available_cols = [col for col in ALL_PERF_COLS if col in df.columns]
    
    if not available_cols:
        return
    
    # Grouper par type de métrique
    groups = {
        'Raw Features': [col for col in RAW_FEATURES if col in available_cols],
        'Delta Features': [col for col in DELTA_FEATURES if col in available_cols],
        'Performance Metrics': [col for col in PERF_METRICS if col in available_cols]
    }
    
    for group_name, cols in groups.items():
        if not cols:
            continue
        
        n_cols = min(len(cols), 4)
        n_rows = (len(cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(cols):
            ax = axes[idx]
            
            # Préparer les données
            train_data = df[df['train'] == True][col].dropna()
            test_data = df[df['train'] == False][col].dropna()
            
            if len(train_data) > 0 or len(test_data) > 0:
                data_to_plot = [train_data, test_data]
                labels = ['Train', 'Test']
                
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Colorier les boxplots
                colors = ['lightblue', 'lightcoral']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{col}', fontsize=10, fontweight='bold')
                ax.set_ylabel('Valeur')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Ajouter les statistiques
                if len(train_data) > 0:
                    ax.text(1, train_data.median(), f'Med={train_data.median():.3f}', 
                           ha='center', va='bottom', fontsize=8)
                if len(test_data) > 0:
                    ax.text(2, test_data.median(), f'Med={test_data.median():.3f}', 
                           ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'{col}\n(Aucune donnée)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(col, fontsize=10)
        
        # Masquer les axes inutilisés
        for idx in range(len(cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{group_name} - Comparaison Train/Test', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = output_dir / f'boxplots_{group_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {output_path}")
        plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Génère une matrice de corrélation."""
    print("\n" + "=" * 80)
    print("GÉNÉRATION DE LA MATRICE DE CORRÉLATION")
    print("=" * 80)
    
    available_cols = [col for col in ALL_PERF_COLS if col in df.columns]
    
    if len(available_cols) < 2:
        print("⚠️  Pas assez de colonnes pour calculer les corrélations")
        return
    
    # Calculer la matrice de corrélation
    corr_matrix = df[available_cols].corr()
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    
    ax.set_title('Matrice de Corrélation - Métriques de Performance', 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = output_dir / 'correlation_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Matrice de corrélation sauvegardée: {output_path}")
    plt.close()
    
    # Afficher les corrélations les plus fortes
    print("\nCorrélations les plus fortes (|r| > 0.5):")
    print("-" * 80)
    
    # Extraire les paires de corrélations
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                corr_pairs.append((col1, col2, corr_val))
    
    if corr_pairs:
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, corr_val in corr_pairs[:10]:  # Top 10
            print(f"  {col1} ↔ {col2}: {corr_val:.4f}")
    else:
        print("  Aucune corrélation forte (|r| > 0.5) trouvée")


def plot_scatter_pairs(df: pd.DataFrame, output_dir: Path, max_pairs: int = 6) -> None:
    """Génère des scatter plots pour les paires de métriques les plus corrélées."""
    print("\n" + "=" * 80)
    print("GÉNÉRATION DES SCATTER PLOTS")
    print("=" * 80)
    
    available_cols = [col for col in ALL_PERF_COLS if col in df.columns]
    
    if len(available_cols) < 2:
        return
    
    # Calculer les corrélations
    corr_matrix = df[available_cols].corr()
    
    # Trouver les paires les plus corrélées
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            corr_pairs.append((col1, col2, abs(corr_val)))
    
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = corr_pairs[:max_pairs]
    
    if not top_pairs:
        return
    
    n_cols = 3
    n_rows = (len(top_pairs) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (col1, col2, corr_val) in enumerate(top_pairs):
        ax = axes[idx]
        
        # Données
        data1 = df[col1].dropna()
        data2 = df[col2].dropna()
        
        # Intersection des indices
        common_idx = data1.index.intersection(data2.index)
        if len(common_idx) > 0:
            x = df.loc[common_idx, col1]
            y = df.loc[common_idx, col2]
            
            # Scatter plot avec couleur selon train/test
            if 'train' in df.columns:
                train_mask = df.loc[common_idx, 'train'] == True
                ax.scatter(x[train_mask], y[train_mask], alpha=0.5, label='Train', s=20)
                ax.scatter(x[~train_mask], y[~train_mask], alpha=0.5, label='Test', s=20)
                ax.legend()
            else:
                ax.scatter(x, y, alpha=0.5, s=20)
            
            # Ligne de régression
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Régression (r={corr_val:.3f})')
            
            ax.set_xlabel(col1, fontsize=10)
            ax.set_ylabel(col2, fontsize=10)
            ax.set_title(f'{col1} vs {col2}\n(r={corr_val:.3f})', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{col1} vs {col2}\n(Aucune donnée)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{col1} vs {col2}', fontsize=10)
    
    # Masquer les axes inutilisés
    for idx in range(len(top_pairs), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Scatter Plots - Paires de Métriques les Plus Corrélées', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'scatter_pairs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {output_path}")
    plt.close()


def check_baseline(df: pd.DataFrame, baseline_prefix: str = 'binc_3b3_709014_6') -> None:
    """Vérifie que le baseline a bien des deltas à zéro."""
    print("\n" + "=" * 80)
    print("VÉRIFICATION DU BASELINE")
    print("=" * 80)
    
    if 'matching' not in df.columns:
        print("⚠️  Colonne 'matching' introuvable")
        return
    
    baseline_rows = df[df['matching'].str.lower() == baseline_prefix.lower()]
    
    if len(baseline_rows) == 0:
        print(f"⚠️  Baseline '{baseline_prefix}' introuvable dans le CSV")
        return
    
    print(f"✓ Baseline trouvé: {baseline_prefix} ({len(baseline_rows)} occurrence(s))")
    
    # Vérifier les deltas
    delta_cols = [col for col in DELTA_FEATURES if col in df.columns]
    
    if delta_cols:
        print("\nValeurs des deltas pour le baseline:")
        print("-" * 80)
        for col in delta_cols:
            values = baseline_rows[col].values
            if len(values) > 0:
                val = values[0]
                status = "✓" if (pd.isna(val) or abs(val) < 1e-6) else "⚠️"
                print(f"  {status} {col}: {val}")
            else:
                print(f"  ⚠️  {col}: Valeur manquante")


def main():
    parser = argparse.ArgumentParser(
        description='Analyse et visualise le CSV de performances',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csv', type=str, default='data/epure/performances.csv',
                       help='Chemin vers le CSV de performances')
    parser.add_argument('--output_dir', type=str, default='scripts/analysis_output',
                       help='Répertoire de sortie pour les graphiques')
    parser.add_argument('--baseline_prefix', type=str, default='binc_3b3_709014_6',
                       help='Préfixe du baseline à vérifier')
    parser.add_argument('--no-plots', action='store_true',
                       help='Ne pas générer de graphiques (statistiques uniquement)')
    
    args = parser.parse_args()
    
    # Chemins
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ANALYSE DU CSV DE PERFORMANCES")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Charger le CSV
    df = load_csv(csv_path)
    
    # Vérifier les colonnes disponibles
    print(f"\nColonnes disponibles ({len(df.columns)}):")
    print(f"  {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    available_perf_cols = [col for col in ALL_PERF_COLS if col in df.columns]
    print(f"\nColonnes de performance trouvées ({len(available_perf_cols)}/{len(ALL_PERF_COLS)}):")
    print(f"  {', '.join(available_perf_cols)}")
    
    # Analyses
    check_missing_values(df)
    print_summary_statistics(df)
    check_baseline(df, args.baseline_prefix)
    
    # Graphiques
    if not args.no_plots:
        plot_distributions(df, output_dir)
        plot_boxplots_by_split(df, output_dir)
        plot_correlation_matrix(df, output_dir)
        plot_scatter_pairs(df, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSE TERMINÉE")
    print("=" * 80)
    print(f"Graphiques sauvegardés dans: {output_dir}")


if __name__ == '__main__':
    main()

