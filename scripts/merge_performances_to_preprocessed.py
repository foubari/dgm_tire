"""
Script pour fusionner les colonnes de performance dans le CSV preprocessed.

Lit dimensions.csv (preprocessed) et performances.csv, puis crée un nouveau CSV
preprocessed/performances.csv avec toutes les colonnes de dimensions.csv + 
les colonnes de performance de performances.csv.
"""

import argparse
import warnings
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Fusionner les colonnes de performance dans le CSV preprocessed',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dimensions_csv',
        type=str,
        default=r'C:\Users\fouad\Desktop\phd\FOUAD\EpureDGM\data\epure\preprocessed\dimensions.csv',
        help='Chemin vers le CSV dimensions (preprocessed)'
    )
    
    parser.add_argument(
        '--performances_csv',
        type=str,
        default=r'C:\Users\fouad\Desktop\phd\FOUAD\EpureDGM\data\epure\performances.csv',
        help='Chemin vers le CSV performances (source)'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default=r'C:\Users\fouad\Desktop\phd\FOUAD\EpureDGM\data\epure\preprocessed\performances.csv',
        help='Chemin vers le CSV de sortie (preprocessed/performances.csv)'
    )
    
    parser.add_argument(
        '--matching_column',
        type=str,
        default='matching',
        help='Nom de la colonne de matching (default: matching)'
    )
    
    args = parser.parse_args()
    
    # Convertir en Path
    dimensions_path = Path(args.dimensions_csv)
    performances_path = Path(args.performances_csv)
    output_path = Path(args.output_csv)
    
    print("=" * 80)
    print("FUSION DES COLONNES DE PERFORMANCE DANS LE CSV PREPROCESSED")
    print("=" * 80)
    print(f"Dimensions CSV (source): {dimensions_path}")
    print(f"Performances CSV (source): {performances_path}")
    print(f"Output CSV: {output_path}")
    print()
    
    # Vérifier que les fichiers existent
    if not dimensions_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {dimensions_path}")
    
    if not performances_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {performances_path}")
    
    # Charger les CSV
    print("Chargement des CSV...")
    df_dimensions = pd.read_csv(dimensions_path)
    df_performances = pd.read_csv(performances_path)
    
    print(f"✓ Dimensions CSV: {len(df_dimensions)} lignes, {len(df_dimensions.columns)} colonnes")
    print(f"✓ Performances CSV: {len(df_performances)} lignes, {len(df_performances.columns)} colonnes")
    print()
    
    # Vérifier que la colonne matching existe
    if args.matching_column not in df_dimensions.columns:
        raise ValueError(
            f"Colonne '{args.matching_column}' introuvable dans dimensions.csv. "
            f"Colonnes disponibles: {list(df_dimensions.columns)}"
        )
    
    if args.matching_column not in df_performances.columns:
        raise ValueError(
            f"Colonne '{args.matching_column}' introuvable dans performances.csv. "
            f"Colonnes disponibles: {list(df_performances.columns)}"
        )
    
    # Identifier les colonnes de performance
    # Colonnes de performance: de 'm_top' à 'd_stab'
    perf_columns = []
    start_idx = None
    end_idx = None
    
    for idx, col in enumerate(df_performances.columns):
        if col == 'm_top':
            start_idx = idx
        if col == 'd_stab':
            end_idx = idx
    
    if start_idx is None or end_idx is None:
        # Essayer de trouver toutes les colonnes de performance
        perf_candidates = ['m_top', 'm_side', 'm_total', 'round_top',
                          'dm_top', 'dm_side', 'dm_total', 'dround_top',
                          'd_cons', 'd_rigid', 'd_life', 'd_stab']
        
        perf_columns = [col for col in perf_candidates if col in df_performances.columns]
        
        if not perf_columns:
            raise ValueError(
                "Colonnes de performance introuvables. "
                "Recherché: m_top, m_side, m_total, round_top, "
                "dm_top, dm_side, dm_total, dround_top, "
                "d_cons, d_rigid, d_life, d_stab"
            )
    else:
        # Extraire toutes les colonnes entre m_top et d_stab (inclus)
        perf_columns = list(df_performances.columns[start_idx:end_idx + 1])
    
    print(f"Colonnes de performance identifiées ({len(perf_columns)}):")
    print(f"  {', '.join(perf_columns)}")
    print()
    
    # Normaliser les colonnes matching en lowercase pour la correspondance
    df_dimensions[args.matching_column] = df_dimensions[args.matching_column].astype(str).str.lower()
    df_performances[args.matching_column] = df_performances[args.matching_column].astype(str).str.lower()
    
    # Créer un dictionnaire de mapping: matching -> colonnes de performance
    print("Création du mapping matching -> performances...")
    perf_dict = {}
    for _, row in df_performances.iterrows():
        matching = str(row[args.matching_column]).lower()
        perf_values = {col: row[col] for col in perf_columns}
        perf_dict[matching] = perf_values
    
    print(f"✓ {len(perf_dict)} matchings trouvés dans performances.csv")
    print()
    
    # Créer le nouveau dataframe en copiant dimensions.csv
    df_output = df_dimensions.copy()
    
    # Ajouter les colonnes de performance (initialisées à NaN)
    for col in perf_columns:
        df_output[col] = pd.NA
    
    # Remplir les colonnes de performance pour chaque ligne
    print("Fusion des colonnes de performance...")
    matched_count = 0
    missing_count = 0
    
    for idx, row in df_dimensions.iterrows():
        matching = str(row[args.matching_column]).lower()
        
        if matching in perf_dict:
            # Copier toutes les colonnes de performance
            for col in perf_columns:
                df_output.at[idx, col] = perf_dict[matching][col]
            matched_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:  # Afficher seulement les 5 premiers
                warnings.warn(f"Matching '{matching}' non trouvé dans performances.csv")
    
    print(f"✓ {matched_count} matchings fusionnés")
    if missing_count > 0:
        print(f"⚠️  {missing_count} matchings non trouvés dans performances.csv")
    print()
    
    # Statistiques sur les valeurs manquantes
    print("Statistiques sur les colonnes de performance:")
    print("-" * 80)
    for col in perf_columns:
        missing = df_output[col].isna().sum()
        missing_pct = (missing / len(df_output) * 100) if len(df_output) > 0 else 0
        print(f"  {col:15s}: {missing:4d} valeurs manquantes ({missing_pct:5.2f}%)")
    print()
    
    # Créer le répertoire de sortie si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder le nouveau CSV
    df_output.to_csv(output_path, index=False)
    
    print(f"✓ CSV sauvegardé: {output_path}")
    print(f"  Total lignes: {len(df_output)}")
    print(f"  Total colonnes: {len(df_output.columns)}")
    print(f"  Colonnes ajoutées: {len(perf_columns)}")
    print()
    print("=" * 80)
    print("TERMINÉ!")
    print("=" * 80)


if __name__ == '__main__':
    main()

