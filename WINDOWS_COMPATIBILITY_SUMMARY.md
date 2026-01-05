# Compatibilité Windows - Résumé Complet

## Problème Original

Sur Windows sans privilèges admin, les symlinks échouent. Le pipeline créait:
- `outputs\ddpm\2026-01-05_20-53-09\check\checkpoint_best.pt` (répertoire timestamp)

Mais cherchait:
- `outputs\ddpm\run_seed0\check\checkpoint_best.pt` (symlink manquant)

## Solutions Appliquées

### 1. `scripts/pipeline/run_pipeline.py`

**Fonction `get_checkpoint_path()` (ligne 170)**:
- ✅ Essaie d'abord le symlink `run_seed{N}` (Unix/Windows admin)
- ✅ Fallback: utilise `state.json['run_directories']` (Windows sans admin)
- ✅ Stocke automatiquement les chemins réels pendant l'entraînement

**Fonction `train_model()` (ligne 279)**:
- ✅ Stocke `run_directories[model][seed] = chemin_timestamp` dans state.json
- ✅ Permet à `get_checkpoint_path()` de retrouver les checkpoints

### 2. `src/scripts/evaluate.py`

**Fonction `main()` (ligne 115)**:
- ✅ Essaie d'abord les dossiers `run_seed{N}` (symlinks)
- ✅ Fallback: utilise les dossiers timestamp triés par date de création
- ✅ Associe automatiquement seed0 → 1er dossier, seed1 → 2e, etc.

### 3. Scripts de Validation

**`validate_models_complete.py`** (déjà compatible):
- ✅ Cherche déjà dans les dossiers timestamp
- ✅ Prend automatiquement le plus récent

## Compatibilité Vérifiée

| Script | Unix | Windows Admin | Windows User | Status |
|--------|------|---------------|--------------|--------|
| Training | ✅ | ✅ | ✅ | OK |
| Sampling | ✅ | ✅ | ✅ | OK |
| Evaluation | ✅ | ✅ | ✅ | OK |
| Validation | ✅ | ✅ | ✅ | OK |

## Structure des Répertoires

### Sur Unix/Linux (ou Windows Admin)
```
outputs/ddpm/
├── 2026-01-05_20-53-09/     # Répertoire timestamp
│   └── check/
│       └── checkpoint_best.pt
└── run_seed0 -> 2026-01-05_20-53-09/  # Symlink
```

### Sur Windows (utilisateur normal)
```
outputs/ddpm/
└── 2026-01-05_20-53-09/     # Répertoire timestamp (pas de symlink)
    └── check/
        └── checkpoint_best.pt

logs/pipeline/2026-01-05_20-53-04/
└── state.json               # Contient run_directories mapping
```

**state.json**:
```json
{
  "run_directories": {
    "ddpm": {
      "0": "outputs\\ddpm\\2026-01-05_20-53-09",
      "1": "outputs\\ddpm\\2026-01-05_21-08-15",
      "2": "outputs\\ddpm\\2026-01-05_21-23-42"
    }
  }
}
```

## Utilisation

### Démarrage Normal (Nouveau Training)
```bash
python scripts/pipeline/run_pipeline.py --dataset epure
```
✅ Fonctionne directement, stocke automatiquement les chemins

### Récupération après Interruption
```bash
# Réparer le state.json (si pipeline interrompu AVANT le fix)
python scripts/pipeline/fix_state_windows.py --dataset epure

# Reprendre le pipeline
python scripts/pipeline/run_pipeline.py --dataset epure
```

### Évaluation Manuelle
```bash
# Évaluer avec seeds spécifiques
python src/scripts/evaluate.py --model ddpm --dataset epure --seeds 0,1,2
```
✅ Trouve automatiquement les bons répertoires (symlinks ou timestamps)

## Scripts d'Aide

### `fix_state_windows.py`
Répare un `state.json` existant en mappant les répertoires timestamp aux seeds.

**Usage**:
```bash
python scripts/pipeline/fix_state_windows.py --dataset epure
```

## Pas d'Action Requise

Tous les scripts fonctionnent maintenant automatiquement sur Windows ET Unix/Linux.

Aucune configuration ou privilège admin nécessaire! 🎉
