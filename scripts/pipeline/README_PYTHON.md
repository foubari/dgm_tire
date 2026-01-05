# EpureDGM Pipeline Python - Guide Complet

Pipeline Python automatisée pour l'entraînement, le sampling et l'évaluation de tous les modèles EpureDGM.

**Avantages Python vs Bash:**
- ✓ Pas de problèmes de fins de ligne Windows/Unix
- ✓ Symlinks gérés automatiquement (cross-platform)
- ✓ Meilleure gestion d'erreurs
- ✓ Plus facile à débugger
- ✓ Logs structurés avec couleurs

---

## Quick Start

### 1. Vérifier l'Environnement

```bash
python scripts/pipeline/verify_pipeline.py --dataset toy
```

Vérifie:
- ✓ Configs existent
- ✓ Données disponibles
- ✓ GPU disponible
- ✓ Espace disque (>150GB)
- ✓ Dépendances Python

### 2. Test Rapide (RECOMMANDÉ)

```bash
# Dry-run (aucune exécution)
python scripts/pipeline/run_pipeline_test.py --dataset toy --models ddpm --dry-run

# Test réel avec 1 modèle, 50 samples
python scripts/pipeline/run_pipeline_test.py --dataset toy --models ddpm --seeds 0
```

### 3. Pipeline Complète

```bash
# TOY dataset
python scripts/pipeline/run_pipeline.py --dataset toy

# EPURE dataset
python scripts/pipeline/run_pipeline.py --dataset epure
```

---

## Scripts Disponibles

### run_pipeline.py - Script Principal

```bash
python scripts/pipeline/run_pipeline.py --dataset {epure|toy} [OPTIONS]
```

**Arguments:**
- `--dataset {epure,toy}` - Dataset (REQUIS)
- `--models MODEL1,MODEL2,...` - Modèles spécifiques (défaut: tous)
- `--seeds SEED1,SEED2,...` - Seeds (défaut: 0,1,2)
- `--num-samples N` - Nombre samples (défaut: 1000)
- `--skip-training` - Skip training
- `--skip-sampling` - Skip sampling
- `--skip-evaluation` - Skip évaluation
- `--dry-run` - Afficher sans exécuter

**Exemples:**

```bash
# Pipeline complète EPURE
python scripts/pipeline/run_pipeline.py --dataset epure

# Seulement certains modèles
python scripts/pipeline/run_pipeline.py --dataset toy --models ddpm,vae,flow_matching

# Seulement 1 seed
python scripts/pipeline/run_pipeline.py --dataset toy --models ddpm --seeds 0

# Évaluer runs existants
python scripts/pipeline/run_pipeline.py --dataset epure --skip-training --skip-sampling

# Dry-run
python scripts/pipeline/run_pipeline.py --dataset toy --dry-run
```

### run_pipeline_test.py - Test Rapide

```bash
python scripts/pipeline/run_pipeline_test.py --dataset {epure|toy} [OPTIONS]
```

Utilise 50 samples au lieu de 1000 pour tests rapides.

**Exemple:**
```bash
python scripts/pipeline/run_pipeline_test.py --dataset toy --models ddpm --seeds 0
```

### verify_pipeline.py - Vérification

```bash
python scripts/pipeline/verify_pipeline.py --dataset {epure|toy}
```

Vérifie tous les prérequis avant d'exécuter la pipeline.

---

## Structure des Fichiers

```
EpureDGM/
├── scripts/pipeline/
│   ├── run_pipeline.py              # Script principal
│   ├── run_pipeline_test.py         # Test rapide
│   ├── verify_pipeline.py           # Vérification
│   └── README_PYTHON.md             # Cette doc
│
├── logs/pipeline/{timestamp}/       # Logs par run
│   ├── pipeline.log                 # Log principal
│   ├── errors.log                   # Erreurs
│   ├── state.json                   # État (pour resume)
│   ├── training/
│   │   ├── ddpm_seed0.log
│   │   └── ...
│   ├── sampling/
│   │   ├── ddpm_seed0_unconditional.log
│   │   └── ...
│   └── evaluation/
│       ├── ddpm_all_seeds.log
│       └── ...
│
├── outputs/{model}/                 # Checkpoints
│   ├── run_seed0 -> 2025-12-21_10-00-00/
│   ├── run_seed1 -> 2025-12-21_12-30-00/
│   └── run_seed2 -> 2025-12-21_15-00-00/
│
├── samples/{model}/                 # Échantillons
│   ├── unconditional/
│   ├── conditional/
│   └── inpainting/{component}/
│
└── evaluation_results/{dataset}/    # Résultats
    ├── {model}/
    │   ├── run_seed0.json
    │   ├── run_seed1.json
    │   └── run_seed2.json
    └── summary.json
```

---

## Workflow Complet

### Étape 1: Vérification

```bash
python scripts/pipeline/verify_pipeline.py --dataset toy
```

Si échecs, corriger avant de continuer.

### Étape 2: Test Rapide

```bash
# Test avec 1 modèle en dry-run
python scripts/pipeline/run_pipeline_test.py --dataset toy --models ddpm --dry-run

# Si OK, test réel
python scripts/pipeline/run_pipeline_test.py --dataset toy --models ddpm --seeds 0
```

Durée: ~10-15min

### Étape 3: Pipeline TOY

```bash
python scripts/pipeline/run_pipeline.py --dataset toy
```

Durée: ~30-40h (9 modèles × 3 seeds × 100 epochs)

### Étape 4: Pipeline EPURE

```bash
python scripts/pipeline/run_pipeline.py --dataset epure
```

Durée: ~30-40h

---

## Monitoring

### Logs en Temps Réel

```bash
# Log principal (Windows PowerShell)
Get-Content logs/pipeline/latest/pipeline.log -Wait

# Linux/Mac
tail -f logs/pipeline/latest/pipeline.log

# Log d'un modèle spécifique
tail -f logs/pipeline/latest/training/ddpm_seed0.log
```

### État de la Pipeline

```python
import json

# Charger état
with open('logs/pipeline/latest/state.json') as f:
    state = json.load(f)

# Voir ce qui est complété
print(state['completed'])

# Voir ce qui a échoué
print(state['failed'])
```

---

## Gestion d'Erreurs

### Erreurs Pendant Training

Si un modèle échoue:
1. L'erreur est loggée dans `errors.log`
2. Le modèle est marqué comme failed dans `state.json`
3. La pipeline **continue** avec les autres modèles

Pour relancer seulement les modèles échoués:
```bash
# Vérifier state.json pour voir les échecs
cat logs/pipeline/latest/state.json

# Relancer seulement le modèle échoué
python scripts/pipeline/run_pipeline.py --dataset toy --models vae --seeds 0
```

### GPU Out of Memory

Réduire batch size dans les configs:
```bash
# Éditer config
vim src/configs/pipeline/toy/ddpm_toy_pipeline.yaml

# Changer batch_size de 64 -> 32
training:
  batch_size: 32
```

### Checkpoint Manquant

```bash
# Vérifier que training a réussi
cat logs/pipeline/latest/training/ddpm_seed0.log

# Vérifier symlink
ls -la outputs/ddpm_toy/run_seed0

# Re-entraîner si nécessaire
python scripts/pipeline/run_pipeline.py --dataset toy --models ddpm --seeds 0
```

---

## Résultats

### Structure Résultats

**Par seed** (`evaluation_results/{dataset}/{model}/run_seed{N}.json`):
```json
{
  "model": "ddpm",
  "dataset": "toy",
  "seed": 0,
  "metrics": {
    "fid": 23.45,
    "iou_dice": {...},
    "rce": {...},
    "com": {...}
  }
}
```

**Agrégé** (`evaluation_results/{dataset}/summary.json`):
```json
{
  "dataset": "toy",
  "model_aggregates": {
    "ddpm": {
      "num_runs": 3,
      "fid": {
        "mean": 23.45,
        "std": 1.23
      }
    }
  },
  "rankings": [...]
}
```

### Afficher Résultats

```python
import json

# Charger summary
with open('evaluation_results/toy/summary.json') as f:
    summary = json.load(f)

# Voir rankings
for i, model in enumerate(summary['rankings'], 1):
    print(f"{i}. {model['model']}: FID = {model['fid_mean']:.2f} ± {model['fid_std']:.2f}")
```

Ou avec le script aggregate:
```bash
python scripts/aggregate_results.py --dataset toy
```

---

## Estimation Temps & Ressources

### Temps par Dataset

| Stage | Durée/Run | Total (27 runs) |
|-------|-----------|-----------------|
| Training | ~3h | ~75h |
| Sampling | ~1h | ~27h |
| Evaluation | ~10min | ~4.5h |
| **TOTAL** | - | **~106h (~4.5 jours)** |

### Espace Disque

- Checkpoints: ~500MB/run
- Samples: ~1GB/run
- Logs: ~50MB/run
- **Total (54 runs)**: ~80GB
- **Recommandé**: 150GB

### GPU

- **Minimum**: 8GB VRAM
- **Recommandé**: 16GB VRAM
- **Optimal**: 24GB VRAM

---

## Configurations

### Configs Pipeline vs Base

Les configs pipeline overrident:
```yaml
training:
  epochs: 100           # vs 1000 défaut
  batch_size: 64/128    # Optimisé par modèle
  check_every: 25
  eval_every: 10

sampling:
  num_samples: 1000     # vs config défaut
```

### Batch Sizes

- Diffusion (ddpm, mdm, flow_matching): 64
- GAN (vqvae, wgan_gp): 64
- VAE (vae, gmrf_mvae, meta_vae): 128
- Multi-modal (mmvaeplus): 128

### Contraintes Normalisées

Par défaut: `normalized: false`

Pour activer:
```bash
# Option 1: Modifier create_pipeline_configs.py
# Ajouter: config['data']['normalized'] = True

# Option 2: Modifier manuellement les configs
vim src/configs/pipeline/toy/ddpm_toy_pipeline.yaml
# Changer: normalized: true
```

---

## Commandes Utiles

### Status

```bash
# Voir dernier log
cat logs/pipeline/latest/pipeline.log

# Compter runs réussis
ls -d outputs/*/run_seed* | wc -l

# Voir échecs
cat logs/pipeline/latest/errors.log

# Voir état JSON
python -m json.tool logs/pipeline/latest/state.json
```

### Nettoyage

```bash
# Supprimer logs anciens
rm -rf logs/pipeline/2025-*

# Supprimer samples (garder checkpoints)
rm -rf samples/*

# Supprimer évaluations
rm -rf evaluation_results/*
```

---

## Troubleshooting

### Import Error

```bash
# Vérifier environnement
python --version
python -c "import torch; print(torch.__version__)"

# Activer bon environnement
conda activate hf_diffusion
```

### Permission Denied (Symlinks Windows)

Les symlinks peuvent nécessiter privilèges admin. Le script utilise un fallback automatique.

### Script Interrompu

La pipeline sauvegarde son état régulièrement. Pour reprendre:

```bash
# Voir état actuel
cat logs/pipeline/latest/state.json

# Identifier dernier modèle réussi
# Puis relancer depuis là (pas encore implémenté dans version Python)
# Pour l'instant, relancer modèles manquants manuellement
python scripts/pipeline/run_pipeline.py --dataset toy --models vae,gmrf_mvae
```

---

## Différences Bash vs Python

| Feature | Bash | Python |
|---------|------|--------|
| Fins de ligne | ✗ Problèmes Windows/Unix | ✓ Pas de problème |
| Symlinks | ✗ Échecs Windows | ✓ Fallback automatique |
| Gestion erreurs | ✗ `set -e` arrête tout | ✓ Continue-on-error |
| Logs | ✗ Basique | ✓ Couleurs, timestamps |
| État | ✗ Non implémenté | ✓ JSON avec save |
| Debugging | ✗ Difficile | ✓ Facile (pdb, print) |

**Recommandation**: Utiliser les scripts **Python** pour tous les nouveaux runs.

---

## Support

Questions ou problèmes:
1. Vérifier logs: `logs/pipeline/latest/errors.log`
2. Vérifier state: `logs/pipeline/latest/state.json`
3. Dry-run: `python scripts/pipeline/run_pipeline.py --dataset toy --dry-run`
4. Test minimal: `python scripts/pipeline/run_pipeline_test.py --dataset toy --models ddpm --seeds 0`

---

## Changelog

- **v2.0** (2025-12-21): Migration vers Python
  - Scripts Python remplacent Bash
  - Gestion erreurs améliorée
  - Cross-platform (Windows/Linux/Mac)
  - Logs structurés avec couleurs

- **v1.0** (2025-12-21): Version Bash initiale
  - Scripts bash avec problèmes Windows
  - Deprecated - utiliser v2.0 Python
