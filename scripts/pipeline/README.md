# EpureDGM Pipeline Complète

Pipeline automatisée pour l'entraînement, le sampling et l'évaluation de tous les modèles EpureDGM.

## Vue d'Ensemble

Cette pipeline gère automatiquement:
- **9 modèles**: ddpm, mdm, flow_matching, vae, gmrf_mvae, meta_vae, vqvae, wgan_gp, mmvaeplus
- **2 datasets**: EPURE (5 composants), TOY (3 composants)
- **3 seeds** par modèle: 0, 1, 2
- **3 modes de sampling**: unconditional, conditional, inpainting

**Total**: 54 runs d'entraînement (9 modèles × 3 seeds × 2 datasets)

---

## Quick Start

### 1. Générer les Configs (REQUIS)

**IMPORTANT**: Utiliser l'environnement conda `hf_diffusion`

```bash
conda activate hf_diffusion
python scripts/create_pipeline_configs.py
```

Cela crée **27 configs**:
- `src/configs/pipeline/epure/{model}_epure_pipeline.yaml` (9 fichiers)
- `src/configs/pipeline/toy/{model}_toy_pipeline.yaml` (9 fichiers)
- `src/configs/pipeline/test/{model}_pipeline_test.yaml` (9 fichiers)

### 2. Vérifier l'Environnement

```bash
./scripts/pipeline/verify_pipeline.sh --dataset epure
```

Vérifie:
- ✓ Configs existent
- ✓ Données disponibles
- ✓ GPU disponible
- ✓ Espace disque suffisant (>150GB)
- ✓ Dépendances Python

### 3. Test Rapide (RECOMMANDÉ)

Tester la pipeline avec **1 epoch, 50 samples** (~10-15min):

```bash
./scripts/pipeline/run_pipeline_test.sh --dataset toy --models ddpm
```

### 4. Pipeline Complète

**TOY dataset** (plus rapide, ~30-40h):
```bash
./scripts/pipeline/run_pipeline.sh --dataset toy
```

**EPURE dataset** (~30-40h):
```bash
./scripts/pipeline/run_pipeline.sh --dataset epure
```

---

## Structure des Fichiers

```
EpureDGM/
├── scripts/
│   ├── pipeline/
│   │   ├── run_pipeline.sh              # Script principal
│   │   ├── run_pipeline_test.sh         # Script de test
│   │   ├── verify_pipeline.sh           # Vérification pré-vol
│   │   └── lib/
│   │       ├── common.sh                # Utils partagés (logging, couleurs)
│   │       ├── train_utils.sh           # Helpers training
│   │       ├── sample_utils.sh          # Helpers sampling
│   │       └── eval_utils.sh            # Helpers evaluation
│   ├── create_pipeline_configs.py       # Générateur configs
│   └── aggregate_results.py             # Agrégation résultats
│
├── src/configs/pipeline/                # Configs générés
│   ├── epure/                           # EPURE (100 epochs)
│   ├── toy/                             # TOY (100 epochs)
│   └── test/                            # Test (1 epoch, 50 samples)
│
├── outputs/{model}/                     # Sorties d'entraînement
│   ├── run_seed0 -> 2025-12-21_14-30-15/  # Symlink stable
│   ├── run_seed1 -> 2025-12-21_16-15-22/
│   └── run_seed2 -> 2025-12-21_18-00-30/
│
├── samples/{model}/                     # Échantillons générés
│   ├── unconditional/
│   ├── conditional/
│   └── inpainting/{component}/
│
├── evaluation_results/{dataset}/        # Résultats d'évaluation
│   ├── {model}/
│   │   ├── run_seed0.json
│   │   ├── run_seed1.json
│   │   └── run_seed2.json
│   └── summary.json                     # Agrégé
│
└── logs/pipeline/{timestamp}/           # Logs pipeline
    ├── pipeline.log                     # Log principal
    ├── errors.log                       # Échecs
    ├── state.json                       # État (resume)
    ├── training_log.csv                 # CSV tracking
    ├── training/                        # Logs training
    ├── sampling/                        # Logs sampling
    └── evaluation/                      # Logs evaluation
```

---

## Usage Détaillé

### run_pipeline.sh - Script Principal

```bash
./scripts/pipeline/run_pipeline.sh [OPTIONS]
```

**Arguments Requis:**
- `--dataset {epure,toy}` - Dataset à utiliser

**Arguments Optionnels:**
- `--models MODEL1,MODEL2,...` - Modèles spécifiques (défaut: tous)
- `--seeds SEED1,SEED2,...` - Seeds spécifiques (défaut: 0,1,2)
- `--skip-training` - Skipper training (utiliser runs existants)
- `--skip-sampling` - Skipper sampling (utiliser samples existants)
- `--skip-evaluation` - Skipper évaluation
- `--resume-from MODEL` - Reprendre depuis un modèle spécifique
- `--force` - Écraser runs existants
- `--dry-run` - Afficher sans exécuter
- `-h, --help` - Aide

**Exemples:**

```bash
# Pipeline complète EPURE (tous modèles, 3 seeds)
./scripts/pipeline/run_pipeline.sh --dataset epure

# Seulement certains modèles
./scripts/pipeline/run_pipeline.sh --dataset toy --models ddpm,vae,flow_matching

# Reprendre depuis vae (après crash)
./scripts/pipeline/run_pipeline.sh --dataset epure --resume-from vae

# Évaluer runs existants (skip training/sampling)
./scripts/pipeline/run_pipeline.sh --dataset epure --skip-training --skip-sampling

# Dry-run (voir ce qui sera exécuté)
./scripts/pipeline/run_pipeline.sh --dataset toy --dry-run
```

### run_pipeline_test.sh - Script de Test

```bash
./scripts/pipeline/run_pipeline_test.sh --dataset {epure|toy} [OPTIONS]
```

Utilise les configs test (1 epoch, 50 samples) pour validation rapide.

**Exemple:**
```bash
# Test rapide sur TOY avec DDPM
./scripts/pipeline/run_pipeline_test.sh --dataset toy --models ddpm
```

### verify_pipeline.sh - Vérification

```bash
./scripts/pipeline/verify_pipeline.sh --dataset {epure|toy}
```

Effectue toutes les vérifications pré-vol et estime le temps total.

---

## Flux d'Exécution

### Stage 1: Training (27 runs par dataset)

Pour chaque modèle et seed:
1. Charger config `src/configs/pipeline/{dataset}/{model}_{dataset}_pipeline.yaml`
2. Lancer `python src/models/{model}/train.py --config {config} --seed {seed}`
3. Créer symlink `outputs/{model}/run_seed{N}` → dossier timestamp
4. Vérifier que `checkpoint_best.pt` existe

**Durée estimée**: ~3h par run × 27 runs = **~75h**

### Stage 2: Sampling (75 mode runs par dataset)

Pour chaque modèle et seed:

1. **Unconditional** (1000 samples):
   ```bash
   python src/models/{model}/sample.py \
       --checkpoint outputs/{model}/run_seed{N}/check/checkpoint_best.pt \
       --mode unconditional \
       --num_samples 1000
   ```

2. **Conditional** (1000 samples):
   ```bash
   python src/models/{model}/sample.py \
       --checkpoint outputs/{model}/run_seed{N}/check/checkpoint_best.pt \
       --mode conditional \
       --num_samples 1000
   ```

3. **Inpainting** (par composant, si supporté):
   - **EPURE**: 5 runs (group_nc, group_km, bt, fpu, tpc)
   - **TOY**: 3 runs (group_nc, group_km, fpu)
   - **Skip**: MDM, WGAN-GP (pas d'inpainting)

**Durée estimée**: ~1h par run × 27 runs = **~27h**

### Stage 3: Evaluation (27 runs par dataset)

Pour chaque modèle (tous seeds):
```bash
python src/scripts/evaluate.py \
    --model {model} \
    --dataset {dataset} \
    --seeds 0,1,2
```

Calcule:
- **FID** (Frechet Inception Distance)
- **IoU/Dice** distributions + Wasserstein distance
- **RCE** (Relative Count Error)
- **CoM** (Center of Mass distributions)

**Durée estimée**: ~10min par modèle × 9 modèles = **~4.5h**

### Agrégation Finale

```bash
python scripts/aggregate_results.py \
    --dataset {dataset} \
    --output evaluation_results/{dataset}/summary.json
```

Génère:
- Moyennes ± std pour toutes métriques
- Rankings par FID
- Résumé textuel

---

## Monitoring

### Logs en Temps Réel

```bash
# Log principal
tail -f logs/pipeline/latest/pipeline.log

# Log d'un modèle spécifique
tail -f logs/pipeline/latest/training/ddpm_seed0.log

# Erreurs seulement
tail -f logs/pipeline/latest/errors.log
```

### État de la Pipeline

```bash
# Voir l'état JSON
cat logs/pipeline/latest/state.json | jq .

# Résumé rapide
cat logs/pipeline/latest/state.json | jq '.completed'
```

### Progress Tracking

Le script affiche en temps réel:
```
=================================================================================
EPUREDGM PIPELINE - EPURE Dataset
=================================================================================
Start: 2025-12-21 10:00:00
Models: ddpm, mdm, flow_matching, vae, gmrf_mvae, meta_vae, vqvae, wgan_gp, mmvaeplus
Seeds: 0, 1, 2
=================================================================================

[Stage 1/3] TRAINING (27 runs)
  [1/27] ddpm seed0 .......................... ✓ SUCCESS (145 min)
  [2/27] ddpm seed1 .......................... [RUNNING] 12/100 epochs
  [3/27] ddpm seed2 .......................... [PENDING]

Progress: [██████░░░░░░░░░░░░░░░░░░░░░░] 7% (2/27)
Elapsed: 3h 15m | ETA: 42h 30m
```

---

## Gestion d'Erreurs

### 3 Niveaux d'Erreurs

1. **Immediate Retry** (erreurs transitoires):
   - GPU OOM, network issues
   - 1 retry automatique après 60s

2. **Continue Pipeline** (erreurs modèle-spécifiques):
   - Si un modèle échoue, continuer au suivant
   - Échec loggé dans `errors.log`
   - Résumé final liste succès/échecs

3. **Abort Pipeline** (erreurs critiques):
   - Pas de GPU disponible
   - Données manquantes
   - Espace disque insuffisant

### Resume après Crash

La pipeline sauvegarde son état dans `state.json`. Pour reprendre:

```bash
# Reprendre depuis le dernier modèle en cours
./scripts/pipeline/run_pipeline.sh --dataset epure --resume-from vae

# Ou manuellement identifier le dernier modèle réussi
cat logs/pipeline/latest/state.json | jq '.completed.training'
```

---

## Configurations

### Configs Générés

Les configs pipeline overrident les configs de base:

```yaml
training:
  epochs: 100              # Standardisé (vs 1000 par défaut)
  batch_size: {optimisé}   # Par modèle (voir ci-dessous)
  check_every: 25          # Sauvegarder tous les 25 epochs
  eval_every: 10           # Évaluer tous les 10 epochs

sampling:
  num_samples: 1000        # Pour tous les modes

paths:
  output_dir: outputs/{model}{_toy}/
  samples_dir: samples/{model}{_toy}/
```

### Batch Sizes Optimisés

- **Diffusion** (ddpm, mdm, flow_matching): **64**
- **GAN** (vqvae, wgan_gp): **64**
- **VAE** (vae, gmrf_mvae, meta_vae): **128**
- **Multi-modal** (mmvaeplus): **128**

### Configs Test

Les configs test (`pipeline/test/`) utilisent:
- **Epochs**: 1
- **Samples**: 50
- **Batch sizes**: Identiques aux configs pipeline

---

## Résultats

### Structure des Résultats

**Par run individuel** (`evaluation_results/{dataset}/{model}/run_seed{N}.json`):
```json
{
  "model": "ddpm",
  "dataset": "epure",
  "seed": 0,
  "run_dir": "outputs/ddpm/run_seed0",
  "samples_dir": "samples/ddpm/conditional/run_seed0",
  "metrics": {
    "fid": 23.45,
    "iou_dice": {
      "average": {
        "iou_wd": [0.027, 0.003]
      }
    },
    "rce": {
      "gen_wd": 0.015
    },
    "com": {
      "overall": {
        "wasserstein": 0.042
      }
    }
  }
}
```

**Agrégé** (`evaluation_results/{dataset}/summary.json`):
```json
{
  "dataset": "epure",
  "num_models": 9,
  "model_aggregates": {
    "ddpm": {
      "num_runs": 3,
      "seeds": [0, 1, 2],
      "fid": {
        "mean": 23.45,
        "std": 1.23,
        "min": 22.10,
        "max": 24.80
      },
      "iou_dice_wd": {
        "mean": 0.027,
        "std": 0.005
      }
    }
  },
  "rankings": [
    {
      "model": "flow_matching",
      "fid_mean": 21.30,
      "fid_std": 0.98
    },
    {
      "model": "ddpm",
      "fid_mean": 23.45,
      "fid_std": 1.23
    }
  ]
}
```

### Afficher Résumé

```bash
# Générer et afficher résumé
python scripts/aggregate_results.py --dataset epure

# Voir rankings
cat evaluation_results/epure/summary.json | jq '.rankings'
```

---

## Estimation Temps & Ressources

### Temps par Dataset

| Stage | Durée par Run | Total (27 runs) |
|-------|---------------|-----------------|
| **Training** | ~3h | ~75h |
| **Sampling** | ~1h | ~27h |
| **Evaluation** | ~10min | ~4.5h |
| **TOTAL** | - | **~106h (~4.5 jours)** |

**Pour 2 datasets**: ~212h (~9 jours)

### Espace Disque

- **Checkpoints**: ~500MB par run
- **Samples**: ~1GB par run (1000 images × 3 modes)
- **Logs**: ~50MB par run
- **Total (54 runs)**: ~80GB
- **Recommandé**: **150GB** (avec marge)

### Requirements GPU

- **Minimum**: 8GB VRAM (batch sizes réduits si nécessaire)
- **Recommandé**: 16GB VRAM
- **Optimal**: 24GB VRAM

### Parallélisation (Avancé)

Pour utilisateurs avec **multiple GPUs**:

```bash
# GPU 0: Modèles diffusion
CUDA_VISIBLE_DEVICES=0 ./scripts/pipeline/run_pipeline.sh \
    --dataset epure --models ddpm,mdm,flow_matching &

# GPU 1: Modèles VAE
CUDA_VISIBLE_DEVICES=1 ./scripts/pipeline/run_pipeline.sh \
    --dataset epure --models vae,gmrf_mvae,meta_vae &

# GPU 2: Autres modèles
CUDA_VISIBLE_DEVICES=2 ./scripts/pipeline/run_pipeline.sh \
    --dataset epure --models vqvae,wgan_gp,mmvaeplus &
```

---

## Troubleshooting

### Problème: Configs Manquants

**Erreur**: `Config not found: src/configs/pipeline/epure/ddpm_epure_pipeline.yaml`

**Solution**:
```bash
conda activate hf_diffusion
python scripts/create_pipeline_configs.py
```

### Problème: GPU OOM (Out of Memory)

**Symptômes**: CUDA out of memory errors pendant training

**Solutions**:
1. Réduire batch size dans les configs
2. Utiliser moins de workers (`num_workers: 0` dans config)
3. Libérer VRAM: `nvidia-smi` puis kill autres processus

### Problème: Checkpoint Manquant

**Erreur**: `Checkpoint not found: outputs/ddpm/run_seed0/check/checkpoint_best.pt`

**Causes possibles**:
- Training échoué silencieusement
- Mauvais chemin de symlink

**Solution**:
```bash
# Vérifier logs training
cat logs/pipeline/latest/training/ddpm_seed0.log

# Vérifier symlink
ls -la outputs/ddpm/run_seed0

# Re-créer symlink manuellement si nécessaire
```

### Problème: Samples Directory Not Found

**Erreur**: `Samples not found: samples/ddpm/conditional/run_seed0`

**Solution**:
- Vérifier que sampling a réussi: `cat logs/pipeline/latest/sampling/ddpm_seed0_conditional.log`
- Re-lancer sampling: `./scripts/pipeline/run_pipeline.sh --dataset epure --skip-training --models ddpm`

### Problème: Conda Environnement

**Erreur**: `conda: command not found`

**Solutions**:
```bash
# Option 1: Initialiser conda pour bash
conda init bash
source ~/.bashrc

# Option 2: Utiliser chemin absolu
/path/to/conda/bin/conda activate hf_diffusion

# Option 3: Utiliser manuellement
export PATH="/path/to/conda/bin:$PATH"
```

### Problème: Espace Disque Insuffisant

**Erreur**: `No space left on device`

**Solutions**:
1. Nettoyer runs précédents: `rm -rf outputs/*/20*` (garder symlinks)
2. Nettoyer samples: `rm -rf samples/*/unconditional/*` (sauf derniers runs)
3. Nettoyer logs anciens: `rm -rf logs/pipeline/20*`

---

## Commandes Utiles

### Vérifications Rapides

```bash
# Vérifier tous les checkpoints existent
ls outputs/*/run_seed*/check/checkpoint_best.pt

# Compter runs réussis
ls -d outputs/*/run_seed* | wc -l

# Vérifier espace disque
df -h .

# Vérifier GPU
nvidia-smi

# Vérifier VRAM utilisée
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Nettoyage

```bash
# Supprimer tous les runs (ATTENTION!)
rm -rf outputs/*/run_seed*
rm -rf outputs/*/20*

# Supprimer samples
rm -rf samples/*

# Supprimer résultats évaluation
rm -rf evaluation_results/*

# Supprimer logs
rm -rf logs/pipeline/*
```

### Status Checks

```bash
# Combien de modèles entraînés?
cat logs/pipeline/latest/state.json | jq '.completed.training | length'

# Combien de runs ont échoué?
cat logs/pipeline/latest/state.json | jq '.failed | length'

# Dernière activité
tail -1 logs/pipeline/latest/pipeline.log
```

---

## Limitations Connues

### MDM et WGAN-GP: Pas d'Inpainting

Ces modèles ne supportent pas le mode inpainting. La pipeline les skip automatiquement pour ce mode.

### Symlinks sur Windows

Sur Windows, les symlinks nécessitent des privilèges administrateur. Le script utilise un fallback (copie) si la création de symlink échoue.

### Conda dans Bash

Conda doit être initialisé pour bash. Si `conda activate` ne fonctionne pas, voir section Troubleshooting.

---

## Checklist Avant Exécution Complète

- [ ] Configs générés (27 fichiers dans `src/configs/pipeline/`)
- [ ] Verify script exécuté et passé
- [ ] Test script exécuté sur 1 modèle avec succès
- [ ] Screen/tmux session active (pour runs longs)
- [ ] Espace disque >150GB disponible
- [ ] GPU disponible et VRAM suffisante
- [ ] Backup des runs précédents (si important)
- [ ] Monitoring en place (`tail -f logs/...`)

---

## Support

Pour questions ou problèmes:
1. Vérifier logs: `logs/pipeline/latest/errors.log`
2. Vérifier state: `logs/pipeline/latest/state.json`
3. Dry-run: `./scripts/pipeline/run_pipeline.sh --dataset epure --dry-run`
4. Test rapide: `./scripts/pipeline/run_pipeline_test.sh --dataset toy --models ddpm`

---

## Changelog

- **v1.0** (2025-12-21): Création initiale de la pipeline complète
  - Support 9 modèles × 2 datasets × 3 seeds
  - 3 stages: training, sampling, evaluation
  - Resume support, error handling, logging complet
