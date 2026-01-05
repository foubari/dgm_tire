# Scripts de Validation EpureDGM

## 📋 Vue d'ensemble

Trois scripts de validation disponibles:

| Script | Usage | Dataset | Rapidité |
|--------|-------|---------|----------|
| `validate_all_models.py` | Validation rapide (TOY uniquement) | TOY | ⚡ Rapide |
| `validate_models_complete.py` | Validation complète (TOY + EPURE) | TOY/EPURE | 🔧 Flexible |
| `run_pipeline.py` | Pipeline complet multi-seeds | TOY/EPURE | 🐢 Long |

---

## 🚀 Script Recommandé: `validate_models_complete.py`

### Fonctionnalités

✅ **Tous les modèles** (9): ddpm, mdm, flow_matching, vae, gmrf_mvae, meta_vae, vqvae, wgan_gp, mmvaeplus
✅ **Datasets**: TOY (rapide) ou EPURE (complet)
✅ **3 phases**: Train → Sample → Evaluate
✅ **Sélection flexible**: Tous ou sous-ensemble de modèles
✅ **Mode dry-run**: Vérification configs sans exécution
✅ **Résumé détaillé**: Temps par étape, statuts, erreurs

### Usage

```bash
# 1. Validation rapide (TOY, tous les modèles)
python scripts/pipeline/validate_models_complete.py

# 2. Modèles spécifiques
python scripts/pipeline/validate_models_complete.py --models ddpm,vae,gmrf_mvae

# 3. Dataset EPURE (plus long, ~30min par modèle)
python scripts/pipeline/validate_models_complete.py --dataset epure

# 4. Dry run (vérification configs uniquement)
python scripts/pipeline/validate_models_complete.py --dry-run

# 5. Mode verbeux (debug)
python scripts/pipeline/validate_models_complete.py --verbose

# 6. Seed personnalisé
python scripts/pipeline/validate_models_complete.py --seed 42
```

### Exemple de Sortie

```
================================================================================
 EPUREDGM MODEL VALIDATION
================================================================================
Models: 9 - ddpm, mdm, flow_matching, vae, gmrf_mvae, meta_vae, vqvae, wgan_gp, mmvaeplus
Dataset: TOY
Mode: FULL VALIDATION
Training: 1 epoch
Sampling: 50 samples per model
================================================================================

[1/9] Starting validation: ddpm
================================================================================
[12:34:56] VALIDATING: ddpm (dataset=toy)
================================================================================

[OK] Config: src/configs/pipeline/test_toy/ddpm_pipeline_test.yaml

[1/3] Training (1 epoch)...
[OK] Training completed (45.2s)
[OK] Checkpoint: outputs/ddpm_toy/2025-12-22_12-34-56/check/checkpoint_epoch0.pt

[2/3] Sampling (50 samples, unconditional)...
[OK] Sampling completed (12.3s)
[OK] Samples: samples/ddpm_toy/unconditional/2025-12-22_12-34-56/

[3/3] Evaluating metrics...
[OK] Evaluation completed (8.9s)

================================================================================
[PASS] ddpm VALIDATED (total: 66.4s)
================================================================================

...

================================================================================
 VALIDATION SUMMARY
================================================================================
Dataset: TOY
Total time: 0:12:34
================================================================================

[PASS] ddpm             66.4s  [Train:45s | Sample:12s | Eval:9s]
[PASS] mdm              52.1s  [Train:38s | Sample:8s | Eval:6s]
[PASS] flow_matching    71.8s  [Train:51s | Sample:13s | Eval:8s]
[PASS] vae              43.2s  [Train:28s | Sample:9s | Eval:6s]
[PASS] gmrf_mvae        58.9s  [Train:41s | Sample:11s | Eval:7s]
[PASS] meta_vae         61.3s  [Train:43s | Sample:12s | Eval:6s]
[PASS] vqvae            79.5s  [Train:58s | Sample:14s | Eval:8s]
[PASS] wgan_gp          38.7s  [Train:25s | Sample:8s | Eval:6s]
[PASS] mmvaeplus        67.2s  [Train:47s | Sample:13s | Eval:7s]

================================================================================
Passed: 9/9
Failed: 0/9
================================================================================
```

---

## 📊 Métriques Évaluées

Toutes les métriques sont calculées pour **TOUS les modes de sampling** (unconditional, conditional, inpainting):

| Métrique | Description | MDM | Autres Modèles |
|----------|-------------|-----|----------------|
| **FID** | Fréchet Inception Distance | ✅ | ✅ |
| **IoU/Dice** | Overlap entre composants | ❌ Skipped | ✅ |
| **CoM** | Center of Mass (localisation) | ✅ | ✅ |
| **RCE** | Relative Count Error | ✅ | ✅ |

**Note**: IoU/Dice est désactivé pour MDM (modèle de segmentation, pas multi-composants).

---

## 🔧 Configuration par Dataset

### TOY Dataset (Rapide)
- **Epochs**: 1
- **Samples**: 50
- **Temps moyen**: ~1 minute par modèle
- **Total (9 modèles)**: ~10-15 minutes

### EPURE Dataset (Complet)
- **Epochs**: 1
- **Samples**: 100
- **Temps moyen**: ~5-30 minutes par modèle (selon GPU)
- **Total (9 modèles)**: ~2-4 heures

---

## 🐛 Corrections Appliquées

### VAE (9 bugs corrigés)
1. ✅ Import `save_checkpoint` manquant
2. ✅ Argument `--seed` manquant
3. ✅ Unicode encoding (Windows)
4. ✅ Dataset constructor incompatible
5. ✅ Argument `type` inattendu
6. ✅ Dimension conditionnement incorrecte (`cond_dim: 2` → `4`)
7. ✅ Structure dossier output incorrecte
8. ✅ Sampling unconditional échoue (decoder forward signature)
9. ✅ **Double sigmoid** (output squashing) - **CRITIQUE**

### MDM
- ✅ IoU/Dice désactivé (modèle de segmentation)

### Architecture Globale
- ✅ Sweep complet: Pas de double sigmoid dans autres modèles VAE (GMRF_MVAE, Meta-VAE, MMVAE+)

Voir [BUGFIXES_VAE.md](../../BUGFIXES_VAE.md) pour détails complets.

---

## 📁 Structure des Outputs

```
EpureDGM/
├── outputs/
│   ├── {model}_toy/
│   │   └── 2025-12-22_12-34-56/
│   │       ├── check/
│   │       │   ├── checkpoint_best.pt
│   │       │   └── checkpoint_epoch0.pt
│   │       └── config.yaml (copié automatiquement)
│   └── {model}/  (pour EPURE, sans suffix _toy)
│
├── samples/
│   ├── {model}_toy/
│   │   └── unconditional/
│   │       └── 2025-12-22_12-34-56/
│   │           ├── full/
│   │           ├── group_nc/
│   │           ├── group_km/
│   │           └── fpu/
│   └── {model}/  (pour EPURE)
│
├── evaluation_results/
│   ├── toy/
│   │   └── {model}/
│   │       └── 2025-12-22_12-34-56.json  (résultats métriques)
│   └── epure/
│
└── evaluation_cache/  (partagé entre modèles)
    ├── toy/
    │   ├── real/  (données réelles, partagées)
    │   │   ├── fid_features.npz
    │   │   ├── iou_dice_distributions.pkl
    │   │   ├── com_positions.pkl
    │   │   └── rce_counts.pkl
    │   └── models/  (données générées, par modèle)
    │       ├── ddpm/
    │       ├── vae/
    │       └── ...
    └── epure/
```

---

## 🔍 Troubleshooting

### Erreur: "Config not found"
```bash
# Vérifier que les configs test existent
ls src/configs/pipeline/test_toy/
ls src/configs/pipeline/test/

# Régénérer si nécessaire
python scripts/create_pipeline_configs.py
```

### Erreur: "Checkpoint not found"
Le training a probablement échoué. Relancer avec `--verbose`:
```bash
python scripts/pipeline/validate_models_complete.py --models {model} --verbose
```

### Erreur: "CUDA out of memory"
Réduire le batch size dans les configs:
```yaml
# src/configs/pipeline/test_toy/{model}_pipeline_test.yaml
training:
  batch_size: 64  # Réduire à 32 ou 16
```

### Erreur: IoU/Dice pour MDM
**Normal** - IoU/Dice est désactivé pour MDM. Le message suivant apparaîtra:
```
[INFO] Skipping IoU/Dice for MDM (segmentation model)
```

---

## 📞 Support

Pour les bugs ou questions:
1. Vérifier [BUGFIXES_VAE.md](../../BUGFIXES_VAE.md)
2. Relancer avec `--verbose` pour logs détaillés
3. Vérifier les logs dans les fichiers de sortie des commandes

---

## 🎯 Prochaines Étapes

Après validation réussie:

1. **Training complet** (dataset EPURE, multiple epochs):
   ```bash
   python src/models/{model}/train.py --config src/configs/pipeline/epure/{model}_pipeline.yaml
   ```

2. **Sampling complet** (3 modes):
   ```bash
   # Unconditional
   python src/models/{model}/sample.py --checkpoint {path} --mode unconditional --num_samples 1000

   # Conditional
   python src/models/{model}/sample.py --checkpoint {path} --mode conditional --num_samples 1000

   # Inpainting (si supporté)
   python src/models/{model}/sample.py --checkpoint {path} --mode inpainting --components group_nc
   ```

3. **Évaluation finale**:
   ```bash
   python src/scripts/evaluate.py --model {model} --dataset epure --run {run_dir} --split test
   ```
