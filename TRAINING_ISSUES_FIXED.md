# Corrections des Problèmes d'Entraînement

## Problèmes Résolus

### 1. ✅ VAE Training Crash (CRITIQUE)

**Problème**: VAE échoue immédiatement (exit code 1) après ~24 secondes

**Cause**: Dimension mismatch dans le config
- `cond_dim: 2` dans le config
- Mais 4 `condition_columns`: `[d_cons_norm, d_rigid_norm, d_life_norm, d_stab_norm]`

**Fix**: Mis à jour `src/configs/pipeline/epure/vae_epure_pipeline.yaml`
```yaml
cond_dim: 4  # Was 2, must match condition_columns count
```

---

### 2. ✅ Checkpoint Discovery pour Évaluation

**Problème**: Pipeline cherche `checkpoint_best.pt` qui n'existe que pour GMRF-MVAE et VAE

**Cause**: DDPM, MDM, Flow Matching, VQ-VAE, MMVAE+ ne sauvegardent QUE `checkpoint_{epoch}.pt`

**Fix**: Mis à jour la logique de découverte des checkpoints avec priorité:
1. **checkpoint_100.pt** (epoch final - pour évaluation)
2. **checkpoint_best.pt** (si existe - GMRF-MVAE, VAE)
3. **Latest checkpoint_{N}.pt** (plus haut numéro d'epoch)

**Fichiers modifiés**:
- `scripts/pipeline/run_pipeline.py` → `get_checkpoint_path()`
- `src/scripts/evaluate.py` → `find_checkpoint_in_dir()` helper

---

### 3. ✅ Fausses Erreurs de Checkpoint Après Training

**Problème**: Logs montrent "ERROR: Checkpoint not found after training" pour tous les modèles sauf GMRF

**Cause**: Le code vérifie `checkpoint_best.pt` qui n'existe pas pour la plupart des modèles

**Fix**: Mis à jour la vérification pour accepter N'IMPORTE QUEL checkpoint dans `check/`
- Ne signale erreur QUE si AUCUN `checkpoint_*.pt` n'existe

---

### 4. ✅ GMRF Logs

**Problème**: "les logs du gmrf ne sont pas dans outputs"

**Clarification**: Les logs SONT capturés par run_pipeline.py dans:
```
logs/pipeline/{pipeline_id}/training/gmrf_mvae_seed{N}.log
```

C'est le comportement attendu. Le GMRF train.py écrit vers stdout/stderr qui est capturé.

---

## Sur l'Autre Machine

### Pull les Corrections

```bash
cd C:\Users\Shadow\Desktop\dmg_tire
git pull origin main
```

### Relancer le Training

Le VAE devrait maintenant s'entraîner correctement:

```bash
python scripts/pipeline/run_pipeline.py --dataset epure
```

**Ce qui a changé**:
- ✅ VAE va s'entraîner (dimension mismatch corrigé)
- ✅ Plus de fausses erreurs "Checkpoint not found" après training
- ✅ Sampling/évaluation utiliseront `checkpoint_100.pt` (epoch final)
- ✅ Si checkpoint_100.pt manque, utilise checkpoint_best.pt ou le plus récent

### Vérifier les Checkpoints Existants

Pour les modèles déjà entraînés (DDPM, MDM, Flow, GMRF), le pipeline trouvera automatiquement:

```
outputs/ddpm/2026-01-05_22-01-42/check/
├── checkpoint_25.pt
├── checkpoint_50.pt
├── checkpoint_75.pt
└── checkpoint_100.pt    ← Utilisé pour évaluation
```

Pour GMRF-MVAE (qui a best):
```
outputs/gmrf_mvae/2026-01-05_23-32-20/check/
├── checkpoint_10.pt
├── checkpoint_20.pt
├── ...
├── checkpoint_100.pt    ← Priorité 1
└── checkpoint_best.pt   ← Priorité 2 si 100.pt manque
```

---

## Récapitulatif des Checkpoints par Modèle

| Modèle | Sauvegarde best.pt? | check_every | Checkpoints typiques |
|--------|---------------------|-------------|---------------------|
| DDPM | Non | 25 | checkpoint_25, 50, 75, 100 |
| MDM | Non | 25 | checkpoint_25, 50, 75, 100 |
| Flow Matching | Non | 25 | checkpoint_25, 50, 75, 100 |
| **VAE** | **Oui** | 25 | best + 25, 50, 75, 100 |
| **GMRF-MVAE** | **Oui** | 25 | best + 25, 50, 75, 100 |
| Meta-VAE | Oui | 50 | best + 50, 100 |
| VQ-VAE | Non | 50 | checkpoint_50, 100 |
| WGAN-GP | Non | 25 | checkpoint_25, 50, 75, 100 |
| MMVAE+ | Non | [50, 100, 150, 250] | Liste spécifique |

---

## État Actuel sur l'Autre Machine

D'après vos logs:

✅ **Entraînés (100 epochs)**:
- DDPM (seeds 0, 1, 2)
- MDM (seeds 0, 1, 2)
- Flow Matching (seeds 0, 1, 2)
- GMRF-MVAE (seeds 0, 1, 2)

❌ **Échoués** (maintenant corrigés):
- VAE (seeds 0, 1, 2) → va marcher avec le fix

⏳ **À entraîner**:
- Meta-VAE
- VQ-VAE
- WGAN-GP
- MMVAE+

---

## Commandes Utiles

### Reprendre Training après Pull

```bash
# Continue le pipeline (skip les modèles déjà entraînés via state.json)
python scripts/pipeline/run_pipeline.py --dataset epure
```

### Sampling/Evaluation Seulement

Si vous voulez juste sample et évaluer les modèles déjà entraînés:

```bash
python scripts/pipeline/run_pipeline.py --dataset epure --skip-training
```

### Évaluer un Modèle Spécifique

```bash
python src/scripts/evaluate.py --model ddpm --dataset epure --seeds 0,1,2
```

Le script trouvera automatiquement les bons checkpoints (checkpoint_100.pt ou best).

---

## Notes Importantes

1. **Évaluation utilise epoch 100**: Pour une comparaison équitable, tous les modèles sont évalués avec checkpoint_100.pt (même nombre d'epochs)

2. **checkpoint_best.pt**: Seuls GMRF-MVAE et VAE ont checkpoint_best.pt, c'est normal

3. **check_every variance**: C'est acceptable que certains modèles sauvegardent tous les 25 epochs et d'autres tous les 50

4. **GMRF seed2**: Le training s'est bien terminé (SUCCESS était correct)

5. **Logs GMRF**: Les logs sont dans `logs/pipeline/{id}/training/gmrf_mvae_seed*.log`, c'est le comportement attendu
