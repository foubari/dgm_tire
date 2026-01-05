# Fix pour Pipeline sur Windows

## Problème Rencontré

Sur Windows sans privilèges admin, les symlinks échouent. Le pipeline a créé `outputs\ddpm\2026-01-05_20-53-09` mais cherche `outputs\ddpm\run_seed0\check\checkpoint_best.pt`.

## Solution Appliquée

Le fichier `scripts/pipeline/run_pipeline.py` a été corrigé pour:
1. Stocker les chemins réels des répertoires dans `state.json`
2. Utiliser ces chemins quand les symlinks n'existent pas

## Étapes pour Reprendre le Pipeline

### 1. Transférer le Code Corrigé

Copiez ces fichiers vers l'autre machine:
- `scripts/pipeline/run_pipeline.py` (corrigé)
- `scripts/pipeline/fix_state_windows.py` (nouveau)

### 2. Réparer le State Existant

Sur l'autre machine, lancez le script de réparation:

```bash
cd C:\Users\Shadow\Desktop\dmg_tire
python scripts/pipeline/fix_state_windows.py --dataset epure
```

Cela va:
- Trouver le dernier fichier `state.json`
- Mapper les répertoires timestamp existants aux seeds
- Sauvegarder une backup du state original
- Créer un state.json corrigé

### 3. Reprendre le Pipeline

Maintenant vous pouvez reprendre le pipeline en skippant les trainings déjà faits:

```bash
python scripts/pipeline/run_pipeline.py --dataset epure --skip-training
```

Ou si vous voulez continuer les trainings non terminés:

```bash
python scripts/pipeline/run_pipeline.py --dataset epure
```

Le script va:
- ✓ Détecter que ddpm seed0 est déjà fait
- ✓ Continuer avec ddpm seed1, seed2, etc.
- ✓ Stocker correctement les chemins pour les futurs trainings

### 4. Alternative: Reprendre Manuellement

Si vous voulez contrôler quels modèles entraîner:

```bash
# Seulement les modèles pas encore entraînés
python scripts/pipeline/run_pipeline.py --dataset epure --models mdm,flow_matching,vae,gmrf_mvae,meta_vae,vqvae,wgan_gp,mmvaeplus

# Ou continuer ddpm seeds manquants puis autres modèles
python scripts/pipeline/run_pipeline.py --dataset epure --models ddpm --seeds 1,2
python scripts/pipeline/run_pipeline.py --dataset epure --models mdm,flow_matching,vae
```

## Vérification

Vérifiez que le checkpoint ddpm seed0 est trouvable:

```bash
python -c "from pathlib import Path; import json; state = json.load(open('logs/pipeline/2026-01-05_20-53-04/state.json')); print(state.get('run_directories', {}).get('ddpm', {}))"
```

Devrait afficher quelque chose comme:
```
{0: 'outputs\\ddpm\\2026-01-05_20-53-09'}
```

## Notes

- Le fix est compatible Unix/Linux (utilise toujours symlinks si disponibles)
- Les futurs trainings stockeront automatiquement leurs chemins
- Le state.json contient maintenant la section `run_directories`
