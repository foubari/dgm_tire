# Commandes d'Entraînement - src_new

## Entraînement Rapide (50 timesteps, 1 epoch)

### DDPM
```powershell
python src_new/scripts/train.py --model ddpm --config src_new/configs/ddpm_quick_test.yaml
```

### MDM
```powershell
python src_new/scripts/train.py --model mdm --config src_new/configs/mdm_quick_test.yaml
```

## Entraînement Complet (configs par défaut)

### DDPM
```powershell
python src_new/scripts/train.py --model ddpm --config src_new/configs/ddpm_default.yaml
```

### MDM
```powershell
python src_new/scripts/train.py --model mdm --config src_new/configs/mdm_default.yaml
```

## Override de Paramètres

Vous pouvez surcharger les valeurs de la config directement :

### DDPM avec paramètres personnalisés
```powershell
python src_new/scripts/train.py --model ddpm --config src_new/configs/ddpm_quick_test.yaml --epochs 5 --batch_size 16 --lr 0.0002
```

### MDM avec paramètres personnalisés
```powershell
python src_new/scripts/train.py --model mdm --config src_new/configs/mdm_quick_test.yaml --epochs 5 --batch_size 16 --lr 0.0002
```

## Scripts PowerShell (Alternative)

### DDPM
```powershell
.\src_new\quick_train_ddpm.ps1
```

### MDM
```powershell
.\src_new\quick_train_mdm.ps1
```

## Notes

- Les checkpoints sont sauvegardés dans `outputs/ddpm/[DATE]/check/` ou `outputs/mdm/[DATE]/check/`
- La date `[DATE]` est au format `YYYY-MM-DD_HH-MM-SS`
- Pour Windows, `num_workers=0` est automatiquement utilisé dans les configs de test rapide

