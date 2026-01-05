# Commandes de Sampling - src_new

## Format Général

```powershell
python src_new/scripts/sample.py --model [MODEL] --checkpoint [CHECKPOINT_PATH] --config [CONFIG_PATH] --mode [MODE] [OPTIONS]
```

Remplacez `[DATE]` par la date du checkpoint (format: `YYYY-MM-DD_HH-MM-SS`)

---

## DDPM - 3 Modes de Sampling

### 1. Unconditional
```powershell
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/[DATE]/check/checkpoint_1.pt --config src_new/configs/ddpm_quick_test.yaml --mode unconditional --num_samples 16 --batch_sz 8
```

### 2. Conditional (avec guidance scale)
```powershell
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/[DATE]/check/checkpoint_1.pt --config src_new/configs/ddpm_quick_test.yaml --mode conditional --num_samples 16 --batch_sz 8 --guidance_scale 2.0
```

### 3. Inpainting (préserve certains composants, génère le reste)
```powershell
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/[DATE]/check/checkpoint_1.pt --config src_new/configs/ddpm_quick_test.yaml --mode inpainting --components fpu group_nc group_km
```

---

## MDM - 2 Modes de Sampling

**Note:** MDM ne supporte pas l'inpainting pour l'instant.

### 1. Unconditional
```powershell
python src_new/scripts/sample.py --model mdm --checkpoint outputs/mdm/[DATE]/check/checkpoint_1.pt --config src_new/configs/mdm_quick_test.yaml --mode unconditional --num_samples 16 --batch_sz 8
```

### 2. Conditional (avec guidance scale)
```powershell
python src_new/scripts/sample.py --model mdm --checkpoint outputs/mdm/[DATE]/check/checkpoint_1.pt --config src_new/configs/mdm_quick_test.yaml --mode conditional --num_samples 16 --batch_sz 8 --guidance_scale 2.0
```

---

## Scripts PowerShell (Alternative)

### DDPM Unconditional
```powershell
.\src_new\quick_sample_ddpm.ps1 "2025-01-20_10-30-45" "unconditional" 16
```

### DDPM Conditional
```powershell
.\src_new\quick_sample_ddpm.ps1 "2025-01-20_10-30-45" "conditional" 16
```

### DDPM Inpainting
```powershell
.\src_new\quick_sample_ddpm.ps1 "2025-01-20_10-30-45" "inpainting"
```

### MDM Unconditional
```powershell
.\src_new\quick_sample_mdm.ps1 "2025-01-20_10-30-45" "unconditional" 16
```

### MDM Conditional
```powershell
.\src_new\quick_sample_mdm.ps1 "2025-01-20_10-30-45" "conditional" 16
```

---

## Exemple Complet - Workflow

```powershell
# 1. Entraîner DDPM rapidement
python src_new/scripts/train.py --model ddpm --config src_new/configs/ddpm_quick_test.yaml --epochs 5 --batch_size 16

# 2. Trouver la date du checkpoint (dans outputs/ddpm/)
# Exemple: 2025-01-20_14-30-45

# 3. Sampler unconditional
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/2025-01-20_14-30-45/check/checkpoint_5.pt --config src_new/configs/ddpm_quick_test.yaml --mode unconditional --num_samples 16

# 4. Sampler conditional
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/2025-01-20_14-30-45/check/checkpoint_5.pt --config src_new/configs/ddpm_quick_test.yaml --mode conditional --num_samples 16 --guidance_scale 2.0

# 5. Sampler inpainting
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/2025-01-20_14-30-45/check/checkpoint_5.pt --config src_new/configs/ddpm_quick_test.yaml --mode inpainting --components fpu group_nc group_km
```

---

## Où sont sauvegardés les échantillons?

- **DDPM Unconditional**: `samples/ddpm/unconditional/[DATE]/`
- **DDPM Conditional**: `samples/ddpm/conditional/[DATE]/`
- **DDPM Inpainting**: `samples/ddpm/inpainting/[DATE]/[COMPONENT]/`
- **MDM Unconditional**: `samples/mdm/unconditional/[DATE]/`
- **MDM Conditional**: `samples/mdm/conditional/[DATE]/`

