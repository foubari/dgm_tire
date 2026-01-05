# Performance Metrics Implementation

This implementation adds performance metrics calculation to the EpureDGM project.

## Overview

The system now calculates performance metrics from binary mask images and stores them in a new `performances.csv` file. All dataloaders have been updated to use this new CSV with backward compatibility.

## Files Created/Modified

### Created Files

1. **`scripts/add_perfs_to_csv.py`** - Main script to calculate performance metrics
2. **`scripts/requirements_perf_script.txt`** - Python dependencies for the script
3. **`scripts/README_PERFORMANCES.md`** - This documentation

### Modified Files

1. **`src/datasets/dataset_epure_conditional.py`** - DDPM dataloader with fallback logic
2. **`src/datasets/dataset_dimensions.py`** - DDPM dataloader with fallback logic
3. **`src/multinomial_diffusion/segmentation_diffusion/epures/create_masks_from_csv.py`** - MDM preprocessing script
4. **`src/multinomial_diffusion/segmentation_diffusion/epures/epures_fast_classifier_free.py`** - MDM dataloader

## Installation

Install the required dependencies:

```bash
pip install -r scripts/requirements_perf_script.txt
```

Or install individually:
```bash
pip install numpy pandas Pillow scikit-image scipy
```

## Usage

### Step 1: Generate Performance CSV

Run the script to calculate performance metrics and create `performances.csv`:

```bash
python scripts/add_perfs_to_csv.py \
  --csv_in data/epure/dimensions.csv \
  --csv_out data/epure/performances.csv \
  --images_root data/epure \
  --baseline_prefix binc_3b3_709014_6
```

**Expected output:**
- Creates `data/epure/performances.csv` with 16 new columns
- Baseline sample (`binc_3b3_709014_6`) will have all delta columns = 0.0
- Same number of rows as input (1126 rows)

### Step 2: Regenerate MDM Preprocessed Files (Optional)

If you're using the Multinomial Diffusion Model (MDM), regenerate the preprocessed CSVs:

```bash
cd src/multinomial_diffusion/segmentation_diffusion/epures
python create_masks_from_csv.py
```

**This creates:**
- `data/epure/preprocessed/train_performances.csv`
- `data/epure/preprocessed/test_performances.csv`
- `data/epure/preprocessed/val_performances.csv`
- Numpy mask files remain unchanged

### Step 3: Use Updated Dataloaders

The dataloaders now automatically use `performances.csv` if it exists, otherwise fall back to `dimensions.csv` with a warning.

**For DDPM models:**
```python
from src.datasets.dataset_epure_conditional import DataLoaderFactory

# Will automatically use performances.csv
factory = DataLoaderFactory(
    root_dir="data/epure",
    conditional=True,
    # Default condition_csv is now "data/epure/performances.csv"
)
train_loader, test_loader = factory.get_loaders()
```

**For MDM models:**
```python
# The dataloader will automatically find and use performance CSVs
# Priority:
# 1. train_performances.csv (if exists)
# 2. train_conditions.csv (backward compatibility)
# 3. perfs_cond.csv (shared)
# 4. dims_cond.csv (fallback)
```

## Performance Metrics Explanation

The script calculates 16 new columns:

### Raw Features (4 columns)
- `m_top`: Mass/area of top component (group_km) × rho_top
- `m_side`: Mass/area of side component (fpu) × rho_side
- `m_total`: Mass/area of full assembly × rho_total
- `round_top`: Roundness of top arc (sagitta/chord ratio)

### Delta Features (4 columns, relative to baseline)
- `dm_top`: Relative change in top mass vs baseline
- `dm_side`: Relative change in side mass vs baseline
- `dm_total`: Relative change in total mass vs baseline
- `dround_top`: Relative change in roundness vs baseline

### Performance Metrics (4 columns, derived)
- `d_cons`: Consumption metric = alpha_cons × dm_total (default: 1.0 × dm_total)
- `d_rigid`: Rigidity metric = alpha_rigid × (w_top×dm_top + w_side×dm_side) (default: 0.5 × (0.6×dm_top + 0.4×dm_side))
- `d_life`: Lifetime metric = alpha_life × dm_top (default: 0.8 × dm_top)
- `d_stab`: Stability metric = alpha_stab × dround_top (default: -1.0 × dround_top, negative because rounder = less stable)

## Advanced Usage

### Custom Coefficients

You can customize the performance coefficients:

```bash
python scripts/add_perfs_to_csv.py \
  --csv_in data/epure/dimensions.csv \
  --csv_out data/epure/performances.csv \
  --images_root data/epure \
  --baseline_prefix binc_3b3_709014_6 \
  --alpha_cons 1.2 \
  --alpha_rigid 0.6 \
  --w_top 0.7 \
  --w_side 0.3 \
  --alpha_life 0.9 \
  --alpha_stab -1.5
```

### Custom Density Coefficients

```bash
python scripts/add_perfs_to_csv.py \
  --rho_top 1.5 \
  --rho_side 1.2 \
  --rho_total 1.0
```

### Roundness Calculation Parameters

```bash
python scripts/add_perfs_to_csv.py \
  --roundness_source full \
  --top_frac 0.25 \
  --mirror_full 0  # 0 = no mirroring (recommended for half-epure)
```

### Debug Mode

Save visualization for a specific sample:

```bash
python scripts/add_perfs_to_csv.py \
  --debug_prefix binc_3b3_1008010_10
```

## Validation

After running the script, verify the output:

```python
import pandas as pd

df = pd.read_csv('data/epure/performances.csv')

# Check column count (original 8 + new 16 = 24 columns)
print(f"Total columns: {len(df.columns)}")

# Check baseline has deltas = 0
baseline = df[df['matching'] == 'binc_3b3_709014_6']
print(baseline[['dm_top', 'dm_side', 'dm_total', 'dround_top']])
# Should show: 0.0, 0.0, 0.0, 0.0

# Check statistics
print(df[['d_cons', 'd_rigid', 'd_life', 'd_stab']].describe())
```

## Backward Compatibility

All changes maintain backward compatibility:

1. **If `performances.csv` doesn't exist:**
   - Dataloaders will use `dimensions.csv` with a warning
   - Training/evaluation continues normally

2. **Existing code:**
   - No changes required to existing training scripts
   - Dataloaders automatically detect and use the best available CSV

3. **Preprocessing:**
   - Old `train_conditions.csv` still works
   - New `train_performances.csv` preferred when available

## Troubleshooting

### Script fails with "Baseline images missing"
- Check that baseline `binc_3b3_709014_6` exists in `data/epure/train/`
- Verify images: `data/epure/train/fpu/mdbk_binc_3b3_709014_6_fpu.png`
- Check all components: fpu, group_km, full

### Dataloader warnings about missing performances.csv
- Run Step 1 to generate the performances CSV
- If using MDM, also run Step 2 to regenerate preprocessed CSVs

### Mismatch between CSV and numpy masks
- This happens if you regenerated performances.csv but not the preprocessed CSVs
- Solution: Run `create_masks_from_csv.py` again (Step 2)

### NaN values in performance columns
- Normal for samples with missing images
- Check warnings during script execution for which samples have missing images

## Contact

For issues or questions, check:
- [C:\Users\fouad\.claude\plans\lucky-orbiting-catmull.md](file:///C:/Users/fouad/.claude/plans/lucky-orbiting-catmull.md) - Full implementation plan
- Script help: `python scripts/add_perfs_to_csv.py --help`
