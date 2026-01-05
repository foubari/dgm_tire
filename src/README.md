# src_new - Modular Diffusion Models

This directory contains a modular implementation of diffusion models (DDPM and MDM) that is independent of specific dataset structures.

## Structure

```
src_new/
├── configs/              # YAML configuration files
├── datasets/            # Modular dataset implementations
│   ├── base.py         # GenericDataset base class
│   ├── ddpm/           # DDPM-specific datasets
│   └── mdm/            # MDM-specific datasets
├── models/              # Model implementations
│   ├── ddpm/           # DDPM model
│   └── mdm/            # MDM model
├── scripts/             # Global dispatcher scripts
└── utils/               # Utility functions
```

## Key Features

- **Modular Datasets**: Generic dataset base class that can be configured for any multi-component dataset
- **YAML Configuration**: All parameters (model, data, training) are specified in YAML files
- **Auto-calculation**: Automatically calculates `channels` from `component_dirs` and `cond_dim` from `condition_columns`
- **Independent**: Completely separate from `src/` - no risk of breaking existing code

## Usage

### Training

#### DDPM
```bash
python src_new/scripts/train.py --model ddpm --config src_new/configs/ddpm_default.yaml
```

#### MDM
```bash
python src_new/scripts/train.py --model mdm --config src_new/configs/mdm_default.yaml
```

#### Override Config Values
```bash
python src_new/scripts/train.py --model ddpm --config src_new/configs/ddpm_default.yaml --epochs 100 --batch_size 32
```

### Sampling

#### DDPM Unconditional
```bash
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/.../checkpoint_100.pt --config src_new/configs/ddpm_default.yaml --mode unconditional --num_samples 100
```

#### DDPM Conditional
```bash
python src_new/scripts/sample.py --model ddpm --checkpoint outputs/ddpm/.../checkpoint_100.pt --config src_new/configs/ddpm_default.yaml --mode conditional --guidance_scale 2.0
```

#### MDM Conditional
```bash
python src_new/scripts/sample.py --model mdm --checkpoint outputs/mdm/.../checkpoint_100.pt --config src_new/configs/mdm_default.yaml --mode conditional --guidance_scale 2.0
```

## Configuration Files

Configuration files are YAML files that specify:
- **model**: Architecture parameters (dim, dim_mults, timesteps, etc.)
- **data**: Dataset parameters (component_dirs, condition_columns, paths, etc.)
- **training**: Training parameters (epochs, batch_size, lr, etc.)
- **paths**: Output directories

See `configs/ddpm_default.yaml`, `configs/mdm_default.yaml`, and `configs/example_new_dataset.yaml` for examples.

## Adding a New Dataset

To use a new dataset:

1. Create a new YAML config file (see `configs/example_new_dataset.yaml`)
2. Specify:
   - `data.component_dirs`: List of component directory names
   - `data.condition_columns`: List of CSV columns to use for conditioning
   - `data.root_dir`: Root directory containing data
   - `data.condition_csv`: Path to CSV file
   - `model.channels`: Will be auto-calculated from `len(component_dirs)` if not specified
   - `model.cond_dim`: Will be auto-calculated from `len(condition_columns)` if not specified

3. Train:
   ```bash
   python src_new/scripts/train.py --model ddpm --config configs/my_new_dataset.yaml
   ```

## Differences from `src/`

- **Modular**: No hardcoded component names or CSV logic
- **Config-driven**: All parameters come from YAML files
- **Auto-calculation**: Automatically calculates model parameters from data configuration
- **Independent**: Completely separate codebase - `src/` remains untouched

