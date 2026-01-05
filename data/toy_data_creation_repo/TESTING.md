# Testing Instructions

## Prerequisites

Make sure you have the required dependencies installed in your `hf_diffusion` environment:

```bash
# Activate your environment
conda activate hf_diffusion  # or: source hf_diffusion/bin/activate

# Install dependencies (if not already installed)
pip install numpy matplotlib scipy PyYAML pandas
```

## Quick Validation Test

Run the validation script to test core functionality:

```bash
python test_implementation.py
```

Expected output:
```
Testing Tire Deformation Benchmark Implementation...
============================================================

1. Testing imports...
   ✓ All imports successful

2. Creating tire with components...
   ✓ Tire created with 3 components

3. Computing mechanical properties...
   K_vert: 41.77
   Mass: 1101
   Performance: 0.0379
   ✓ Properties computed successfully

4. Applying deformation...
   Deflection δ: 4.8 px
   K_vert: 41.77
   ✓ Deformation applied successfully

5. Testing metrics system...
   Available metrics: ['vertical_stiffness', 'mass', 'performance_ratio']
   K_vert = 41.77
   Mass = 1101
   Performance = 0.0379
   ✓ Metrics system working

6. Testing registries...
   Registered components: ['carcass', 'crown', 'flanks']
   Registered metrics: ['vertical_stiffness', 'mass', 'performance_ratio']
   ✓ Registries working correctly

============================================================
✅ ALL TESTS PASSED!
============================================================
```

## Run Examples

Try the example scripts:

```bash
# Basic usage
python examples/01_basic_usage.py

# Custom metrics
python examples/02_custom_metrics.py

# Dataset generation (small test)
python examples/03_generate_dataset.py

# Model evaluation
python examples/04_evaluate_model.py
```

## Reproduce Notebook Results

The implementation should reproduce the same results as the original notebook. Compare:

```python
# Original notebook values (sample_00000):
# K_vert: 41.77
# Mass: 1101
# Performance: 0.0379
# F=200 → δ≈4.8px
```

## Common Issues

### ModuleNotFoundError
- Ensure you're using the correct Python environment
- Install missing packages: `pip install <package>`

### Import Errors
- Check that you're running from the repository root directory
- Try: `pip install -e .` to install in development mode

### Visualization Issues
- Make sure matplotlib is installed: `pip install matplotlib`
- If running headless, use `matplotlib.use('Agg')` before importing pyplot
