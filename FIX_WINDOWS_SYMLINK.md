# Fix pour Windows - Symlinks Pipeline

## Problème
Sur Windows, la création de symlinks échoue sans privilèges admin, donc `get_checkpoint_path()` ne trouve pas les checkpoints.

## Solution

Modifier la méthode `get_checkpoint_path()` dans `scripts/pipeline/run_pipeline.py` (ligne 170):

### Remplacer:
```python
def get_checkpoint_path(self, model: str, seed: int) -> Path:
    """Get checkpoint path for model and seed."""
    suffix = "_toy" if self.dataset == "toy" else ""
    checkpoint_dir = Path("outputs") / f"{model}{suffix}" / f"run_seed{seed}" / "check"
    checkpoint_file = checkpoint_dir / "checkpoint_best.pt"

    return checkpoint_file
```

### Par:
```python
def get_checkpoint_path(self, model: str, seed: int) -> Path:
    """Get checkpoint path for model and seed."""
    suffix = "_toy" if self.dataset == "toy" else ""
    output_base = Path("outputs") / f"{model}{suffix}"
    symlink_name = output_base / f"run_seed{seed}"

    # If symlink exists, use it
    if symlink_name.exists():
        checkpoint_file = symlink_name / "check" / "checkpoint_best.pt"
        return checkpoint_file

    # Otherwise, find the corresponding timestamp directory
    # Look for directory created for this seed in state
    if model in self.state.get('completed', {}).get('training', {}):
        if seed in self.state['completed']['training'][model]:
            # Find most recent directory matching this model
            if output_base.exists():
                timestamp_dirs = sorted([
                    d for d in output_base.iterdir()
                    if d.is_dir() and d.name.startswith("20")
                ])
                if timestamp_dirs:
                    # Match by seed index
                    seed_index = self.seeds.index(seed) if seed in self.seeds else 0
                    if seed_index < len(timestamp_dirs):
                        latest_dir = timestamp_dirs[seed_index]
                        checkpoint_file = latest_dir / "check" / "checkpoint_best.pt"
                        return checkpoint_file

    # Fallback: try symlink path anyway
    checkpoint_file = symlink_name / "check" / "checkpoint_best.pt"
    return checkpoint_file
```

## Alternative Plus Simple (Recommandée)

Stocker le mapping seed → directory dans le state lors du training:

### Dans `train_model()`, après ligne 261, ajouter:
```python
# Store the actual directory path for this seed
if 'run_directories' not in self.state:
    self.state['run_directories'] = {}
if model not in self.state['run_directories']:
    self.state['run_directories'][model] = {}
self.state['run_directories'][model][seed] = str(latest_dir)
self.save_state()
```

### Puis dans `get_checkpoint_path()`:
```python
def get_checkpoint_path(self, model: str, seed: int) -> Path:
    """Get checkpoint path for model and seed."""
    suffix = "_toy" if self.dataset == "toy" else ""
    output_base = Path("outputs") / f"{model}{suffix}"
    symlink_name = output_base / f"run_seed{seed}"

    # Try symlink first
    if symlink_name.exists():
        checkpoint_file = symlink_name / "check" / "checkpoint_best.pt"
        return checkpoint_file

    # Try stored path from state
    if 'run_directories' in self.state:
        if model in self.state['run_directories']:
            if seed in self.state['run_directories'][model]:
                run_dir = Path(self.state['run_directories'][model][seed])
                checkpoint_file = run_dir / "check" / "checkpoint_best.pt"
                return checkpoint_file

    # Fallback: return expected symlink path
    checkpoint_file = symlink_name / "check" / "checkpoint_best.pt"
    return checkpoint_file
```

## Application

1. Éditez `scripts/pipeline/run_pipeline.py`
2. Appliquez l'une des deux solutions ci-dessus
3. Relancez le pipeline avec `--skip-training` pour sauter les trainings déjà faits:
   ```bash
   python scripts/pipeline/run_pipeline.py --dataset epure --skip-training
   ```
