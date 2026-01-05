#!/usr/bin/env python3
"""
Vérification pré-vol pour la pipeline EpureDGM.

Vérifie:
- Configs existent
- Données disponibles
- GPU disponible
- Espace disque suffisant
- Dépendances Python

Usage:
    python scripts/pipeline/verify_pipeline.py --dataset epure
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# All models
ALL_MODELS = [
    'ddpm', 'mdm', 'flow_matching',
    'vae', 'gmrf_mvae', 'meta_vae',
    'vqvae', 'wgan_gp', 'mmvaeplus'
]


def check_configs(dataset: str) -> bool:
    """Check if all pipeline configs exist."""
    print("Checking configuration files...")

    config_dir = Path("src/configs/pipeline") / dataset
    missing = 0

    for model in ALL_MODELS:
        config_file = config_dir / f"{model}_{dataset}_pipeline.yaml"
        if not config_file.exists():
            print(f"  [X] Missing: {config_file}")
            missing += 1

    if missing == 0:
        print(f"  [OK] All configuration files found ({len(ALL_MODELS)} configs)")
        return True
    else:
        print(f"  [X] {missing} configuration files missing")
        print("    Run: python scripts/create_pipeline_configs.py")
        return False


def check_data(dataset: str) -> bool:
    """Check if data directories exist."""
    print("\nChecking data directories...")

    if dataset == "epure":
        data_root = Path("data/epure")
    else:
        data_root = Path("data/toy_epure")

    errors = []

    # Check train/test directories
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.exists():
        errors.append(f"Training data not found: {train_dir}")

    if not test_dir.exists():
        errors.append(f"Test data not found: {test_dir}")

    # Check performances CSV
    csv_paths = [
        data_root / "performances.csv",
        data_root.parent / "performances.csv"
    ]

    csv_found = any(p.exists() for p in csv_paths)
    if not csv_found:
        errors.append("Performances CSV not found")

    if errors:
        for err in errors:
            print(f"  [X] {err}")
        return False
    else:
        print(f"  [OK] Data directories found ({data_root})")
        return True


def check_gpu() -> bool:
    """Check if GPU is available."""
    print("\nChecking GPU...")

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = result.stdout.strip().split('\n')
        print(f"  [OK] Found {len(gpus)} GPU(s)")

        for i, gpu in enumerate(gpus):
            name, memory = gpu.split(',')
            memory_mb = int(memory.strip().split()[0])
            print(f"    GPU {i}: {name.strip()} ({memory_mb} MB)")

            if memory_mb < 8000:
                print(f"    [!] Low VRAM (<8GB), consider reducing batch sizes")

        return True

    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  [X] nvidia-smi not found (no GPU detected)")
        return False


def check_disk_space() -> bool:
    """Check available disk space."""
    print("\nChecking disk space...")

    usage = shutil.disk_usage(".")
    available_gb = usage.free // (1024 ** 3)

    if available_gb > 150:
        print(f"  [OK] Sufficient disk space: {available_gb}GB available")
        return True
    elif available_gb > 80:
        print(f"  [!] Limited disk space: {available_gb}GB available (recommended: 150GB)")
        return True
    else:
        print(f"  [X] Insufficient disk space: {available_gb}GB available (need: 150GB)")
        return False


def check_python_deps() -> bool:
    """Check Python dependencies."""
    print("\nChecking Python dependencies...")

    required_packages = ['torch', 'yaml', 'PIL', 'numpy', 'pandas']
    missing = []

    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  [X] Missing packages: {', '.join(missing)}")
        return False
    else:
        print(f"  [OK] All required packages available")

        # Show Python version
        import sys
        print(f"    Python: {sys.version.split()[0]}")

        # Show PyTorch version and CUDA
        try:
            import torch
            print(f"    PyTorch: {torch.__version__}")
            if torch.cuda.is_available():
                print(f"    CUDA: {torch.version.cuda} (available)")
            else:
                print(f"    CUDA: Not available (CPU only)")
        except:
            pass

        return True


def estimate_time(dataset: str):
    """Estimate pipeline runtime."""
    print("\nEstimating runtime...")

    num_models = len(ALL_MODELS)
    num_seeds = 3

    # Rough estimates (hours per run)
    avg_train_hours = 3
    avg_sample_hours = 1
    avg_eval_hours = 0.2

    total_runs = num_models * num_seeds
    total_hours = total_runs * (avg_train_hours + avg_sample_hours + avg_eval_hours)

    print(f"  Training runs: {total_runs}")
    print(f"  Estimated total time: ~{total_hours:.0f} hours (~{total_hours/24:.1f} days)")


def main():
    parser = argparse.ArgumentParser(description="Verify EpureDGM pipeline prerequisites")
    parser.add_argument('--dataset', required=True, choices=['epure', 'toy'],
                       help='Dataset name')
    args = parser.parse_args()

    print("=" * 80)
    print(" EPUREDGM PIPELINE VERIFICATION")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print("=" * 80)
    print()

    checks_passed = 0
    checks_failed = 0

    # Run all checks
    if check_configs(args.dataset):
        checks_passed += 1
    else:
        checks_failed += 1

    if check_data(args.dataset):
        checks_passed += 1
    else:
        checks_failed += 1

    if check_gpu():
        checks_passed += 1
    else:
        checks_failed += 1

    if check_disk_space():
        checks_passed += 1
    else:
        checks_failed += 1

    if check_python_deps():
        checks_passed += 1
    else:
        checks_failed += 1

    # Estimate time
    estimate_time(args.dataset)

    # Summary
    print()
    print("=" * 80)
    print(" VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Passed: {checks_passed}")
    print(f"Failed: {checks_failed}")
    print()

    if checks_failed == 0:
        print("[OK] All checks passed! Ready to run pipeline.")
        print()
        print("Next steps:")
        print(f"  1. Run test pipeline (quick validation):")
        print(f"     python scripts/pipeline/run_pipeline_test.py --dataset {args.dataset} --models ddpm --dry-run")
        print()
        print(f"  2. Run full pipeline:")
        print(f"     python scripts/pipeline/run_pipeline.py --dataset {args.dataset}")
        print()
        sys.exit(0)
    else:
        print(f"[X] {checks_failed} check(s) failed. Please fix issues before running pipeline.")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()
