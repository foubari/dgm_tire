#!/usr/bin/env python3
"""
Script de validation complet pour tous les modèles EpureDGM.

Valide: Train → Sample → Evaluate pour chaque modèle sur TOY ou EPURE datasets.

Usage:
    # Tous les modèles, dataset TOY
    python scripts/pipeline/validate_models_complete.py

    # Modèles spécifiques
    python scripts/pipeline/validate_models_complete.py --models ddpm,vae,gmrf_mvae

    # Dataset EPURE (plus long)
    python scripts/pipeline/validate_models_complete.py --dataset epure

    # Dry run (vérification configs uniquement)
    python scripts/pipeline/validate_models_complete.py --dry-run

    # Mode verbeux
    python scripts/pipeline/validate_models_complete.py --verbose
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

ALL_MODELS = [
    'ddpm', 'mdm', 'flow_matching',
    'vae', 'gmrf_mvae', 'meta_vae',
    'vqvae', 'wgan_gp', 'mmvaeplus'
]

DATASET_CONFIGS = {
    'toy': {
        'config_dir': 'src/configs/pipeline/test_toy',
        'config_suffix': '_pipeline_test.yaml',
        'epochs': 1,
        'num_samples': 50,
        'output_suffix': '_toy'
    },
    'epure': {
        'config_dir': 'src/configs/pipeline/epure',
        'config_suffix': '_pipeline.yaml',
        'epochs': 10,
        'num_samples': 1000,
        'output_suffix': ''
    },
    'test': {  # EPURE light - validation rapide complète
        'config_dir': 'src/configs/pipeline/test',
        'config_suffix': '_pipeline_test.yaml',
        'epochs': 1,
        'num_samples': 50,
        'output_suffix': ''
    }
}


class ValidationResult:
    """Store validation results for one model."""

    def __init__(self, model: str):
        self.model = model
        self.train_success = False
        self.train_time = 0.0
        self.sample_success = False
        self.sample_time = 0.0
        self.eval_success = False
        self.eval_time = 0.0
        self.error_message = None
        self.checkpoint_path = None
        self.samples_dir = None

    @property
    def success(self) -> bool:
        return self.train_success and self.sample_success and self.eval_success

    @property
    def total_time(self) -> float:
        return self.train_time + self.sample_time + self.eval_time

    def __repr__(self):
        status = "[PASS]" if self.success else "[FAIL]"
        time_str = f"{self.total_time:.1f}s" if self.success else "N/A"
        return f"{status} {self.model:15} ({time_str})"


def check_config_exists(model: str, dataset: str) -> Optional[Path]:
    """Check if config file exists for model and dataset."""
    config_dir = DATASET_CONFIGS[dataset]['config_dir']
    config_suffix = DATASET_CONFIGS[dataset]['config_suffix']
    config_file = Path(f"{config_dir}/{model}{config_suffix}")

    return config_file if config_file.exists() else None


def run_training(model: str, config_file: Path, seed: int, verbose: bool) -> tuple:
    """
    Run training for one epoch.

    Returns:
        (success: bool, duration: float, error_msg: str)
    """
    cmd = [
        sys.executable,
        f"src/models/{model}/train.py",
        "--config", str(config_file),
        "--seed", str(seed)
    ]

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None
    )
    duration = time.time() - start

    if result.returncode != 0:
        error = result.stderr.decode() if result.stderr else "Unknown error"
        return False, duration, error

    return True, duration, None


def find_latest_checkpoint(model: str, dataset: str) -> Optional[Path]:
    """Find the latest checkpoint for a model."""
    output_suffix = DATASET_CONFIGS[dataset]['output_suffix']
    output_base = Path("outputs") / f"{model}{output_suffix}"

    if not output_base.exists():
        return None

    # Find timestamped directories
    checkpoint_dirs = sorted([
        d for d in output_base.iterdir()
        if d.is_dir() and d.name.startswith("20")
    ])

    if not checkpoint_dirs:
        return None

    latest_run = checkpoint_dirs[-1]
    check_dir = latest_run / "check"

    if not check_dir.exists():
        return None

    # Find checkpoint files
    checkpoint_files = list(check_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        return None

    # Return the latest one
    return max(checkpoint_files, key=lambda p: p.stat().st_mtime)


def run_sampling(model: str, checkpoint: Path, config_file: Path, num_samples: int,
                batch_sz: int, verbose: bool) -> tuple:
    """
    Run unconditional sampling.

    Returns:
        (success: bool, duration: float, samples_dir: Path, error_msg: str)
    """
    cmd = [
        sys.executable,
        f"src/models/{model}/sample.py",
        "--checkpoint", str(checkpoint),
        "--config", str(config_file),  # CRITICAL: Pass config for correct paths
        "--mode", "unconditional",
        "--num_samples", str(num_samples),
        "--batch_sz", str(batch_sz)
    ]

    # Only add --seed for models that support it (VAE-based models)
    if model in ['vae', 'gmrf_mvae', 'meta_vae']:
        cmd.extend(["--seed", "42"])

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None
    )
    duration = time.time() - start

    if result.returncode != 0:
        error = result.stderr.decode() if result.stderr else "Unknown error"
        return False, duration, None, error

    # Infer samples directory
    # Format: samples/{model}_{dataset}/unconditional/{timestamp}/
    run_name = checkpoint.parent.parent.name  # Extract timestamp (e.g., "2025-12-25_11-03-20")
    model_dataset = checkpoint.parent.parent.parent.name  # e.g., "vae_toy"
    samples_dir = Path(f"samples/{model_dataset}/unconditional/{run_name}")

    return True, duration, samples_dir, None


def run_evaluation(model: str, dataset: str, run_dir: Path,
                  seed: int, verbose: bool) -> tuple:
    """
    Run evaluation.

    Returns:
        (success: bool, duration: float, error_msg: str)
    """
    # Map 'test' dataset to 'epure' for evaluator (test uses EPURE data)
    eval_dataset = 'epure' if dataset == 'test' else dataset

    cmd = [
        sys.executable,
        "src/scripts/evaluate.py",
        "--model", model,
        "--dataset", eval_dataset,
        "--run", str(run_dir),
        "--split", "test"
    ]

    if verbose:
        print(f"  Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if not verbose else None,
        stderr=subprocess.PIPE if not verbose else None
    )
    duration = time.time() - start

    if result.returncode != 0:
        error = result.stderr.decode() if result.stderr else "Unknown error"
        return False, duration, error

    return True, duration, None


def validate_model(model: str, dataset: str, seed: int = 0,
                  dry_run: bool = False, verbose: bool = False) -> ValidationResult:
    """
    Validate one model: Train → Sample → Evaluate.

    Args:
        model: Model name
        dataset: 'toy' or 'epure'
        seed: Random seed
        dry_run: Only check configs, don't run
        verbose: Print detailed output

    Returns:
        ValidationResult object
    """
    result = ValidationResult(model)

    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] VALIDATING: {model} (dataset={dataset})")
    print(f"{'='*80}\n")

    # Check config exists
    config_file = check_config_exists(model, dataset)
    if config_file is None:
        result.error_message = f"Config not found for {model}/{dataset}"
        print(f"[FAIL] {result.error_message}")
        return result

    print(f"[OK] Config: {config_file}")

    if dry_run:
        print("[DRY RUN] Would execute: Train -> Sample -> Evaluate")
        result.train_success = True
        result.sample_success = True
        result.eval_success = True
        return result

    # Stage 1: Training
    print(f"\n[1/3] Training ({DATASET_CONFIGS[dataset]['epochs']} epoch)...")
    success, duration, error = run_training(model, config_file, seed, verbose)
    result.train_time = duration

    if not success:
        result.error_message = f"Training failed: {error[:200]}"
        print(f"[FAIL] Training failed ({duration:.1f}s)")
        if verbose:
            print(f"  Error: {error}")
        return result

    result.train_success = True
    print(f"[OK] Training completed ({duration:.1f}s)")

    # Find checkpoint
    checkpoint = find_latest_checkpoint(model, dataset)
    if checkpoint is None:
        result.error_message = "Checkpoint not found after training"
        print(f"[FAIL] {result.error_message}")
        return result

    result.checkpoint_path = checkpoint
    print(f"[OK] Checkpoint: {checkpoint}")

    # Stage 2: Sampling
    num_samples = DATASET_CONFIGS[dataset]['num_samples']
    print(f"\n[2/3] Sampling ({num_samples} samples, unconditional)...")
    success, duration, samples_dir, error = run_sampling(
        model, checkpoint, config_file, num_samples, batch_sz=16, verbose=verbose
    )
    result.sample_time = duration

    if not success:
        result.error_message = f"Sampling failed: {error[:200]}"
        print(f"[FAIL] Sampling failed ({duration:.1f}s)")
        if verbose:
            print(f"  Error: {error}")
        return result

    result.sample_success = True
    result.samples_dir = samples_dir
    print(f"[OK] Sampling completed ({duration:.1f}s)")
    print(f"[OK] Samples: {samples_dir}")

    # Copy samples to conditional directory for evaluation
    # evaluate.py expects samples in conditional/{timestamp}/
    output_suffix = DATASET_CONFIGS[dataset]['output_suffix']
    samples_base = Path("samples") / f"{model}{output_suffix}"
    timestamp_dir = checkpoint.parent.parent.name  # Get timestamp from checkpoint path
    src_samples = samples_base / "unconditional" / timestamp_dir
    dst_samples = samples_base / "conditional" / timestamp_dir

    if src_samples.exists():
        import shutil
        dst_samples.parent.mkdir(parents=True, exist_ok=True)

        # Remove old destination if exists
        if dst_samples.exists():
            shutil.rmtree(dst_samples)

        # Copy samples (preserve structure)
        shutil.copytree(src_samples, dst_samples)
        print(f"[OK] Copied samples to conditional/{timestamp_dir}")
    else:
        print(f"[WARN] Samples not found at {src_samples}")

    # Stage 3: Evaluation
    print(f"\n[3/3] Evaluating metrics...")
    run_dir = checkpoint.parent.parent  # Get run directory (timestamped folder)
    success, duration, error = run_evaluation(
        model, dataset, run_dir, seed, verbose
    )
    result.eval_time = duration

    if not success:
        result.error_message = f"Evaluation failed: {error[:200]}"
        print(f"[FAIL] Evaluation failed ({duration:.1f}s)")
        if verbose:
            print(f"  Error: {error}")
        return result

    result.eval_success = True
    print(f"[OK] Evaluation completed ({duration:.1f}s)")

    print(f"\n{'='*80}")
    print(f"[PASS] {model} VALIDATED (total: {result.total_time:.1f}s)")
    print(f"{'='*80}")

    return result


def print_summary(results: List[ValidationResult], dataset: str, elapsed_time: float):
    """Print validation summary."""
    print(f"\n{'='*80}")
    print(" VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Total time: {str(timedelta(seconds=int(elapsed_time)))}")
    print(f"{'='*80}\n")

    passed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Print results
    for r in results:
        status_icon = "[PASS]" if r.success else "[FAIL]"
        time_str = f"{r.total_time:6.1f}s" if r.success else "   N/A"

        stages = []
        if r.train_success:
            stages.append(f"Train:{r.train_time:.0f}s")
        if r.sample_success:
            stages.append(f"Sample:{r.sample_time:.0f}s")
        if r.eval_success:
            stages.append(f"Eval:{r.eval_time:.0f}s")

        stages_str = " | ".join(stages) if stages else "No stages completed"

        print(f"{status_icon} {r.model:15} {time_str}  [{stages_str}]")

        if r.error_message and not r.success:
            print(f"       Error: {r.error_message[:60]}")

    print(f"\n{'='*80}")
    print(f"Passed: {len(passed)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    print(f"{'='*80}\n")

    # Failed details
    if failed:
        print("Failed models:")
        for r in failed:
            print(f"  - {r.model}: {r.error_message}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate all EpureDGM models on TOY or EPURE dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--models',
        type=str,
        help=f'Comma-separated models to validate (default: all). Options: {", ".join(ALL_MODELS)}'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['toy', 'epure', 'test'],
        default='toy',
        help='Dataset: toy (fast), epure (full), test (epure light for complete validation)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed (default: 0)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only check configs exist, don\'t run validation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output during validation'
    )

    args = parser.parse_args()

    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
        invalid = [m for m in models if m not in ALL_MODELS]
        if invalid:
            print(f"ERROR: Invalid models: {', '.join(invalid)}")
            print(f"Valid models: {', '.join(ALL_MODELS)}")
            sys.exit(1)
    else:
        models = ALL_MODELS

    # Print header
    print("="*80)
    print(" EPUREDGM MODEL VALIDATION")
    print("="*80)
    print(f"Models: {len(models)} - {', '.join(models)}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Mode: {'DRY RUN (config check only)' if args.dry_run else 'FULL VALIDATION'}")
    if not args.dry_run:
        config = DATASET_CONFIGS[args.dataset]
        print(f"Training: {config['epochs']} epoch")
        print(f"Sampling: {config['num_samples']} samples per model")
    print("="*80)

    # Run validation
    start_time = time.time()
    results = []

    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Starting validation: {model}")
        result = validate_model(
            model=model,
            dataset=args.dataset,
            seed=args.seed,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        results.append(result)

        # Stop if critical failure and not dry run
        if not result.success and not args.dry_run:
            print(f"\n[WARN] {model} failed, continuing with next model...")

    elapsed_time = time.time() - start_time

    # Print summary
    print_summary(results, args.dataset, elapsed_time)

    # Exit code
    failed = [r for r in results if not r.success]
    sys.exit(len(failed))


if __name__ == '__main__':
    main()
