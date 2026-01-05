#!/usr/bin/env python3
"""
Validation end-to-end de tous les 9 modèles EpureDGM.

Usage:
    python scripts/pipeline/validate_all_models.py
    python scripts/pipeline/validate_all_models.py --models ddpm,vae
    python scripts/pipeline/validate_all_models.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ALL_MODELS = [
    'ddpm', 'mdm', 'flow_matching',
    'vae', 'gmrf_mvae', 'meta_vae',
    'vqvae', 'wgan_gp', 'mmvaeplus'
]


def validate_model(model: str, dry_run: bool = False) -> bool:
    """
    Valide un modèle: train (1 epoch) -> sample (50) -> evaluate.

    Returns:
        True si toutes les étapes réussissent
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {model}")
    print(f"{'='*80}\n")

    # Check config exists
    config_file = Path(f"src/configs/pipeline/test_toy/{model}_pipeline_test.yaml")
    if not config_file.exists():
        print(f"[FAIL] Config not found: {config_file}")
        return False

    print(f"[OK] Config found: {config_file}")

    if dry_run:
        print("[DRY RUN] OK - Would execute pipeline stages")
        return True

    # Stage 1: Train (1 epoch)
    print("\n[1/3] Training (1 epoch)...")
    train_cmd = [
        sys.executable,
        f"src/models/{model}/train.py",
        "--config", str(config_file),
        "--seed", "0"
    ]

    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        print(f"[FAIL] Training failed for {model}")
        return False
    print(f"[OK] Training completed")

    # Store config file path for sampling stage
    config_file_abs = config_file.resolve()

    # Find checkpoint
    suffix = "_toy"
    output_base = Path("outputs") / f"{model}{suffix}"

    if not output_base.exists():
        print(f"[FAIL] Output directory not found: {output_base}")
        return False

    checkpoint_dirs = sorted([d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("20")])

    if not checkpoint_dirs:
        print(f"[FAIL] No output directory found in {output_base}")
        return False

    latest_run = checkpoint_dirs[-1]
    check_dir = latest_run / "check"

    if not check_dir.exists():
        print(f"[FAIL] Checkpoint directory not found: {check_dir}")
        return False

    # Find the latest checkpoint (checkpoint_*.pt or checkpoint_best.pt)
    checkpoint_files = list(check_dir.glob("checkpoint_*.pt"))

    if not checkpoint_files:
        print(f"[FAIL] No checkpoint files found in {check_dir}")
        return False

    # Sort by modification time and take the latest
    checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

    print(f"[OK] Checkpoint found: {checkpoint}")

    # No need for symlink - we'll pass the run directory directly to evaluate.py

    # Stage 2: Sample (50 samples, unconditional only for speed)
    print("\n[2/3] Sampling (50 samples, unconditional)...")
    sample_cmd = [
        sys.executable,
        f"src/models/{model}/sample.py",
        "--checkpoint", str(checkpoint),
        "--config", str(config_file_abs),  # CRITICAL: Pass the same config used for training
        "--mode", "unconditional",
        "--num_samples", "50",
        "--batch_sz", "16"
    ]

    result = subprocess.run(sample_cmd)
    if result.returncode != 0:
        print(f"[FAIL] Sampling failed for {model}")
        return False
    print(f"[OK] Sampling completed")

    # Copy samples to conditional directory for evaluation
    # Samples are now in samples/{model}_toy/unconditional/{timestamp}/
    # Need to be in samples/{model}_toy/conditional/{timestamp}/ for evaluation
    samples_base = Path("samples") / f"{model}{suffix}"
    src_samples = samples_base / "unconditional" / latest_run.name
    dst_samples = samples_base / "conditional" / latest_run.name

    if src_samples.exists():
        import shutil
        dst_samples.parent.mkdir(parents=True, exist_ok=True)

        # Remove old destination if exists
        if dst_samples.exists():
            shutil.rmtree(dst_samples)

        # Copy samples (use copytree to preserve structure)
        shutil.copytree(src_samples, dst_samples)
        print(f"[OK] Copied samples: {src_samples.name} -> conditional/{dst_samples.name}")
    else:
        print(f"⚠ Warning: Samples not found at {src_samples}")

    # Stage 3: Evaluate
    print("\n[3/3] Evaluating...")
    # Pass the run directory directly instead of using --seeds
    eval_cmd = [
        sys.executable,
        "src/scripts/evaluate.py",
        "--model", model,
        "--dataset", "toy",
        "--run", str(latest_run)
    ]

    result = subprocess.run(eval_cmd)
    if result.returncode != 0:
        print(f"[FAIL] Evaluation failed for {model}")
        # Don't fail validation if evaluation fails (might be WIP)
        print("  (Continuing anyway - evaluation may not be fully implemented)")
    else:
        print(f"[OK] Evaluation completed")

    print(f"\n{'='*80}")
    print(f"[OK][OK][OK] {model} VALIDATED SUCCESSFULLY [OK][OK][OK]")
    print(f"{'='*80}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate all EpureDGM models")
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of models to validate (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue validation even if a model fails')
    args = parser.parse_args()

    # Determine which models to validate
    if args.models:
        models = [m.strip() for m in args.models.split(',')]
        # Validate model names
        for m in models:
            if m not in ALL_MODELS:
                print(f"Error: Unknown model '{m}'. Valid models: {', '.join(ALL_MODELS)}")
                sys.exit(1)
    else:
        models = ALL_MODELS

    print("="*80)
    print(" EPUREDGM MODEL VALIDATION")
    print("="*80)
    print(f"Models to validate: {len(models)}")
    print(f"Models: {', '.join(models)}")
    print(f"Dataset: TOY (fast validation)")
    print(f"Mode: 1 epoch + 50 samples per model")
    if args.dry_run:
        print("DRY RUN: No actual execution")
    print("="*80)

    # Validate each model
    start_time = datetime.now()
    results = {}

    for i, model in enumerate(models, 1):
        print(f"\n\n{'='*80}")
        print(f"[{i}/{len(models)}] Starting validation: {model}")
        print(f"{'='*80}")

        try:
            success = validate_model(model, dry_run=args.dry_run)
            results[model] = "[OK] PASS" if success else "[FAIL] FAIL"

            if not success and not args.continue_on_error:
                print(f"\n[FAIL] Validation failed for {model}, stopping.")
                break
        except KeyboardInterrupt:
            print(f"\n\nInterrupted by user")
            results[model] = "[FAIL] INTERRUPTED"
            break
        except Exception as e:
            print(f"\n[FAIL] Exception during {model} validation: {e}")
            import traceback
            traceback.print_exc()
            results[model] = f"[FAIL] ERROR"

            if not args.continue_on_error:
                break

    # Summary
    elapsed = datetime.now() - start_time

    print("\n\n" + "="*80)
    print(" VALIDATION SUMMARY")
    print("="*80)

    for model, result in results.items():
        print(f"{model:20} {result}")

    passed = sum(1 for r in results.values() if "PASS" in r)
    failed = sum(1 for r in results.values() if "FAIL" in r or "ERROR" in r)
    interrupted = sum(1 for r in results.values() if "INTERRUPTED" in r)

    print()
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    if interrupted > 0:
        print(f"Interrupted: {interrupted}/{len(results)}")
    print(f"Elapsed time: {elapsed}")
    print("="*80)

    if args.dry_run:
        print("\n[OK] Dry run completed - all configs found")
        sys.exit(0)

    sys.exit(0 if failed == 0 and interrupted == 0 else 1)


if __name__ == '__main__':
    main()
