#!/usr/bin/env python3
"""
Script d'évaluation pour EpureDGM.

Supports both conditional and inpainting evaluation modes.

Usage:
    # Evaluer un run spécifique (both modes by default)
    python src/scripts/evaluate.py --model ddpm --dataset epure --run outputs/ddpm/run_seed0

    # Evaluer tous les runs d'un modèle
    python src/scripts/evaluate.py --model ddpm --dataset epure --all-runs

    # Evaluer plusieurs seeds
    python src/scripts/evaluate.py --model ddpm --dataset epure --seeds 0,1,2

    # Evaluer uniquement le mode conditional
    python src/scripts/evaluate.py --model ddpm --dataset epure --all-runs --mode conditional

    # Evaluer uniquement le mode inpainting
    python src/scripts/evaluate.py --model ddpm --dataset epure --all-runs --mode inpainting

Output structure:
    evaluation_results/{dataset}/
    ├── {model}_conditional_all_runs.json
    ├── {model}_inpainting_all_runs.json
    └── {model}/
        ├── conditional/{run_name}.json
        └── inpainting/{run_name}.json
"""

import os
# Fix OMP conflict between Intel MKL and LLVM OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np

# Add src_new to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import ModelEvaluator


def find_all_runs(model_name: str, dataset_name: str) -> list:
    """Trouve tous les runs pour un modèle/dataset."""
    if dataset_name == "toy":
        base_dir = Path("outputs") / f"{model_name}_toy"
    else:
        base_dir = Path("outputs") / model_name

    if not base_dir.exists():
        return []

    # Chercher dossiers de runs (skip check, samples, etc.)
    runs = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name not in ['check', 'samples', 'logs', 'fid_cache']
    ]
    return sorted(runs)


def find_samples_dir(model_name: str, dataset_name: str, run_name: str, mode: str = "conditional") -> Path:
    """Trouve le dossier samples pour un run donné.

    Args:
        model_name: Model type
        dataset_name: Dataset name
        run_name: Run directory name
        mode: 'conditional' or 'inpainting'

    Returns:
        Path to samples directory
    """
    if dataset_name == "toy":
        samples_base = Path("samples") / f"{model_name}_toy"
    else:
        samples_base = Path("samples") / model_name

    # Try different locations based on mode
    candidates = [
        samples_base / mode / run_name,
        samples_base / run_name / mode,
        samples_base / run_name,
    ]

    for cand in candidates:
        if cand.exists():
            # For conditional, check if 'full' exists
            if mode == "conditional" and (cand / "full").exists():
                return cand
            # For inpainting, check if any component folder exists
            elif mode == "inpainting":
                # Inpainting has structure: {run_name}/{component}/full
                subdirs = [d for d in cand.iterdir() if d.is_dir()]
                if subdirs and any((d / "full").exists() for d in subdirs):
                    return cand

    # If not found, return most likely location
    return samples_base / mode / run_name


def get_inpainting_components(inpainting_dir: Path) -> list:
    """Get list of preserved components in inpainting directory."""
    if not inpainting_dir.exists():
        return []

    components = []
    for subdir in sorted(inpainting_dir.iterdir()):
        if subdir.is_dir() and (subdir / "full").exists():
            components.append(subdir.name)

    return components


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EpureDGM models")

    parser.add_argument('--model', required=True,
                       choices=['ddpm', 'mdm', 'flow_matching', 'vqvae', 'wgan_gp', 'mmvaeplus', 'vae', 'gmrf_mvae', 'meta_vae'],
                       help='Model type')
    parser.add_argument('--dataset', required=True,
                       choices=['epure', 'toy'],
                       help='Dataset name')
    parser.add_argument('--run', type=str,
                       help='Specific run directory (e.g., outputs/ddpm_toy/run_seed0)')
    parser.add_argument('--all-runs', action='store_true',
                       help='Evaluate all runs for this model/dataset')
    parser.add_argument('--seeds', type=str,
                       help='Comma-separated seed values (e.g., 0,1,2)')
    parser.add_argument('--split', default='test', choices=['test', 'train'],
                       help='Data split to evaluate on')
    parser.add_argument('--mode', default='all', choices=['conditional', 'inpainting', 'all'],
                       help='Evaluation mode: conditional, inpainting, or all (default: all)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable FID caching')
    parser.add_argument('--num-bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')
    parser.add_argument('--inception-path', type=str,
                       default='data/pt_inception-2015-12-05-6726825d.pth',
                       help='Path to Inception model weights')

    return parser.parse_args()


def find_checkpoint_in_dir(run_dir: Path) -> Optional[Path]:
    """Find best checkpoint in run directory with priority order.

    Priority:
    1. checkpoint_100.pt (final epoch)
    2. checkpoint_best.pt (if exists)
    3. Latest checkpoint_{N}.pt
    """
    check_dir = run_dir / "check"
    if not check_dir.exists():
        return None

    # Priority 1: checkpoint_100.pt
    checkpoint_100 = check_dir / "checkpoint_100.pt"
    if checkpoint_100.exists():
        return checkpoint_100

    # Priority 2: checkpoint_best.pt
    checkpoint_best = check_dir / "checkpoint_best.pt"
    if checkpoint_best.exists():
        return checkpoint_best

    # Priority 3: Latest checkpoint
    checkpoints = list(check_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[1])
            except:
                return 0
        return max(checkpoints, key=extract_epoch)

    return None


def evaluate_mode(args, runs_to_evaluate: list, mode: str) -> list:
    """Evaluate a specific mode (conditional or inpainting).

    Args:
        args: Parsed arguments
        runs_to_evaluate: List of run directories
        mode: 'conditional' or 'inpainting'

    Returns:
        List of results
    """
    all_results = []

    print(f"\n{'#'*70}")
    print(f"# Evaluating {mode.upper()} mode")
    print(f"{'#'*70}")

    for run_dir in runs_to_evaluate:
        # Verify checkpoint exists
        checkpoint_file = find_checkpoint_in_dir(run_dir)
        if not checkpoint_file or not checkpoint_file.exists():
            print(f"\nWARNING: No valid checkpoint found for {run_dir.name}, skipping...")
            continue

        # Find samples directory for this mode
        samples_dir = find_samples_dir(args.model, args.dataset, run_dir.name, mode)

        if not samples_dir.exists():
            print(f"\nWARNING: {mode.capitalize()} samples not found for {run_dir.name}, skipping...")
            print(f"  Expected at: {samples_dir}")
            continue

        # Extract seed from run directory name
        seed = None
        for part in run_dir.name.split('_'):
            if 'seed' in part:
                try:
                    seed = int(part.replace('seed', ''))
                except ValueError:
                    pass

        if mode == "conditional":
            # Standard conditional evaluation
            print(f"\n{'='*70}")
            print(f"Evaluating run: {run_dir.name} (conditional)")
            print(f"  Checkpoint: {checkpoint_file.name}")
            print(f"  Samples: {samples_dir}")
            print(f"{'='*70}")

            evaluator = ModelEvaluator(
                model_name=args.model,
                dataset_name=args.dataset,
                run_dir=run_dir,
                samples_dir=samples_dir,
                seed=seed,
                inception_path=args.inception_path
            )

            try:
                results = evaluator.evaluate_all_metrics(
                    split=args.split,
                    use_cache=not args.no_cache,
                    num_bootstrap=args.num_bootstrap
                )
                results['mode'] = 'conditional'
                all_results.append(results)

                # Save individual results
                output_file = Path(args.output_dir) / args.dataset / args.model / "conditional" / f"{run_dir.name}.json"
                evaluator.save_results(results, output_file)

            except Exception as e:
                print(f"\nERROR evaluating {run_dir.name}: {e}")
                import traceback
                traceback.print_exc()

        else:  # inpainting
            # For inpainting, evaluate each preserved component
            preserved_components = get_inpainting_components(samples_dir)

            if not preserved_components:
                print(f"\nWARNING: No inpainting components found in {samples_dir}")
                continue

            print(f"\n{'='*70}")
            print(f"Evaluating run: {run_dir.name} (inpainting)")
            print(f"  Checkpoint: {checkpoint_file.name}")
            print(f"  Samples: {samples_dir}")
            print(f"  Preserved components: {preserved_components}")
            print(f"{'='*70}")

            run_results = {
                'model': args.model,
                'dataset': args.dataset,
                'run_dir': str(run_dir),
                'seed': seed,
                'split': args.split,
                'mode': 'inpainting',
                'components': {}
            }

            for comp in preserved_components:
                comp_samples_dir = samples_dir / comp
                print(f"\n  --- Evaluating preserved component: {comp} ---")

                evaluator = ModelEvaluator(
                    model_name=args.model,
                    dataset_name=args.dataset,
                    run_dir=run_dir,
                    samples_dir=comp_samples_dir,
                    seed=seed,
                    inception_path=args.inception_path
                )

                try:
                    comp_results = evaluator.evaluate_all_metrics(
                        split=args.split,
                        use_cache=not args.no_cache,
                        num_bootstrap=args.num_bootstrap
                    )
                    run_results['components'][comp] = comp_results['metrics']

                except Exception as e:
                    print(f"\nERROR evaluating {comp}: {e}")
                    import traceback
                    traceback.print_exc()
                    run_results['components'][comp] = None

            # Compute average metrics across components
            run_results['average'] = compute_average_metrics(run_results['components'])
            all_results.append(run_results)

            # Save individual results
            output_file = Path(args.output_dir) / args.dataset / args.model / "inpainting" / f"{run_dir.name}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(run_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            print(f"\n[OK] Results saved to: {output_file}")

    return all_results


def compute_average_metrics(components_results: dict) -> dict:
    """Compute average metrics across inpainting components."""
    avg = {}

    # Collect valid results
    valid_results = {k: v for k, v in components_results.items() if v is not None}
    if not valid_results:
        return avg

    # FID average
    fids = [r.get('fid') for r in valid_results.values() if r.get('fid') is not None]
    if fids:
        avg['fid'] = {'mean': float(np.mean(fids)), 'std': float(np.std(fids))}

    # IoU WD average
    iou_wds = []
    for r in valid_results.values():
        if r.get('iou_dice') and r['iou_dice'].get('average'):
            iou_wd = r['iou_dice']['average'].get('iou_wd')
            if iou_wd:
                iou_wds.append(iou_wd[0] if isinstance(iou_wd, (list, tuple)) else iou_wd)
    if iou_wds:
        avg['iou_wd'] = {'mean': float(np.mean(iou_wds)), 'std': float(np.std(iou_wds))}

    # RCE average
    rces = []
    for r in valid_results.values():
        if r.get('rce') and r['rce'].get('gen_wd') is not None:
            rces.append(r['rce']['gen_wd'])
    if rces:
        avg['rce_wd'] = {'mean': float(np.mean(rces)), 'std': float(np.std(rces))}

    # CoM average
    coms = []
    for r in valid_results.values():
        if r.get('com') and r['com'].get('overall'):
            wd = r['com']['overall'].get('wasserstein')
            if wd is not None:
                coms.append(wd)
    if coms:
        avg['com_wd'] = {'mean': float(np.mean(coms)), 'std': float(np.std(coms))}

    return avg


def print_summary(all_results: list, mode: str):
    """Print summary statistics for results."""
    print(f"\nSummary Statistics ({mode})")
    print(f"{'='*70}")

    if mode == "conditional":
        # FID
        fids = [r['metrics']['fid'] for r in all_results if r['metrics'].get('fid') is not None]
        if fids:
            print(f"FID: {np.mean(fids):.2f} +/- {np.std(fids):.2f}")

        # IoU WD
        iou_wds = [
            r['metrics']['iou_dice']['average']['iou_wd'][0]
            for r in all_results
            if r['metrics'].get('iou_dice') is not None
        ]
        if iou_wds:
            print(f"IoU WD: {np.mean(iou_wds):.4f} +/- {np.std(iou_wds):.4f}")

        # RCE
        rce_wds = [
            r['metrics']['rce']['gen_wd']
            for r in all_results
            if r['metrics'].get('rce') is not None
        ]
        if rce_wds:
            print(f"RCE WD: {np.mean(rce_wds):.4f} +/- {np.std(rce_wds):.4f}")

        # CoM
        com_wds = [
            r['metrics']['com']['overall']['wasserstein']
            for r in all_results
            if r['metrics'].get('com') is not None
        ]
        if com_wds:
            print(f"CoM WD: {np.mean(com_wds):.4f} +/- {np.std(com_wds):.4f}")

    else:  # inpainting
        # Average across runs (each run already has averages across components)
        fids = [r['average'].get('fid', {}).get('mean') for r in all_results if r.get('average', {}).get('fid')]
        if fids:
            print(f"FID: {np.mean(fids):.2f} +/- {np.std(fids):.2f}")

        iou_wds = [r['average'].get('iou_wd', {}).get('mean') for r in all_results if r.get('average', {}).get('iou_wd')]
        if iou_wds:
            print(f"IoU WD: {np.mean(iou_wds):.4f} +/- {np.std(iou_wds):.4f}")

        rce_wds = [r['average'].get('rce_wd', {}).get('mean') for r in all_results if r.get('average', {}).get('rce_wd')]
        if rce_wds:
            print(f"RCE WD: {np.mean(rce_wds):.4f} +/- {np.std(rce_wds):.4f}")

        com_wds = [r['average'].get('com_wd', {}).get('mean') for r in all_results if r.get('average', {}).get('com_wd')]
        if com_wds:
            print(f"CoM WD: {np.mean(com_wds):.4f} +/- {np.std(com_wds):.4f}")

    print(f"\n[Evaluated {len(all_results)} run(s) successfully]")


def main():
    args = parse_args()

    # Determine which runs to evaluate
    runs_to_evaluate = []

    if args.run:
        runs_to_evaluate = [Path(args.run)]
    elif args.all_runs:
        runs_to_evaluate = find_all_runs(args.model, args.dataset)
    elif args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]

        suffix = "_toy" if args.dataset == "toy" else ""
        base_dir = Path("outputs") / f"{args.model}{suffix}"

        if base_dir.exists():
            seed_folders = [base_dir / f"run_seed{s}" for s in seeds]
            if all(f.exists() for f in seed_folders):
                runs_to_evaluate = seed_folders
            else:
                all_runs = find_all_runs(args.model, args.dataset)
                all_runs = sorted(all_runs, key=lambda d: d.stat().st_mtime)
                runs_to_evaluate = all_runs[:len(seeds)]
        else:
            print(f"ERROR: Output directory not found: {base_dir}")
            sys.exit(1)
    else:
        print("ERROR: Must specify --run, --all-runs, or --seeds")
        sys.exit(1)

    if not runs_to_evaluate:
        print(f"ERROR: No runs found for {args.model} on {args.dataset}")
        sys.exit(1)

    print(f"\n[Found {len(runs_to_evaluate)} run(s) to evaluate]")
    print(f"[Mode: {args.mode}]")

    # Determine modes to evaluate
    if args.mode == 'all':
        modes = ['conditional', 'inpainting']
    else:
        modes = [args.mode]

    all_results = {'conditional': [], 'inpainting': []}

    # Evaluate each mode
    for mode in modes:
        results = evaluate_mode(args, runs_to_evaluate, mode)
        all_results[mode] = results

        # Save aggregate results for this mode
        if results:
            aggregate_path = Path(args.output_dir) / args.dataset / f"{args.model}_{mode}_all_runs.json"
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)

            with open(aggregate_path, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

            print(f"\n{'='*70}")
            print(f"[OK] {mode.capitalize()} results saved to: {aggregate_path}")
            print_summary(results, mode)

    # Final summary
    total_evaluated = sum(len(r) for r in all_results.values())
    if total_evaluated == 0:
        print("\nERROR: No runs were successfully evaluated")
        sys.exit(1)


if __name__ == '__main__':
    main()
