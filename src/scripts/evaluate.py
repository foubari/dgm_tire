#!/usr/bin/env python3
"""
Script d'évaluation pour EpureDGM.

Usage:
    # Evaluer un run spécifique
    python src_new/scripts/evaluate.py --model ddpm --dataset toy --run outputs/ddpm_toy/run_seed0

    # Evaluer tous les runs d'un modèle
    python src_new/scripts/evaluate.py --model ddpm --dataset toy --all-runs

    # Evaluer plusieurs seeds
    python src_new/scripts/evaluate.py --model ddpm --dataset epure --seeds 0,1,2
"""

import os
# Fix OMP conflict between Intel MKL and LLVM OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
import sys
from pathlib import Path
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


def find_samples_dir(model_name: str, dataset_name: str, run_name: str) -> Path:
    """Trouve le dossier samples pour un run donné."""
    if dataset_name == "toy":
        samples_base = Path("samples") / f"{model_name}_toy"
    else:
        samples_base = Path("samples") / model_name

    # Try different locations
    candidates = [
        samples_base / "conditional" / run_name,
        samples_base / run_name / "conditional",
        samples_base / run_name,
    ]

    for cand in candidates:
        if cand.exists() and (cand / "full").exists():
            return cand

    # If not found, return most likely location
    return samples_base / "conditional" / run_name


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


def main():
    args = parse_args()

    # Déterminer quels runs évaluer
    runs_to_evaluate = []

    if args.run:
        runs_to_evaluate = [Path(args.run)]
    elif args.all_runs:
        runs_to_evaluate = find_all_runs(args.model, args.dataset)
    elif args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]

        # Try to find run_seed folders (symlinks on Unix or actual folders)
        suffix = "_toy" if args.dataset == "toy" else ""
        base_dir = Path("outputs") / f"{args.model}{suffix}"

        if base_dir.exists():
            # First, try finding run_seed folders
            seed_folders = [base_dir / f"run_seed{s}" for s in seeds]
            if all(f.exists() for f in seed_folders):
                runs_to_evaluate = seed_folders
            else:
                # Fallback: use timestamp folders in chronological order
                all_runs = find_all_runs(args.model, args.dataset)
                # Sort by modification time (oldest first)
                all_runs = sorted(all_runs, key=lambda d: d.stat().st_mtime)
                # Take first N runs matching number of seeds
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

    # Évaluer chaque run
    all_results = []

    for run_dir in runs_to_evaluate:
        # Trouver samples directory
        samples_dir = find_samples_dir(args.model, args.dataset, run_dir.name)

        if not samples_dir.exists():
            print(f"\nWARNING: Samples not found for {run_dir.name}, skipping...")
            print(f"  Expected at: {samples_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating run: {run_dir.name}")
        print(f"  Samples: {samples_dir}")
        print(f"{'='*70}")

        # Extract seed from run directory name (if present)
        seed = None
        for part in run_dir.name.split('_'):
            if 'seed' in part:
                try:
                    seed = int(part.replace('seed', ''))
                except ValueError:
                    pass

        # Créer evaluator
        evaluator = ModelEvaluator(
            model_name=args.model,
            dataset_name=args.dataset,
            run_dir=run_dir,
            samples_dir=samples_dir,
            seed=seed,
            inception_path=args.inception_path
        )

        # Évaluer
        try:
            results = evaluator.evaluate_all_metrics(
                split=args.split,
                use_cache=not args.no_cache,
                num_bootstrap=args.num_bootstrap
            )
            all_results.append(results)

            # Sauvegarder résultats individuels
            output_file = Path(args.output_dir) / args.dataset / args.model / f"{run_dir.name}.json"
            evaluator.save_results(results, output_file)

        except Exception as e:
            print(f"\nERROR evaluating {run_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Sauvegarder résultats agrégés
    if all_results:
        aggregate_path = Path(args.output_dir) / args.dataset / f"{args.model}_all_runs.json"
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)

        with open(aggregate_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"[OK] Aggregate results saved to: {aggregate_path}")
        print(f"{'='*70}")

        # Afficher statistiques
        print(f"\nSummary Statistics")
        print(f"{'='*70}")

        # FID
        fids = [r['metrics']['fid'] for r in all_results if r['metrics'].get('fid') is not None]
        if fids:
            print(f"FID: {np.mean(fids):.2f} ± {np.std(fids):.2f}")

        # IoU WD
        iou_wds = [
            r['metrics']['iou_dice']['average']['iou_wd'][0]
            for r in all_results
            if r['metrics'].get('iou_dice') is not None
        ]
        if iou_wds:
            print(f"IoU WD: {np.mean(iou_wds):.4f} ± {np.std(iou_wds):.4f}")

        # RCE
        rce_wds = [
            r['metrics']['rce']['gen_wd']
            for r in all_results
            if r['metrics'].get('rce') is not None
        ]
        if rce_wds:
            print(f"RCE WD: {np.mean(rce_wds):.4f} ± {np.std(rce_wds):.4f}")

        # CoM
        com_wds = [
            r['metrics']['com']['overall']['wasserstein']
            for r in all_results
            if r['metrics'].get('com') is not None
        ]
        if com_wds:
            print(f"CoM WD: {np.mean(com_wds):.4f} ± {np.std(com_wds):.4f}")

        print(f"\n[Evaluated {len(all_results)} run(s) successfully]")

    else:
        print("\nERROR: No runs were successfully evaluated")
        sys.exit(1)


if __name__ == '__main__':
    main()
