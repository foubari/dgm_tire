#!/usr/bin/env python3
"""
Aggregate evaluation results across multiple runs.

This script:
- Loads individual evaluation JSON files
- Computes aggregate statistics (mean ± std)
- Generates summary JSON with rankings

Usage:
    python scripts/aggregate_results.py --dataset epure --output results/summary.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Any


def load_results(dataset: str) -> Dict[str, List[Dict]]:
    """
    Load all evaluation results for a dataset.

    Args:
        dataset: 'epure' or 'toy'

    Returns:
        Dict mapping model names to lists of run results
    """
    results_dir = Path(f"evaluation_results/{dataset}")

    if not results_dir.exists():
        print(f"Warning: Results directory not found: {results_dir}")
        return {}

    model_results = {}

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        model_results[model_name] = []

        # Load all run JSON files
        for json_file in model_dir.glob("run_seed*.json"):
            try:
                with open(json_file) as f:
                    result = json.load(f)
                    model_results[model_name].append(result)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

    return model_results


def aggregate_metrics(runs: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple runs.

    Args:
        runs: List of run result dictionaries

    Returns:
        Aggregated metrics with mean ± std
    """
    if not runs:
        return {}

    # Extract FID scores
    fids = [r['metrics']['fid'] for r in runs if 'fid' in r['metrics']]

    # Extract IoU/Dice averages
    iou_wds = []
    for r in runs:
        if 'iou_dice' in r['metrics'] and 'average' in r['metrics']['iou_dice']:
            avg = r['metrics']['iou_dice']['average']
            if isinstance(avg, dict) and 'iou_wd' in avg:
                iou_wds.append(avg['iou_wd'][0])  # Mean value
            elif isinstance(avg, list) and len(avg) > 0:
                iou_wds.append(avg[0])

    # Extract RCE
    rce_gen_wds = []
    for r in runs:
        if 'rce' in r['metrics'] and 'gen_wd' in r['metrics']['rce']:
            rce_gen_wds.append(r['metrics']['rce']['gen_wd'])

    # Extract CoM
    com_overall_wds = []
    for r in runs:
        if 'com' in r['metrics'] and 'overall' in r['metrics']['com']:
            overall = r['metrics']['com']['overall']
            if 'wasserstein' in overall:
                wd = overall['wasserstein']
                if isinstance(wd, list):
                    com_overall_wds.append(wd[0])
                else:
                    com_overall_wds.append(wd)

    # Compute aggregate statistics
    aggregated = {
        'num_runs': len(runs),
        'seeds': [r.get('seed', None) for r in runs]
    }

    if fids:
        aggregated['fid'] = {
            'mean': float(np.mean(fids)),
            'std': float(np.std(fids)),
            'min': float(np.min(fids)),
            'max': float(np.max(fids))
        }

    if iou_wds:
        aggregated['iou_dice_wd'] = {
            'mean': float(np.mean(iou_wds)),
            'std': float(np.std(iou_wds))
        }

    if rce_gen_wds:
        aggregated['rce_gen_wd'] = {
            'mean': float(np.mean(rce_gen_wds)),
            'std': float(np.std(rce_gen_wds))
        }

    if com_overall_wds:
        aggregated['com_overall_wd'] = {
            'mean': float(np.mean(com_overall_wds)),
            'std': float(np.std(com_overall_wds))
        }

    return aggregated


def rank_models(model_aggregates: Dict[str, Dict]) -> List[Dict]:
    """
    Rank models by FID score (lower is better).

    Args:
        model_aggregates: Dict mapping model names to aggregated metrics

    Returns:
        List of models ranked by FID
    """
    # Extract models with FID scores
    models_with_fid = [
        {
            'model': model,
            'fid_mean': agg['fid']['mean'],
            'fid_std': agg['fid']['std'],
            **agg
        }
        for model, agg in model_aggregates.items()
        if 'fid' in agg
    ]

    # Sort by FID (ascending - lower is better)
    ranked = sorted(models_with_fid, key=lambda x: x['fid_mean'])

    return ranked


def generate_summary(dataset: str) -> Dict:
    """
    Generate summary of all evaluation results.

    Args:
        dataset: 'epure' or 'toy'

    Returns:
        Summary dictionary
    """
    # Load all results
    model_results = load_results(dataset)

    # Aggregate per model
    model_aggregates = {}
    for model, runs in model_results.items():
        model_aggregates[model] = aggregate_metrics(runs)

    # Rank models
    rankings = rank_models(model_aggregates)

    # Create summary
    summary = {
        'dataset': dataset,
        'num_models': len(model_aggregates),
        'model_aggregates': model_aggregates,
        'rankings': rankings
    }

    return summary


def print_summary(summary: Dict):
    """Print human-readable summary."""
    print("=" * 80)
    print(f" EVALUATION SUMMARY - {summary['dataset'].upper()} Dataset")
    print("=" * 80)
    print()

    print(f"Models evaluated: {summary['num_models']}")
    print()

    if summary['rankings']:
        print("Rankings (by FID, lower is better):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<20} {'FID':<20} {'IoU/Dice WD':<15}")
        print("-" * 80)

        for i, model_data in enumerate(summary['rankings'], 1):
            fid_str = f"{model_data['fid_mean']:.2f} ± {model_data['fid_std']:.2f}"

            iou_str = "N/A"
            if 'iou_dice_wd' in model_data:
                iou_mean = model_data['iou_dice_wd']['mean']
                iou_std = model_data['iou_dice_wd']['std']
                iou_str = f"{iou_mean:.4f} ± {iou_std:.4f}"

            print(f"{i:<6} {model_data['model']:<20} {fid_str:<20} {iou_str:<15}")

        print("-" * 80)
    else:
        print("No models ranked (no FID scores available)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['epure', 'toy'],
                       help='Dataset name')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: evaluation_results/{dataset}/summary.json)')
    args = parser.parse_args()

    # Generate summary
    print(f"Aggregating results for {args.dataset} dataset...")
    summary = generate_summary(args.dataset)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"evaluation_results/{args.dataset}/summary.json")

    # Save summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {output_path}")
    print()

    # Print summary
    print_summary(summary)


if __name__ == '__main__':
    main()
