#!/usr/bin/env python3
"""
Plot evaluation results for all models.

Creates one figure per metric, comparing all models across their runs.
Each model has a consistent color and marker shape across all figures.

Usage:
    python plot_evaluation_results.py --dataset epure
    python plot_evaluation_results.py --dataset toy
    python plot_evaluation_results.py --dataset epure --mode conditional
    python plot_evaluation_results.py --dataset epure --mode inpainting
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Add src to path
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ============================================================================
# CONFIGURATION: Model styles (consistent across all figures)
# ============================================================================

# Models in display order
MODELS = [
    'ddpm',
    'flow_matching',
    'mdm',
    'vae',
    'meta_vae',
    'gmrf_mvae',
    'mmvaeplus',
    'wgan_gp',
]

# Display names for models
MODEL_DISPLAY_NAMES = {
    'ddpm': 'DDPM',
    'flow_matching': 'Flow Matching',
    'mdm': 'MDM',
    'vae': 'VAE',
    'meta_vae': 'Meta-VAE',
    'gmrf_mvae': 'GMRF-MVAE',
    'mmvaeplus': 'MMVAE+',
    'wgan_gp': 'WGAN-GP',
}

# Colors for each model (colorblind-friendly palette)
MODEL_COLORS = {
    'ddpm': '#1f77b4',        # Blue
    'flow_matching': '#ff7f0e',  # Orange
    'mdm': '#2ca02c',         # Green
    'vae': '#d62728',         # Red
    'meta_vae': '#9467bd',    # Purple
    'gmrf_mvae': '#8c564b',   # Brown
    'mmvaeplus': '#e377c2',   # Pink
    'wgan_gp': '#7f7f7f',     # Gray
}

# Markers for each model
MODEL_MARKERS = {
    'ddpm': 'o',          # Circle
    'flow_matching': 's', # Square
    'mdm': '^',           # Triangle up
    'vae': 'D',           # Diamond
    'meta_vae': 'v',      # Triangle down
    'gmrf_mvae': 'p',     # Pentagon
    'mmvaeplus': 'h',     # Hexagon
    'wgan_gp': '*',       # Star
}

# Marker size
MARKER_SIZE = 100

# ============================================================================
# Data loading functions
# ============================================================================

def load_all_runs_json(results_dir: Path, model: str, mode: str) -> Optional[List[Dict]]:
    """Load the aggregated all_runs JSON file for a model and mode."""
    filename = f"{model}_{mode}_all_runs.json"
    filepath = results_dir / filename

    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        return json.load(f)


def get_available_models(results_dir: Path, mode: str) -> List[str]:
    """Get list of models that have results for the given mode."""
    available = []
    for model in MODELS:
        data = load_all_runs_json(results_dir, model, mode)
        if data and len(data) > 0:
            available.append(model)
    return available


# ============================================================================
# Metric extraction functions
# ============================================================================

def extract_fid(run_data: Dict) -> Optional[float]:
    """Extract FID from a run's metrics."""
    metrics = run_data.get('metrics', {})
    return metrics.get('fid')


def extract_com_overall(run_data: Dict) -> Optional[float]:
    """Extract COM overall Wasserstein distance."""
    metrics = run_data.get('metrics', {})
    com = metrics.get('com', {})
    overall = com.get('overall', {})
    return overall.get('wasserstein')


def extract_rce(run_data: Dict) -> Optional[Tuple[float, float]]:
    """Extract RCE (real_wd, gen_wd) from a run's metrics."""
    metrics = run_data.get('metrics', {})
    rce = metrics.get('rce', {})
    real_wd = rce.get('real_wd')
    gen_wd = rce.get('gen_wd')
    if real_wd is not None and gen_wd is not None:
        return (real_wd, gen_wd)
    return None


def extract_iou_overall(run_data: Dict) -> Optional[float]:
    """Extract average IoU Wasserstein distance across all component pairs."""
    metrics = run_data.get('metrics', {})
    iou_dice = metrics.get('iou_dice', {})

    if not iou_dice:
        return None

    iou_values = []
    for pair_name, pair_data in iou_dice.items():
        iou_wd = pair_data.get('iou_wd', [])
        if iou_wd and len(iou_wd) > 0:
            iou_values.append(iou_wd[0])  # Mean

    if iou_values:
        return np.mean(iou_values)
    return None


def extract_dice_overall(run_data: Dict) -> Optional[float]:
    """Extract average Dice Wasserstein distance across all component pairs."""
    metrics = run_data.get('metrics', {})
    iou_dice = metrics.get('iou_dice', {})

    if not iou_dice:
        return None

    dice_values = []
    for pair_name, pair_data in iou_dice.items():
        dice_wd = pair_data.get('dice_wd', [])
        if dice_wd and len(dice_wd) > 0:
            dice_values.append(dice_wd[0])  # Mean

    if dice_values:
        return np.mean(dice_values)
    return None


# ============================================================================
# Inpainting-specific extraction (averaged across preserved components)
# ============================================================================

def extract_inpainting_fid_avg(run_data: Dict) -> Optional[float]:
    """Extract average FID across all preserved components for inpainting."""
    components = run_data.get('components', {})
    if not components:
        return None

    fid_values = []
    for comp_name, comp_data in components.items():
        fid = comp_data.get('fid')
        if fid is not None:
            fid_values.append(fid)

    if fid_values:
        return np.mean(fid_values)
    return None


def extract_inpainting_com_avg(run_data: Dict) -> Optional[float]:
    """Extract average COM overall Wasserstein across all preserved components."""
    components = run_data.get('components', {})
    if not components:
        return None

    com_values = []
    for comp_name, comp_data in components.items():
        com = comp_data.get('com', {})
        overall = com.get('overall', {})
        wd = overall.get('wasserstein')
        if wd is not None:
            com_values.append(wd)

    if com_values:
        return np.mean(com_values)
    return None


def extract_inpainting_iou_avg(run_data: Dict) -> Optional[float]:
    """Extract average IoU WD across all components and pairs for inpainting."""
    components = run_data.get('components', {})
    if not components:
        return None

    all_iou_values = []
    for comp_name, comp_data in components.items():
        iou_dice = comp_data.get('iou_dice', {})
        for pair_name, pair_data in iou_dice.items():
            iou_wd = pair_data.get('iou_wd', [])
            if iou_wd and len(iou_wd) > 0:
                all_iou_values.append(iou_wd[0])

    if all_iou_values:
        return np.mean(all_iou_values)
    return None


def extract_inpainting_dice_avg(run_data: Dict) -> Optional[float]:
    """Extract average Dice WD across all components and pairs for inpainting."""
    components = run_data.get('components', {})
    if not components:
        return None

    all_dice_values = []
    for comp_name, comp_data in components.items():
        iou_dice = comp_data.get('iou_dice', {})
        for pair_name, pair_data in iou_dice.items():
            dice_wd = pair_data.get('dice_wd', [])
            if dice_wd and len(dice_wd) > 0:
                all_dice_values.append(dice_wd[0])

    if all_dice_values:
        return np.mean(all_dice_values)
    return None


# ============================================================================
# Plotting functions
# ============================================================================

def collect_metric_data(
    results_dir: Path,
    models: List[str],
    mode: str,
    extract_fn
) -> Dict[str, List[float]]:
    """
    Collect metric values for all models and their runs.

    Returns dict: model -> list of metric values (one per run)
    """
    data = {}

    for model in models:
        runs = load_all_runs_json(results_dir, model, mode)
        if not runs:
            continue

        values = []
        for run in runs:
            value = extract_fn(run)
            if value is not None:
                values.append(value)

        if values:
            data[model] = values

    return data


def plot_metric(
    data: Dict[str, List[float]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
    title: Optional[str] = None,
    lower_is_better: bool = True,
    log_scale: bool = False
):
    """
    Create a scatter plot for a single metric.

    X-axis: models
    Y-axis: metric values (one point per run, jittered for visibility)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get models in order
    models_with_data = [m for m in MODELS if m in data]

    if not models_with_data:
        print(f"  No data available for {metric_name}")
        plt.close(fig)
        return

    # X positions for each model
    x_positions = {model: i for i, model in enumerate(models_with_data)}

    # Plot each model's runs
    for model in models_with_data:
        values = data[model]
        x = x_positions[model]

        # Add small jitter for visibility when multiple runs overlap
        jitter = np.random.uniform(-0.1, 0.1, len(values))
        xs = [x + j for j in jitter]

        ax.scatter(
            xs, values,
            c=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            s=MARKER_SIZE,
            label=MODEL_DISPLAY_NAMES[model],
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        # Add mean line
        mean_val = np.mean(values)
        ax.hlines(
            mean_val,
            x - 0.3, x + 0.3,
            colors=MODEL_COLORS[model],
            linestyles='--',
            linewidths=2,
            alpha=0.7
        )

    # Configure axes
    ax.set_xticks(range(len(models_with_data)))
    ax.set_xticklabels([MODEL_DISPLAY_NAMES[m] for m in models_with_data], rotation=45, ha='right')
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{metric_name}" + (" (lower is better)" if lower_is_better else " (higher is better)"))

    if log_scale:
        ax.set_yscale('log')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color='w',
               markerfacecolor=MODEL_COLORS[m], markersize=10,
               markeredgecolor='black', markeredgewidth=0.5,
               label=MODEL_DISPLAY_NAMES[m])
        for m in models_with_data
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_metrics(
    results_dir: Path,
    output_dir: Path,
    mode: str,
    dataset: str
):
    """Plot all metrics for a given mode (conditional or inpainting)."""

    print(f"\nPlotting {mode} metrics for {dataset}...")

    # Get available models
    models = get_available_models(results_dir, mode)
    print(f"  Models with data: {models}")

    if not models:
        print(f"  No models found for {mode} mode.")
        return

    # Create output directory
    mode_output_dir = output_dir / mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    # Define metrics to plot
    if mode == 'conditional':
        metrics = [
            ('FID', 'FID Score', extract_fid, True, False),
            ('COM', 'COM Wasserstein Distance', extract_com_overall, True, False),
            ('IoU_WD', 'IoU Wasserstein Distance (avg)', extract_iou_overall, True, False),
            ('Dice_WD', 'Dice Wasserstein Distance (avg)', extract_dice_overall, True, False),
        ]
    else:  # inpainting
        metrics = [
            ('FID', 'FID Score (avg over components)', extract_inpainting_fid_avg, True, False),
            ('COM', 'COM Wasserstein Distance (avg)', extract_inpainting_com_avg, True, False),
            ('IoU_WD', 'IoU Wasserstein Distance (avg)', extract_inpainting_iou_avg, True, False),
            ('Dice_WD', 'Dice Wasserstein Distance (avg)', extract_inpainting_dice_avg, True, False),
        ]

    # Plot each metric
    for metric_name, ylabel, extract_fn, lower_is_better, log_scale in metrics:
        data = collect_metric_data(results_dir, models, mode, extract_fn)

        output_path = mode_output_dir / f"{metric_name.lower()}_{dataset}_{mode}.png"

        plot_metric(
            data,
            metric_name,
            ylabel,
            output_path,
            title=f"{metric_name} - {dataset.upper()} ({mode.capitalize()})",
            lower_is_better=lower_is_better,
            log_scale=log_scale
        )


def create_summary_table(
    results_dir: Path,
    output_dir: Path,
    mode: str,
    dataset: str
):
    """Create a summary table with mean and std for each model."""

    models = get_available_models(results_dir, mode)
    if not models:
        return

    # Define metrics
    if mode == 'conditional':
        metrics = [
            ('FID', extract_fid),
            ('COM', extract_com_overall),
            ('IoU_WD', extract_iou_overall),
            ('Dice_WD', extract_dice_overall),
        ]
    else:
        metrics = [
            ('FID', extract_inpainting_fid_avg),
            ('COM', extract_inpainting_com_avg),
            ('IoU_WD', extract_inpainting_iou_avg),
            ('Dice_WD', extract_inpainting_dice_avg),
        ]

    # Collect data
    summary = {}
    for model in models:
        summary[model] = {}
        for metric_name, extract_fn in metrics:
            data = collect_metric_data(results_dir, [model], mode, extract_fn)
            if model in data and data[model]:
                values = data[model]
                summary[model][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n_runs': len(values)
                }

    # Save as JSON
    output_path = output_dir / mode / f"summary_{dataset}_{mode}.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {output_path}")

    # Print table
    print(f"\n{'='*80}")
    print(f"Summary: {dataset.upper()} - {mode.capitalize()}")
    print(f"{'='*80}")
    print(f"{'Model':<15}", end='')
    for metric_name, _ in metrics:
        print(f"{metric_name:>15}", end='')
    print(f"{'N_runs':>10}")
    print("-" * 80)

    for model in models:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        print(f"{display_name:<15}", end='')
        for metric_name, _ in metrics:
            if metric_name in summary.get(model, {}):
                mean = summary[model][metric_name]['mean']
                std = summary[model][metric_name]['std']
                print(f"{mean:>10.4f}+{std:.2f}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        n_runs = summary.get(model, {}).get(metrics[0][0], {}).get('n_runs', 0)
        print(f"{n_runs:>10}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--dataset', type=str, required=True, choices=['epure', 'toy'],
                       help='Dataset to plot results for')
    parser.add_argument('--mode', type=str, default='all', choices=['conditional', 'inpainting', 'all'],
                       help='Which mode to plot (default: all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: evaluation_results/<dataset>/plots)')

    args = parser.parse_args()

    # Setup paths
    results_dir = _PROJECT_ROOT / 'evaluation_results' / args.dataset

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / 'plots'

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Plot for requested modes
    modes = ['conditional', 'inpainting'] if args.mode == 'all' else [args.mode]

    for mode in modes:
        plot_all_metrics(results_dir, output_dir, mode, args.dataset)
        create_summary_table(results_dir, output_dir, mode, args.dataset)

    print(f"\nDone! Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
