#!/usr/bin/env python3
"""
Fix pipeline state for Windows - Map existing training directories to seeds
Run this after interrupted pipeline to recover training runs.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def find_latest_state_file():
    """Find the most recent state.json file in logs/pipeline/"""
    logs_dir = Path("logs/pipeline")
    if not logs_dir.exists():
        print("No logs/pipeline directory found")
        return None

    state_files = list(logs_dir.glob("*/state.json"))
    if not state_files:
        print("No state.json files found")
        return None

    # Sort by modification time, most recent first
    state_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return state_files[0]


def fix_state(state_file: Path, dataset: str = "epure"):
    """Fix state.json by mapping timestamp directories to seeds."""

    print(f"Reading state from: {state_file}")

    with open(state_file, 'r') as f:
        state = json.load(f)

    # Initialize run_directories if not present
    if 'run_directories' not in state:
        state['run_directories'] = {}

    suffix = "_toy" if dataset == "toy" else ""
    outputs_base = Path("outputs")

    fixed_count = 0

    # For each completed training, find its directory
    for model, seeds in state.get('completed', {}).get('training', {}).items():
        model_output = outputs_base / f"{model}{suffix}"

        if not model_output.exists():
            print(f"Warning: {model_output} not found")
            continue

        # Find timestamp directories
        timestamp_dirs = sorted([
            d for d in model_output.iterdir()
            if d.is_dir() and d.name.startswith("20")
        ], key=lambda d: d.stat().st_mtime)

        if not timestamp_dirs:
            print(f"Warning: No timestamp directories found for {model}")
            continue

        if model not in state['run_directories']:
            state['run_directories'][model] = {}

        # Map seeds to directories by order
        for i, seed in enumerate(sorted(seeds)):
            if i < len(timestamp_dirs):
                dir_path = timestamp_dirs[i]
                state['run_directories'][model][seed] = str(dir_path)
                print(f"Mapped {model} seed{seed} -> {dir_path.name}")
                fixed_count += 1
            else:
                print(f"Warning: Not enough directories for {model} seed{seed}")

    # Save fixed state
    backup_file = state_file.with_suffix('.json.backup')
    print(f"\nBacking up original state to: {backup_file}")
    with open(backup_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"Writing fixed state to: {state_file}")
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n✓ Fixed {fixed_count} training directory mappings")
    return state_file


def main():
    parser = argparse.ArgumentParser(description="Fix pipeline state for Windows")
    parser.add_argument('--state-file', type=Path, help='Path to state.json (auto-detect if not provided)')
    parser.add_argument('--dataset', default='epure', choices=['epure', 'toy'], help='Dataset name')
    args = parser.parse_args()

    state_file = args.state_file
    if not state_file:
        state_file = find_latest_state_file()
        if not state_file:
            print("Could not find state.json file")
            return

    if not state_file.exists():
        print(f"State file not found: {state_file}")
        return

    fix_state(state_file, args.dataset)

    print("\nYou can now resume the pipeline:")
    print(f"  python scripts/pipeline/run_pipeline.py --dataset {args.dataset} --skip-training")


if __name__ == '__main__':
    main()
