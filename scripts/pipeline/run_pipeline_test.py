#!/usr/bin/env python3
"""
Pipeline de test rapide - 1 epoch, 50 samples

Usage:
    python scripts/pipeline/run_pipeline_test.py --dataset toy --models ddpm
"""

import sys
import subprocess
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run pipeline in test mode with small parameters."""

    print("=" * 80)
    print(" EPUREDGM PIPELINE TEST MODE")
    print("=" * 80)
    print("Using reduced parameters for quick validation:")
    print("  - num_samples: 50 (instead of 1000)")
    print("  - Recommended: Use test configs with 1 epoch")
    print("=" * 80)
    print()

    # Forward all args to main pipeline but with reduced samples
    args = sys.argv[1:]  # Skip script name

    # Add --num-samples if not present
    if '--num-samples' not in args:
        args.extend(['--num-samples', '50'])

    # Call main pipeline
    cmd = [sys.executable, 'scripts/pipeline/run_pipeline.py'] + args

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
