#!/usr/bin/env python3
"""
Pipeline de test ultra-rapide - 1 epoch avec configs test

Usage:
    python scripts/pipeline/run_pipeline_quick_test.py --dataset toy --models ddpm
"""

import sys
from pathlib import Path

# Modify sys.path to import run_pipeline
sys.path.insert(0, str(Path(__file__).parent))

# Import the Pipeline class
from run_pipeline import Pipeline, parse_args, ALL_MODELS


class QuickTestPipeline(Pipeline):
    """Pipeline variant that uses test configs (1 epoch, 50 samples)."""

    def get_config_path(self, model: str) -> Path:
        """Get TEST config path for model (dataset-aware)."""
        if self.dataset == "toy":
            config_dir = Path("src/configs/pipeline/test_toy")
        else:
            config_dir = Path("src/configs/pipeline/test")

        config_file = config_dir / f"{model}_pipeline_test.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Test config not found: {config_file}")

        return config_file


def main():
    print("=" * 80)
    print(" EPUREDGM QUICK TEST MODE")
    print("=" * 80)
    print("Using test configs (1 epoch, 50 samples)")
    print("Expected duration: ~2-5 minutes per model")
    print("=" * 80)
    print()

    args = parse_args()

    # Force num_samples to 50 for quick testing
    args.num_samples = 50

    pipeline = QuickTestPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
