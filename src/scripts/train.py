#!/usr/bin/env python3
"""
Global training script for all diffusion models - Modular version.

This script dispatches training to the appropriate model-specific training script
based on the model type and configuration file.

Usage:
    python scripts/train.py --model ddpm --config configs/ddpm_default.yaml
    python scripts/train.py --model mdm --config configs/mdm_default.yaml
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add src_new to path
_THIS_FILE = Path(__file__).resolve()
_SRC_NEW_DIR = _THIS_FILE.parent.parent
_PROJECT_ROOT = _SRC_NEW_DIR.parent

if str(_SRC_NEW_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_NEW_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from utils.config import load_config


def get_model_script(model_type):
    """Get the path to the model-specific training script."""
    model_scripts = {
        'ddpm': 'models/ddpm/train.py',
        'mdm': 'models/mdm/train.py',
        'multinomial': 'models/mdm/train.py',  # Alias for mdm
        'flow_matching': 'models/flow_matching/train.py',
        'vqvae': 'models/vqvae/train.py',
        'wgan_gp': 'models/wgan_gp/train.py',
        'mmvaeplus': 'models/mmvaeplus/train.py',
        'gmrf_mvae': 'models/gmrf_mvae/train.py',
        'meta_vae': 'models/meta_vae/train.py',
        'vae': 'models/vae/train.py',
    }
    
    if model_type not in model_scripts:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: {list(model_scripts.keys())}"
        )
    
    script_path = _SRC_NEW_DIR / model_scripts[model_type]
    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description='Global training dispatcher for diffusion models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['ddpm', 'mdm', 'multinomial', 'flow_matching', 'vqvae', 'wgan_gp', 'mmvaeplus', 'gmrf_mvae','meta_vae', 'vae'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    
    # Parse known args and get remaining args
    args, remaining_args = parser.parse_known_args()
    
    # Get model-specific script
    script_path = get_model_script(args.model)
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        '--config', args.config,
    ] + remaining_args
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the script
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()

