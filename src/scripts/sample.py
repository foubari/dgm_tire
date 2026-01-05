#!/usr/bin/env python3
"""
Global sampling script for all diffusion models - Modular version.

This script dispatches sampling to the appropriate model-specific sampling script
based on the model type and configuration file.

Usage:
    python scripts/sample.py --model ddpm --checkpoint outputs/.../checkpoint_100.pt --config configs/ddpm_default.yaml --mode unconditional
    python scripts/sample.py --model mdm --checkpoint outputs/.../checkpoint_100.pt --config configs/mdm_default.yaml --mode conditional
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


def get_model_script(model_type):
    """Get the path to the model-specific sampling script."""
    model_scripts = {
        'ddpm': 'models/ddpm/sample.py',
        'mdm': 'models/mdm/sample.py',
        'multinomial': 'models/mdm/sample.py',  # Alias for mdm
        'flow_matching': 'models/flow_matching/sample.py',
        'vqvae': 'models/vqvae/sample.py',
        'wgan_gp': 'models/wgan_gp/sample.py',
        'mmvaeplus': 'models/mmvaeplus/sample.py',
        'gmrf_mvae': 'models/gmrf_mvae/sample.py',
        'meta_vae': 'models/meta_vae/sample.py',
        'vae': 'models/vae/sample.py',
    }
    
    if model_type not in model_scripts:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: {list(model_scripts.keys())}"
        )
    
    script_path = _SRC_NEW_DIR / model_scripts[model_type]
    if not script_path.exists():
        raise FileNotFoundError(f"Sampling script not found: {script_path}")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description='Global sampling dispatcher for diffusion models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['ddpm', 'mdm', 'multinomial', 'flow_matching', 'vqvae', 'wgan_gp', 'mmvaeplus', 'gmrf_mvae','meta_vae', 'vae'],
                       help='Model type to sample from')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file (optional)')
    
    # Parse known args and get remaining args
    args, remaining_args = parser.parse_known_args()
    
    # Get model-specific script
    script_path = get_model_script(args.model)
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        '--checkpoint', args.checkpoint,
    ]
    
    if args.config:
        cmd.extend(['--config', args.config])
    
    cmd.extend(remaining_args)
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the script
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()

