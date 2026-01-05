#!/usr/bin/env python3
"""
Script to migrate checkpoints from old logs/ structure to new outputs/ structure.

This script copies checkpoints from:
  logs/log/flow/epures_classifier_free/multinomial_diffusion/multistep/YYYY-MM-DD_HH-MM/72/
to:
  outputs/multinomial_diffusion/epures_classifier_free/YYYY-MM-DD_HH-MM/
"""

import os
import shutil
from pathlib import Path

# Project root
project_root = Path(__file__).parent.parent

# Old logs path
old_logs_base = project_root / 'src' / 'multinomial_diffusion' / 'segmentation_diffusion' / 'logs' / 'log' / 'flow'

# New outputs path
new_outputs_base = project_root / 'outputs' / 'multinomial_diffusion'


def migrate_checkpoints():
    """Migrate checkpoints from old structure to new structure."""
    print("=" * 70)
    print("MIGRATION DES CHECKPOINTS")
    print("=" * 70)
    
    # Find all experiment directories
    if not old_logs_base.exists():
        print(f"Old logs directory not found: {old_logs_base}")
        return
    
    # Look for epures_classifier_free experiments
    epures_cf_path = old_logs_base / 'epures_classifier_free' / 'multinomial_diffusion' / 'multistep'
    
    if not epures_cf_path.exists():
        print(f"Source directory not found: {epures_cf_path}")
        return
    
    # Find all experiment directories
    experiments = [d for d in epures_cf_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\nFound {len(experiments)} experiments to migrate")
    
    for exp_dir in experiments:
        if exp_dir.name == 'delete':
            continue
        
        # Look for subfolder (usually "72")
        subfolders = [d for d in exp_dir.iterdir() if d.is_dir()]
        
        for subfolder in subfolders:
            # Check if it contains checkpoints
            check_dir = subfolder / 'check'
            if not check_dir.exists():
                continue
            
            checkpoints = list(check_dir.glob('checkpoint_*.pt'))
            if not checkpoints:
                continue
            
            # Create new directory structure
            new_exp_dir = new_outputs_base / 'epures_classifier_free' / exp_dir.name
            new_check_dir = new_exp_dir / 'check'
            
            print(f"\nMigrating: {exp_dir.name}")
            print(f"  From: {subfolder}")
            print(f"  To: {new_exp_dir}")
            
            # Create directories
            new_check_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy checkpoints
            for checkpoint in checkpoints:
                dest = new_check_dir / checkpoint.name
                if not dest.exists():
                    shutil.copy2(checkpoint, dest)
                    print(f"    Copied: {checkpoint.name}")
                else:
                    print(f"    Skipped (exists): {checkpoint.name}")
            
            # Copy args.pickle and other files
            for file in ['args.pickle', 'args_table.txt', 'metrics_train.pickle', 
                        'metrics_eval.pickle', 'metrics_train.txt', 'metrics_eval.txt']:
                src = subfolder / file
                if src.exists():
                    dest = new_exp_dir / file
                    if not dest.exists():
                        shutil.copy2(src, dest)
                        print(f"    Copied: {file}")
    
    print("\n" + "=" * 70)
    print("MIGRATION TERMINÉE!")
    print("=" * 70)
    print(f"Checkpoints migrés vers: {new_outputs_base}")


if __name__ == '__main__':
    migrate_checkpoints()

