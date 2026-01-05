#!/usr/bin/env python3
"""
Global sampling dispatcher.

Routes sampling requests to model-specific sampling scripts.

Usage:
    python scripts/sample.py --model multinomial --checkpoint outputs/.../checkpoint.pt --mode conditional
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Model-specific sampling modules
MODELS = {
    'multinomial': 'models.multinomial.sample',
}


def main():
    parser = argparse.ArgumentParser(description='Global sampling dispatcher')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODELS.keys()),
                       help='Model to sample from')
    
    args, remaining_args = parser.parse_known_args()
    
    # Import and run model-specific sampling script
    if args.model == 'multinomial':
        from models.multinomial import sample as sample_module
        # Create new argv with remaining args
        sys.argv = [sys.argv[0]] + remaining_args
        sample_module.main()
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == '__main__':
    main()

