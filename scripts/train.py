#!/usr/bin/env python3
"""
Global training dispatcher.

Routes training requests to model-specific training scripts.

Usage:
    python scripts/train.py --model multinomial --epochs 1000 --batch_size 64
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Model-specific training modules
MODELS = {
    'multinomial': 'models.multinomial.train',
}


def main():
    parser = argparse.ArgumentParser(description='Global training dispatcher')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODELS.keys()),
                       help='Model to train')
    
    args, remaining_args = parser.parse_known_args()
    
    # Import and run model-specific training script
    module_name = MODELS[args.model]
    
    # Import the module
    if args.model == 'multinomial':
        from models.multinomial import train as train_module
        # Create new argv with remaining args
        sys.argv = [sys.argv[0]] + remaining_args
        train_module.main()
    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == '__main__':
    main()

