#!/usr/bin/env python3
"""
Standalone inference script for Low-Rank Transformer models
"""

import argparse
import torch
from train import load_joint, ar_decode, set_seed, DECODE_PRESETS

def main():
    parser = argparse.ArgumentParser(description='Inference for Low-Rank Transformer')
    parser.add_argument('--ckpt', required=True, help='Model checkpoint path')
    parser.add_argument('--preset', default='base17', choices=['small', 'smallx2', 'base', 'base17'])
    parser.add_argument('--prompt', required=True, help='Input prompt')
    parser.add_argument('--max_new', type=int, default=256, help='Max new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--decode_preset', choices=['det', 'balanced', 'creative'], default='balanced')
    parser.add_argument('--fp8_only', action='store_true', help='Use FP8 precision')
    
    args = parser.parse_args()
    
    # Load model
    core, ar_head = load_joint(args.ckpt, args.preset)
    set_seed(args.seed)
    
    # Apply decode preset
    preset_config = DECODE_PRESETS[args.decode_preset]
    
    # Run generation
    ar_decode(
        core, ar_head, 
        args.prompt, args.max_new,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_fp8=args.fp8_only,
        **preset_config
    )

if __name__ == '__main__':
    main()
