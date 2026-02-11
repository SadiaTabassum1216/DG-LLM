"""
Example script to run multi-seed experiments with statistical analysis.
This demonstrates how to use the statistical rigor framework.
"""

import subprocess
import sys
import os

def run_multiseed_experiment(
    dataset='PEMSD04',
    num_seeds=5,
    epochs=50,
    batch_size=8
):
    """
    Run a multi-seed experiment with statistical analysis.
    
    Args:
        dataset: Dataset name
        num_seeds: Number of random seeds to run
        epochs: Number of training epochs per seed
        batch_size: Batch size
    """
    cmd = [
        sys.executable,  # python
        'main.py',
        '--data', dataset,
        '--num_seeds', str(num_seeds),
        '--seed_start', '42',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--save_stats'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_single_seed_experiment(
    dataset='PEMSD04',
    seed=42,
    epochs=50,
    batch_size=8
):
    """
    Run a single-seed experiment (original behavior).
    
    Args:
        dataset: Dataset name
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
    """
    cmd = [
        sys.executable,
        'main.py',
        '--data', dataset,
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run DG-LLM experiments with statistical rigor')
    parser.add_argument('--mode', type=str, default='multi', choices=['single', 'multi'],
                        help='Experiment mode: single or multi-seed')
    parser.add_argument('--dataset', type=str, default='PEMSD04',
                        help='Dataset name')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of seeds for multi-seed mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    
    args = parser.parse_args()
    
    if args.mode == 'multi':
        print(f"\n{'='*70}")
        print(f"  Running Multi-Seed Experiment")
        print(f"  Dataset: {args.dataset}")
        print(f"  Seeds: {args.num_seeds}")
        print(f"  Epochs per seed: {args.epochs}")
        print(f"{'='*70}\n")
        
        run_multiseed_experiment(
            dataset=args.dataset,
            num_seeds=args.num_seeds,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    else:
        print(f"\n{'='*70}")
        print(f"  Running Single-Seed Experiment")
        print(f"  Dataset: {args.dataset}")
        print(f"  Epochs: {args.epochs}")
        print(f"{'='*70}\n")
        
        run_single_seed_experiment(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    print(f"\n{'='*70}")
    print(f"  Experiment Complete!")
    print(f"{'='*70}\n")
