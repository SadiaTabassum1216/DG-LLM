"""
Example script to run multi-seed experiments with statistical analysis.
This demonstrates how to use the statistical rigor framework.
"""

import subprocess
import sys
import os


def parse_horizon_pairs(horizon_pairs):
    """Parse horizon strings like ['12:12', '48:48'] into list of (input_len, output_len)."""
    parsed = []
    for pair in horizon_pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid horizon pair '{pair}'. Expected format 'input:output', e.g. '48:48'.")
        input_str, output_str = pair.split(":", 1)
        input_len = int(input_str)
        output_len = int(output_str)
        if input_len <= 0 or output_len <= 0:
            raise ValueError(f"Invalid horizon pair '{pair}'. Lengths must be positive integers.")
        parsed.append((input_len, output_len))
    return parsed


def build_log_dir(base_log_dir, dataset, input_len, output_len, mode):
    """Build a deterministic log directory per dataset/horizon/mode to avoid overwrite."""
    run_name = f"{dataset}_{input_len}in_{output_len}out_{mode}"
    return os.path.join(base_log_dir, run_name)

def run_multiseed_experiment(
    dataset='PEMSD04',
    num_seeds=5,
    epochs=50,
    batch_size=8,
    input_len=12,
    output_len=12,
    base_log_dir='./logs'
):
    """
    Run a multi-seed experiment with statistical analysis.
    
    Args:
        dataset: Dataset name
        num_seeds: Number of random seeds to run
        epochs: Number of training epochs per seed
        batch_size: Batch size
        input_len: Input sequence length
        output_len: Output sequence length
        base_log_dir: Root logging directory
    """
    log_dir = build_log_dir(base_log_dir, dataset, input_len, output_len, mode='multi')
    cmd = [
        sys.executable,  # python
        'main.py',
        '--data', dataset,
        '--num_seeds', str(num_seeds),
        '--seed_start', '42',
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--input_len', str(input_len),
        '--output_len', str(output_len),
        '--log_dir', log_dir,
        '--save_stats',
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_single_seed_experiment(
    dataset='PEMSD04',
    seed=42,
    epochs=50,
    batch_size=8,
    input_len=12,
    output_len=12,
    base_log_dir='./logs'
):
    """
    Run a single-seed experiment (original behavior).
    
    Args:
        dataset: Dataset name
        seed: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        input_len: Input sequence length
        output_len: Output sequence length
        base_log_dir: Root logging directory
    """
    log_dir = build_log_dir(base_log_dir, dataset, input_len, output_len, mode='single')
    cmd = [
        sys.executable,
        'main.py',
        '--data', dataset,
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--input_len', str(input_len),
        '--output_len', str(output_len),
        '--log_dir', log_dir,
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
    parser.add_argument('--input_len', type=int, default=12,
                        help='Input sequence length for single run (default: 12)')
    parser.add_argument('--output_len', type=int, default=12,
                        help='Output sequence length for single run (default: 12)')
    parser.add_argument('--horizon_pairs', type=str, nargs='*', default=None,
                        help="Optional list of horizon pairs for sweep, e.g. --horizon_pairs 12:12 48:48 96:96")
    parser.add_argument('--base_log_dir', type=str, default='./logs',
                        help='Base directory for run logs/checkpoints (default: ./logs)')
    
    args = parser.parse_args()
    
    if args.horizon_pairs:
        horizons = parse_horizon_pairs(args.horizon_pairs)
    else:
        horizons = [(args.input_len, args.output_len)]

    print(f"\n{'='*70}")
    print(f"  Running {args.mode.capitalize()} Experiment")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    if args.mode == 'multi':
        print(f"  Seeds: {args.num_seeds}")
    print(f"  Horizons: {', '.join([f'{i}:{o}' for i, o in horizons])}")
    print(f"{'='*70}\n")

    for input_len, output_len in horizons:
        print(f"\n--- Running horizon {input_len}:{output_len} ---")
        if args.mode == 'multi':
            run_multiseed_experiment(
                dataset=args.dataset,
                num_seeds=args.num_seeds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                input_len=input_len,
                output_len=output_len,
                base_log_dir=args.base_log_dir,
            )
        else:
            run_single_seed_experiment(
                dataset=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                input_len=input_len,
                output_len=output_len,
                base_log_dir=args.base_log_dir,
            )
    
    print(f"\n{'='*70}")
    print(f"  Experiment Complete!")
    print(f"{'='*70}\n")
