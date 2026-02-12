"""
Aggregate Results from Multiple Single-Seed Runs

After running training with different seeds separately (e.g., on Kaggle),
use this script to combine results and compute statistical significance.

Usage:
    python aggregate_seeds.py --results_dir results/taxi_drop
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats


def compute_statistics(values, confidence_level=0.95):
    """Compute mean, std, and confidence interval."""
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    se = std / np.sqrt(n)  # Standard error
    
    # Confidence interval using t-distribution
    t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
    ci_lower = mean - t_val * se
    ci_upper = mean + t_val * se
    
    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n': n
    }


def load_seed_results(results_dir):
    """Load results from multiple seed directories."""
    results_dir = Path(results_dir)
    
    all_results = {}
    seed_files = list(results_dir.glob('seed_*/results.json'))
    
    if not seed_files:
        print(f"❌ No results found in {results_dir}")
        print(f"   Expected files like: seed_42/results.json")
        return None
    
    print(f"Found {len(seed_files)} seed results:")
    
    for seed_file in sorted(seed_files):
        seed = seed_file.parent.name  # e.g., "seed_42"
        print(f"  - {seed}")
        
        with open(seed_file, 'r') as f:
            data = json.load(f)
        
        # Aggregate metrics
        for metric, value in data.items():
            if isinstance(value, (int, float)):
                if metric not in all_results:
                    all_results[metric] = []
                all_results[metric].append(value)
    
    return all_results


def print_statistics(stats_dict, num_seeds):
    """Pretty print statistical results."""
    print("\n" + "="*70)
    print(f"  STATISTICAL ANALYSIS ({num_seeds} seeds)")
    print("="*70)
    
    print(f"\n{'Metric':<15} | {'Mean':<12} | {'Std':<12} | {'95% CI':<25}")
    print("-"*70)
    
    for metric in ['mae', 'rmse', 'mape']:
        if metric in stats_dict:
            s = stats_dict[metric]
            ci_str = f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
            print(f"{metric.upper():<15} | {s['mean']:<12.4f} | {s['std']:<12.4f} | {ci_str:<25}")
    
    print("="*70)


def save_aggregated_results(stats_dict, output_file):
    """Save aggregated statistics to JSON."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    serializable = {}
    for metric, stats_values in stats_dict.items():
        serializable[metric] = {
            k: float(v) if isinstance(v, np.floating) else v 
            for k, v in stats_values.items()
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n✓ Aggregated results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed results")
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing seed_*/results.json files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for aggregated results (default: results_dir/aggregated_stats.json)')
    
    args = parser.parse_args()
    
    # Load results
    all_results = load_seed_results(args.results_dir)
    
    if all_results is None:
        return
    
    # Compute statistics
    stats_dict = {}
    for metric, values in all_results.items():
        stats_dict[metric] = compute_statistics(values, confidence_level=0.95)
    
    # Print results
    num_seeds = len(next(iter(all_results.values())))
    print_statistics(stats_dict, num_seeds)
    
    # Save aggregated results
    if args.output is None:
        args.output = Path(args.results_dir) / 'aggregated_stats.json'
    save_aggregated_results(stats_dict, args.output)
    
    print("\n" + "="*70)
    print("  Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
