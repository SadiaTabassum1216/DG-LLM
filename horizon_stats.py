"""
Utilities for aggregating per-horizon metrics across multiple seeds.
"""

import numpy as np
from typing import Dict, List
from scipy import stats


def aggregate_per_horizon_metrics(
    all_horizon_results: Dict[str, List[List[float]]],
    confidence_level: float = 0.95
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Aggregate per-horizon metrics across multiple seeds.
    
    Args:
        all_horizon_results: Dict mapping metric names to list of per-seed horizon lists
                            e.g., {'horizon_mae': [[seed1_h1, seed1_h2, ...], [seed2_h1, seed2_h2, ...], ...]}
        confidence_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        Dict with structure: {metric: {horizon_idx: {mean, std, ci_lower, ci_upper}}}
    """
    aggregated = {}
    alpha = 1 - confidence_level
    
    for metric_name, seed_horizons in all_horizon_results.items():
        # seed_horizons: list of lists, where each inner list is horizons for one seed
        # Convert to array: [num_seeds, num_horizons]
        horizon_array = np.array(seed_horizons)
        num_seeds, num_horizons = horizon_array.shape
        
        aggregated[metric_name] = {}
        
        for h_idx in range(num_horizons):
            # Get all values for this horizon across seeds
            horizon_values = horizon_array[:, h_idx]
            
            mean = np.mean(horizon_values)
            std = np.std(horizon_values, ddof=1)
            sem = std / np.sqrt(num_seeds)
            
            # Compute confidence interval
            t_critical = stats.t.ppf(1 - alpha/2, df=num_seeds-1)
            ci_lower = mean - t_critical * sem
            ci_upper = mean + t_critical * sem
            
            aggregated[metric_name][h_idx] = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            }
    
    return aggregated


def print_per_horizon_statistics(
    horizon_stats: Dict[str, Dict[int, Dict[str, float]]],
    num_seeds: int,
    metrics: List[str] = None
) -> None:
    """
    Print formatted per-horizon statistics table.
    
    Args:
        horizon_stats: Aggregated horizon statistics
        num_seeds: Number of seeds
        metrics: List of metrics to display (default: ['horizon_mae', 'horizon_rmse', 'horizon_mape'])
    """
    if metrics is None:
        metrics = ['horizon_mae', 'horizon_rmse', 'horizon_mape']
    
    # Filter to requested metrics that exist
    metrics = [m for m in metrics if m in horizon_stats]
    
    if not metrics:
        return
    
    print(f"\n{'='*95}")
    print(f"  PER-HORIZON STATISTICS (averaged across {num_seeds} seeds)")
    print(f"{'='*95}")
    
    # Determine number of horizons
    num_horizons = len(horizon_stats[metrics[0]])
    
    # Print header
    header = f"{'Horizon':<10}"
    for metric in metrics:
        metric_label = metric.replace('horizon_', '').upper()
        header += f" | {metric_label + ' (Mean±Std)':<25}"
    print(f"\n{header}")
    print(f"{'-'*95}")
    
    # Print each horizon
    for h_idx in range(num_horizons):
        row = f"Step {h_idx+1:02d}   "
        for metric in metrics:
            stats_dict = horizon_stats[metric][h_idx]
            mean = stats_dict['mean']
            std = stats_dict['std']
            row += f" | {mean:6.4f} ± {std:5.4f}        "
        print(row)
    
    # Print average across horizons
    print(f"{'-'*95}")
    avg_row = f"AVERAGE   "
    for metric in metrics:
        all_means = [horizon_stats[metric][h]['mean'] for h in range(num_horizons)]
        avg_mean = np.mean(all_means)
        all_stds = [horizon_stats[metric][h]['std'] for h in range(num_horizons)]
        avg_std = np.mean(all_stds)
        avg_row += f" | {avg_mean:6.4f} ± {avg_std:5.4f}        "
    print(avg_row)
    print(f"{'='*95}")


def save_per_horizon_statistics(
    horizon_stats: Dict[str, Dict[int, Dict[str, float]]],
    save_path: str
) -> None:
    """
    Save per-horizon statistics to JSON file.
    
    Args:
        horizon_stats: Aggregated horizon statistics
        save_path: Path to save JSON file
    """
    import json
    import os
    
    # Convert integer keys to strings for JSON serialization
    serializable = {}
    for metric, horizons in horizon_stats.items():
        serializable[metric] = {f"horizon_{h}": stats for h, stats in horizons.items()}
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n✓ Per-horizon statistics saved to: {save_path}")
