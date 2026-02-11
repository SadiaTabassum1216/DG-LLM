"""
Experiment utilities for multi-seed experiments and statistical analysis.
Addresses reviewer concerns about statistical rigor and reproducibility.
"""

import torch
import numpy as np
import random
import os
import json
from scipy import stats
from typing import Dict, List, Tuple, Any
import pickle


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_with_seeds(
    train_fn,
    seeds: List[int],
    save_dir: str,
    experiment_name: str,
    **kwargs
) -> Dict[str, List[float]]:
    """
    Run experiment across multiple seeds and aggregate results.
    
    Args:
        train_fn: Training function that takes seed and kwargs, returns metrics dict
        seeds: List of random seeds to use
        save_dir: Directory to save per-seed results
        experiment_name: Name of experiment for logging
        **kwargs: Additional arguments to pass to train_fn
    
    Returns:
        Dictionary mapping metric names to lists of values across seeds
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"  Multi-Seed Experiment: {experiment_name}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*70}\n")
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'─'*70}")
        print(f"  Seed {seed_idx}/{len(seeds)}: {seed}")
        print(f"{'─'*70}")
        
        # Set seed
        seed_everything(seed)
        
        # Run training
        seed_results = train_fn(seed=seed, **kwargs)
        
        # Save per-seed results
        seed_file = os.path.join(save_dir, f"{experiment_name}_seed_{seed}.json")
        with open(seed_file, 'w') as f:
            json.dump(seed_results, f, indent=2)
        
        # Accumulate results
        for metric_name, value in seed_results.items():
            if metric_name not in all_results:
                all_results[metric_name] = []
            all_results[metric_name].append(value)
        
        print(f"\n  Seed {seed} Results:")
        for metric_name, value in seed_results.items():
            if isinstance(value, (int, float)):
                print(f"    {metric_name}: {value:.4f}")
    
    return all_results


def compute_statistics(
    results: Dict[str, List[float]],
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical measures across multiple runs.
    
    Args:
        results: Dictionary mapping metric names to lists of values
        confidence_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        Dictionary with mean, std, CI lower/upper bounds for each metric
    """
    stats_dict = {}
    alpha = 1 - confidence_level
    
    for metric_name, values in results.items():
        if not isinstance(values[0], (int, float)):
            continue
            
        values = np.array(values)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample std
        sem = std / np.sqrt(n)  # Standard error of mean
        
        # Compute confidence interval using t-distribution
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        ci_lower = mean - t_critical * sem
        ci_upper = mean + t_critical * sem
        
        stats_dict[metric_name] = {
            'mean': float(mean),
            'std': float(std),
            'sem': float(sem),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n': int(n),
            'raw_values': values.tolist()
        }
    
    return stats_dict


def paired_t_test(
    results_a: List[float],
    results_b: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float, float]:
    """
    Perform paired t-test to compare two sets of results.
    
    Args:
        results_a: Results from method A (same seeds)
        results_b: Results from method B (same seeds)
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Tuple of (t_statistic, p_value, effect_size)
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(results_a, results_b, alternative=alternative)
    
    # Compute Cohen's d (effect size)
    diff = results_a - results_b
    effect_size = np.mean(diff) / np.std(diff, ddof=1)
    
    return float(t_stat), float(p_value), float(effect_size)


def statistical_comparison(
    baseline_results: Dict[str, List[float]],
    method_results: Dict[str, List[float]],
    metrics_to_compare: List[str],
    alpha: float = 0.05,
    better_lower: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare method against baseline with statistical significance testing.
    
    Args:
        baseline_results: Results from baseline method
        method_results: Results from proposed method
        metrics_to_compare: List of metric names to compare
        alpha: Significance threshold (default: 0.05)
        better_lower: List of metrics where lower is better (e.g., MAE, RMSE)
    
    Returns:
        Dictionary with comparison statistics for each metric
    """
    if better_lower is None:
        better_lower = ['mae', 'rmse', 'mape', 'loss']
    
    comparison = {}
    
    for metric in metrics_to_compare:
        if metric not in baseline_results or metric not in method_results:
            continue
        
        baseline_vals = baseline_results[metric]
        method_vals = method_results[metric]
        
        # Determine test direction
        is_lower_better = any(lb.lower() in metric.lower() for lb in better_lower)
        alternative = 'less' if is_lower_better else 'greater'
        
        # Perform statistical test
        t_stat, p_value, effect_size = paired_t_test(
            method_vals, baseline_vals, alternative=alternative
        )
        
        # Compute descriptive statistics
        baseline_stats = compute_statistics({metric: baseline_vals})
        method_stats = compute_statistics({metric: method_vals})
        
        # Determine significance
        is_significant = p_value < alpha
        
        # Compute improvement percentage
        improvement = (
            (baseline_stats[metric]['mean'] - method_stats[metric]['mean']) 
            / baseline_stats[metric]['mean'] * 100
        )
        
        comparison[metric] = {
            'baseline_mean': baseline_stats[metric]['mean'],
            'baseline_std': baseline_stats[metric]['std'],
            'baseline_ci': [baseline_stats[metric]['ci_lower'], baseline_stats[metric]['ci_upper']],
            'method_mean': method_stats[metric]['mean'],
            'method_std': method_stats[metric]['std'],
            'method_ci': [method_stats[metric]['ci_lower'], method_stats[metric]['ci_upper']],
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'is_significant': is_significant,
            'improvement_percent': improvement,
            'significance_marker': get_significance_marker(p_value)
        }
    
    return comparison


def get_significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def format_result_with_ci(mean: float, std: float, ci_lower: float, ci_upper: float) -> str:
    """Format result as: mean ± std [CI_lower, CI_upper]"""
    return f"{mean:.4f} ± {std:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"


def save_statistical_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = 'json'
) -> None:
    """
    Save statistical results to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save file
        format: 'json' or 'pickle'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == 'json':
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"\n✓ Statistical results saved to: {save_path}")


def generate_latex_table(
    comparison_results: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: List[str],
    methods: List[str],
    caption: str = "Performance comparison with statistical significance",
    label: str = "tab:results"
) -> str:
    """
    Generate LaTeX table from statistical results.
    
    Args:
        comparison_results: Nested dict {method: {metric: stats}}
        metrics: List of metric names (columns)
        methods: List of method names (rows)
        caption: Table caption
        label: LaTeX label
    
    Returns:
        LaTeX table string
    """
    n_cols = len(metrics) + 1
    col_format = 'l' + 'c' * len(metrics)
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex.append("\\toprule")
    
    # Header
    header = "Method & " + " & ".join(metrics) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # Rows
    for method in methods:
        row = [method]
        for metric in metrics:
            if method in comparison_results and metric in comparison_results[method]:
                stats = comparison_results[method][metric]
                mean = stats['method_mean']
                std = stats['method_std']
                sig = stats.get('significance_marker', '')
                row.append(f"{mean:.2f} $\\pm$ {std:.2f}{sig}")
            else:
                row.append("--")
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\multicolumn{" + str(n_cols) + "}{l}{")
    latex.append("\\footnotesize $***$: $p < 0.001$, $**$: $p < 0.01$, $*$: $p < 0.05$")
    latex.append("} \\\\")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def print_statistical_report(
    stats: Dict[str, Dict[str, float]],
    title: str = "Statistical Report"
) -> None:
    """Print formatted statistical report to console."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    
    for metric, values in stats.items():
        print(f"\n{metric}:")
        print(f"  Mean:       {values['mean']:.6f}")
        print(f"  Std:        {values['std']:.6f}")
        print(f"  SEM:        {values['sem']:.6f}")
        print(f"  95% CI:     [{values['ci_lower']:.6f}, {values['ci_upper']:.6f}]")
        print(f"  n:          {values['n']}")


def print_comparison_report(
    comparison: Dict[str, Dict[str, Any]],
    method_name: str = "Proposed Method",
    baseline_name: str = "Baseline"
) -> None:
    """Print formatted comparison report to console."""
    print(f"\n{'='*70}")
    print(f"  Statistical Comparison: {method_name} vs {baseline_name}")
    print(f"{'='*70}")
    
    for metric, results in comparison.items():
        print(f"\n{metric.upper()}:")
        print(f"  {baseline_name:20s}: {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
        print(f"  {method_name:20s}: {results['method_mean']:.4f} ± {results['method_std']:.4f}")
        print(f"  Improvement:         {results['improvement_percent']:+.2f}%")
        print(f"  t-statistic:         {results['t_statistic']:.4f}")
        print(f"  p-value:             {results['p_value']:.6f} {results['significance_marker']}")
        print(f"  Effect size (d):     {results['effect_size']:.4f}")
        
        if results['is_significant']:
            print(f"  ✓ Statistically significant at α = 0.05")
        else:
            print(f"  ✗ Not statistically significant at α = 0.05")
