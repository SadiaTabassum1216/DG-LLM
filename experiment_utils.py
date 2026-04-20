"""
Experiment utilities for reproducibility and statistical reporting.
"""

import json
import os
import pickle
import random
from typing import Any, Dict, List

import numpy as np
import torch
from scipy import stats


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_statistics(
    results: Dict[str, List[float]],
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """Compute descriptive statistics and confidence intervals across runs."""
    stats_dict = {}
    alpha = 1 - confidence_level

    for metric_name, values in results.items():
        if not values or not isinstance(values[0], (int, float)):
            continue

        values = np.array(values, dtype=float)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        sem = std / np.sqrt(n)

        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_lower = mean - t_critical * sem
        ci_upper = mean + t_critical * sem

        stats_dict[metric_name] = {
            "mean": float(mean),
            "std": float(std),
            "sem": float(sem),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n": int(n),
            "raw_values": values.tolist(),
        }

    return stats_dict


def get_significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def save_statistical_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = "json",
) -> None:
    """Save statistical results to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if format == "json":
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    elif format == "pickle":
        with open(save_path, "wb") as handle:
            pickle.dump(results, handle)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"\nSaved statistical results to: {save_path}")


def generate_latex_table(
    comparison_results: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: List[str],
    methods: List[str],
    caption: str = "Performance comparison with statistical significance",
    label: str = "tab:results",
) -> str:
    """Generate a LaTeX table from statistical comparison results."""
    n_cols = len(metrics) + 1
    col_format = "l" + "c" * len(metrics)

    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
        "Method & " + " & ".join(metrics) + " \\\\",
        "\\midrule",
    ]

    for method in methods:
        row = [method]
        for metric in metrics:
            if method in comparison_results and metric in comparison_results[method]:
                stat_row = comparison_results[method][metric]
                mean = stat_row["method_mean"]
                std = stat_row["method_std"]
                sig = stat_row.get("significance_marker", "")
                row.append(f"{mean:.2f} $\\pm$ {std:.2f}{sig}")
            else:
                row.append("--")
        latex.append(" & ".join(row) + " \\\\")

    latex.extend(
        [
            "\\bottomrule",
            "\\multicolumn{" + str(n_cols) + "}{l}{",
            "\\footnotesize $***$: $p < 0.001$, $**$: $p < 0.01$, $*$: $p < 0.05$",
            "} \\\\",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(latex)


def print_statistical_report(
    stats_dict: Dict[str, Dict[str, float]],
    title: str = "Statistical Report",
) -> None:
    """Print a formatted statistical report to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    for metric, values in stats_dict.items():
        print(f"\n{metric}:")
        print(f"  Mean:       {values['mean']:.6f}")
        print(f"  Std:        {values['std']:.6f}")
        print(f"  SEM:        {values['sem']:.6f}")
        print(f"  95% CI:     [{values['ci_lower']:.6f}, {values['ci_upper']:.6f}]")
        print(f"  n:          {values['n']}")
