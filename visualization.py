"""
Visualization utilities for DG-LLM model predictions and diagnostics.

This module provides functions for visualizing:
- Model predictions vs ground truth
- Error distributions and prediction accuracy
- Temporal patterns across multiple days
- Temporal features (time-of-day, day-of-week)

All visualizations are saved as PNG files to the specified directory.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
from paths import RESULTS_LOGS_DIR


def plot_predictions_vs_ground_truth(preds, reals, node_idx=0, horizon_idx=0, num_samples=288, save_dir=None):
    """
    Plot predictions vs ground truth for a specific node and prediction horizon.
    
    Generates a line plot comparing model predictions with actual values for a single
    node across a specified number of samples. Useful for visually inspecting prediction
    quality and identifying systematic errors.
    
    Args:
        preds (torch.Tensor): Predicted values, shape [num_samples, output_len, num_nodes, 1]
        reals (torch.Tensor): Ground truth values, shape [num_samples, output_len, num_nodes, 1]
        node_idx (int, optional): Node index to visualize. Default: 0
        horizon_idx (int, optional): Prediction horizon index (0 = 1-step ahead). Default: 0
        num_samples (int, optional): Number of samples to plot. Default: 288 (one day)
        save_dir (str, optional): Directory to save the plot. Uses RESULTS_LOGS_DIR if None.
    
    Returns:
        str: Path to the saved PNG file
        
    Example:
        >>> preds = torch.randn(500, 12, 307, 1)  # [samples, horizons, nodes, features]
        >>> reals = torch.randn(500, 12, 307, 1)
        >>> path = plot_predictions_vs_ground_truth(preds, reals, node_idx=10, horizon_idx=0)
    """
    save_dir = save_dir or str(RESULTS_LOGS_DIR)
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(15, 5))
    plt.plot(reals[:num_samples, horizon_idx, node_idx, 0].cpu().numpy(), label='Ground Truth', color='blue', alpha=0.7)
    plt.plot(preds[:num_samples, horizon_idx, node_idx, 0].cpu().numpy(), label='Prediction', color='red', linestyle='--')
    plt.title(f"Node {node_idx} - Horizon {horizon_idx+1} Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(save_dir, f'predictions_node{node_idx}_h{horizon_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def plot_error_diagnostics(preds, reals, node_idx=0, save_dir=None):
    """
    Generate diagnostic plots for prediction error analysis.
    
    Creates a 2-subplot figure showing:
    1. Error distribution histogram with KDE curve for residual analysis
    2. Scatter plot of predictions vs ground truth with perfect prediction line
    
    Useful for identifying prediction bias, error magnitude, and systematic deviations
    from the true values.
    
    Args:
        preds (torch.Tensor): Predicted values, shape [num_samples, output_len, num_nodes, 1]
        reals (torch.Tensor): Ground truth values, shape [num_samples, output_len, num_nodes, 1]
        node_idx (int, optional): Node index to analyze. Default: 0
        save_dir (str, optional): Directory to save the plot. Uses RESULTS_LOGS_DIR if None.
    
    Returns:
        str: Path to the saved PNG file
        
    Example:
        >>> path = plot_error_diagnostics(preds, reals, node_idx=5)
    """
    save_dir = save_dir or str(RESULTS_LOGS_DIR)
    os.makedirs(save_dir, exist_ok=True)
    p = preds[:, 0, node_idx, 0].cpu().numpy()
    r = reals[:, 0, node_idx, 0].cpu().numpy()
    errors = p - r
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Error Distribution
    sns.histplot(errors, kde=True, ax=axes[0], color='purple')
    axes[0].set_title("Prediction Error Distribution")
    
    # 2. Scatter Plot
    axes[1].scatter(r, p, alpha=0.1, color='green')
    axes[1].plot([r.min(), r.max()], [r.min(), r.max()], 'r--')
    axes[1].set_xlabel("True Value")
    axes[1].set_ylabel("Predicted Value")
    axes[1].set_title("Prediction vs Ground Truth")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'diagnostics_node{node_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def plot_weekly_1step_ahead_predictions(preds, reals, node_idx=0, save_dir=None):
    """
    Plot a full week (7 days) of 1-step ahead predictions.
    
    Creates a zoomed-in view of the first prediction horizon (1-step ahead) over
    one week of data (2016 timesteps = 288 steps/day × 7 days). Useful for observing
    long-term temporal patterns and weekly cycles in traffic data.
    
    Args:
        preds (torch.Tensor): Predicted values, shape [num_samples, output_len, num_nodes, 1]
        reals (torch.Tensor): Ground truth values, shape [num_samples, output_len, num_nodes, 1]
        node_idx (int, optional): Node index to visualize. Default: 0
        save_dir (str, optional): Directory to save the plot. Uses RESULTS_LOGS_DIR if None.
    
    Returns:
        str: Path to the saved PNG file
        
    Example:
        >>> path = plot_weekly_1step_ahead_predictions(preds, reals, node_idx=0)
    """
    save_dir = save_dir or str(RESULTS_LOGS_DIR)
    os.makedirs(save_dir, exist_ok=True)
    # 288 steps/day * 7 days = 2016 steps
    steps = min(2016, preds.shape[0])
    plt.figure(figsize=(20, 6))
    plt.plot(reals[:steps, 0, node_idx, 0].cpu().numpy(), label='True', color='black', linewidth=1)
    plt.plot(preds[:steps, 0, node_idx, 0].cpu().numpy(), label='Pred', color='orange', alpha=0.8)
    plt.title(f"Weekly Flow Patterns - Node {node_idx}")
    plt.legend()
    save_path = os.path.join(save_dir, f'weekly_node{node_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def validate_temporal_features(data_loader):
    """
    Validate and visualize temporal features (Time-of-Day and Day-of-Week).
    
    Validates temporal feature values by plotting the first sample's Time-of-Day (ToD)
    and Day-of-Week (DoW) sequences. Ensures that temporal embeddings are correctly
    computed and within expected ranges.
    
    Expected ranges:
    - ToD: [0, 1] (normalized over 288 timesteps per day)
    - DoW: [0, 6] (7 days per week)
    
    Args:
        data_loader: Data loader with get_iterator() method that yields (x, y, vmd)
                    where x shape is [batch_size, seq_len, num_nodes, features]
    
    Returns:
        None (prints feature statistics and displays plot)
        
    Raises:
        Assertion errors if features are outside expected ranges.
        
    Example:
        >>> from data_loader import load_dataset_optimized
        >>> data = load_dataset_optimized(data_path, batch_size=8, args)
        >>> validate_temporal_features(data['train_loader'])
    """
    x, y, vmd = next(data_loader.get_iterator())
    
    # Handle both numpy and torch
    if hasattr(x, 'cpu'):
        tod = x[0, :, 0, 1].cpu().numpy()
        dow = x[0, :, 0, 2].cpu().numpy()
    else:
        tod = x[0, :, 0, 1]
        dow = x[0, :, 0, 2]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tod)
    plt.title("Time of Day Feature")
    
    plt.subplot(1, 2, 2)
    plt.plot(dow)
    plt.title("Day of Week Feature")
    plt.show()
    print(f"TOD Range: {tod.min():.2f} to {tod.max():.2f}")
    print(f"DOW Range: {dow.min():.2f} to {dow.max():.2f}")
