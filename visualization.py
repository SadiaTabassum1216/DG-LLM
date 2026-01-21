import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import os

def visualize_model_predictions(preds, reals, node_idx=0, horizon_idx=0, num_samples=288, save_dir='./logs'):
    """Compare Predicted vs Real values for a specific node and horizon."""
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

def visualize_advanced_diagnostics(preds, reals, node_idx=0, save_dir='./logs'):
    """Plot Error Distribution and Prediction vs Truth Scatter."""
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

def visualize_weekly_horizon1(preds, reals, node_idx=0, save_dir='./logs'):
    """Zoomed in view of 1-week of forecasts."""
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

def verify_temporal_features(data_loader):
    """Sanity check for TOD and DOW features."""
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
