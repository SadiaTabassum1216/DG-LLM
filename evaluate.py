"""
Enhanced evaluation module with statistical rigor.
Provides comprehensive evaluation with multi-seed support and statistical analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from utils import MAE_torch, MAPE_torch, RMSE_torch
from experiment_utils import compute_statistics, print_statistical_report


def evaluate_model_statistical(
    trainer,
    dataloader,
    device,
    scaler,
    output_len: int,
    num_seeds: int = 1,
    current_seed: int = None
) -> Dict[str, float]:
    """
    Evaluate model and return comprehensive metrics.
    
    Args:
        trainer: Model trainer instance
        dataloader: Test data loader
        device: Torch device
        scaler: Data scaler for inverse transform
        output_len: Prediction horizon length
        num_seeds: Total number of seeds (for logging)
        current_seed: Current seed value (for logging)
    
    Returns:
        Dictionary containing average MAE, RMSE, MAPE and per-horizon metrics
    """
    trainer.model.eval()
    
    # Per-horizon metrics storage
    horizon_mae = [[] for _ in range(output_len)]
    horizon_mape = [[] for _ in range(output_len)]
    horizon_rmse = [[] for _ in range(output_len)]
    
    # Overall metrics storage
    all_maes = []
    all_mapes = []
    all_rmses = []
    
    with torch.no_grad():
        for x, y, vmd in dataloader.get_iterator():
            tx = torch.Tensor(x).to(device).transpose(1, 3)
            ty = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
            tvmd = torch.Tensor(vmd).to(device)
            
            x_in = tx.permute(0, 3, 2, 1)  # [B, T, N, F]
            preds, _ = trainer.model(tvmd, x_in)
            
            # Scale back to original range
            preds_scaled = scaler.inverse_transform(preds)
            real_scaled = ty.permute(0, 2, 1).unsqueeze(-1)
            
            # Compute per-horizon metrics
            for t in range(output_len):
                p = preds_scaled[:, t, ...]
                r = real_scaled[:, t, ...]
                
                horizon_mae[t].append(MAE_torch(p, r, 0).item())
                horizon_mape[t].append(MAPE_torch(p, r, 0).item())
                horizon_rmse[t].append(RMSE_torch(p, r, 0).item())
            
            # Compute overall metrics for this batch
            all_maes.append(MAE_torch(preds_scaled, real_scaled, 0).item())
            all_mapes.append(MAPE_torch(preds_scaled, real_scaled, 0).item())
            all_rmses.append(RMSE_torch(preds_scaled, real_scaled, 0).item())
    
    # Aggregate per-horizon metrics
    horizon_mae_avg = [np.mean(h) for h in horizon_mae]
    horizon_mape_avg = [np.mean(h) for h in horizon_mape]
    horizon_rmse_avg = [np.mean(h) for h in horizon_rmse]
    
    # Overall average metrics
    avg_mae = np.mean(all_maes)
    avg_mape = np.mean(all_mapes)
    avg_rmse = np.mean(all_rmses)
    
    # Print results
    if current_seed is not None:
        print(f"\n  Evaluation Results (Seed {current_seed}):")
    else:
        print(f"\n  Evaluation Results:")
    
    print(f"    Overall MAE:  {avg_mae:.4f}")
    print(f"    Overall RMSE: {avg_rmse:.4f}")
    print(f"    Overall MAPE: {avg_mape:.4f}")
    
    # Return comprehensive results
    results = {
        'mae': avg_mae,
        'rmse': avg_rmse,
        'mape': avg_mape,
        'horizon_mae': horizon_mae_avg,
        'horizon_mape': horizon_mape_avg,
        'horizon_rmse': horizon_rmse_avg
    }
    
    return results


def test_model_with_statistics(
    trainer,
    dataloader_dict,
    device,
    model_path: str,
    output_len: int,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Test model with detailed per-horizon statistics.
    
    Args:
        trainer: Model trainer
        dataloader_dict: Dictionary containing test_loader and scaler
        device: Torch device
        model_path: Path to saved model weights
        output_len: Prediction horizon length
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with test metrics
    """
    # Load model
    if verbose:
        print(f"\n>> Loading model from {model_path}...")
    trainer.model.load_state_dict(torch.load(model_path, weights_only=False))
    trainer.model.eval()
    
    # Evaluate
    results = evaluate_model_statistical(
        trainer,
        dataloader_dict['test_loader'],
        device,
        dataloader_dict['scaler'],
        output_len,
        num_seeds=1,
        current_seed=None
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{'Horizon':<10} | {'MAE':<10} | {'MAPE':<10} | {'RMSE':<10}")
        print(f"{'-'*60}")
        
        for i in range(output_len):
            print(f"Step {i+1:02d}    | "
                  f"{results['horizon_mae'][i]:<10.4f} | "
                  f"{results['horizon_mape'][i]:<10.4f} | "
                  f"{results['horizon_rmse'][i]:<10.4f}")
        
        print(f"{'-'*60}")
        print(f"AVERAGE    | "
              f"{results['mae']:<10.4f} | "
              f"{results['mape']:<10.4f} | "
              f"{results['rmse']:<10.4f}")
        print(f"{'='*60}")
    
    return results


def aggregate_multi_seed_results(
    all_results: Dict[str, List[float]],
    save_path: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple seeds into statistical measures.
    
    Args:
        all_results: Dictionary mapping metric names to lists of values across seeds
        save_path: Optional path to save aggregated results
    
    Returns:
        Statistical measures for each metric
    """
    stats = compute_statistics(all_results, confidence_level=0.95)
    
    # Print report
    print_statistical_report(stats, title="Multi-Seed Results")
    
    # Save if path provided
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✓ Results saved to: {save_path}")
    
    return stats
