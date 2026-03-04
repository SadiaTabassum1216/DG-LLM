"""
Curriculum Graph Ablation Experiment for DG-LLM Rebuttal

Tests the necessity of the curriculum-inspired graph blending strategy by
comparing three configurations:
  1. "curriculum" (default): mix_hi=0.6 → mix_lo=0.2 (static-heavy → dynamic-heavy)
  2. "pure_dynamic": mix_hi=0.0, mix_lo=0.0 (100% dynamic from epoch 1, no static prior)
  3. "pure_static": mix_hi=1.0, mix_lo=1.0 (100% static graph, dynamic graph ignored)
  4. "no_dynamic": use_dynamic_graph=False (completely disable dynamic graph module)

This proves the curriculum blending is necessary for training stability.

Usage:
    python curriculum_ablation.py --data taxi_drop --epochs 50
    python curriculum_ablation.py --data PEMSD04 --epochs 50
    python curriculum_ablation.py --data taxi_drop --epochs 5 --quick

Output:
    - Console: Training logs and comparison table
    - File: results/curriculum_ablation_<dataset>.json
    - Figure: rebuttal_figures/curriculum_ablation_<dataset>.png
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse
import time
import copy
from tqdm import tqdm

from data_loader import load_dataset_optimized
from model import DGLLM
from utils import load_pickle, Ranger, MAE_torch, MAPE_torch, RMSE_torch
from experiment_utils import seed_everything
from evaluate import evaluate_model_statistical


# =============================================================================
# Ablation Configurations
# =============================================================================

ABLATION_CONFIGS = {
    "curriculum": {
        "description": "Default curriculum blend (mix_hi=0.6 → mix_lo=0.2)",
        "mix_hi": 0.6,
        "mix_lo": 0.2,
        "use_dynamic_graph": True,
        "warmup_steps": 500,
    },
    "pure_dynamic": {
        "description": "Pure dynamic graph (mix=0.0, no static prior)",
        "mix_hi": 0.0,
        "mix_lo": 0.0,
        "use_dynamic_graph": True,
        "warmup_steps": 0,  # No warmup union with static either
    },
    "pure_static": {
        "description": "Pure static graph (mix=1.0, dynamic ignored)",
        "mix_hi": 1.0,
        "mix_lo": 1.0,
        "use_dynamic_graph": True,  # Still builds dynamic but blends 100% static
        "warmup_steps": 500,
    },
    "no_dynamic": {
        "description": "Dynamic graph completely disabled",
        "mix_hi": 0.6,
        "mix_lo": 0.2,
        "use_dynamic_graph": False,
        "warmup_steps": 500,
    },
}


def apply_graph_config(model, config):
    """Apply a graph ablation configuration to all ModeProcessors in the model."""
    for mode_proc in model.mode_models:
        mode_proc.use_dynamic_graph = config["use_dynamic_graph"]
        mode_proc.mix_hi = config["mix_hi"]
        mode_proc.mix_lo = config["mix_lo"]
        mode_proc.warmup_steps = config["warmup_steps"]
        # Reset graph state buffers
        mode_proc.global_step.zero_()
        mode_proc.ema_A.zero_()
        mode_proc.prev_A.zero_()


# =============================================================================
# Training & Evaluation (adapted from main.py)
# =============================================================================

def run_ablation_variant(args, config_name, config, data, adj_mx):
    """Train and evaluate one ablation variant."""
    print(f"\n{'='*65}")
    print(f"  Config: {config_name.upper()}")
    print(f"  {config['description']}")
    print(f"{'='*65}")

    device = args.device
    seed_everything(args.seed)

    # Create fresh model
    model = DGLLM(
        device, adj_mx, args.input_dim, args.num_nodes,
        args.input_len, args.output_len, args.llm_layer, args.U,
        vmd_K=args.vmd_k
    ).to(device)

    # Apply ablation config
    apply_graph_config(model, config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    optimizer = Ranger(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scaler = data['scaler']

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    epoch_history = {
        'train_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'epochs': [],
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        for x, y, vmd in data['train_loader'].get_iterator():
            tx = x.to(device, non_blocking=True).transpose(1, 3)  # [B, F, N, T]
            ty = y.to(device, non_blocking=True).transpose(1, 3)[:, 0, :, :]  # [B, N, T]
            tvmd = vmd.to(device, non_blocking=True)

            # Permute to [B, T, N, F] — matching VMD_Trainer.train_step
            x_in = tx.permute(0, 3, 2, 1)

            optimizer.zero_grad()

            preds, adj_list = model(tvmd, x_in)
            preds = preds.transpose(1, 3)  # Match trainer output shape
            preds = scaler.inverse_transform(preds)
            real_scaled = torch.unsqueeze(ty, 1)

            loss = MAE_torch(preds, real_scaled, 0.0)

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    [WARN] NaN/Inf loss at epoch {epoch}, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        epoch_history['train_loss'].append(float(avg_train_loss))
        epoch_history['epochs'].append(epoch)

        # Validation
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            model.eval()
            val_maes, val_rmses = [], []

            with torch.no_grad():
                for x, y, vmd in data['val_loader'].get_iterator():
                    tx = x.to(device, non_blocking=True).transpose(1, 3)
                    ty = y.to(device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
                    tvmd = vmd.to(device, non_blocking=True)

                    x_in = tx.permute(0, 3, 2, 1)
                    preds, _ = model(tvmd, x_in)
                    preds = preds.transpose(1, 3)
                    preds = scaler.inverse_transform(preds)
                    real_scaled = torch.unsqueeze(ty, 1)

                    mae_val = MAE_torch(preds, real_scaled, 0.0).item()
                    rmse_val = RMSE_torch(preds, real_scaled, 0.0).item()
                    val_maes.append(mae_val)
                    val_rmses.append(rmse_val)

            avg_val_mae = np.mean(val_maes)
            avg_val_rmse = np.mean(val_rmses)
            epoch_history['val_mae'].append(float(avg_val_mae))
            epoch_history['val_rmse'].append(float(avg_val_rmse))

            if avg_val_mae < best_val_loss:
                best_val_loss = avg_val_mae
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

            if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
                print(f"    Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | "
                      f"Val MAE: {avg_val_mae:.4f} | Val RMSE: {avg_val_rmse:.4f}")

    train_time = time.time() - start_time

    # Test on best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_maes, test_rmses, test_mapes = [], [], []
    with torch.no_grad():
        for x, y, vmd in data['test_loader'].get_iterator():
            tx = x.to(device, non_blocking=True).transpose(1, 3)
            ty = y.to(device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
            tvmd = vmd.to(device, non_blocking=True)

            x_in = tx.permute(0, 3, 2, 1)
            preds, _ = model(tvmd, x_in)
            preds = preds.transpose(1, 3)
            preds = scaler.inverse_transform(preds)
            real_scaled = torch.unsqueeze(ty, 1)

            test_maes.append(MAE_torch(preds, real_scaled, 0.0).item())
            test_rmses.append(RMSE_torch(preds, real_scaled, 0.0).item())
            test_mapes.append(MAPE_torch(preds, real_scaled, 0.0).item())

    test_mae = np.mean(test_maes)
    test_rmse = np.mean(test_rmses)
    test_mape = np.mean(test_mapes) * 100

    print(f"\n  Best Validation @ Epoch {best_epoch}")
    print(f"  Test: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, MAPE={test_mape:.2f}%")
    print(f"  Time: {train_time/60:.1f} min")

    return {
        'config': config_name,
        'description': config['description'],
        'best_epoch': int(best_epoch),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'train_time_min': float(train_time / 60),
        'epoch_history': epoch_history,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_ablation_results(results, dataset, save_dir="rebuttal_figures"):
    """Generate visualization of ablation results."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    colors = {
        'curriculum':   '#2563EB',
        'pure_dynamic': '#DC2626',
        'pure_static':  '#F59E0B',
        'no_dynamic':   '#6B7280',
    }
    labels = {
        'curriculum':   'Curriculum Blend (default)',
        'pure_dynamic': 'Pure Dynamic (no static prior)',
        'pure_static':  'Pure Static (no dynamic)',
        'no_dynamic':   'Dynamic Disabled',
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Training Loss Curves ---
    ax = axes[0]
    for r in results:
        name = r['config']
        epochs = r['epoch_history']['epochs']
        losses = r['epoch_history']['train_loss']
        ax.plot(epochs, losses, color=colors.get(name, 'gray'),
                linewidth=2, label=labels.get(name, name), alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MAE)')
    ax.set_title('Training Convergence', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 2: Validation MAE Curves ---
    ax = axes[1]
    for r in results:
        name = r['config']
        val_mae = r['epoch_history']['val_mae']
        # Validation may not happen every epoch
        val_epochs = [e for i, e in enumerate(r['epoch_history']['epochs'])
                      if i < len(val_mae)][:len(val_mae)]
        ax.plot(val_epochs, val_mae, 'o-', color=colors.get(name, 'gray'),
                linewidth=2, markersize=3, label=labels.get(name, name), alpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MAE')
    ax.set_title('Validation Performance', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel 3: Final Test Performance Bar Chart ---
    ax = axes[2]
    names = [r['config'] for r in results]
    maes = [r['test_mae'] for r in results]
    bar_colors = [colors.get(n, 'gray') for n in names]

    bars = ax.bar(range(len(names)), maes, color=bar_colors, alpha=0.9,
                  edgecolor='white', linewidth=1)

    # Value labels on bars
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mae:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Highlight best
    best_idx = np.argmin(maes)
    bars[best_idx].set_edgecolor('#059669')
    bars[best_idx].set_linewidth(3)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([labels.get(n, n).split('(')[0].strip() for n in names],
                       fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Test MAE')
    ax.set_title('Final Test Performance', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(f'Curriculum Graph Ablation — {dataset}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'curriculum_ablation_{dataset}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\n  ✓ Figure saved: {save_path}")


def print_comparison_table(results):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"  CURRICULUM GRAPH ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"  {'Config':<20} {'MAE':>10} {'RMSE':>10} {'MAPE%':>10} {'Best Ep':>10} {'Time':>8}")
    print(f"  {'-'*75}")

    best_mae = min(r['test_mae'] for r in results)

    for r in results:
        marker = ' ★' if r['test_mae'] == best_mae else ''
        print(f"  {r['config']:<20} {r['test_mae']:>10.4f} {r['test_rmse']:>10.4f} "
              f"{r['test_mape']:>9.2f}% {r['best_epoch']:>10d} {r['train_time_min']:>7.1f}m{marker}")

    print(f"{'='*80}")

    # Compute degradation from curriculum
    curriculum_mae = next(r['test_mae'] for r in results if r['config'] == 'curriculum')
    print(f"\n  Degradation vs Curriculum Blend:")
    for r in results:
        if r['config'] == 'curriculum':
            continue
        deg = (r['test_mae'] - curriculum_mae) / curriculum_mae * 100
        print(f"    {r['config']:<20}: {'+' if deg > 0 else ''}{deg:.2f}% MAE")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Curriculum Graph Ablation Experiment')

    parser.add_argument('--data', type=str, default='taxi_drop',
                        choices=['PEMSD04', 'PEMSD08', 'bike_drop', 'bike_pick',
                                 'taxi_drop', 'taxi_pick'])
    parser.add_argument('--root_path', type=str, default='./Dataset/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-5)
    parser.add_argument('--llm_layer', type=int, default=6)
    parser.add_argument('--U', type=int, default=1)
    parser.add_argument('--vmd_k', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_interval', type=int, default=5)
    parser.add_argument('--quick', action='store_true', help='Quick test (5 epochs)')
    parser.add_argument('--configs', nargs='+',
                        default=['curriculum', 'pure_dynamic', 'pure_static', 'no_dynamic'],
                        help='Configs to run')

    args = parser.parse_args()

    if args.quick:
        args.epochs = 5
        args.val_interval = 1

    # Dataset config
    args.data_path = os.path.join(args.root_path, args.data, 'processed')
    if 'PEMSD04' in args.data:
        args.num_nodes = 307
    elif 'PEMSD08' in args.data:
        args.num_nodes = 170
    elif 'bike' in args.data:
        args.num_nodes = 250
    elif 'taxi' in args.data:
        args.num_nodes = 266

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  CURRICULUM GRAPH ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"  Dataset: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Configs: {args.configs}")
    print(f"  Device: {args.device}")
    print("=" * 70)

    # Load data once
    print("\nLoading dataset...")
    adj_path = os.path.join(args.root_path, args.data, 'adj_mx.pkl')
    adj_mx = load_pickle(adj_path)

    data = load_dataset_optimized(args.data_path, args.batch_size, args)

    # Run all configs
    results = []
    for config_name in args.configs:
        if config_name not in ABLATION_CONFIGS:
            print(f"\n  [WARN] Unknown config: {config_name}, skipping")
            continue

        try:
            result = run_ablation_variant(
                args, config_name, ABLATION_CONFIGS[config_name], data, adj_mx
            )
            results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] Config {config_name} failed: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n  [ERROR] No results generated!")
        return

    # Print comparison
    print_comparison_table(results)

    # Plot
    try:
        plot_ablation_results(results, args.data)
    except Exception as e:
        print(f"\n  [WARN] Plotting failed: {e}")

    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = f'results/curriculum_ablation_{args.data}.json'
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': args.data,
            'epochs': args.epochs,
            'seed': args.seed,
            'configs': {r['config']: r for r in results},
        }, f, indent=2)
    print(f"\n  Results saved: {output_path}")


if __name__ == '__main__':
    main()
