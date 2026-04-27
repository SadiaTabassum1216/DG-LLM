"""
Single-Sample Inference Demo for DG-LLM
========================================
Runs a single test sample through the model and displays:
  1. Input (last 12 timesteps) for selected nodes
  2. VMD decomposition (computed on-the-fly for this sample)
  3. Mode-wise predictions (VMD mode 1, 2, 3)
  4. Final combined prediction vs ground truth
  5. Per-horizon error metrics (all 12 horizons)

Computes VMD sample-wise — no pre-computed cache needed.

Usage:
  python single_sample_inference.py --data taxi_drop --sample_idx 42 --nodes 0,5,10
  python single_sample_inference.py --data PEMSD08 --sample_idx 100
"""

import torch
import numpy as np
import argparse
import os
import time
import glob
from types import SimpleNamespace

from trainer import Trainer
from utils import StandardScaler, MAE_torch, RMSE_torch, MAPE_torch, load_pickle
from vmd_utils import decompose_single_window
from paths import DATASET_DIR, MODELS_DIR, RESULTS_FIGURES_DIR, RESULTS_LOGS_DIR


DATASET_NUM_NODES = {
    "PEMSD04": 307,
    "PEMSD08": 170,
    "bike_drop": 250,
    "bike_pick": 250,
    "taxi_drop": 266,
    "taxi_pick": 266,
}


def parse_args():
    parser = argparse.ArgumentParser(description='DG-LLM Single Sample Inference')
    parser.add_argument('--data', type=str, default='taxi_drop',
                        choices=list(DATASET_NUM_NODES.keys()))
    parser.add_argument('--root_path', type=str, default=str(DATASET_DIR))
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to best_model.pth (default: models/<data>/best_model.pth)')
    parser.add_argument('--sample_idx', type=int, default=42,
                        help='Index of test sample to visualize')
    parser.add_argument('--nodes', type=str, default=None,
                        help='Comma-separated node indices to display (default: top-5 by flow)')
    parser.add_argument('--vmd_k', type=int, default=3)
    parser.add_argument('--llm_layer', type=int, default=6)
    parser.add_argument('--U', type=int, default=1)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--output_len', type=int, default=12)

    args = parser.parse_args()
    args.num_nodes = DATASET_NUM_NODES[args.data]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is None:
        args.model_path = os.path.join(str(MODELS_DIR), args.data, "best_model.pth")

    # Kaggle-specific auto-detection if default path fails
    if not os.path.exists(args.model_path) and os.path.exists('/kaggle'):
        print(f"  >> Checkpoint not found at {args.model_path}. Searching in /kaggle/input...")
        # Pattern matching the user's structure: /kaggle/input/.../models/<data>/best_model.pth
        search_pattern = f"/kaggle/input/**/models/{args.data}/best_model.pth"
        found_paths = glob.glob(search_pattern, recursive=True)
        if found_paths:
            args.model_path = found_paths[0]
            print(f"  >> Found checkpoint at: {args.model_path}")
        else:
            # Fallback: search for any best_model.pth inside a directory named after the dataset
            search_pattern_alt = f"/kaggle/input/**/{args.data}/best_model.pth"
            found_paths_alt = glob.glob(search_pattern_alt, recursive=True)
            if found_paths_alt:
                args.model_path = found_paths_alt[0]
                print(f"  >> Found checkpoint at: {args.model_path}")

    return args


def load_adjacency(root_path, dataset, num_nodes):
    adj_path = os.path.join(root_path, dataset, "adj_mx.pkl")
    if not os.path.exists(adj_path):
        print(f"  [Warning] Adjacency not found at {adj_path}. Using identity.")
        return np.eye(num_nodes, dtype=np.float32)
    adj_data = load_pickle(adj_path)
    if isinstance(adj_data, list):
        return adj_data[2]
    return adj_data


def load_model_weights(trainer, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    raw = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    else:
        state = raw

    # --- Compatibility Mapping ---
    # Handles name changes between different model versions
    mapping_rules = {
        "mode_models.": "mode_processors.",
        "fusion_value.": "fusion_query.",
        "residual_proj.": "flow_residual_projection.",
        "global_scale": "global_flow_residual_scale",
        # ModeProcessor internal renames
        "node_emb": "node_identity_emb",
        "global_step": "total_training_steps",
        "ema_A": "purely_dynamic_graph",
    }
    
    new_state = {}
    mapped_log = []
    for k, v in state.items():
        new_key = k
        for old, new in mapping_rules.items():
            if old in k:
                new_key = k.replace(old, new)
                mapped_log.append(f"{k} -> {new_key}")
                break
        new_state[new_key] = v
    
    if mapped_log:
        print(f"  >> Mapped {len(mapped_log)} keys for compatibility")
        # If fusion_query was mapped but fusion_key is missing, duplicate it
        if "fusion_query.weight" in new_state and "fusion_key.weight" not in state:
            new_state["fusion_key.weight"] = new_state["fusion_query.weight"].clone()
            new_state["fusion_key.bias"] = new_state["fusion_query.bias"].clone()
            print("  >> Initialized fusion_key from fusion_query weights")

    state = new_state
    # -----------------------------

    missing, unexpected = trainer.model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [Warning] Missing keys: {len(missing)}")
        print(f"    Examples (first 10): {missing[:10]}")
    if unexpected:
        print(f"  [Warning] Unexpected keys: {len(unexpected)}")
        print(f"    Examples (first 10): {unexpected[:10]}")


def main():
    args = parse_args()

    print("=" * 70)
    print(f"  DG-LLM Single Sample Inference — {args.data}")
    print(f"  Device: {args.device} | Nodes: {args.num_nodes} | VMD K: {args.vmd_k}")
    print("=" * 70)

    # ─── Load raw test + train data ──────────────────────────────────────
    data_dir = os.path.join(args.root_path, args.data, "processed")

    print("\n>> Loading data...")
    test_data = np.load(os.path.join(data_dir, "test.npz"))
    x_test = test_data["x"]  # [samples, 12, N, F]
    y_test = test_data["y"]  # [samples, 12, N, 1]
    print(f"   Test set: {x_test.shape[0]} samples")
    print(f"   x shape: {x_test.shape}  y shape: {y_test.shape}")

    # Scaler from training data
    train_data = np.load(os.path.join(data_dir, "train.npz"))
    x_train = train_data["x"]
    mean_val = float(x_train[..., 0].mean())
    std_val = float(x_train[..., 0].std())
    scaler = StandardScaler(mean_val, std_val)
    print(f"   Scaler: mean={mean_val:.4f}, std={std_val:.4f}")

    # Add temporal features if missing (PeMS datasets only have flow)
    # Matches data_loader.py lines 275-290
    if x_test.shape[-1] < 3:
        print("   Adding temporal features (ToD, DoW)...")
        # Compute offset: train + val samples come before test
        val_data = np.load(os.path.join(data_dir, "val.npz"))
        cumulative_offset = x_train.shape[0] + val_data["x"].shape[0]

        num_samples, T_len, num_nodes, _ = x_test.shape
        sample_starts = np.arange(num_samples) + cumulative_offset
        step_indices = sample_starts[:, None] + np.arange(T_len)[None, :]

        time_of_day = (step_indices % 288) / 288.0
        time_of_day = np.tile(time_of_day[:, :, None, None], (1, 1, num_nodes, 1))

        day_of_week = ((step_indices // 288) % 7).astype(np.float32)
        day_of_week = np.tile(day_of_week[:, :, None, None], (1, 1, num_nodes, 1))

        x_test = np.concatenate([x_test[..., 0:1], time_of_day, day_of_week], axis=-1)
        print(f"   x_test expanded: {x_test.shape}")
    else:
        print(f"   Temporal features already present (F={x_test.shape[-1]})")

    # ─── Select sample ──────────────────────────────────────────────────
    idx = args.sample_idx
    if idx >= x_test.shape[0]:
        print(f"   WARNING: sample_idx={idx} > max ({x_test.shape[0]-1}), using 0")
        idx = 0

    x_sample = x_test[idx]  # [12, N, F]
    y_sample = y_test[idx]  # [12, N, 1]

    # Pick display nodes
    if args.nodes:
        display_nodes = [int(n) for n in args.nodes.split(',')]
    else:
        avg_flow = x_sample[:, :, 0].mean(axis=0)
        display_nodes = np.argsort(avg_flow)[-5:][::-1].tolist()

    print(f"\n>> Sample #{idx}")
    print(f"   Display nodes: {display_nodes}")

    # ─── Compute VMD on-the-fly for this single sample ───────────────────
    print(f"\n>> Computing VMD decomposition (K={args.vmd_k}) for sample #{idx}...")
    flow_window = x_sample[:, :, 0:1]  # [T, N, 1] — flow channel only

    t_vmd_start = time.time()
    vmd_modes = decompose_single_window(flow_window, K=args.vmd_k)  # [K, T, N, 1]
    t_vmd = time.time() - t_vmd_start

    print(f"   VMD computed in {t_vmd:.2f}s  ({args.num_nodes} nodes × {args.input_len} timesteps)")
    print(f"   VMD output shape: {vmd_modes.shape}  (K, T, N, 1)")

    # Verify: sum of modes ≈ original signal
    mode_sum = vmd_modes.sum(axis=0)  # [T, N, 1]
    reconstruction_err = np.abs(mode_sum - flow_window).mean()
    print(f"   Reconstruction error (|sum(modes) - original|): {reconstruction_err:.6f}")

    # ─── Show VMD decomposition for display nodes ────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  VMD DECOMPOSITION (input signal → {args.vmd_k} modes)")
    print(f"{'─' * 70}")
    for n in display_nodes:
        print(f"\n  ── Node {n} ──")
        header = f"  {'t':>4} | {'Original':>10}"
        for k in range(args.vmd_k):
            header += f" | {'Mode'+str(k+1):>10}"
        header += f" | {'Sum':>10}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for t in range(args.input_len):
            orig = flow_window[t, n, 0]
            row = f"  {t+1:>4} | {orig:10.2f}"
            for k in range(args.vmd_k):
                row += f" | {vmd_modes[k, t, n, 0]:10.2f}"
            row += f" | {mode_sum[t, n, 0]:10.2f}"
            print(row)

    # ─── Prepare tensors ─────────────────────────────────────────────────
    # Normalize x using scaler (flow channel only)
    x_normalized = x_sample.copy()
    x_normalized[:, :, 0] = (x_normalized[:, :, 0] - mean_val) / (std_val + 1e-8)

    # Normalize VMD modes the same way
    vmd_normalized = (vmd_modes - mean_val) / (std_val + 1e-8)

    # Build tensors matching the pipeline:
    #   tx = x.transpose(1,3) on [B, T, N, F] → but x from .npz is [T, N, F]
    #   Need: [1, T, N, F] → transpose(1,3) → [1, F, N, T]
    #   Then: x_in = tx.permute(0,3,2,1) → [1, T, N, F]
    x_tensor = torch.from_numpy(x_normalized[np.newaxis]).float()  # [1, T, N, F]
    tx = x_tensor.transpose(1, 3)       # [1, F, N, T]
    x_in = tx.permute(0, 3, 2, 1)       # [1, T, N, F]

    # y is NOT normalized in the pipeline
    y_tensor = torch.from_numpy(y_sample[np.newaxis]).float()  # [1, T, N, 1]
    ty = y_tensor.transpose(1, 3)[:, 0, :, :]  # [1, N, T]
    real_unscaled = ty.permute(0, 2, 1).unsqueeze(-1)  # [1, T, N, 1]

    # VMD tensor: [1, K, T, N, 1] — normalized
    vmd_tensor = torch.from_numpy(vmd_normalized[np.newaxis]).float()  # [1, K, T, N, 1]

    # ─── Load Model ──────────────────────────────────────────────────────
    print(f"\n>> Loading model from {args.model_path}...")
    adj_mx = load_adjacency(args.root_path, args.data, args.num_nodes)

    runtime_args = SimpleNamespace(
        data=args.data,
        input_dim=args.input_dim,
        input_len=args.input_len,
        output_len=args.output_len,
        llm_layer=args.llm_layer,
        U=args.U,
        vmd_k=args.vmd_k,
        num_nodes=args.num_nodes,
        device=args.device,
        log_dir=str(RESULTS_LOGS_DIR),
        lrate=1e-3,
        wdecay=1e-5,
        grad_accum_steps=1,
        enable_compile=False,
        use_amp=False,
        use_bf16=False,
        batch_size=1,
    )

    trainer = Trainer(runtime_args, scaler, adj_mx, args.device)
    load_model_weights(trainer, args.model_path, args.device)
    trainer.model.eval()
    print(f"   Model loaded ({trainer.model.param_num():,} parameters)")

    # ─── Run Inference ───────────────────────────────────────────────────
    print("\n>> Running inference...")
    tx_dev = tx.to(args.device)
    x_in_dev = x_in.to(args.device)
    tvmd_dev = vmd_tensor.to(args.device)

    t_inf_start = time.time()

    with torch.no_grad():
        # Full model prediction
        pred, graphs = trainer.model(tvmd_dev, x_in_dev)
        pred_unscaled = scaler.inverse_transform(pred).cpu()

        # Mode-wise predictions
        K = args.vmd_k
        time_feats = x_in_dev[..., 1:]
        mode_preds_unscaled = []
        for k in range(K):
            mode_flow = tvmd_dev[:, k, ...]
            mode_in = torch.cat([mode_flow, time_feats], dim=-1)
            mode_pred_k, _ = trainer.model.mode_processors[k](mode_in)
            mode_pred_k = scaler.inverse_transform(mode_pred_k).cpu()
            mode_preds_unscaled.append(mode_pred_k)

    t_inference = time.time() - t_inf_start
    print(f"   Inference complete in {t_inference*1000:.1f}ms")

    real_cpu = real_unscaled

    # ─── Display: Input History ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    x_raw = x_sample  # original unnormalized [T, N, F]

    print(f"\n{'─' * 70}")
    print(f"  INPUT HISTORY (last {args.input_len} timesteps, raw flow)")
    print(f"{'─' * 70}")
    _print_header(["Step"] + [f"Node {n}" for n in display_nodes])
    for t in range(args.input_len):
        row = f"  t-{args.input_len - t:>2} "
        for n in display_nodes:
            row += f" | {x_raw[t, n, 0]:10.2f}"
        print(row)

    # ─── Display: Mode-wise Predictions ──────────────────────────────────
    for k in range(K):
        print(f"\n{'─' * 70}")
        print(f"  VMD MODE {k+1} PREDICTION (12 horizons)")
        print(f"{'─' * 70}")
        _print_header(["Step"] + [f"Node {n}" for n in display_nodes])
        for t in range(args.output_len):
            row = f"  h={t+1:>2} "
            for n in display_nodes:
                row += f" | {mode_preds_unscaled[k][0, t, n, 0].item():10.2f}"
            print(row)

    # ─── Display: Final Prediction ───────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  FINAL PREDICTION (attention fusion + residual)")
    print(f"{'─' * 70}")
    _print_header(["Step"] + [f"Node {n}" for n in display_nodes])
    for t in range(args.output_len):
        row = f"  h={t+1:>2} "
        for n in display_nodes:
            row += f" | {pred_unscaled[0, t, n, 0].item():10.2f}"
        print(row)

    # ─── Display: Ground Truth ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  GROUND TRUTH")
    print(f"{'─' * 70}")
    _print_header(["Step"] + [f"Node {n}" for n in display_nodes])
    for t in range(args.output_len):
        row = f"  h={t+1:>2} "
        for n in display_nodes:
            row += f" | {real_cpu[0, t, n, 0].item():10.2f}"
        print(row)

    # ─── Display: Side-by-side Pred vs Truth per node ────────────────────
    print(f"\n{'─' * 70}")
    print(f"  PREDICTION vs GROUND TRUTH (per node, all 12 horizons)")
    print(f"{'─' * 70}")
    for n in display_nodes:
        print(f"\n  ── Node {n} ──")
        print(f"  {'Step':>6} | {'Predicted':>10} | {'Truth':>10} | {'Error':>10} | {'%Err':>8}")
        print(f"  {'-'*55}")
        node_errs = []
        for t in range(args.output_len):
            p = pred_unscaled[0, t, n, 0].item()
            r = real_cpu[0, t, n, 0].item()
            err = abs(p - r)
            pct = (err / abs(r) * 100) if abs(r) > 1e-6 else 0.0
            node_errs.append(err)
            print(f"  h={t+1:>2}  | {p:10.2f} | {r:10.2f} | {err:10.2f} | {pct:7.1f}%")
        node_mae = np.mean(node_errs)
        node_rmse = np.sqrt(np.mean(np.array(node_errs)**2))
        print(f"  {'-'*55}")
        print(f"  MAE={node_mae:.2f}  RMSE={node_rmse:.2f}")

    # ─── Display: Per-horizon error across ALL nodes ─────────────────────
    print(f"\n{'─' * 70}")
    print(f"  PER-HORIZON ERROR (across all {args.num_nodes} nodes)")
    print(f"{'─' * 70}")
    print(f"  {'Step':>6} | {'MAE':>8} | {'RMSE':>8} | {'MAPE':>8}")
    print(f"  {'-'*42}")

    all_mae, all_rmse, all_mape = [], [], []
    for t in range(args.output_len):
        p = pred_unscaled[:, t, :, :]
        r = real_cpu[:, t, :, :]
        mae = MAE_torch(p, r, 0.0).item()
        rmse = RMSE_torch(p, r, 0.0).item()
        mape = MAPE_torch(p, r, 0.0).item()
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_mape.append(mape)
        print(f"  h={t+1:>2}  | {mae:8.2f} | {rmse:8.2f} | {mape:8.4f}")

    print(f"  {'-'*42}")
    print(f"  AVG   | {np.mean(all_mae):8.2f} | {np.mean(all_rmse):8.2f} | {np.mean(all_mape):8.4f}")

    # ─── Display: Mode contribution ──────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  MODE CONTRIBUTION (avg across all nodes per horizon)")
    print(f"{'─' * 70}")
    mh = f"  {'Step':>6}"
    for k in range(K):
        mh += f" | {'Mode '+str(k+1):>10}"
    mh += f" | {'Final':>10} | {'Truth':>10}"
    print(mh)
    print(f"  {'-'*(len(mh)-2)}")

    for t in range(args.output_len):
        row = f"  h={t+1:>2}  "
        for k in range(K):
            avg_val = mode_preds_unscaled[k][0, t, :, 0].mean().item()
            row += f" | {avg_val:10.2f}"
        final_avg = pred_unscaled[0, t, :, 0].mean().item()
        gt_avg = real_cpu[0, t, :, 0].mean().item()
        row += f" | {final_avg:10.2f} | {gt_avg:10.2f}"
        print(row)

    # ─── Plot: Ground Truth vs Prediction ────────────────────────────────
    import matplotlib.pyplot as plt

    num_display = len(display_nodes)
    fig, axes = plt.subplots(num_display, 1, figsize=(12, 4 * num_display), squeeze=False)
    fig.suptitle(f"DG-LLM Prediction vs Ground Truth — {args.data} Sample #{idx}", fontsize=14, fontweight='bold')

    horizon_labels = [f"h{t+1}" for t in range(args.output_len)]
    input_labels = [f"t-{args.input_len - t}" for t in range(args.input_len)]
    all_labels = input_labels + horizon_labels
    x_ticks = np.arange(len(all_labels))

    mode_colors = ['#2ca02c', '#9467bd', '#ff7f0e']

    for i, n in enumerate(display_nodes):
        ax = axes[i, 0]

        # Input history (flow)
        input_vals = [x_raw[t, n, 0] for t in range(args.input_len)]

        # Predictions & truth (forecast horizon)
        pred_vals = [pred_unscaled[0, t, n, 0].item() for t in range(args.output_len)]
        truth_vals = [real_cpu[0, t, n, 0].item() for t in range(args.output_len)]

        # Plot input history
        ax.plot(range(args.input_len), input_vals, 'o-', color='#1f77b4', linewidth=2, markersize=5, label='Input (history)')

        # Plot mode predictions
        for k in range(K):
            mode_vals = [mode_preds_unscaled[k][0, t, n, 0].item() for t in range(args.output_len)]
            ax.plot(range(args.input_len, args.input_len + args.output_len), mode_vals,
                    '--', color=mode_colors[k], linewidth=1.2, alpha=0.6, label=f'Mode {k+1}')

        # Plot final prediction
        ax.plot(range(args.input_len, args.input_len + args.output_len), pred_vals,
                's-', color='#d62728', linewidth=2.5, markersize=6, label='Prediction')

        # Plot ground truth
        ax.plot(range(args.input_len, args.input_len + args.output_len), truth_vals,
                'D-', color='#1f77b4', linewidth=2.5, markersize=6, label='Ground Truth')

        # Shade error
        ax.fill_between(range(args.input_len, args.input_len + args.output_len),
                         pred_vals, truth_vals, alpha=0.15, color='#d62728')

        # Vertical line separating input from forecast
        ax.axvline(x=args.input_len - 0.5, color='gray', linestyle=':', linewidth=1.5)
        ax.text(args.input_len - 1, ax.get_ylim()[1] * 0.95, '← History | Forecast →',
                ha='center', fontsize=9, color='gray')

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(all_labels, fontsize=8, rotation=45)
        ax.set_ylabel('Flow', fontsize=11)
        ax.set_title(f'Node {n}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(alpha=0.25)

    plt.tight_layout()

    # Save plot
    plot_dir = os.path.join(str(RESULTS_FIGURES_DIR), "vmd", args.data)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"single_sample_{idx}.png")
    fig.savefig(plot_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"\n>> Plot saved to: {plot_path}")

    # ─── Timing summary ─────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  TIMING SUMMARY")
    print(f"{'─' * 70}")
    print(f"  VMD decomposition : {t_vmd*1000:8.1f} ms  ({args.num_nodes} nodes)")
    print(f"  Model inference   : {t_inference*1000:8.1f} ms")
    print(f"  Total per sample  : {(t_vmd + t_inference)*1000:8.1f} ms")

    print("\n" + "=" * 70)
    print(f"  Done! Sample #{idx} from {args.data}")
    print("=" * 70)


def _print_header(headers):
    line = f"{headers[0]:>6}"
    for h in headers[1:]:
        line += f" | {h:>10}"
    print(line)
    print("-" * len(line))


if __name__ == '__main__':
    main()
