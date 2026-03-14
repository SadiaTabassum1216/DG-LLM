"""
Comprehensive DG-LLM test-time visualization script.

This script loads a trained checkpoint, runs test inference once, and generates:
1) Daily/weekly pattern plots for horizon-1 predictions
2) Full test sequence ground-truth vs prediction at horizon-1
3) Dynamic graph diagnostics (metrics, snapshots, optional GIF)
4) Per-node day-window prediction (horizon 1, 288 samples)
5) Error distribution histogram + prediction vs ground-truth scatter
6) 1-week zoomed forecast for a selected node
"""

import argparse
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from data_loader import load_dataset_optimized
from trainer import VMD_Trainer
from utils import MAE_torch, MAPE_torch, RMSE_torch, load_pickle
from visualization import (
    visualize_model_predictions,
    visualize_advanced_diagnostics,
    visualize_weekly_horizon1,
)


DATASET_NUM_NODES = {
    "PEMSD04": 307,
    "PEMSD08": 170,
    "bike_drop": 250,
    "bike_pick": 250,
    "taxi_drop": 266,
    "taxi_pick": 266,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DG-LLM test inference and generate comprehensive visualizations"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="PEMSD04",
        choices=list(DATASET_NUM_NODES.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./Dataset",
        help="Root folder containing dataset directories",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for test inference",
    )
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--llm_layer", type=int, default=6)
    parser.add_argument("--U", type=int, default=1)
    parser.add_argument("--vmd_k", type=int, default=3)

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to checkpoint/state_dict file (defaults to models/<data>/best_model.pth)",
    )
    parser.add_argument(
        "--vmd_cache_dir",
        type=str,
        default=None,
        help="Path to VMD cache dir (defaults to models/<data>/vmd_cache)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder for plots (defaults to vmd_visualizations/<data>/test_bundle)",
    )

    parser.add_argument(
        "--node_idx",
        type=int,
        default=0,
        help="Node index to visualize for full horizon-1 line plots",
    )
    parser.add_argument(
        "--graph_nodes",
        type=int,
        default=64,
        help="Number of leading nodes for graph heatmap/GIF visualization",
    )
    parser.add_argument(
        "--graph_snapshots",
        type=int,
        default=6,
        help="Number of dynamic graph snapshots to save",
    )
    parser.add_argument(
        "--max_graph_batches",
        type=int,
        default=200,
        help="Max number of test batches used for dynamic graph tracking",
    )
    parser.add_argument(
        "--save_graph_gif",
        action="store_true",
        help="Save dynamic adjacency GIF over test batches",
    )

    args = parser.parse_args()
    args.num_nodes = DATASET_NUM_NODES[args.data]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path is None:
        args.model_path = os.path.join("models", args.data, "best_model.pth")
    if args.vmd_cache_dir is None:
        args.vmd_cache_dir = os.path.join("models", args.data, "vmd_cache")
    if args.output_dir is None:
        args.output_dir = os.path.join("vmd_visualizations", args.data, "test_bundle")

    return args


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_adjacency(root_path, dataset, num_nodes):
    adj_path = os.path.join(root_path, dataset, "adj_mx.pkl")
    if not os.path.exists(adj_path):
        print(f"[Warning] Adjacency not found at {adj_path}. Using identity matrix.")
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

    missing, unexpected = trainer.model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warning] Missing keys while loading model: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys while loading model: {len(unexpected)}")


def horizon1_time_indices(last_tod, last_dow, slots_per_day=288):
    last_slot = np.clip(np.round(last_tod * slots_per_day).astype(int), 0, slots_per_day - 1)
    next_slot = (last_slot + 1) % slots_per_day
    day_roll = (last_slot + 1) // slots_per_day
    next_dow = (last_dow.astype(int) + day_roll) % 7
    return next_slot, next_dow


def run_test_inference(trainer, data, args):
    all_preds = []
    all_reals = []
    all_tod_idx = []
    all_dow_idx = []
    graph_series = []

    test_loader = data["test_loader"]
    scaler = data["scaler"]

    trainer.model.eval()
    with torch.no_grad():
        for batch_i, (x, y, vmd) in enumerate(tqdm(test_loader.get_iterator(), desc="Testing")):
            tx = x.to(args.device, non_blocking=True).transpose(1, 3)
            ty = y.to(args.device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
            tvmd = vmd.to(args.device, non_blocking=True)
            x_in = tx.permute(0, 3, 2, 1)

            pred, graphs = trainer.model(tvmd, x_in)

            pred_unscaled = scaler.inverse_transform(pred).cpu()
            real_unscaled = scaler.inverse_transform(ty.permute(0, 2, 1).unsqueeze(-1)).cpu()

            all_preds.append(pred_unscaled)
            all_reals.append(real_unscaled)

            last_tod = x[:, -1, 0, 1].cpu().numpy()
            last_dow = x[:, -1, 0, 2].cpu().numpy()
            h1_tod, h1_dow = horizon1_time_indices(last_tod, last_dow)
            all_tod_idx.append(h1_tod)
            all_dow_idx.append(h1_dow)

            if batch_i < args.max_graph_batches:
                # Average K mode-specific adjacencies into one dynamic graph per batch.
                g = torch.stack([gg.detach().cpu() for gg in graphs], dim=0).float().mean(dim=0)
                graph_series.append(g.numpy())

    preds = torch.cat(all_preds, dim=0)
    reals = torch.cat(all_reals, dim=0)
    tod_idx = np.concatenate(all_tod_idx, axis=0)
    dow_idx = np.concatenate(all_dow_idx, axis=0)
    graphs = np.stack(graph_series, axis=0) if graph_series else None

    return {
        "preds": preds,
        "reals": reals,
        "tod_idx": tod_idx,
        "dow_idx": dow_idx,
        "graphs": graphs,
    }


def compute_metrics(preds, reals):
    overall = {
        "mae": MAE_torch(preds, reals, 0).item(),
        "rmse": RMSE_torch(preds, reals, 0).item(),
        "mape": MAPE_torch(preds, reals, 0).item(),
    }

    p_h1 = preds[:, 0, :, :]
    r_h1 = reals[:, 0, :, :]
    h1 = {
        "mae": MAE_torch(p_h1, r_h1, 0).item(),
        "rmse": RMSE_torch(p_h1, r_h1, 0).item(),
        "mape": MAPE_torch(p_h1, r_h1, 0).item(),
    }
    return {"overall": overall, "horizon_1": h1}


def plot_full_horizon1_series(preds_np, reals_np, node_idx, output_dir):
    h1_pred_node = preds_np[:, 0, node_idx, 0]
    h1_real_node = reals_np[:, 0, node_idx, 0]
    h1_pred_global = preds_np[:, 0, :, 0].mean(axis=1)
    h1_real_global = reals_np[:, 0, :, 0].mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(h1_real_node, color="#1f77b4", linewidth=1.0, label="Ground Truth")
    axes[0].plot(h1_pred_node, color="#d62728", linewidth=1.0, alpha=0.85, label="Prediction")
    axes[0].set_title(f"Horizon-1 Full Test Sequence (Node {node_idx})")
    axes[0].set_ylabel("Traffic Flow")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(h1_real_global, color="#2ca02c", linewidth=1.0, label="Ground Truth (mean over nodes)")
    axes[1].plot(
        h1_pred_global,
        color="#ff7f0e",
        linewidth=1.0,
        alpha=0.85,
        label="Prediction (mean over nodes)",
    )
    axes[1].set_title("Horizon-1 Full Test Sequence (Network Mean)")
    axes[1].set_xlabel("Test Sample Index")
    axes[1].set_ylabel("Traffic Flow")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(output_dir, "horizon1_full_series.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_pattern_tables(h1_pred_mean, h1_real_mean, tod_idx, dow_idx):
    slots_per_day = 288
    daily_pred = np.full(slots_per_day, np.nan, dtype=np.float32)
    daily_real = np.full(slots_per_day, np.nan, dtype=np.float32)
    daily_err = np.full(slots_per_day, np.nan, dtype=np.float32)

    for s in range(slots_per_day):
        mask = tod_idx == s
        if np.any(mask):
            daily_pred[s] = float(np.mean(h1_pred_mean[mask]))
            daily_real[s] = float(np.mean(h1_real_mean[mask]))
            daily_err[s] = float(np.mean(np.abs(h1_pred_mean[mask] - h1_real_mean[mask])))

    weekly_pred = np.full((7, slots_per_day), np.nan, dtype=np.float32)
    weekly_real = np.full((7, slots_per_day), np.nan, dtype=np.float32)
    weekly_err = np.full((7, slots_per_day), np.nan, dtype=np.float32)

    for d in range(7):
        for s in range(slots_per_day):
            mask = (dow_idx == d) & (tod_idx == s)
            if np.any(mask):
                weekly_pred[d, s] = float(np.mean(h1_pred_mean[mask]))
                weekly_real[d, s] = float(np.mean(h1_real_mean[mask]))
                weekly_err[d, s] = float(np.mean(np.abs(h1_pred_mean[mask] - h1_real_mean[mask])))

    return daily_pred, daily_real, daily_err, weekly_pred, weekly_real, weekly_err


def plot_daily_weekly_patterns(preds_np, reals_np, tod_idx, dow_idx, output_dir):
    h1_pred_mean = preds_np[:, 0, :, 0].mean(axis=1)
    h1_real_mean = reals_np[:, 0, :, 0].mean(axis=1)

    daily_pred, daily_real, daily_err, weekly_pred, weekly_real, weekly_err = build_pattern_tables(
        h1_pred_mean, h1_real_mean, tod_idx, dow_idx
    )

    slots = np.arange(288)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.3, 1.3])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(slots, daily_real, color="#1f77b4", linewidth=2.0, label="Ground Truth")
    ax1.plot(slots, daily_pred, color="#d62728", linewidth=2.0, alpha=0.9, label="Prediction")
    ax1.fill_between(slots, 0, daily_err, color="#ffbb78", alpha=0.35, label="|Error|")
    ax1.set_title("Daily Pattern Capture (Horizon-1, Mean Over Nodes)")
    ax1.set_ylabel("Traffic Flow")
    ax1.legend()
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        weekly_real,
        ax=ax2,
        cmap="viridis",
        cbar=True,
        xticklabels=48,
        yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    ax2.set_title("Weekly Pattern: Ground Truth (Dow x Time-of-Day)")
    ax2.set_ylabel("Day of Week")

    ax3 = fig.add_subplot(gs[2, 0])
    sns.heatmap(
        weekly_err,
        ax=ax3,
        cmap="magma",
        cbar=True,
        xticklabels=48,
        yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    ax3.set_title("Weekly Pattern: Absolute Error (Dow x Time-of-Day)")
    ax3.set_xlabel("Time-of-Day Slot (5-minute intervals)")
    ax3.set_ylabel("Day of Week")

    fig.tight_layout()
    path = os.path.join(output_dir, "daily_weekly_patterns_h1.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return path


def dynamic_graph_stats(graphs_np):
    if graphs_np is None or len(graphs_np) == 0:
        return None

    B, N, _ = graphs_np.shape
    bin_graphs = (graphs_np >= 0.5).astype(np.float32)
    off_diag = N * (N - 1)

    density = []
    mean_degree = []
    max_degree = []
    edge_change = [0.0]

    for i in range(B):
        g = bin_graphs[i].copy()
        np.fill_diagonal(g, 0.0)
        density.append(float(g.sum() / max(off_diag, 1)))
        degree = g.sum(axis=1)
        mean_degree.append(float(np.mean(degree)))
        max_degree.append(float(np.max(degree)))

        if i > 0:
            prev = bin_graphs[i - 1].copy()
            np.fill_diagonal(prev, 0.0)
            change = np.abs(g - prev).sum() / max(off_diag, 1)
            edge_change.append(float(change))

    return {
        "density": np.array(density),
        "mean_degree": np.array(mean_degree),
        "max_degree": np.array(max_degree),
        "edge_change": np.array(edge_change),
        "bin_graphs": bin_graphs,
    }


def plot_dynamic_graph_metrics(graph_stats, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    x = np.arange(len(graph_stats["density"]))

    axes[0].plot(x, graph_stats["density"], color="#1f77b4", linewidth=1.8)
    axes[0].set_title("Dynamic Graph Density Over Test Batches")
    axes[0].set_ylabel("Edge Density")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, graph_stats["mean_degree"], color="#2ca02c", linewidth=1.8, label="Mean Degree")
    axes[1].plot(x, graph_stats["max_degree"], color="#d62728", linewidth=1.8, label="Max Degree")
    axes[1].set_title("Dynamic Graph Degree Statistics")
    axes[1].set_ylabel("Degree")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, graph_stats["edge_change"], color="#9467bd", linewidth=1.8)
    axes[2].set_title("Inter-Batch Graph Change Rate")
    axes[2].set_ylabel("Change Ratio")
    axes[2].set_xlabel("Test Batch Index")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(output_dir, "dynamic_graph_metrics.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dynamic_graph_snapshots(graphs_np, graph_nodes, graph_snapshots, output_dir):
    total = len(graphs_np)
    if total == 0:
        return None

    graph_nodes = min(graph_nodes, graphs_np.shape[1])
    picks = np.linspace(0, total - 1, num=min(graph_snapshots, total), dtype=int)
    rows = len(picks)

    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.6 * rows), squeeze=False)
    for i, idx in enumerate(picks):
        ax = axes[i, 0]
        sns.heatmap(
            graphs_np[idx, :graph_nodes, :graph_nodes],
            ax=ax,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=(i == 0),
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(f"Dynamic Adjacency Snapshot - Batch {idx}")

    fig.tight_layout()
    path = os.path.join(output_dir, "dynamic_graph_snapshots.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_dynamic_graph_gif(graphs_np, graph_nodes, output_dir):
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as ex:
        print(f"[Warning] GIF export skipped (animation backend unavailable): {ex}")
        return None

    if graphs_np is None or len(graphs_np) < 2:
        return None

    graph_nodes = min(graph_nodes, graphs_np.shape[1])
    anim_data = graphs_np[:, :graph_nodes, :graph_nodes]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(anim_data[0], cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    title = ax.set_title("Dynamic Adjacency - Batch 0")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame_idx):
        im.set_data(anim_data[frame_idx])
        title.set_text(f"Dynamic Adjacency - Batch {frame_idx}")
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(anim_data), interval=180, blit=False)
    path = os.path.join(output_dir, "dynamic_graph_evolution.gif")

    try:
        ani.save(path, writer=PillowWriter(fps=6))
    except Exception as ex:
        print(f"[Warning] GIF export failed: {ex}")
        path = None

    plt.close(fig)
    return path


def save_metrics_json(metrics, graph_stats, output_dir):
    payload = {
        "overall_metrics": metrics["overall"],
        "horizon_1_metrics": metrics["horizon_1"],
    }

    if graph_stats is not None:
        payload["graph_metrics"] = {
            "avg_density": float(np.mean(graph_stats["density"])),
            "avg_mean_degree": float(np.mean(graph_stats["mean_degree"])),
            "avg_max_degree": float(np.mean(graph_stats["max_degree"])),
            "avg_change_rate": float(np.mean(graph_stats["edge_change"])),
        }

    path = os.path.join(output_dir, "test_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    print("=" * 70)
    print(f"DG-LLM Test + Visualization | Dataset: {args.data}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.model_path}")
    print(f"VMD cache: {args.vmd_cache_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)

    data_path = os.path.join(args.root_path, args.data, "processed")

    # Build a minimal training-arg namespace for existing loader/trainer APIs.
    runtime_args = SimpleNamespace(
        data=args.data,
        data_path=data_path,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        input_len=args.input_len,
        output_len=args.output_len,
        llm_layer=args.llm_layer,
        U=args.U,
        vmd_k=args.vmd_k,
        num_nodes=args.num_nodes,
        device=args.device,
        log_dir=args.output_dir,
        lrate=1e-3,
        wdecay=1e-5,
        grad_accum_steps=1,
        enable_compile=False,
        use_amp=False,
        use_bf16=False,
        vmd_cache_dir=args.vmd_cache_dir,
    )

    print("\n[1/5] Loading test data (with VMD cache)...")
    data = load_dataset_optimized(runtime_args.data_path, runtime_args.batch_size, runtime_args)

    print("\n[2/5] Building model and loading checkpoint...")
    adj_mx = load_adjacency(args.root_path, args.data, args.num_nodes)
    trainer = VMD_Trainer(runtime_args, data["scaler"], adj_mx, args.device)
    load_model_weights(trainer, args.model_path, args.device)

    print("\n[3/5] Running test inference...")
    outputs = run_test_inference(trainer, data, args)

    preds = outputs["preds"]
    reals = outputs["reals"]
    tod_idx = outputs["tod_idx"]
    dow_idx = outputs["dow_idx"]
    graphs = outputs["graphs"]

    node_idx = int(np.clip(args.node_idx, 0, args.num_nodes - 1))
    metrics = compute_metrics(preds, reals)

    print("\n[4/5] Creating visualizations...")
    preds_np = preds.numpy()
    reals_np = reals.numpy()
    generated_files = []

    # --- full horizon-1 series + daily/weekly heatmaps (new) ---
    generated_files.append(plot_full_horizon1_series(preds_np, reals_np, node_idx, args.output_dir))
    generated_files.append(plot_daily_weekly_patterns(preds_np, reals_np, tod_idx, dow_idx, args.output_dir))

    # --- per-node / per-horizon day-window plot (from visualization.py) ---
    generated_files.append(
        visualize_model_predictions(
            preds, reals,
            node_idx=node_idx, horizon_idx=0,
            num_samples=288,
            save_dir=args.output_dir,
        )
    )

    # --- error distribution + scatter (from visualization.py) ---
    generated_files.append(
        visualize_advanced_diagnostics(
            preds, reals,
            node_idx=node_idx,
            save_dir=args.output_dir,
        )
    )

    # --- 1-week zoomed forecast (from visualization.py) ---
    generated_files.append(
        visualize_weekly_horizon1(
            preds, reals,
            node_idx=node_idx,
            save_dir=args.output_dir,
        )
    )

    graph_stats = dynamic_graph_stats(graphs)
    if graph_stats is not None:
        generated_files.append(plot_dynamic_graph_metrics(graph_stats, args.output_dir))
        snapshot_path = plot_dynamic_graph_snapshots(
            graph_stats["bin_graphs"], args.graph_nodes, args.graph_snapshots, args.output_dir
        )
        if snapshot_path is not None:
            generated_files.append(snapshot_path)
        if args.save_graph_gif:
            gif_path = save_dynamic_graph_gif(graph_stats["bin_graphs"], args.graph_nodes, args.output_dir)
            if gif_path is not None:
                generated_files.append(gif_path)

    print("\n[5/5] Saving summary...")
    summary_path = save_metrics_json(metrics, graph_stats, args.output_dir)
    generated_files.append(summary_path)

    print("\n" + "=" * 70)
    print("Completed: Test inference + visualization bundle")
    print("Overall Metrics:")
    print(
        f"  MAE={metrics['overall']['mae']:.4f}, "
        f"RMSE={metrics['overall']['rmse']:.4f}, "
        f"MAPE={metrics['overall']['mape']:.4f}"
    )
    print("Horizon-1 Metrics:")
    print(
        f"  MAE={metrics['horizon_1']['mae']:.4f}, "
        f"RMSE={metrics['horizon_1']['rmse']:.4f}, "
        f"MAPE={metrics['horizon_1']['mape']:.4f}"
    )
    print("Generated files:")
    for p in generated_files:
        print(f"  - {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
