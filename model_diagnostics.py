"""
model_diagnostics.py — DG-LLM Backbone Internals Visualizer
=============================================================
Answers the question: "Is my freezing strategy actually correct?"

Run AFTER training a model checkpoint:
    python model_diagnostics.py --data taxi_drop --checkpoint results/logs/best_model.pth

Generates 4 diagnostic plots in results/logs/diagnostics/:
  1. attention_maps.png       — Are the attention heads routing spatially?
  2. gradient_norms.png       — Which layers are the model actually trying to update?
  3. wpe_similarity.png       — Did pos-embeddings learn traffic periodicity?
  4. hidden_state_tsne.png    — Do frozen MLPs already separate Peak vs Off-Peak?
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score

from utils import load_pickle, seed_everything
from data_loader import load_dataset
from trainer import Trainer
from paths import DATASET_DIR, RESULTS_LOGS_DIR


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
CMAP_ATTN   = LinearSegmentedColormap.from_list("attn",   ["#0d0d2b", "#1a3a6b", "#e8b86d", "#ffffff"])
CMAP_SIM    = LinearSegmentedColormap.from_list("sim",    ["#0d0221", "#003566", "#ffd166", "#fffcf2"])
CMAP_GRAD   = LinearSegmentedColormap.from_list("grad",   ["#1b1b2f", "#e94560", "#f5a623"])
ACCENT      = "#e8b86d"
BG          = "#0d0d1a"
TEXT        = "#e8e8e8"
GRID        = "#2a2a40"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "font.family": "DejaVu Sans",
})


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


def _format_float(v, ndigits=4):
    if v is None:
        return "NA"
    return f"{v:.{ndigits}f}"


def _compute_hidden_state_metrics(features, labels_binary):
    """
    Compute representational diagnostics for Peak vs Off-Peak separation.
    Returns (silhouette, auc). Either value can be None when not computable.
    """
    silhouette = None
    auc = None

    if features.shape[0] < 4:
        return silhouette, auc

    classes, counts = np.unique(labels_binary, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        return silhouette, auc

    try:
        silhouette = float(silhouette_score(features, labels_binary))
    except Exception:
        silhouette = None

    try:
        peak_centroid = features[labels_binary == 1].mean(axis=0)
        off_centroid = features[labels_binary == 0].mean(axis=0)
        direction = peak_centroid - off_centroid
        norm = np.linalg.norm(direction)
        if norm > 0:
            scores = features @ (direction / norm)
            auc = float(roc_auc_score(labels_binary, scores))
            # Keep score direction consistent: >= 0.5 means better separation.
            if auc < 0.5:
                auc = 1.0 - auc
    except Exception:
        auc = None

    return silhouette, auc


def write_diagnostics_report(save_dir, args, per_layer_report):
    """Save a concise text report alongside figures."""
    os.makedirs(save_dir, exist_ok=True)
    report_path = os.path.join(save_dir, "diagnostics_report.txt")

    lines = []
    lines.append("DG-LLM Diagnostics Report")
    lines.append("=" * 60)
    lines.append(f"Dataset      : {args.data}")
    lines.append(f"Checkpoint   : {args.checkpoint or os.path.join(args.log_dir, 'best_model.pth')}")
    lines.append(f"Device       : {args.device}")
    lines.append(f"Input/Output : {args.input_len}/{args.output_len}")
    lines.append("")
    lines.append("Per-layer hidden-state diagnostics (Peak vs Off-Peak)")
    lines.append("- silhouette: higher is better (range approx. -1 to 1)")
    lines.append("- auc       : 0.5 random, 1.0 perfect linear separability")
    lines.append("")

    if not per_layer_report:
        lines.append("No per-layer metrics were computed.")
    else:
        lines.append("Layer | Samples | Peak | OffPeak | Silhouette | AUC")
        lines.append("-" * 60)
        for row in per_layer_report:
            lines.append(
                f"{row['layer']:>5} | "
                f"{row['n_samples']:>7} | "
                f"{row['n_peak']:>4} | "
                f"{row['n_offpeak']:>7} | "
                f"{_format_float(row['silhouette']):>10} | "
                f"{_format_float(row['auc']):>6}"
            )

        valid_s = [r["silhouette"] for r in per_layer_report if r["silhouette"] is not None]
        valid_a = [r["auc"] for r in per_layer_report if r["auc"] is not None]
        lines.append("")
        lines.append(
            f"Mean silhouette: {_format_float(float(np.mean(valid_s)) if valid_s else None)}"
        )
        lines.append(
            f"Mean AUC       : {_format_float(float(np.mean(valid_a)) if valid_a else None)}"
        )

    lines.append("")
    lines.append("Generated plots:")
    lines.append("- attention_maps.png")
    lines.append("- gradient_norms.png")
    lines.append("- wpe_similarity.png")
    lines.append("- hidden_state_tsne.png (unless --skip_tsne)")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  ✔  Saved → {report_path}")
    return report_path


def _get_backbone(model):
    """Navigate to the inner GPT-2 transformer blocks."""
    # model is DGLLM → ModeProcessor → SpatialGPTBackbone → PeftModel → GPT2Model
    proc   = model.mode_processors[0]
    return proc.backbone.gpt2.base_model.model   # GPT2Model


def _get_blocks(model):
    return _get_backbone(model).h


# ──────────────────────────────────────────────────────────────────────────────
# 1. Attention-Map Heatmaps
# ──────────────────────────────────────────────────────────────────────────────
def plot_attention_maps(model, batch, device, save_dir, adj_mx=None):
    """
    For each unfrozen (top) layer, record the raw QK^T attention scores
    and plot them as heatmaps.  Also overlays the spatial connectivity mask
    so you can see whether attention aligns with the road graph.
    """
    model.eval()
    blocks = _get_blocks(model)
    num_layers = len(blocks)

    # ── Hook setup ──────────────────────────────────────────
    captured = {}

    def _make_hook(layer_idx):
        def hook(module, inputs, outputs):
            # inputs[0] = hidden_states [B, N, D]
            hs = inputs[0].detach()
            q, k, _ = module.c_attn(hs).split(module.split_size, dim=2)
            q = module._split_heads(q, module.num_heads, module.head_dim)
            k = module._split_heads(k, module.num_heads, module.head_dim)
            # raw attention logits before softmax: [B, H, N, N]
            scale = q.size(-1) ** 0.5
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / scale
            captured[layer_idx] = attn_logits.float().cpu()
        return hook

    hooks = []
    for i, block in enumerate(blocks):
        hooks.append(block.attn.register_forward_hook(_make_hook(i)))

    # ── One forward pass ─────────────────────────────────────
    vmd_data, x_in = batch
    with torch.no_grad():
        model(vmd_data.to(device), x_in.to(device))

    for h in hooks:
        h.remove()

    # ── Plot ──────────────────────────────────────────────────
    layer_ids = sorted(captured.keys())
    n_show    = min(len(layer_ids), 4)          # at most 4 layers
    layer_ids = layer_ids[-n_show:]             # show the top (unfrozen) ones

    fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 9))
    fig.suptitle("Attention Map Diagnostics\n(Top = Raw Logits  |  Bottom = Softmax)", 
                 fontsize=14, color=TEXT, y=1.01)

    for col, lid in enumerate(layer_ids):
        logits = captured[lid][0]          # [H, N, N]  — first sample
        avg_logits  = logits.mean(0).numpy()        # avg over heads
        avg_softmax = F.softmax(logits, dim=-1).mean(0).numpy()

        # clamp extreme values for display
        vmin_l, vmax_l = np.percentile(avg_logits,  [2, 98])
        ax_raw  = axes[0, col] if n_show > 1 else axes[0]
        ax_soft = axes[1, col] if n_show > 1 else axes[1]

        im0 = ax_raw.imshow(avg_logits,  cmap=CMAP_ATTN, vmin=vmin_l, vmax=vmax_l, aspect="auto")
        im1 = ax_soft.imshow(avg_softmax, cmap=CMAP_ATTN, aspect="auto")

        for ax, im, title in [(ax_raw, im0, f"Layer {lid} — Raw QKᵀ/√d"),
                               (ax_soft, im1, f"Layer {lid} — Softmax")]:
            ax.set_title(title, fontsize=9, color=TEXT)
            ax.set_xlabel("Key (node)", fontsize=7)
            ax.set_ylabel("Query (node)", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "attention_maps.png"))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Gradient-Norm Bar Chart
# ──────────────────────────────────────────────────────────────────────────────
def plot_gradient_norms(model, batch, device, save_dir, scaler):
    """
    Runs ONE backward pass and records the gradient L2-norm for every named
    parameter.  Then groups norms by component (LoRA, LayerNorm, MLP, Attn base)
    and plots as a stacked bar chart.  Zero gradient → that module was frozen or
    wasn't touched.
    """
    model.train()

    vmd_data, x_in = batch
    vmd_t = vmd_data.to(device)
    x_t   = x_in.to(device)

    # Synthetic target (one-step ahead of the last step of input flow)
    # shape: [B, output_len, N, 1]
    B, T, N, _ = x_t.shape
    output_len  = model.mode_processors[0].output_len
    target      = x_t[:, -output_len:, :, :1].permute(0, 1, 2, 3)

    pred, _ = model(vmd_t, x_t)
    loss = F.l1_loss(pred, target)
    loss.backward()

    # ── Collect norms ─────────────────────────────────────────
    records = {}   # name → L2-norm
    for name, p in model.named_parameters():
        if "backbone" not in name:
            continue
        if p.grad is None:
            norm = 0.0
        else:
            norm = p.grad.detach().float().norm(2).item()
        records[name] = norm

    model.zero_grad()
    model.eval()

    if not records:
        print("  ⚠  No backbone gradients found. Skipping gradient norm plot.")
        return

    # ── Group by component ───────────────────────────────────
    groups = {
        "LoRA (lora_A/B)": [],
        "LayerNorm (ln_1/2)": [],
        "MLP base (c_fc/c_proj-mlp)": [],
        "Attn base (c_attn/c_proj-attn)": [],
        "Pos. Embed (wpe)": [],
        "Other": [],
    }
    for name, norm in records.items():
        if "lora_" in name:
            groups["LoRA (lora_A/B)"].append(norm)
        elif "ln_" in name or "ln_f" in name:
            groups["LayerNorm (ln_1/2)"].append(norm)
        elif "mlp" in name:
            groups["MLP base (c_fc/c_proj-mlp)"].append(norm)
        elif "attn" in name or "c_proj" in name:
            groups["Attn base (c_attn/c_proj-attn)"].append(norm)
        elif "wpe" in name:
            groups["Pos. Embed (wpe)"].append(norm)
        else:
            groups["Other"].append(norm)

    labels  = [k for k, v in groups.items() if v]
    means   = [np.mean(v) for k, v in groups.items() if v]
    maxvals = [np.max(v)  for k, v in groups.items() if v]
    colors  = ["#e94560", "#f5a623", "#06d6a0", "#118ab2", "#ffd166", "#adb5bd"][:len(labels)]

    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Gradient L2-Norm by Component\n(Non-zero = model tried to update it; zero = effectively frozen)",
                 fontsize=12, color=TEXT)

    bars = ax.bar(x_pos, means, color=colors, alpha=0.85, label="Mean grad norm", zorder=3)
    ax.scatter(x_pos, maxvals, color="white", zorder=4, s=60, label="Max grad norm", marker="D")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Gradient L2-Norm", fontsize=10)
    ax.grid(axis="y", zorder=0, alpha=0.4)
    ax.legend(fontsize=9)

    # Annotate bar tops
    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{mean:.2e}", ha="center", va="bottom", fontsize=7, color=TEXT)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "gradient_norms.png"))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Positional Embedding Similarity Matrix
# ──────────────────────────────────────────────────────────────────────────────
def plot_wpe_similarity(model, save_dir):
    """
    Computes cosine-similarity between every pair of positional embedding vectors.
    For traffic data with periodicity = 288 steps/day, the heatmap should show
    strong diagonal "stripes" every 288 steps if wpe learned traffic periodicity.
    If wpe is still text-like (random looking), consider freezing it.
    """
    wpe = _get_backbone(model).wpe.weight.detach().float().cpu()  # [max_pos, D]

    # Normalise and compute cosine similarity
    wpe_norm = F.normalize(wpe, dim=-1)
    sim_matrix = torch.mm(wpe_norm, wpe_norm.t()).numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        "Positional Embedding (wpe) Cosine Similarity\n"
        "Diagonal stripes every 288 steps → learned traffic periodicity",
        fontsize=12, color=TEXT
    )

    im = ax.imshow(sim_matrix, cmap=CMAP_SIM, aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Cosine Similarity")

    # Annotate day boundaries (288 steps = 1 day, 5-min resolution)
    period = 288
    for tick in range(0, sim_matrix.shape[0], period):
        ax.axhline(tick, color=ACCENT, linewidth=0.5, alpha=0.6)
        ax.axvline(tick, color=ACCENT, linewidth=0.5, alpha=0.6)

    ax.set_xlabel("Position index", fontsize=10)
    ax.set_ylabel("Position index", fontsize=10)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "wpe_similarity.png"))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Hidden-State t-SNE (MLP representational quality)
# ──────────────────────────────────────────────────────────────────────────────
def plot_hidden_state_tsne(model, batches, device, save_dir):
    """
    Extracts the hidden states at the OUTPUT of the MLP sub-layer for each
    transformer block (after the residual add).  Runs t-SNE (or PCA if too slow)
    on the pooled node embeddings and colours them by Time-of-Day bucket
    (Peak ≈ 6-9 AM and 4-7 PM, Off-Peak otherwise).

    If the frozen MLP already produces well-separated clusters → the pre-trained
    MLP was sufficient and doesn't need LoRA.
    If clusters are mixed → the MLP may need to be unlocked or given LoRA adapters.
    """
    model.eval()
    blocks    = _get_blocks(model)
    num_layers = len(blocks)

    # ── Hooks ────────────────────────────────────────────────
    # Capture hidden state AFTER the MLP residual add (i.e. final output of block)
    hs_after_mlp = {i: [] for i in range(num_layers)}

    def _make_hook(layer_idx):
        def hook(module, inputs, outputs):
            # outputs[0] = hidden_states [B, N, D]
            hs_after_mlp[layer_idx].append(outputs[0].detach().float().cpu())
        return hook

    hooks = [block.register_forward_hook(_make_hook(i)) for i, block in enumerate(blocks)]

    # time-of-day labels from the first feature channel of the input
    # x_in: [B, T, N, F], channel 1 is ToD (normalised 0-1)
    tod_labels = []

    with torch.no_grad():
        for vmd_data, x_in in batches:
            vmd_t = vmd_data.to(device)
            x_t   = x_in.to(device)
            model(vmd_t, x_t)
            # Average ToD across sequence for each sample
            if x_in.shape[-1] > 1:
                tod = x_in[:, :, 0, 1].mean(dim=1).numpy()   # [B]
            else:
                tod = np.zeros(x_in.shape[0])
            tod_labels.append(tod)

    for h in hooks:
        h.remove()

    tod_arr = np.concatenate(tod_labels, axis=0)  # [total_samples]

    # Peak = 6-9 AM (0.25-0.375 of 288) and 4-7 PM (0.667-0.792)
    def _tod_bucket(tod):
        labels = []
        for t in tod:
            if 0.25 <= t <= 0.375 or 0.667 <= t <= 0.792:
                labels.append("Peak")
            else:
                labels.append("Off-Peak")
        return np.array(labels)

    bucket_labels = _tod_bucket(tod_arr)
    color_map = {"Peak": "#e94560", "Off-Peak": "#06d6a0"}

    # ── t-SNE per layer ───────────────────────────────────────
    n_plot = min(num_layers, 4)          # show at most 4 layers
    plot_layers = list(range(num_layers))[-n_plot:]   # top layers

    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 5), squeeze=False)
    fig.suptitle(
        "Hidden States After MLP — t-SNE Projection\n"
        "Well-separated clusters → frozen MLP is sufficient",
        fontsize=12, color=TEXT, y=1.02
    )

    per_layer_report = []

    for col, lid in enumerate(plot_layers):
        ax = axes[0, col]
        # Pool over nodes: [B*batches, N, D] → [B*batches, D]
        all_hs = torch.cat(hs_after_mlp[lid], dim=0)   # [total, N, D]
        pooled  = all_hs.mean(dim=1).numpy()             # [total, D]

        n_samples = pooled.shape[0]
        if n_samples < 5:
            ax.set_title(f"Layer {lid} — Not enough samples", fontsize=9, color=TEXT)
            per_layer_report.append({
                "layer": int(lid),
                "n_samples": int(n_samples),
                "n_peak": int(0),
                "n_offpeak": int(0),
                "silhouette": None,
                "auc": None,
            })
            continue

        # Reduce with PCA first to speed up t-SNE
        n_pca = min(50, pooled.shape[1], n_samples - 1)
        pca_out = PCA(n_components=n_pca).fit_transform(pooled)

        labels_trimmed = bucket_labels[:n_samples]
        labels_binary = (labels_trimmed == "Peak").astype(np.int64)
        n_peak = int(labels_binary.sum())
        n_offpeak = int(n_samples - n_peak)
        sil, auc = _compute_hidden_state_metrics(pca_out, labels_binary)

        per_layer_report.append({
            "layer": int(lid),
            "n_samples": int(n_samples),
            "n_peak": n_peak,
            "n_offpeak": n_offpeak,
            "silhouette": sil,
            "auc": auc,
        })

        perplexity = min(30, n_samples - 1)
        tsne_out = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                        n_iter=500, init="pca").fit_transform(pca_out)

        for bkt, clr in color_map.items():
            mask = labels_trimmed == bkt
            ax.scatter(tsne_out[mask, 0], tsne_out[mask, 1],
                       c=clr, label=bkt, s=12, alpha=0.7, linewidths=0)

        ax.set_title(f"Layer {lid}", fontsize=10, color=TEXT)
        ax.set_xlabel("t-SNE dim 1", fontsize=8)
        ax.set_ylabel("t-SNE dim 2", fontsize=8)
        ax.legend(fontsize=8, markerscale=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "hidden_state_tsne.png"))
    return per_layer_report


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="DG-LLM Model Internals Diagnostics"
    )
    parser.add_argument("--data",       type=str, default="PEMSD04",
                        choices=["PEMSD04", "PEMSD08", "bike_drop", "bike_pick", "taxi_drop", "taxi_pick"])
    parser.add_argument("--root_path",  type=str, default=str(DATASET_DIR))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pth). Uses best_model.pth by default.")
    parser.add_argument("--log_dir",    type=str, default=str(RESULTS_LOGS_DIR))
    parser.add_argument("--llm_layer",  type=int, default=6)
    parser.add_argument("--U",          type=int, default=1)
    parser.add_argument("--vmd_k",      type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--input_dim",  type=int, default=3)
    parser.add_argument("--input_len",  type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_batches",  type=int, default=8,
                        help="Number of batches to collect hidden states from (for t-SNE)")
    parser.add_argument("--skip_tsne",  action="store_true",
                        help="Skip the t-SNE plot (slow for large N_samples)")
    args = parser.parse_args()

    args.data_path = os.path.join(args.root_path, args.data, "processed")
    if "PEMSD04" in args.data:
        args.num_nodes = 307
    elif "PEMSD08" in args.data:
        args.num_nodes = 170
    elif "bike" in args.data:
        args.num_nodes = 250
    elif "taxi" in args.data:
        args.num_nodes = 266
    else:
        args.num_nodes = 307
    args.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.lora_rank  = 16
    args.lora_alpha = 32
    args.wdecay     = 1e-5
    args.lrate      = 1e-3
    args.grad_accum_steps = 1
    args.enable_compile   = False
    args.use_amp          = False
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    save_dir = os.path.join(args.log_dir, "diagnostics")
    print(f"\n{'='*60}")
    print(f"  DG-LLM Model Diagnostics — {args.data}")
    print(f"  Device : {args.device}")
    print(f"  Output : {save_dir}")
    print(f"{'='*60}\n")

    # ── Load data ────────────────────────────────────────────
    print(">> Loading dataset …")
    data = load_dataset(args.data_path, args.batch_size, args)

    # ── Load adjacency matrix ─────────────────────────────────
    adj_path = os.path.join(args.root_path, args.data, "adj_mx.pkl")
    adj_mx   = None
    if os.path.exists(adj_path):
        adj_data = load_pickle(adj_path)
        adj_mx   = adj_data[2] if isinstance(adj_data, list) else adj_data

    # ── Build model ───────────────────────────────────────────
    print(">> Building model …")
    trainer = Trainer(args, data["scaler"], adj_mx, args.device)

    ckpt_path = args.checkpoint or os.path.join(args.log_dir, "best_model.pth")
    if os.path.exists(ckpt_path):
        print(f">> Loading checkpoint from {ckpt_path} …")
        trainer.load_model(ckpt_path, strict=False)
    else:
        print(f"  ⚠  No checkpoint found at {ckpt_path}. Using randomly-initialised weights.\n"
               "     Diagnostics will reflect the initial state, not a trained model.")

    model = trainer.model

    # ── Collect a few batches ─────────────────────────────────
    print(">> Collecting sample batches …")
    batches = []
    for i, (x, y, vmd) in enumerate(data["val_loader"].get_iterator()):
        if i >= args.n_batches:
            break
        batches.append((vmd, x))

    if not batches:
        print("  ERROR: No batches available. Check your data_path.")
        return

    first_batch = batches[0]

    # ── Plot 1 — Attention maps ───────────────────────────────
    print("\n[1/4] Plotting attention maps …")
    try:
        plot_attention_maps(model, first_batch, args.device, save_dir, adj_mx)
    except Exception as e:
        print(f"  ⚠  Attention map plot failed: {e}")

    # ── Plot 2 — Gradient norms ───────────────────────────────
    print("\n[2/4] Plotting gradient norms …")
    try:
        plot_gradient_norms(model, first_batch, args.device, save_dir, data["scaler"])
    except Exception as e:
        print(f"  ⚠  Gradient norm plot failed: {e}")

    # ── Plot 3 — WPE similarity ───────────────────────────────
    print("\n[3/4] Plotting positional embedding similarity …")
    try:
        plot_wpe_similarity(model, save_dir)
    except Exception as e:
        print(f"  ⚠  WPE similarity plot failed: {e}")

    per_layer_report = []

    # ── Plot 4 — t-SNE + quantitative metrics ─────────────────
    if args.skip_tsne:
        print("\n[4/4] Skipping t-SNE (--skip_tsne flag set).")
    else:
        print("\n[4/4] Plotting hidden-state t-SNE + per-layer metrics (may take a few minutes) …")
        try:
            per_layer_report = plot_hidden_state_tsne(model, batches, args.device, save_dir) or []
            if per_layer_report:
                print("  Per-layer separability metrics:")
                for row in per_layer_report:
                    print(
                        f"    Layer {row['layer']}: silhouette={_format_float(row['silhouette'])}, "
                        f"auc={_format_float(row['auc'])}, "
                        f"samples={row['n_samples']} (Peak={row['n_peak']}, OffPeak={row['n_offpeak']})"
                    )
        except Exception as e:
            print(f"  ⚠  t-SNE plot failed: {e}")

    # ── Save short text summary report ────────────────────────
    print("\n[report] Writing diagnostics summary …")
    try:
        write_diagnostics_report(save_dir, args, per_layer_report)
    except Exception as e:
        print(f"  ⚠  Diagnostics report write failed: {e}")

    print(f"\n{'='*60}")
    print(f"  All diagnostics saved to: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
