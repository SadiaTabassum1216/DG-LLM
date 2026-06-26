import os
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from utils import Ranger, MAE_torch, MAPE_torch, RMSE_torch, compute_metrics
from model import DGLLM

class Trainer:
    def __init__(self, args, scaler, adj_mx, device):
        self.args = args
        self.device = device
        self.scaler = scaler       

        self.model = DGLLM(
            device,
            adj_mx,
            args.input_dim,
            args.num_nodes,
            args.input_len,
            args.output_len,
            args.llm_layer,
            args.U,
            unfrozen_bottom_layers=getattr(args, "unfrozen_bottom_layers", 2),
            middle_lora_layers=getattr(args, "middle_lora_layers", None),
            vmd_K=args.vmd_k,
        ).to(device)

        if hasattr(args, "enable_compile") and args.enable_compile:
            if hasattr(torch, "compile"):
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("  >> PyTorch compilation ENABLED (~30% speedup)")
            else:
                print("  >> WARNING: torch.compile not available (requires PyTorch 2.0+)")

        self.optimizer = Ranger(self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        self.loss_fn = MAE_torch

        self.grad_accum_steps = getattr(args, "grad_accum_steps", 1)
        if self.grad_accum_steps > 1:
            effective_batch = args.batch_size * self.grad_accum_steps
            print(
                f"  >> Gradient Accumulation: {self.grad_accum_steps} steps "
                f"(Effective batch = {effective_batch})"
            )

        self.use_amp = getattr(args, "use_amp", False) or getattr(args, "use_bf16", False)
        self.amp_dtype = None
        self.grad_scaler = None

        if self.use_amp and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 8:
                self.amp_dtype = torch.bfloat16
                print(f"  >> AMP ENABLED: BF16 (GPU compute {cc[0]}.{cc[1]}, no overflow risk)")
            else:
                self.amp_dtype = torch.float16
                self.grad_scaler = torch.amp.GradScaler("cuda")
                print(f"  >> AMP ENABLED: FP16 + GradScaler (GPU compute {cc[0]}.{cc[1]})")
        else:
            print("  >> Mixed Precision DISABLED")

        self.log_dir = args.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_val_loss = float("inf")

    def probe_layer_importance(self, data_loader, n_batches: int = 10, save_dir: str = None) -> dict:
        """
        Counterfactual gradient probe: answers "what would each frozen layer learn if unfrozen?"

        Procedure
        ---------
        1. Save the current requires_grad state of every parameter.
        2. Temporarily enable gradients for ALL GPT blocks (frozen and unfrozen alike).
        3. Run `n_batches` of real forward+backward without an optimizer step.
        4. Record per-GPT-layer gradient L2 norms.
        5. Restore every parameter to its original requires_grad state.
        6. Save a bar chart comparing "probe" (all-unfrozen) vs "actual" (your current config).

        The resulting plot tells you:
          - If a frozen layer shows a HIGH probe grad norm → you may be suppressing learning.
          - If a frozen layer shows a LOW probe grad norm  → freezing it was the right call.

        Args:
            data_loader : DataLoader with a get_iterator() method (your standard loader).
            n_batches   : Number of batches to average over (10 is usually enough).
            save_dir    : Where to save the plot. Defaults to self.log_dir/diagnostics/.

        Returns:
            dict with keys 'probe_norms' and 'actual_norms', each a list indexed by GPT layer.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = save_dir or os.path.join(self.log_dir, "diagnostics")
        os.makedirs(save_dir, exist_ok=True)

        # ── 1. Snapshot current requires_grad state ────────────────────────────
        saved_states = {
            name: param.requires_grad
            for name, param in self.model.named_parameters()
        }

        gpt_blocks = self.model.mode_processors[0].backbone.gpt2.base_model.model.h
        n_layers = len(gpt_blocks)

        # ── 2. Temporarily enable ALL GPT block gradients ─────────────────────
        for processor in self.model.mode_processors:
            for block in processor.backbone.gpt2.base_model.model.h:
                for param in block.parameters():
                    param.requires_grad = True

        # ── 3. Accumulate gradients over n_batches ────────────────────────────
        self.model.train()
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None

        batches_run = 0
        gat_weight = getattr(self.args, "gat_aux_weight", 0.01)

        for x, y, vmd in data_loader.get_iterator():
            if batches_run >= n_batches:
                break
            tx   = x.to(self.device, non_blocking=True)
            ty   = y.to(self.device, non_blocking=True)
            tvmd = vmd.to(self.device, non_blocking=True)

            preds, _, gat_aux_loss = self.model(tvmd, tx)
            preds_scaled = self.scaler.inverse_transform(preds)
            task_loss    = self.loss_fn(preds_scaled, ty, 0.0)
            loss         = task_loss + gat_weight * gat_aux_loss
            loss.backward()
            batches_run += 1

        # ── 4. Collect per-layer probe grad norms (averaged across VMD modes) ──
        probe_norms = []
        attn_norms = []
        mlp_norms = []
        ln_norms = []
        
        for layer_idx in range(n_layers):
            total_norm_sq = 0.0
            total_attn_sq = 0.0
            total_mlp_sq = 0.0
            total_ln_sq = 0.0
            for processor in self.model.mode_processors:
                block = processor.backbone.gpt2.base_model.model.h[layer_idx]
                for name, param in block.named_parameters():
                    if param.grad is not None:
                        norm_sq = param.grad.norm().item() ** 2
                        total_norm_sq += norm_sq
                        if "attn" in name:
                            total_attn_sq += norm_sq
                        elif "mlp" in name:
                            total_mlp_sq += norm_sq
                        elif "ln" in name:
                            total_ln_sq += norm_sq
            
            num_procs = len(self.model.mode_processors)
            probe_norms.append(total_norm_sq ** 0.5 / num_procs)
            attn_norms.append(total_attn_sq ** 0.5 / num_procs)
            mlp_norms.append(total_mlp_sq ** 0.5 / num_procs)
            ln_norms.append(total_ln_sq ** 0.5 / num_procs)

        # ── 5. Restore original requires_grad state ───────────────────────────
        for name, param in self.model.named_parameters():
            param.requires_grad = saved_states[name]
            if param.grad is not None:
                param.grad = None           # clear probe grads

        # ── 6. Collect "actual" norms (under current freeze config) ───────────
        #      Run n_batches again with the real freeze config to get actual norms.
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None

        batches_run = 0
        for x, y, vmd in data_loader.get_iterator():
            if batches_run >= n_batches:
                break
            tx   = x.to(self.device, non_blocking=True)
            ty   = y.to(self.device, non_blocking=True)
            tvmd = vmd.to(self.device, non_blocking=True)

            preds, _, gat_aux_loss = self.model(tvmd, tx)
            preds_scaled = self.scaler.inverse_transform(preds)
            task_loss    = self.loss_fn(preds_scaled, ty, 0.0)
            loss         = task_loss + gat_weight * gat_aux_loss
            loss.backward()
            batches_run += 1

        actual_norms = []
        for layer_idx in range(n_layers):
            total_norm_sq = 0.0
            for processor in self.model.mode_processors:
                block = processor.backbone.gpt2.base_model.model.h[layer_idx]
                for param in block.parameters():
                    if param.grad is not None:
                        total_norm_sq += param.grad.norm().item() ** 2
            actual_norms.append(total_norm_sq ** 0.5 / len(self.model.mode_processors))

        # Clear grads after probe is done
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None

        # ── 7. Plot ───────────────────────────────────────────────────────────
        labels       = [f"L{i}" for i in range(n_layers)]
        x_pos        = range(n_layers)
        width        = 0.38
        frozen_start = n_layers - self.args.U   # first unfrozen layer index

        fig, ax = plt.subplots(figsize=(max(8, n_layers * 1.1), 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        bars_probe  = ax.bar([p - width / 2 for p in x_pos], probe_norms,
                             width, label="Probe (all unfrozen)", color="#58a6ff", alpha=0.85)
        bars_actual = ax.bar([p + width / 2 for p in x_pos], actual_norms,
                             width, label=f"Actual (U={self.args.U} top unfrozen)", color="#3fb950", alpha=0.85)

        # Shade frozen region
        ax.axvspan(-0.5, frozen_start - 0.5, color="#ff7b72", alpha=0.07, label="Frozen zone")
        ax.axvline(frozen_start - 0.5, color="#ff7b72", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.text(frozen_start - 0.5, ax.get_ylim()[1] * 0.95, "  freeze boundary",
                color="#ff7b72", fontsize=8, va="top")

        for bar, val in zip(bars_probe, probe_norms):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1e}", ha="center", va="bottom", fontsize=6.5, color="#c9d1d9")
        for bar, val in zip(bars_actual, actual_norms):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1e}", ha="center", va="bottom", fontsize=6.5, color="#c9d1d9")

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, color="#c9d1d9")
        ax.set_ylabel("Gradient L2-Norm", color="#c9d1d9")
        ax.set_xlabel("GPT Layer Index", color="#c9d1d9")
        ax.tick_params(colors="#c9d1d9")
        ax.set_title(
            "Layer-by-Layer Gradient Probe\n"
            "Blue = what each layer WOULD learn if unfrozen | "
            "Green = what it actually learns now",
            color="#e6edf3", fontsize=10, pad=10,
        )
        legend = ax.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
        ax.spines[:].set_color("#30363d")

        plt.tight_layout()
        out_path = os.path.join(save_dir, "layer_importance_probe.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  >> Layer importance probe saved to: {out_path}")

        # ── 8. Print summary table ─────────────────────────────────────────────
        print(f"\n  {'Layer':<8} {'Probe norm':>12} | {'Attn norm':>11} | {'MLP norm':>11} | {'LN norm':>11}")
        print(f"  {'-'*65}")
        for i, (pn, attn_n, mlp_n, ln_n) in enumerate(zip(probe_norms, attn_norms, mlp_norms, ln_norms)):
            print(f"  L{i:<7} {pn:>12.4e} | {attn_n:>11.4e} | {mlp_n:>11.4e} | {ln_n:>11.4e}")

        # Summary of which component has the most info across all layers
        total_attn = sum(attn_norms)
        total_mlp = sum(mlp_norms)
        total_ln = sum(ln_norms)
        print(f"\n  [Summary of Components across all layers]")
        print(f"  Total Attention Norm : {total_attn:.4e}")
        print(f"  Total MLP Norm       : {total_mlp:.4e}")
        print(f"  Total LayerNorm Norm : {total_ln:.4e}")
        
        # Max layer
        max_layer = np.argmax(probe_norms)
        print(f"\n  [Conclusion]")
        print(f"  Layer storing most info (highest grad norm): L{max_layer} ({probe_norms[max_layer]:.4e})")
        print()

        return {"probe_norms": probe_norms, "actual_norms": actual_norms}

    def collect_epoch_grad_norms(self) -> dict:
        """
        Collect per-GPT-layer gradient L2 norms for the current epoch.
        Call this immediately AFTER loss.backward() and BEFORE optimizer.zero_grad(),
        i.e., at the END of each training epoch (after the last batch).

        Returns a dict mapping layer index to its current grad norm.
        Used to build a trend chart showing which layers are actively updating over time.
        """
        layer_norms = {}
        for processor in self.model.mode_processors:
            blocks = processor.backbone.gpt2.base_model.model.h
            for i, block in enumerate(blocks):
                norm_sq = sum(
                    p.grad.norm().item() ** 2
                    for p in block.parameters()
                    if p.grad is not None
                )
                layer_norms[i] = layer_norms.get(i, 0.0) + norm_sq ** 0.5
        # Average across VMD modes
        n_modes = len(self.model.mode_processors)
        return {i: v / n_modes for i, v in layer_norms.items()}

    def save_grad_norm_trend(self, grad_norm_history: list, save_dir: str = None):
        """
        Save a per-layer gradient norm trend plot from the history collected during training.

        Args:
            grad_norm_history : list of dicts, one per epoch, from collect_epoch_grad_norms().
            save_dir          : where to save. Defaults to self.log_dir/diagnostics/.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = save_dir or os.path.join(self.log_dir, "diagnostics")
        os.makedirs(save_dir, exist_ok=True)

        if not grad_norm_history:
            return

        n_layers = max(max(d.keys()) for d in grad_norm_history) + 1
        epochs   = list(range(1, len(grad_norm_history) + 1))
        frozen_start = n_layers - self.args.U

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#161b22")

        palette_frozen  = ["#ff7b72", "#ffa198", "#ffb8b1"]
        palette_trained = ["#3fb950", "#58a6ff", "#d2a8ff", "#f0883e"]

        for layer_idx in range(n_layers):
            norms = [epoch_dict.get(layer_idx, 0.0) for epoch_dict in grad_norm_history]
            if max(norms) < 1e-10:
                continue  # completely frozen — skip to avoid visual noise
            is_frozen = (layer_idx < frozen_start)
            color  = palette_frozen[layer_idx % len(palette_frozen)] if is_frozen \
                     else palette_trained[(layer_idx - frozen_start) % len(palette_trained)]
            style  = "--" if is_frozen else "-"
            label  = f"L{layer_idx} [frozen]" if is_frozen else f"L{layer_idx} [train]"
            ax.plot(epochs, norms, linestyle=style, color=color, linewidth=1.6,
                    marker="o", markersize=3, label=label)

        ax.set_xlabel("Epoch", color="#c9d1d9")
        ax.set_ylabel("Gradient L2-Norm", color="#c9d1d9")
        ax.set_title(
            "Per-Layer Gradient Norm Trend During Training\n"
            "Solid = trainable layers | Dashed = frozen layers",
            color="#e6edf3", fontsize=10,
        )
        ax.tick_params(colors="#c9d1d9")
        ax.spines[:].set_color("#30363d")
        legend = ax.legend(facecolor="#21262d", edgecolor="#30363d",
                           labelcolor="#c9d1d9", fontsize=8, ncol=2)
        plt.tight_layout()

        out_path = os.path.join(save_dir, "grad_norm_trend.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  >> Grad norm trend saved to: {out_path}")


    def save_checkpoint(self, epoch, val_loss, path):
        """Save training state for resuming later."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": val_loss,
        }
        torch.save(state, path)
        print(f"--- Checkpoint saved to {path} (Epoch {epoch}) ---")

    def load_model(self, path, strict=False):
        """Load either a raw state dict or a checkpoint containing model weights."""
        print(f"--- Loading model weights from {path} ---")
        payload = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload

        missing, unexpected = self.model.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"Warning: Missing keys when loading model: {len(missing)}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading model: {len(unexpected)}")

        return payload

    def load_checkpoint(self, path):
        """Load checkpoint including optimizer state."""
        checkpoint = self.load_model(path, strict=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint at {path} does not contain optimizer/training state.")

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            print("Warning: Optimizer state could not be fully loaded. Resetting optimizer.")

        return checkpoint["epoch"], checkpoint["best_val_loss"]

    def train(self, x, y_real, vmd_data, accumulation_step=0, is_last_batch=False):
        """Train the model for one epoch with optional gradient accumulation and mixed precision."""
        self.model.train()

        if accumulation_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        x_in = x

        # Enable mixed precision if configured
        ctx = (
            torch.amp.autocast("cuda", dtype=self.amp_dtype)
            if self.use_amp
            else nullcontext()
        )
        with ctx:
            preds, _, gat_aux_loss = self.model(vmd_data, x_in)
            preds_scaled = self.scaler.inverse_transform(preds)
            real_scaled = y_real
            task_loss = self.loss_fn(preds_scaled, real_scaled, 0.0)
            # Add GAT entropy aux loss to keep gradients flowing into gat_q/gat_k/gat_a.
            # gat_aux_weight defaults to 0.01; set to 0.0 in args to disable.
            gat_weight = getattr(self.args, "gat_aux_weight", 0.01)
            loss = task_loss + gat_weight * gat_aux_loss

        loss = loss / self.grad_accum_steps

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = ((accumulation_step + 1) % self.grad_accum_steps == 0) or is_last_batch
        if should_step:
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if torch.isfinite(grad_norm):
                    self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

        return (loss.item() * self.grad_accum_steps), compute_metrics(preds_scaled, real_scaled)

    def eval(self, x, y_real, vmd_data):
        """Run one evaluation step and return aggregate metrics."""
        self.model.eval()
        x_in = x

        ctx = (
            torch.amp.autocast("cuda", dtype=self.amp_dtype)
            if self.use_amp
            else nullcontext()
        )
        with torch.no_grad(), ctx:
            preds, _, _ = self.model(vmd_data, x_in)

        preds_scaled = self.scaler.inverse_transform(preds)
        real_scaled = y_real

        # Use pure task loss for validation so metrics stay comparable across runs.
        loss = self.loss_fn(preds_scaled, real_scaled, 0.0).item()
        metrics = compute_metrics(preds_scaled, real_scaled)

        return loss, metrics

    def test(self, test_loader, model_path=None):
        """Evaluate the test split and report per-horizon metrics."""
        if model_path is not None:
            self.load_model(model_path, strict=False)

        self.model.eval()

        horizon_mae = [[] for _ in range(self.args.output_len)]
        horizon_mape = [[] for _ in range(self.args.output_len)]
        horizon_rmse = [[] for _ in range(self.args.output_len)]

        print(">> Starting Detailed Horizon Evaluation...")

        for x, y, vmd in tqdm(test_loader.get_iterator(), desc="Testing"):
            tx = x.to(self.device, non_blocking=True)
            ty = y.to(self.device, non_blocking=True)
            tvmd = vmd.to(self.device, non_blocking=True)

            x_in = tx
            with torch.no_grad():
                preds, _, _ = self.model(tvmd, x_in)

            preds_scaled = self.scaler.inverse_transform(preds)
            real_scaled = ty

            for t in range(self.args.output_len):
                p = preds_scaled[:, t, ...]
                r = real_scaled[:, t, ...]

                horizon_mae[t].append(MAE_torch(p, r, 0).item())
                horizon_mape[t].append(MAPE_torch(p, r, 0).item())
                horizon_rmse[t].append(RMSE_torch(p, r, 0).item())

        print("\n" + "=" * 50)
        print(f"{'Horizon':<10} | {'MAE':<10} | {'MAPE':<10} | {'RMSE':<10}")
        print("-" * 50)

        total_mae, total_mape, total_rmse = [], [], []
        for i in range(self.args.output_len):
            m_mae = np.mean(horizon_mae[i])
            m_mape = np.mean(horizon_mape[i])
            m_rmse = np.mean(horizon_rmse[i])

            total_mae.append(m_mae)
            total_mape.append(m_mape)
            total_rmse.append(m_rmse)

            print(f"Step {i + 1:02d}    | {m_mae:<10.4f} | {m_mape:<10.4f} | {m_rmse:<10.4f}")

        print("-" * 50)
        print(
            f"AVERAGE    | {np.mean(total_mae):<10.4f} | "
            f"{np.mean(total_mape):<10.4f} | {np.mean(total_rmse):<10.4f}"
        )
        print("=" * 50)

        return {
            "mae": float(np.mean(total_mae)),
            "rmse": float(np.mean(total_rmse)),
            "mape": float(np.mean(total_mape)),
        }
