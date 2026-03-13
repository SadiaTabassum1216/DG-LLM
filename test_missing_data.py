"""Test DG-LLM under controlled missing-data settings (no training).

This script:
1) validates dataset/checkpoint files,
2) loads a trained model,
3) evaluates on test data at 10/20/30/50% missingness,
4) saves all outputs to JSON.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data_loader import OptimizedDataLoader, load_dataset_optimized
from evaluate import evaluate_model_statistical
from experiment_utils import seed_everything
from trainer import VMD_Trainer
from utils import load_pickle
from vmd_utils import precompute_vmd


DATASETS = ["PEMSD04", "PEMSD08", "bike_drop", "bike_pick", "taxi_drop", "taxi_pick"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test DG-LLM on missing-data test sets")

    parser.add_argument("--data", type=str, required=True, choices=DATASETS)
    parser.add_argument("--root_path", type=str, default="./Dataset/")
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to trained model checkpoint (defaults to ./models/<dataset>/best_model.pth)")
    parser.add_argument("--output_json", type=str, default="",
                        help="Output JSON path")

    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                        help="Missing rates in [0, 1]")
    parser.add_argument("--pattern", type=str, default="mcar", choices=["mcar", "block"])
    parser.add_argument("--fill_method", type=str, default="mean", choices=["zero", "mean", "ffill"])
    parser.add_argument("--block_len", type=int, default=3)
    parser.add_argument("--include_clean", action="store_true",
                        help="Also evaluate 0% missing")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vmd_workers", type=int, default=4)

    # Model args required by VMD_Trainer
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--llm_layer", type=int, default=6)
    parser.add_argument("--U", type=int, default=1)
    parser.add_argument("--vmd_k", type=int, default=3)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--log_dir", type=str, default="./logs_missing_eval")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--use_amp", action="store_true")

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

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.model_path:
        args.model_path = os.path.join(args.root_path, "..", "models", args.data, "best_model.pth")
    args.vmd_cache_dir = os.path.join(args.root_path, "..", "models", args.data, "vmd_cache")

    if not args.output_json:
        args.output_json = f"./results/missing_test_{args.data}.json"

    return args


def check_dataset_and_model(args: argparse.Namespace) -> None:
    needed = [
        os.path.join(args.data_path, "train.npz"),
        os.path.join(args.data_path, "val.npz"),
        os.path.join(args.data_path, "test.npz"),
    ]
    for p in needed:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing dataset file: {p}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")


def load_adjacency(args: argparse.Namespace) -> np.ndarray:
    adj_path = os.path.join(args.root_path, args.data, "adj_mx.pkl")
    if not os.path.exists(adj_path):
        print(f"[Warning] Adjacency not found at {adj_path}, using identity.")
        return np.eye(args.num_nodes, dtype=np.float32)

    adj_data = load_pickle(adj_path)
    return adj_data[2] if isinstance(adj_data, list) else adj_data


def build_block_mask(shape: Tuple[int, int, int], rate: float, block_len: int, rng: np.random.Generator) -> np.ndarray:
    s, t, n = shape
    block_len = max(1, min(block_len, t))
    n_slots = max(1, t // block_len)
    effective_t = block_len * n_slots
    slot_rate = min(1.0, rate * t / effective_t) if effective_t > 0 else 0.0
    slot_mask = rng.random((s, n_slots, n)) < slot_rate
    mask = np.repeat(slot_mask, block_len, axis=1)
    return mask[:, :t, :]


def forward_fill(values: np.ndarray, mask: np.ndarray, node_mean: np.ndarray) -> np.ndarray:
    out = values.copy()
    s, t, n = out.shape
    for si in range(s):
        for ni in range(n):
            last = node_mean[ni]
            for ti in range(t):
                if mask[si, ti, ni]:
                    out[si, ti, ni] = last
                else:
                    last = out[si, ti, ni]
    return out


def apply_missing(
    x: np.ndarray,
    rate: float,
    pattern: str,
    fill_method: str,
    node_mean: np.ndarray,
    block_len: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    out = x.copy()
    flow = out[..., 0]

    if pattern == "mcar":
        mask = rng.random(flow.shape) < rate
    else:
        mask = build_block_mask(flow.shape, rate, block_len, rng)

    if fill_method == "zero":
        flow[mask] = 0.0
    elif fill_method == "mean":
        mean_expand = np.broadcast_to(node_mean[None, None, :], flow.shape)
        flow[mask] = mean_expand[mask]
    elif fill_method == "ffill":
        flow = forward_fill(flow, mask, node_mean)
    else:
        raise ValueError(f"Unsupported fill method: {fill_method}")

    out[..., 0] = flow
    return out, float(mask.mean())


def load_model_weights(trainer: VMD_Trainer, model_path: str, device: torch.device) -> None:
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        trainer.model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        trainer.model.load_state_dict(state, strict=False)
    trainer.model.eval()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    check_dataset_and_model(args)

    print("=" * 70)
    print("DG-LLM Missing-Data Testing")
    print("=" * 70)
    print(f"Dataset   : {args.data}")
    print(f"Model     : {args.model_path}")
    print(f"Device    : {args.device}")
    print(f"Pattern   : {args.pattern}")
    print(f"Fill      : {args.fill_method}")
    print(f"Rates     : {args.rates}")

    print("\n[1/4] Loading dataset...")
    data = load_dataset_optimized(args.data_path, args.batch_size, args)

    print("[2/4] Building model and loading checkpoint...")
    adj_mx = load_adjacency(args)
    trainer = VMD_Trainer(args, data["scaler"], adj_mx, args.device)
    load_model_weights(trainer, args.model_path, args.device)

    x_test = data["x_test"]
    y_test = data["y_test"]
    node_mean = data["x_train"][..., 0].mean(axis=(0, 1)).astype(np.float32)

    evaluations: List[Dict[str, Any]] = []

    if args.include_clean:
        print("[3/4] Evaluating clean set (0%)...")
        clean_results = evaluate_model_statistical(
            trainer,
            data["test_loader"],
            args.device,
            data["scaler"],
            args.output_len,
            current_seed=args.seed,
        )
        evaluations.append({
            "rate": 0.0,
            "actual_missing_rate": 0.0,
            "results": clean_results,
        })

    print("[4/4] Evaluating missing rates...")
    for rate in args.rates:
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"Rate must be in [0, 1], got {rate}")

        print(f"\n  -> Rate {rate:.2f}")
        rng = np.random.default_rng(args.seed)
        x_missing, actual = apply_missing(
            x=x_test,
            rate=rate,
            pattern=args.pattern,
            fill_method=args.fill_method,
            node_mean=node_mean,
            block_len=args.block_len,
            rng=rng,
        )

        # Cache VMD for missing datasets to avoid 2+ minute recalculations
        missing_vmd_filename = f"vmd_test_{args.data}_missing_{int(rate*100)}_{args.pattern}_{args.fill_method}_seed{args.seed}_K{args.vmd_k}.npy"
        missing_vmd_path = os.path.join(args.vmd_cache_dir, missing_vmd_filename)
        
        if os.path.exists(missing_vmd_path):
            print(f"  [Cache Hit] Loading missing data VMD from {missing_vmd_path}...")
            vmd_missing = np.load(missing_vmd_path)
        else:
            print(f"  [Cache Miss] Computing VMD for {rate*100}% missing data...")
            vmd_missing = precompute_vmd(x_missing, vmd_k=args.vmd_k, max_workers=args.vmd_workers)
            # Save for future runs
            os.makedirs(args.vmd_cache_dir, exist_ok=True)
            np.save(missing_vmd_path, vmd_missing)
            
        test_loader = OptimizedDataLoader(x_missing, y_test, vmd_missing, args.batch_size, shuffle=False)

        results = evaluate_model_statistical(
            trainer,
            test_loader,
            args.device,
            data["scaler"],
            args.output_len,
            current_seed=args.seed,
        )

        evaluations.append({
            "rate": float(rate),
            "actual_missing_rate": float(actual),
            "results": results,
        })

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": args.data,
        "model_path": args.model_path,
        "pattern": args.pattern,
        "fill_method": args.fill_method,
        "rates": [float(r) for r in args.rates],
        "include_clean": bool(args.include_clean),
        "seed": int(args.seed),
        "device": str(args.device),
        "evaluations": evaluations,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    print("\n" + "=" * 70)
    print("Testing complete")
    print(f"Saved JSON: {args.output_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
