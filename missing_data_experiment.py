"""
Missing-Data Benchmark Utility for DG-LLM.

This script helps you:
1) Generate corrupted dataset variants with controlled missingness.
2) Optionally run DG-LLM on each variant.
3) Collect metrics into a single CSV summary.

Example usage:
    python missing_data_experiment.py --dataset PEMSD04 --generate_only

    python missing_data_experiment.py \
        --dataset PEMSD04 \
        --rates 0.1 0.2 0.3 0.5 \
        --patterns mcar block \
        --apply_to test \
        --fill_method ffill \
        --run_experiments --num_seeds 5 --epochs 50
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default locations where a trained best_model.pth may live (searched in order)
_DEFAULT_MODEL_SEARCH_DIRS = ["./log", "./logs", "./results"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_split_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {"x": data["x"].astype(np.float32), "y": data["y"].astype(np.float32)}


def _save_split_npz(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    np.savez(path, x=x.astype(np.float32), y=y.astype(np.float32))


def _build_mask_mcar(shape: Tuple[int, int, int], rate: float, rng: np.random.Generator) -> np.ndarray:
    # Mask shape: [S, T, N], True indicates missing.
    return rng.random(shape) < rate


def _build_mask_block(shape: Tuple[int, int, int], rate: float, block_len: int, rng: np.random.Generator) -> np.ndarray:
    # Vectorised block missing: partition T into non-overlapping slots of length
    # block_len and randomly corrupt each slot independently per (sample, node).
    s, t, n = shape
    block_len = max(1, min(block_len, t))
    n_slots = max(1, t // block_len)

    # Derive per-slot corrupt probability so overall missing ≈ rate.
    # actual_rate ≈ slot_rate * (block_len * n_slots) / t
    effective_t = block_len * n_slots  # timesteps covered by whole slots
    slot_rate = min(1.0, rate * t / effective_t) if effective_t > 0 else 0.0

    # [S, n_slots, N] boolean: randomly mark slots to corrupt
    slot_mask = rng.random((s, n_slots, n)) < slot_rate

    # Expand each slot back to block_len steps: [S, n_slots*block_len, N]
    full_mask = np.repeat(slot_mask, block_len, axis=1)

    # Trim trailing timesteps from the last partial slot (if t % block_len != 0)
    full_mask = full_mask[:, :t, :]

    return full_mask


def _fill_missing(
    values: np.ndarray,
    mask: np.ndarray,
    method: str,
    fallback_mean_per_node: np.ndarray,
) -> np.ndarray:
    # values shape: [S, T, N], mask shape: [S, T, N]
    filled = values.copy()

    if method == "zero":
        filled[mask] = 0.0
        return filled

    if method == "mean":
        # Use node-wise means as replacement.
        mean_expand = np.broadcast_to(fallback_mean_per_node[None, None, :], filled.shape)
        filled[mask] = mean_expand[mask]
        return filled

    if method == "ffill":
        # Forward-fill along time for each sample/node; fallback to node mean.
        s, t, n = filled.shape
        for si in range(s):
            for ni in range(n):
                last_val = fallback_mean_per_node[ni]
                for ti in range(t):
                    if mask[si, ti, ni]:
                        filled[si, ti, ni] = last_val
                    else:
                        last_val = filled[si, ti, ni]
        return filled

    raise ValueError(f"Unsupported fill method: {method}")


def _node_mean_from_train(train_x: np.ndarray, channel_idx: int = 0) -> np.ndarray:
    # train_x shape: [S, T, N, F]
    v = train_x[..., channel_idx]
    return v.mean(axis=(0, 1)).astype(np.float32)


def _corrupt_flow_channel(
    x: np.ndarray,
    pattern: str,
    rate: float,
    fill_method: str,
    rng: np.random.Generator,
    node_mean: np.ndarray,
    block_len: int,
) -> Tuple[np.ndarray, float]:
    # x shape: [S, T, N, F]
    out = x.copy()
    flow = out[..., 0]  # [S, T, N]

    if pattern == "mcar":
        mask = _build_mask_mcar(flow.shape, rate, rng)
    elif pattern == "block":
        mask = _build_mask_block(flow.shape, rate, block_len, rng)
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    flow_filled = _fill_missing(flow, mask, fill_method, node_mean)
    out[..., 0] = flow_filled

    actual_rate = float(mask.mean())
    return out, actual_rate


def generate_missing_variants(
    dataset: str,
    source_root: Path,
    output_root: Path,
    rates: List[float],
    patterns: List[str],
    apply_to: str,
    fill_method: str,
    seed: int,
    block_len: int,
) -> List[Dict[str, str]]:
    """Generate variant datasets and return variant descriptors."""
    src_processed = source_root / dataset / "processed"
    if not src_processed.exists():
        raise FileNotFoundError(f"Processed dataset not found: {src_processed}")

    splits = {
        "train": _load_split_npz(src_processed / "train.npz"),
        "val": _load_split_npz(src_processed / "val.npz"),
        "test": _load_split_npz(src_processed / "test.npz"),
    }

    node_mean = _node_mean_from_train(splits["train"]["x"], channel_idx=0)
    variant_rows = []

    for pattern in patterns:
        for rate in rates:
            variant_name = f"{pattern}_r{int(round(rate * 100)):02d}"
            variant_root = output_root / variant_name
            dst_processed = variant_root / dataset / "processed"
            _ensure_dir(dst_processed)

            rng = np.random.default_rng(seed)
            metadata = {
                "dataset": dataset,
                "pattern": pattern,
                "target_rate": rate,
                "apply_to": apply_to,
                "fill_method": fill_method,
                "seed": seed,
                "block_len": block_len,
                "actual_missing_rate": {},
            }

            for split in ["train", "val", "test"]:
                x = splits[split]["x"]
                y = splits[split]["y"]

                should_corrupt = (
                    (apply_to == "all")
                    or (apply_to == "val_test" and split in ["val", "test"])
                    or (apply_to == "test" and split == "test")
                )

                if should_corrupt:
                    x_corrupt, actual = _corrupt_flow_channel(
                        x=x,
                        pattern=pattern,
                        rate=rate,
                        fill_method=fill_method,
                        rng=rng,
                        node_mean=node_mean,
                        block_len=block_len,
                    )
                    metadata["actual_missing_rate"][split] = actual
                else:
                    x_corrupt = x
                    metadata["actual_missing_rate"][split] = 0.0

                _save_split_npz(dst_processed / f"{split}.npz", x_corrupt, y)

            with open(variant_root / dataset / "missing_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(
                f"[Generated] {variant_name} | apply_to={apply_to} | "
                f"test_missing={metadata['actual_missing_rate']['test']:.4f}"
            )

            variant_rows.append(
                {
                    "variant": variant_name,
                    "pattern": pattern,
                    "rate": f"{rate:.4f}",
                    "dataset": dataset,
                    "root_path": str(variant_root),
                    "metadata_path": str(variant_root / dataset / "missing_metadata.json"),
                }
            )

    return variant_rows


def run_experiments(
    dataset: str,
    variant_rows: List[Dict[str, str]],
    logs_root: Path,
    epochs: int,
    batch_size: int,
    num_seeds: int,
    seed_start: int,
    extra_args: List[str],
) -> List[Dict[str, str]]:
    _ensure_dir(logs_root)
    output_rows = []

    for row in variant_rows:
        variant = row["variant"]
        root_path = row["root_path"]
        variant_log_dir = logs_root / dataset / variant
        _ensure_dir(variant_log_dir)

        cmd = [
            sys.executable,
            "main.py",
            "--data",
            dataset,
            "--root_path",
            str(root_path),
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--num_seeds",
            str(num_seeds),
            "--seed_start",
            str(seed_start),
            "--save_stats",
            "--log_dir",
            str(variant_log_dir),
        ]

        cmd.extend(extra_args)

        print("\n[Run]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        stats_path = variant_log_dir / f"{dataset}_multiseed_stats.json"
        mae_mean, rmse_mean, mape_mean = np.nan, np.nan, np.nan

        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            mae_mean = stats.get("mae", {}).get("mean", np.nan)
            rmse_mean = stats.get("rmse", {}).get("mean", np.nan)
            mape_mean = stats.get("mape", {}).get("mean", np.nan)

        output_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "pattern": row["pattern"],
                "rate": row["rate"],
                "mae_mean": f"{float(mae_mean):.6f}" if not np.isnan(mae_mean) else "",
                "rmse_mean": f"{float(rmse_mean):.6f}" if not np.isnan(rmse_mean) else "",
                "mape_mean": f"{float(mape_mean):.6f}" if not np.isnan(mape_mean) else "",
                "log_dir": str(variant_log_dir),
            }
        )

    return output_rows


def _find_model(model_dir: Optional[str], dataset: str) -> Path:
    """Locate best_model.pth: explicit dir > dataset-named subdir > default search dirs."""
    search = []
    if model_dir:
        search.append(Path(model_dir))
    # Also try <model_dir>/<dataset>/ in case per-dataset subdirs are used
    if model_dir:
        search.append(Path(model_dir) / dataset)
    for d in _DEFAULT_MODEL_SEARCH_DIRS:
        search.append(Path(d))
        search.append(Path(d) / dataset)
    for p in search:
        candidate = p / "best_model.pth"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find best_model.pth for dataset '{dataset}'.\n"
        f"Searched: {[str(p / 'best_model.pth') for p in search]}\n"
        f"Use --model_dir to specify the directory explicitly."
    )


def run_test_only_experiments(
    dataset: str,
    variant_rows: List[Dict[str, str]],
    logs_root: Path,
    model_dir: Optional[str],
    batch_size: int,
    extra_args: List[str],
) -> List[Dict[str, str]]:
    """Re-use an existing trained model to evaluate on each missing-data variant."""
    _ensure_dir(logs_root)
    output_rows = []

    model_src = _find_model(model_dir, dataset)
    print(f"[Model] Using: {model_src}")

    for row in variant_rows:
        variant = row["variant"]
        root_path = row["root_path"]
        variant_log_dir = logs_root / dataset / variant
        _ensure_dir(variant_log_dir)

        # Copy model into variant log dir so main.py --test_only finds it
        dst_model = variant_log_dir / "best_model.pth"
        if not dst_model.exists():
            shutil.copy2(model_src, dst_model)

        cmd = [
            sys.executable,
            "main.py",
            "--data", dataset,
            "--root_path", str(root_path),
            "--batch_size", str(batch_size),
            "--log_dir", str(variant_log_dir),
            "--test_only",
        ]
        cmd.extend(extra_args)

        print(f"\n[Test-Only] {variant}")
        print("  ", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # Parse results saved by main.py
        results_path = variant_log_dir / "results.json"
        mae, rmse, mape = np.nan, np.nan, np.nan
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            mae = res.get("mae", np.nan)
            rmse = res.get("rmse", np.nan)
            mape = res.get("mape", np.nan)
        else:
            print(f"  [WARN] results.json not found at {results_path}")

        output_rows.append({
            "dataset": dataset,
            "variant": variant,
            "pattern": row["pattern"],
            "rate": row["rate"],
            "mae": f"{float(mae):.6f}" if not np.isnan(mae) else "",
            "rmse": f"{float(rmse):.6f}" if not np.isnan(rmse) else "",
            "mape": f"{float(mape):.6f}" if not np.isnan(mape) else "",
            "log_dir": str(variant_log_dir),
        })

    return output_rows


def save_csv(rows: List[Dict[str, str]], path: Path) -> None:
    if not rows:
        return
    _ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Missing-data benchmark utility for DG-LLM")

    parser.add_argument("--dataset", type=str, default=None,
                        choices=["PEMSD04", "PEMSD08", "bike_drop", "bike_pick", "taxi_drop", "taxi_pick"],
                        help="Dataset name (required unless --all_datasets is set)")
    parser.add_argument("--source_root", type=str, default="./Dataset",
                        help="Root containing clean processed data")
    parser.add_argument("--output_root", type=str, default="./Dataset_missing",
                        help="Root to store missing-data variants")

    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                        help="Missing rates in [0, 1]")
    parser.add_argument("--patterns", type=str, nargs="+", default=["mcar", "block"],
                        choices=["mcar", "block"],
                        help="Missingness patterns")
    parser.add_argument("--apply_to", type=str, default="test", choices=["test", "val_test", "all"],
                        help="Which splits to corrupt")
    parser.add_argument("--fill_method", type=str, default="ffill", choices=["zero", "mean", "ffill"],
                        help="How to fill missing values after masking")
    parser.add_argument("--block_len", type=int, default=3,
                        help="Temporal block length for block pattern")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for missingness generation")

    parser.add_argument("--all_datasets", action="store_true",
                        help="Run for all 6 datasets (ignores --dataset)")

    parser.add_argument("--generate_only", action="store_true",
                        help="Only generate datasets, do not launch any experiments")
    parser.add_argument("--test_only", action="store_true",
                        help="Evaluate existing trained model on each variant (no retraining)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory containing best_model.pth (auto-detected if omitted)")
    parser.add_argument("--run_experiments", action="store_true",
                        help="Full train+test for each variant (expensive)")

    parser.add_argument("--logs_root", type=str, default="./logs_missing",
                        help="Root directory for missing-data experiment logs")
    parser.add_argument("--summary_csv", type=str, default="./results/missing_data_summary.csv",
                        help="CSV file to save combined experiment summary")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--extra_main_args", type=str, nargs="*", default=[],
                        help="Extra args forwarded directly to main.py")

    return parser.parse_args()


ALL_DATASETS = ["PEMSD04", "PEMSD08", "bike_drop", "bike_pick", "taxi_drop", "taxi_pick"]


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    logs_root = Path(args.logs_root)
    summary_csv = Path(args.summary_csv)

    if not args.all_datasets and args.dataset is None:
        raise SystemExit("error: --dataset is required unless --all_datasets is set")

    for r in args.rates:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"Rate must be in [0, 1], got {r}")

    datasets = ALL_DATASETS if args.all_datasets else [args.dataset]

    all_variant_rows: List[Dict[str, str]] = []
    all_result_rows: List[Dict[str, str]] = []

    for dataset in datasets:
        print("\n" + "=" * 70)
        print(f"Missing-Data Benchmark — {dataset}")
        print("=" * 70)
        print(f"Rates        : {args.rates}")
        print(f"Patterns     : {args.patterns}")
        print(f"Apply To     : {args.apply_to}")
        print(f"Fill Method  : {args.fill_method}")
        print(f"Output Root  : {output_root}")

        variants = generate_missing_variants(
            dataset=dataset,
            source_root=source_root,
            output_root=output_root,
            rates=args.rates,
            patterns=args.patterns,
            apply_to=args.apply_to,
            fill_method=args.fill_method,
            seed=args.seed,
            block_len=args.block_len,
        )
        all_variant_rows.extend(variants)

        if args.generate_only:
            continue

        if args.test_only:
            rows = run_test_only_experiments(
                dataset=dataset,
                variant_rows=variants,
                logs_root=logs_root,
                model_dir=args.model_dir,
                batch_size=args.batch_size,
                extra_args=args.extra_main_args or [],
            )
            all_result_rows.extend(rows)

        elif args.run_experiments:
            rows = run_experiments(
                dataset=dataset,
                variant_rows=variants,
                logs_root=logs_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_seeds=args.num_seeds,
                seed_start=args.seed_start,
                extra_args=args.extra_main_args or [],
            )
            all_result_rows.extend(rows)

    save_csv(all_variant_rows, summary_csv.with_name("missing_variants.csv"))
    print(f"\nSaved variant manifest to: {summary_csv.with_name('missing_variants.csv')}")

    if args.generate_only:
        print("\nGeneration complete (--generate_only mode).")
        print("Next: run with --test_only to evaluate existing models, no retraining needed.")
        return

    if all_result_rows:
        save_csv(all_result_rows, summary_csv)
        print(f"\nSaved summary table to: {summary_csv}")
        print("\nAll missing-data runs complete.")
    else:
        print("\nGeneration done. Add --test_only or --run_experiments to evaluate.")


if __name__ == "__main__":
    main()
