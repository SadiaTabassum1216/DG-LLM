"""Compatible-only zero-shot cross-dataset evaluation for DG-LLM.

This script builds the model for a target dataset, then loads only the
checkpoint tensors whose names and shapes match the target model. This allows
cross-dataset transfer evaluation even when source and target datasets have
different numbers of nodes.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from data_loader import load_dataset_optimized
from evaluate import evaluate_model_statistical
from experiment_utils import seed_everything
from trainer import VMD_Trainer
from utils import load_pickle


DATASET_NUM_NODES = {
    "PEMSD04": 307,
    "PEMSD08": 170,
    "bike_drop": 250,
    "bike_pick": 250,
    "taxi_drop": 266,
    "taxi_pick": 266,
}

DEFAULT_MODEL_SEARCH_DIRS = ["./models", "./logs", "."]
DEFAULT_MODEL_FILENAMES = ["best_model.pth", "latest_checkpoint.pth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compatible-only zero-shot cross-dataset evaluation for DG-LLM"
    )

    parser.add_argument(
        "--target_data",
        type=str,
        required=True,
        choices=list(DATASET_NUM_NODES.keys()),
        help="Dataset used for target-domain zero-shot evaluation",
    )
    parser.add_argument(
        "--source_data",
        type=str,
        default=None,
        choices=list(DATASET_NUM_NODES.keys()),
        help="Optional source dataset label used for auto-locating checkpoints and reporting",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./Dataset",
        help="Root folder containing dataset directories",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Explicit checkpoint path. Overrides --model_dir and --source_data search.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Optional directory containing best_model.pth or latest_checkpoint.pth",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Where to save evaluation metrics and transfer report",
    )
    parser.add_argument(
        "--vmd_cache_dir",
        type=str,
        default="",
        help="Target-side VMD cache directory",
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llm_layer", type=int, default=6)
    parser.add_argument("--U", type=int, default=1)
    parser.add_argument("--vmd_k", type=int, default=3)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--log_dir", type=str, default="./logs_zero_shot_eval")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--use_amp", action="store_true")

    args = parser.parse_args()

    args.data = args.target_data
    args.num_nodes = DATASET_NUM_NODES[args.target_data]
    args.data_path = os.path.join(args.root_path, args.target_data, "processed")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(os.path.join(args.data_path, "test.npz")):
        raise FileNotFoundError(f"Target dataset not found at: {args.data_path}")

    if not args.model_path:
        args.model_path = find_checkpoint(args.model_dir, args.source_data)

    source_label = args.source_data or infer_source_label(args.model_path)
    args.source_label = source_label

    if not args.vmd_cache_dir:
        args.vmd_cache_dir = os.path.join(
            ".",
            "models",
            args.target_data,
            f"vmd_cache_zero_shot_{source_label}",
        )

    if not args.output_json:
        args.output_json = os.path.join(
            ".",
            "results",
            f"zero_shot_{source_label}_to_{args.target_data}.json",
        )

    args.log_dir = os.path.join(args.log_dir, f"{source_label}_to_{args.target_data}")
    return args


def infer_source_label(model_path: str) -> str:
    parent = Path(model_path).resolve().parent.name
    stem = Path(model_path).resolve().stem
    for candidate in [parent, stem]:
        if candidate in DATASET_NUM_NODES:
            return candidate
    return parent or "source_model"


def find_checkpoint(model_dir: str, source_data: Optional[str]) -> str:
    if not source_data and not model_dir:
        raise ValueError("Provide --model_path directly, or use --source_data for auto-discovery.")

    search_roots: List[Path] = []
    if model_dir:
        search_roots.append(Path(model_dir))
        if source_data:
            search_roots.append(Path(model_dir) / source_data)

    for root in DEFAULT_MODEL_SEARCH_DIRS:
        search_roots.append(Path(root))
        if source_data:
            search_roots.append(Path(root) / source_data)

    seen = set()
    deduped_roots = []
    for root in search_roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            deduped_roots.append(root)

    searched = []
    for root in deduped_roots:
        for filename in DEFAULT_MODEL_FILENAMES:
            candidate = root / filename
            searched.append(str(candidate))
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "Could not find a checkpoint automatically. "
        f"Searched: {searched}. Use --model_path to specify it explicitly."
    )


def load_adjacency(root_path: str, dataset: str, num_nodes: int) -> np.ndarray:
    adj_path = os.path.join(root_path, dataset, "adj_mx.pkl")
    if not os.path.exists(adj_path):
        print(f"[Warning] Adjacency not found at {adj_path}. Using identity matrix.")
        return np.eye(num_nodes, dtype=np.float32)

    adj_data = load_pickle(adj_path)
    if isinstance(adj_data, list):
        return adj_data[2]
    return adj_data


def normalize_state_dict_keys(
    checkpoint_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if any(key in model_state for key in checkpoint_state):
        return checkpoint_state

    if all(key.startswith("module.") for key in checkpoint_state):
        stripped = {key[len("module."):]: value for key, value in checkpoint_state.items()}
        if any(key in model_state for key in stripped):
            return stripped

    return checkpoint_state


def tensor_shape(value: torch.Tensor) -> List[int]:
    return list(value.shape)


def load_compatible_weights(
    model: torch.nn.Module,
    model_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    raw = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        checkpoint_state = raw["model_state_dict"]
        checkpoint_type = "trainer_checkpoint"
    else:
        checkpoint_state = raw
        checkpoint_type = "state_dict"

    model_state = model.state_dict()
    checkpoint_state = normalize_state_dict_keys(checkpoint_state, model_state)

    compatible_state = {}
    loaded_keys = []
    missing_in_model = []
    shape_mismatches = []

    for key, value in checkpoint_state.items():
        if key not in model_state:
            missing_in_model.append(key)
            continue

        if model_state[key].shape != value.shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": tensor_shape(value),
                    "model_shape": tensor_shape(model_state[key]),
                }
            )
            continue

        compatible_state[key] = value
        loaded_keys.append(key)

    missing_after_load, unexpected_after_load = model.load_state_dict(compatible_state, strict=False)

    loaded_param_count = sum(model_state[key].numel() for key in loaded_keys)
    total_model_param_count = sum(param.numel() for param in model.parameters())
    checkpoint_param_count = sum(value.numel() for value in checkpoint_state.values())

    return {
        "checkpoint_type": checkpoint_type,
        "checkpoint_tensor_count": len(checkpoint_state),
        "checkpoint_parameter_count": int(checkpoint_param_count),
        "loaded_tensor_count": len(loaded_keys),
        "loaded_parameter_count": int(loaded_param_count),
        "total_model_parameter_count": int(total_model_param_count),
        "loaded_parameter_fraction": float(loaded_param_count / max(total_model_param_count, 1)),
        "loaded_keys": loaded_keys,
        "skipped_missing_in_model": missing_in_model,
        "skipped_shape_mismatches": shape_mismatches,
        "missing_after_load": sorted(missing_after_load),
        "unexpected_after_load": sorted(unexpected_after_load),
    }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def print_transfer_summary(report: Dict[str, Any]) -> None:
    print("\nCompatible transfer summary")
    print("-" * 70)
    print(f"Checkpoint tensors        : {report['checkpoint_tensor_count']}")
    print(f"Loaded tensors            : {report['loaded_tensor_count']}")
    print(f"Shape mismatches skipped  : {len(report['skipped_shape_mismatches'])}")
    print(f"Missing-in-model skipped  : {len(report['skipped_missing_in_model'])}")
    print(f"Loaded parameter fraction : {report['loaded_parameter_fraction']:.2%}")

    if report["skipped_shape_mismatches"]:
        print("\nFirst shape mismatches:")
        for item in report["skipped_shape_mismatches"][:10]:
            print(
                f"  {item['key']}: checkpoint={item['checkpoint_shape']} "
                f"target={item['model_shape']}"
            )

    if report["missing_after_load"]:
        print("\nFirst missing target tensors after load:")
        for key in report["missing_after_load"][:10]:
            print(f"  {key}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print("=" * 70)
    print("DG-LLM Zero-Shot Cross-Dataset Evaluation")
    print("=" * 70)
    print(f"Source    : {args.source_label}")
    print(f"Target    : {args.target_data}")
    print(f"Checkpoint: {args.model_path}")
    print(f"Data path : {args.data_path}")
    print(f"Device    : {args.device}")

    print("\n[1/4] Loading target dataset...")
    data = load_dataset_optimized(args.data_path, args.batch_size, args)

    print("[2/4] Building target-side model...")
    adj_mx = load_adjacency(args.root_path, args.target_data, args.num_nodes)
    trainer = VMD_Trainer(args, data["scaler"], adj_mx, args.device)

    print("[3/4] Loading compatible checkpoint weights...")
    transfer_report = load_compatible_weights(trainer.model, args.model_path, args.device)
    print_transfer_summary(transfer_report)

    print("\n[4/4] Running zero-shot evaluation on target test set...")
    results = evaluate_model_statistical(
        trainer,
        data["test_loader"],
        args.device,
        data["scaler"],
        args.output_len,
        current_seed=args.seed,
    )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_data": args.source_label,
        "target_data": args.target_data,
        "model_path": args.model_path,
        "target_data_path": args.data_path,
        "num_nodes": args.num_nodes,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "vmd_k": args.vmd_k,
        "transfer_report": transfer_report,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)

    print("\nSaved JSON:", args.output_json)


if __name__ == "__main__":
    main()