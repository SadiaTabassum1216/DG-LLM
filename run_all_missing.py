import argparse
import os
import subprocess
import sys
import time


DATASETS = ["PEMSD04", "PEMSD08", "bike_drop", "bike_pick", "taxi_drop", "taxi_pick"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run missing-data testing for all datasets")
    parser.add_argument("--model_root", type=str, default="./models",
                        help="Root containing per-dataset model folders, e.g. ./models/PEMSD04/best_model.pth")
    parser.add_argument("--results_root", type=str, default="./results",
                        help="Directory to store output JSON files")
    parser.add_argument("--pattern", type=str, default="mcar", choices=["mcar", "block"])
    parser.add_argument("--fill_method", type=str, default="mean", choices=["zero", "mean", "ffill"])
    parser.add_argument("--block_len", type=int, default=3)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.results_root, exist_ok=True)

    print("[RUN_ALL] Starting missing-data evaluation for all datasets...")
    start_time = time.time()
    failed = []

    for ds in DATASETS:
        print(f"\n{'-' * 60}")
        print(f"[{ds}] Starting evaluation")
        print(f"{'-' * 60}")

        model_path = os.path.join(args.model_root, ds, "best_model.pth")
        output_json = os.path.join(
            args.results_root,
            f"missing_test_{ds}_{args.pattern}.json"
        )

        cmd = [
            sys.executable,
            "test_missing_data.py",
            "--data", ds,
            "--model_path", model_path,
            "--output_json", output_json,
            "--pattern", args.pattern,
            "--fill_method", args.fill_method,
            "--block_len", str(args.block_len),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed),
            "--rates",
        ]
        cmd.extend([str(r) for r in args.rates])
        if args.include_clean:
            cmd.append("--include_clean")

        try:
            subprocess.run(cmd, check=True)
            print(f"[{ds}] SUCCESS -> {output_json}")
        except subprocess.CalledProcessError as e:
            failed.append(ds)
            print(f"[{ds}] ERROR: {e}")

    total_time = time.time() - start_time
    print(f"\n[RUN_ALL] Finished in {total_time:.1f}s")
    if failed:
        print(f"[RUN_ALL] Failed datasets: {failed}")
    else:
        print("[RUN_ALL] All datasets completed successfully.")


if __name__ == "__main__":
    main()
