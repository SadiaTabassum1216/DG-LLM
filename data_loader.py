import torch
import numpy as np
import os
from utils import StandardScaler, create_sliding_windows
from vmd_utils import precompute_vmd


def _get_writable_dir(preferred_dir, dataset_name):
    """Get a writable cache directory, trying fallbacks if preferred is read-only."""
    try:
        os.makedirs(preferred_dir, exist_ok=True)
        test_path = os.path.join(preferred_dir, ".write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return preferred_dir
    except OSError:
        pass

    fallback_candidates = []
    kaggle_working = os.environ.get("KAGGLE_WORKING_DIR", "")
    if kaggle_working:
        fallback_candidates.append(os.path.join(kaggle_working, "vmd_cache", dataset_name))
    fallback_candidates.append(os.path.join("/kaggle/working", "vmd_cache", dataset_name))
    fallback_candidates.append(os.path.join(".", "vmd_cache_fallback", dataset_name))

    for cand in fallback_candidates:
        try:
            os.makedirs(cand, exist_ok=True)
            return cand
        except OSError:
            continue

    raise OSError(
        f"Could not create writable cache directory. Tried: {preferred_dir} and {fallback_candidates}"
    )


class DataLoaderClass:
    """Memory-efficient DataLoader with pinned-memory tensors for fast GPU transfer."""
    def __init__(self, data_x, data_y, vmd_data, batch_size, shuffle=False):
        # Pre-convert numpy → tensor once (avoids per-batch conversion overhead)
        # Pin memory so .to(device, non_blocking=True) can overlap with GPU compute
        use_pin = torch.cuda.is_available()
        self.data_x = torch.from_numpy(data_x).float()
        self.data_y = torch.from_numpy(data_y).float()
        self.vmd_data = torch.from_numpy(vmd_data).float()
        if use_pin:
            self.data_x = self.data_x.pin_memory()
            self.data_y = self.data_y.pin_memory()
            self.vmd_data = self.vmd_data.pin_memory()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = self.data_x.shape[0]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def get_iterator(self):
        """Returns an iterator that yields pre-built tensors (pinned for non_blocking transfer)."""
        indices = torch.randperm(self.num_samples) if self.shuffle else torch.arange(self.num_samples)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.data_x[batch_indices], self.data_y[batch_indices], self.vmd_data[batch_indices]

def load_dataset_optimized(dataset_dir, batch_size, args, force_recompute=False):
    """
    Load and preprocess traffic dataset with VMD decomposition.
    
    Automatically handles reprocessing if input_len/output_len differ from stored data.
    """
    # Cache directories - respect explicit args if provided
    preferred_cache_dir = getattr(args, "vmd_cache_dir", "./vmd_cache")
    output_cache_dir = _get_writable_dir(preferred_cache_dir, args.data)
    input_cache_dir = getattr(args, "vmd_cache_dir", f"./vmd_cache_{args.data}")
    if output_cache_dir != preferred_cache_dir:
        print(f"  [Warning] Cache dir not writable: {preferred_cache_dir}")
        print(f"            Using writable cache dir: {output_cache_dir}")

    data = {}
    
    # Check if lengths match expected
    sample_path = os.path.join(dataset_dir, "train.npz")
    print(f"  Checking stored data at: {sample_path}")
    if os.path.exists(sample_path):
        sample_data = np.load(sample_path)
        stored_input_len = sample_data["x"].shape[1]
        stored_output_len = sample_data["y"].shape[1]
        print(f"  Stored: x.shape={sample_data['x'].shape}, y.shape={sample_data['y'].shape}")
        print(f"  Stored input_len={stored_input_len}, output_len={stored_output_len}")
        print(f"  Requested input_len={args.input_len}, output_len={args.output_len}")
        
        if stored_input_len != args.input_len or stored_output_len != args.output_len:
            raise ValueError(
                f"\n[SHAPE MISMATCH] Requested input_len={args.input_len}, output_len={args.output_len}, "
                f"but stored data has input_len={stored_input_len}, output_len={stored_output_len}.\n"
                f"Please re-run preprocess_data.py with --input_len {args.input_len} --output_len {args.output_len}."
            )
    
    # Load normally
    for category in ["train", "val", "test"]:
        path = os.path.join(dataset_dir, category + ".npz")
        if not os.path.exists(path):
            print(f"  [Warning] {path} not found. Skipping...")
            continue

        cat_data = np.load(path)
        data["x_" + category] = cat_data["x"]
        data["y_" + category] = cat_data["y"]

    # Shape validation - catch input_len mismatches BEFORE they cause cryptic Conv2d errors
    actual_T = data["x_train"].shape[1]
    if actual_T != args.input_len:
        raise ValueError(
            f"\n[SHAPE MISMATCH] x_train has time dimension T={actual_T}, "
            f"but args.input_len={args.input_len}.\n"
            f"The stored data in {dataset_dir}/train.npz likely has a different input_len.\n"
            f"The model's start_conv expects {args.input_dim}*{args.input_len}={args.input_dim*args.input_len} channels, "
            f"but data would produce {args.input_dim}*{actual_T}={args.input_dim*actual_T} channels.\n"
            f"Fix: Delete VM cache and ensure reprocessing triggers, or use --input_len {actual_T}."
        )
    print(f"  Data shapes: x_train={data['x_train'].shape}, y_train={data['y_train'].shape}")

    # Scaling
    scaler = StandardScaler(mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std())
    for category in ["train", "val", "test"]:
        if "x_" + category in data:
            data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

    # VMD Caching Helper
    def get_or_compute_vmd(split_name, data_input):
        config_id = f"{args.data}_T{args.input_len}_K{args.vmd_k}"
        filename = f"vmd_{split_name}_{config_id}.npy"
        
        target_path = os.path.join(output_cache_dir, filename)
        input_path = os.path.join(input_cache_dir, filename)

        if os.path.exists(input_path) and not force_recompute:
            print(f"  [External Cache Hit] Loading {split_name} from {input_path}...")
            return np.load(input_path)
        elif os.path.exists(target_path) and not force_recompute:
            print(f"  [Local Cache Hit] Loading {split_name} from {target_path}...")
            return np.load(target_path)
        else:
            print(f"  [Cache Miss] Computing VMD for {split_name} (K={args.vmd_k})...")
            vmd_result = precompute_vmd(data_input, vmd_k=args.vmd_k, max_workers=4)
            try:
                np.save(target_path, vmd_result)
            except OSError:
                # Last-resort fallback in case the selected cache path becomes unwritable.
                backup_dir = _get_writable_dir(os.path.join(".", "vmd_cache_fallback", args.data), args.data)
                backup_path = os.path.join(backup_dir, filename)
                np.save(backup_path, vmd_result)
                print(f"  [Warning] Could not save cache to {target_path}. Saved to {backup_path}")
            return vmd_result

    print("Checking VMD Cache...")
    data["vmd_train"] = get_or_compute_vmd("train", data["x_train"])
    data["vmd_val"]   = get_or_compute_vmd("val", data["x_val"])
    data["vmd_test"]  = get_or_compute_vmd("test", data["x_test"])

    # Shuffling
    print("Shuffling Training Data...")
    perm = np.random.permutation(len(data["x_train"]))
    data["x_train"] = data["x_train"][perm]
    data["y_train"] = data["y_train"][perm]
    data["vmd_train"] = data["vmd_train"][perm]

    data["train_loader"] = DataLoaderClass(data["x_train"], data["y_train"], data["vmd_train"], batch_size, shuffle=True)
    data["val_loader"] = DataLoaderClass(data["x_val"], data["y_val"], data["vmd_val"], batch_size)
    data["test_loader"] = DataLoaderClass(data["x_test"], data["y_test"], data["vmd_test"], batch_size)
    data["scaler"] = scaler
    return data
