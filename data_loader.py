import os
import numpy as np
import torch

from utils import StandardScaler
from vmd_utils import precompute_vmd


SPLITS = ("train", "val", "test")


# Selects a writable cache directory using fallback locations when needed.
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


class TrafficDataLoader:
    """
    A memory-efficient data loader that provides batches of traffic sequences.
    Uses pinned memory for significantly faster transfers to the GPU.
    """
    # Converts arrays to tensors and prepares batched iteration settings.
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

    # Returns the number of batches for the current dataset size.
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    # Yields batched tensors in order or shuffled order.
    def get_iterator(self):
        """Returns an iterator that yields pre-built tensors (pinned for non_blocking transfer)."""
        indices = torch.randperm(self.num_samples) if self.shuffle else torch.arange(self.num_samples)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.data_x[batch_indices], self.data_y[batch_indices], self.vmd_data[batch_indices]



# Validates that stored train split sequence lengths match current arguments.
def _validate_stored_sequence_lengths(sample_path, args):
    print(f"  Checking stored data at: {sample_path}")
    if not os.path.exists(sample_path):
        return

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


# Loads train, validation, and test arrays from available NPZ split files.
def _load_data_splits(dataset_dir):
    data = {}
    for split in SPLITS:
        path = os.path.join(dataset_dir, f"{split}.npz")
        if not os.path.exists(path):
            print(f"  [Warning] {path} not found. Skipping...")
            continue

        split_data = np.load(path)
        data[f"x_{split}"] = split_data["x"]
        data[f"y_{split}"] = split_data["y"]
    return data



# Validates runtime input sequence length to prevent downstream shape errors.
def _validate_runtime_input_length(data, dataset_dir, args):
    # Catch input_len mismatches before they cause cryptic Conv2d channel errors.
    actual_t = data["x_train"].shape[1]
    if actual_t != args.input_len:
        raise ValueError(
            f"\n[SHAPE MISMATCH] x_train has time dimension T={actual_t}, "
            f"but args.input_len={args.input_len}.\n"
            f"The stored data in {dataset_dir}/train.npz likely has a different input_len.\n"
            f"The model's start_conv expects {args.input_dim}*{args.input_len}={args.input_dim*args.input_len} channels, "
            f"but data would produce {args.input_dim}*{actual_t}={args.input_dim*actual_t} channels.\n"
            f"Fix: Delete VM cache and ensure reprocessing triggers, or use --input_len {actual_t}."
        )

    print(f"  Data shapes: x_train={data['x_train'].shape}, y_train={data['y_train'].shape}")


# Normalizes the primary traffic flow feature (channel 0) to a standard range.
def _normalize_traffic_flow(data):
    scaler = StandardScaler(mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std())
    for split in SPLITS:
        x_key = f"x_{split}"
        if x_key in data:
            data[x_key][..., 0] = scaler.transform(data[x_key][..., 0])
    return scaler


# Retrieves pre-processed VMD modes from the cache or calculates them on the fly.
def _load_or_extract_vmd_modes(split_name, split_input, args, input_cache_dir, output_cache_dir, force_recompute=False):
    config_id = f"{args.data}_T{args.input_len}_K{args.vmd_k}"
    filename = f"vmd_{split_name}_{config_id}.npy"

    target_path = os.path.join(output_cache_dir, filename)
    input_path = os.path.join(input_cache_dir, filename)

    if os.path.exists(input_path) and not force_recompute:
        print(f"  [External Cache Hit] Loading {split_name} from {input_path}...")
        return np.load(input_path)

    if os.path.exists(target_path) and not force_recompute:
        print(f"  [Local Cache Hit] Loading {split_name} from {target_path}...")
        return np.load(target_path)



    # Cache missing: compute VMD.
    print(f"  [Cache Miss] Computing VMD for {split_name} (K={args.vmd_k})...")
    vmd_result = precompute_vmd(split_input, vmd_k=args.vmd_k, max_workers=4)
    
    try:
        np.save(target_path, vmd_result)
    except OSError:
        # Last-resort fallback in case the selected cache path becomes unwritable.
        backup_dir = _get_writable_dir(os.path.join(".", "vmd_cache_fallback", args.data), args.data)
        backup_path = os.path.join(backup_dir, filename)
        np.save(backup_path, vmd_result)
        print(f"  [Warning] Could not save cache to {target_path}. Saved to {backup_path}")
    return vmd_result


# Randomly shuffles training arrays using one shared permutation.
def _shuffle_train_data(data):
    print("Shuffling Training Data...")
    permutation = np.random.permutation(len(data["x_train"]))
    data["x_train"] = data["x_train"][permutation]
    data["y_train"] = data["y_train"][permutation]
    data["vmd_train"] = data["vmd_train"][permutation]


# Creates the training, validation, and test iterators.
def _create_data_iterators(data, batch_size):
    data["train_loader"] = TrafficDataLoader(
        data["x_train"], data["y_train"], data["vmd_train"], batch_size, shuffle=True
    )
    data["val_loader"] = TrafficDataLoader(data["x_val"], data["y_val"], data["vmd_val"], batch_size)
    data["test_loader"] = TrafficDataLoader(data["x_test"], data["y_test"], data["vmd_test"], batch_size)




# Orchestrates full dataset loading, validation, scaling, VMD, and loader creation.
def load_dataset(dataset_dir, batch_size, args, force_recompute=False):
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

    sample_path = os.path.join(dataset_dir, "train.npz")
    _validate_stored_sequence_lengths(sample_path, args)

    data = _load_data_splits(dataset_dir)
    _validate_runtime_input_length(data, dataset_dir, args)

    scaler = _normalize_traffic_flow(data)

    print("Checking VMD Cache...")
    data["vmd_train"] = _load_or_extract_vmd_modes(
        "train", data["x_train"], args, input_cache_dir, output_cache_dir, force_recompute
    )
    data["vmd_val"] = _load_or_extract_vmd_modes(
        "val", data["x_val"], args, input_cache_dir, output_cache_dir, force_recompute
    )
    data["vmd_test"] = _load_or_extract_vmd_modes(
        "test", data["x_test"], args, input_cache_dir, output_cache_dir, force_recompute
    )

    _shuffle_train_data(data)

    _create_data_iterators(data, batch_size)
    data["scaler"] = scaler
    return data
