import torch
import numpy as np
import os
from utils import StandardScaler
from vmd_utils import precompute_vmd


def reconstruct_raw_from_windows(x, y, original_input_len=12, original_output_len=12):
    """
    Reconstruct raw time series from overlapping sliding windows (step=1).
    
    Args:
        x: [samples, input_len, N, F] - overlapping windows
        y: [samples, output_len, N, 1] - target windows
        original_input_len: original input sequence length
        original_output_len: original output sequence length
        
    Returns:
        raw: [total_timesteps, N, F] - reconstructed raw data
    """
    num_samples = x.shape[0]
    
    # First window contributes all its timesteps
    raw_list = [x[0, t] for t in range(original_input_len)]
    
    # Each subsequent window adds 1 new timestep (the last one)
    for i in range(1, num_samples):
        raw_list.append(x[i, -1])
    
    # Add the output horizon from the last sample
    for t in range(original_output_len):
        raw_list.append(y[-1, t])
    
    raw = np.stack(raw_list, axis=0)
    return raw


def create_sliding_windows(raw_data, input_len, output_len):
    """
    Create sliding window samples from raw time series.
    
    Args:
        raw_data: [total_timesteps, N, F]
        input_len: desired input sequence length
        output_len: desired output sequence length
        
    Returns:
        x: [samples, input_len, N, F]
        y: [samples, output_len, N, 1]
    """
    total_len = input_len + output_len
    num_samples = raw_data.shape[0] - total_len + 1
    
    if num_samples <= 0:
        raise ValueError(
            f"Insufficient data: {raw_data.shape[0]} timesteps, "
            f"need at least {total_len} (input={input_len} + output={output_len})"
        )
    
    x_list = []
    y_list = []
    
    for i in range(num_samples):
        x_list.append(raw_data[i : i + input_len])
        # Target: only first feature (flow/demand)
        y_list.append(raw_data[i + input_len : i + total_len, :, 0:1])
    
    x = np.stack(x_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    
    return x, y


def reprocess_with_new_lengths(dataset_dir, args, original_input_len=12, original_output_len=12):
    """
    Reconstruct raw data from existing windows and re-create with new lengths.
    
    Args:
        dataset_dir: path to processed data
        args: arguments with input_len, output_len
        original_input_len: the input_len used when data was preprocessed
        original_output_len: the output_len used when data was preprocessed
        
    Returns:
        dict with x_train, y_train, x_val, y_val, x_test, y_test
    """
    print(f"\n{'='*60}")
    print(f"Re-processing data: {original_input_len}->{original_output_len} to {args.input_len}->{args.output_len}")
    print(f"{'='*60}")
    
    # Load existing data
    splits_raw = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(dataset_dir, f"{split}.npz")
        data = np.load(path)
        splits_raw[split] = {"x": data["x"], "y": data["y"]}
        print(f"  Loaded {split}: x={data['x'].shape}, y={data['y'].shape}")
    
    # Reconstruct raw data for each split
    print("\nReconstructing raw time series...")
    raw_splits = {}
    for split in ["train", "val", "test"]:
        raw = reconstruct_raw_from_windows(
            splits_raw[split]["x"], 
            splits_raw[split]["y"],
            original_input_len,
            original_output_len
        )
        raw_splits[split] = raw
        print(f"  {split}: {raw.shape[0]} timesteps")
    
    # Create new windows with desired lengths
    print(f"\nCreating new windows (input={args.input_len}, output={args.output_len})...")
    result = {}
    for split in ["train", "val", "test"]:
        x, y = create_sliding_windows(raw_splits[split], args.input_len, args.output_len)
        result[f"x_{split}"] = x
        result[f"y_{split}"] = y
        print(f"  {split}: {x.shape[0]} samples")
    
    print(f"{'='*60}\n")
    return result


class OptimizedDataLoader:
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

    def __iter__(self):
        indices = torch.randperm(self.num_samples) if self.shuffle else torch.arange(self.num_samples)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.data_x[batch_indices], self.data_y[batch_indices], self.vmd_data[batch_indices]

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
    # Cache directories - generalized for local use
    output_cache_dir = "./vmd_cache"
    input_cache_dir = f"./vmd_cache_{args.data}"
    os.makedirs(output_cache_dir, exist_ok=True)

    data = {}
    cumulative_offset = 0
    
    # Check if reprocessing is needed
    needs_reprocess = False
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
            print(f"\n[Auto-Reprocess] Data has {stored_input_len}->{stored_output_len}, "
                  f"but requested {args.input_len}->{args.output_len}")
            needs_reprocess = True
            original_input_len = stored_input_len
            original_output_len = stored_output_len
    
    # Reprocess if needed
    if needs_reprocess:
        reprocessed = reprocess_with_new_lengths(
            dataset_dir, args, 
            original_input_len=original_input_len,
            original_output_len=original_output_len
        )
        
        # Add temporal features to reprocessed data
        for category in ["train", "val", "test"]:
            x_raw = reprocessed[f"x_{category}"]
            y_raw = reprocessed[f"y_{category}"]
            
            if x_raw.shape[-1] < 3:
                print(f"[{category}] Adding temporal features (offset={cumulative_offset})...")
                num_samples, T_len, num_nodes, _ = x_raw.shape
                sample_starts = np.arange(num_samples) + cumulative_offset
                step_indices = sample_starts[:, None] + np.arange(T_len)[None, :]

                time_of_day = (step_indices % 288) / 288.0
                time_of_day = time_of_day[:, :, None, None]
                time_of_day = np.tile(time_of_day, (1, 1, num_nodes, 1))

                day_of_week = (step_indices // 288) % 7
                day_of_week = day_of_week[:, :, None, None].astype(np.float32)
                day_of_week = np.tile(day_of_week, (1, 1, num_nodes, 1))

                x_raw = np.concatenate([x_raw[..., 0:1], time_of_day, day_of_week], axis=-1)
                cumulative_offset += num_samples
            
            data[f"x_{category}"] = x_raw
            data[f"y_{category}"] = y_raw
    else:
        # Load normally
        for category in ["train", "val", "test"]:
            path = os.path.join(dataset_dir, category + ".npz")
            if not os.path.exists(path):
                print(f"  [Warning] {path} not found. Skipping...")
                continue

            cat_data = np.load(path)
            x_raw = cat_data["x"]
            y_raw = cat_data["y"]

            # Temporal Feature Generation
            if x_raw.shape[-1] < 3:
                print(f"[{category}] Adding temporal features (offset={cumulative_offset})...")
                num_samples, T_len, num_nodes, _ = x_raw.shape
                sample_starts = np.arange(num_samples) + cumulative_offset
                step_indices = sample_starts[:, None] + np.arange(T_len)[None, :] 

                time_of_day = (step_indices % 288) / 288.0
                time_of_day = time_of_day[:, :, None, None] 
                time_of_day = np.tile(time_of_day, (1, 1, num_nodes, 1))

                day_of_week = (step_indices // 288) % 7
                day_of_week = day_of_week[:, :, None, None].astype(np.float32)
                day_of_week = np.tile(day_of_week, (1, 1, num_nodes, 1))

                x_raw = np.concatenate([x_raw[..., 0:1], time_of_day, day_of_week], axis=-1)
                cumulative_offset += num_samples

            data["x_" + category] = x_raw
            data["y_" + category] = y_raw

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
            np.save(target_path, vmd_result)
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

    data["train_loader"] = OptimizedDataLoader(data["x_train"], data["y_train"], data["vmd_train"], batch_size, shuffle=True)
    data["val_loader"] = OptimizedDataLoader(data["x_val"], data["y_val"], data["vmd_val"], batch_size)
    data["test_loader"] = OptimizedDataLoader(data["x_test"], data["y_test"], data["vmd_test"], batch_size)
    data["scaler"] = scaler
    return data
