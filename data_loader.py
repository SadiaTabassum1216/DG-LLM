import torch
import numpy as np
import os
from utils import StandardScaler
from vmd_utils import precompute_vmd

class OptimizedDataLoader:
    """Memory-efficient DataLoader with padding for VMD mode processing."""
    def __init__(self, data_x, data_y, vmd_data, batch_size, shuffle=False):
        self.data_x = data_x # [Samples, T, N, F]
        self.data_y = data_y # [Samples, H, N, 1]
        self.vmd_data = vmd_data # [Samples, K, T, N, 1]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = data_x.shape[0]

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            
            x = torch.from_numpy(self.data_x[batch_indices]).float()
            y = torch.from_numpy(self.data_y[batch_indices]).float()
            vmd = torch.from_numpy(self.vmd_data[batch_indices]).float()
            
            yield x, y, vmd

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def load_dataset_optimized(dataset_dir, batch_size, args, force_recompute=False):
    """
    Load and preprocess traffic dataset with VMD decomposition.
    """
    # Cache directories - generalized for local use
    output_cache_dir = "./vmd_cache"
    input_cache_dir = f"./vmd_cache_{args.data}" # Can point to a shared cache
    os.makedirs(output_cache_dir, exist_ok=True)

    data = {}
    cumulative_offset = 0 

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

    # Scaling
    scaler = StandardScaler(mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std())
    for category in ["train", "val", "test"]:
        if "x_" + category in data:
            data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

    # VMD Caching Helper
    def get_or_compute_vmd(split_name, data_input):
        config_id = f"{args.data}_T{args.input_len}"
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
            print(f"  [Cache Miss] Computing VMD for {split_name}...")
            vmd_result = precompute_vmd(data_input, vmd_k=3, max_workers=4)
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
