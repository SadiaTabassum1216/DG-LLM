import numpy as np
import os
from vmdpy import VMD
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def decompose_single_window(window_data, K=3, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    Decomposes a single historical window [T, N, 1] into K frequency modes.
    
    Args:
        window_data: Input traffic signal [Time, Nodes, Features].
        K: Number of modes to extract.
    """
    T, N, _ = window_data.shape
    modes_out = np.zeros((K, T, N, 1), dtype=np.float32)
    
    for n in range(N):
        signal = window_data[:, n, 0] 
        
        if np.std(signal) < 1e-6:
            modes_out[0, :, n, 0] = signal
            continue 

        # Symmetric padding
        pad_width = T 
        signal_padded = np.pad(signal, (pad_width, pad_width), mode='symmetric')
        
        try:
            u, _, _ = VMD(signal_padded, alpha, tau, K, DC, init, tol)
            modes_out[:, :, n, 0] = u[:, pad_width : pad_width + T] 
            
        except Exception:
            modes_out[0, :, n, 0] = signal
            
    return modes_out

def precompute_vmd(data_x, vmd_k=3, max_workers=None):
    """
    Applies Sliding Window VMD across the dataset using only the input window.
    
    Input: [Samples, Time, Nodes, Features]
    Output: [Samples, K, Time, Nodes, 1]
    """
    num_samples, T, N, _ = data_x.shape
    
    cpu_count = os.cpu_count() or 4
    actual_workers = min(max_workers, cpu_count) if max_workers else cpu_count
    
    print(f"  > Initializing 12-step VMD (K={vmd_k})")
    vmd_storage = np.zeros((num_samples, vmd_k, T, N, 1), dtype=np.float32)
    
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = {
            executor.submit(decompose_single_window, data_x[i], K=vmd_k): i 
            for i in range(num_samples)
        }
        
        for future in tqdm(as_completed(futures), total=num_samples, desc="    VMD Decomposing"):
            idx = futures[future]
            try:
                vmd_storage[idx] = future.result()
            except Exception as e:
                vmd_storage[idx, 0, :, :, 0] = data_x[idx, :, :, 0]

    return vmd_storage