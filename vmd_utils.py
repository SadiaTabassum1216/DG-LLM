import numpy as np
from vmdpy import VMD
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def decompose_single_window(window_data, K=3, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    Decomposes a single sample window [T, N, 1].
    Includes checks for Constant Signals and NaNs.
    """
    T, N, _ = window_data.shape
    modes_out = np.zeros((K, T, N, 1), dtype=np.float32)
    
    for n in range(N):
        signal = window_data[:, n, 0] 
        
        # --- CHECK 1: Constant / Flat Signal ---
        if np.std(signal) < 1e-6:
            modes_out[0, :, n, 0] = signal
            continue 

        # Mirror Padding
        pad_width = T 
        signal_padded = np.pad(signal, (pad_width, pad_width), mode='symmetric')
        
        try:
            # Run VMD
            u, _, _ = VMD(signal_padded, alpha, tau, K, DC, init, tol)
            
            if np.any(np.isnan(u)):
                raise ValueError("VMD returned NaNs")

            # Crop center
            u_cropped = u[:, pad_width : pad_width + T] 
            modes_out[:, :, n, 0] = u_cropped
            
        except Exception:
            modes_out[0, :, n, 0] = signal
            
    return modes_out

def precompute_vmd(data_x, vmd_k=3, max_workers=4):
    """
    Applies Sliding Window VMD on the entire dataset.
    Input: data_x [Samples, T, N, F]
    Output: vmd_storage [Samples, K, T, N, 1]
    """
    num_samples = data_x.shape[0]
    T = data_x.shape[1]
    N = data_x.shape[2]
    
    print(f"  > Starting VMD on {num_samples} windows (Len={T})...")
    print(f"  > Mode: Sample-wise")
    print(f"  > Workers: {max_workers}")
    
    vmd_storage = np.zeros((num_samples, vmd_k, T, N, 1), dtype=np.float32)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(decompose_single_window, data_x[i], K=vmd_k): i 
            for i in range(num_samples)
        }
        
        for future in tqdm(as_completed(futures), total=num_samples, desc="VMD Progress"):
            idx = futures[future]
            try:
                result = future.result()
                vmd_storage[idx] = result
            except Exception as e:
                print(f"  [Error] Sample {idx} failed: {e}")
                vmd_storage[idx, 0, :, :, 0] = data_x[idx, :, :, 0]

    return vmd_storage
