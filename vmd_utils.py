import numpy as np
import os
from vmdpy import VMD
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def decompose_single_window(window_data, target_len=12, K=3, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    """
    Decomposes a window [Context + Target, Nodes, 1] into K frequency modes.
    Returns only the modes for the last [target_len] steps.
    """
    total_T, N, _ = window_data.shape
    modes_out = np.zeros((K, target_len, N, 1), dtype=np.float32)
    
    for n in range(N):
        signal = window_data[:, n, 0] 
        
        if np.std(signal) < 1e-6:
            modes_out[0, :, n, 0] = signal[-target_len:]
            continue 

        # Symmetric padding
        pad_width = total_T 
        signal_padded = np.pad(signal, (pad_width, pad_width), mode='symmetric')
        
        try:
            u, _, _ = VMD(signal_padded, alpha, tau, K, DC, init, tol)
            
            # Crop the target length from the end of the original signal area
            # Original signal is at [pad_width : pad_width + total_T]
            # Target is the last target_len of that
            start_idx = pad_width + total_T - target_len
            end_idx = pad_width + total_T
            modes_out[:, :, n, 0] = u[:, start_idx : end_idx] 
            
        except Exception:
            modes_out[0, :, n, 0] = signal[-target_len:]
            
    return modes_out

def precompute_vmd(raw_data, vmd_k=3, input_len=12, context_len=100, max_workers=None):
    """
    Applies Causal Sliding Window VMD across the dataset.
    
    Args:
        raw_data: [Total_Steps, Nodes, Features]
        vmd_k: Number of modes
        input_len: The sequence length for the model (e.g., 12)
        context_len: Historical steps to include for stability (e.g., 100)
    
    Output: [Num_Samples, K, input_len, Nodes, 1]
    """
    total_steps, N, _ = raw_data.shape
    num_samples = total_steps - input_len + 1 # Matches sliding window logic
    
    cpu_count = os.cpu_count() or 4
    actual_workers = min(max_workers, cpu_count) if max_workers else cpu_count
    
    print(f"  > Initializing Causal VMD (K={vmd_k}, Context={context_len})")
    vmd_storage = np.zeros((num_samples, vmd_k, input_len, N, 1), dtype=np.float32)
    
    # We can only compute VMD for samples that have enough history
    # For samples at the very beginning, we use whatever history is available
    
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = {}
        for i in range(num_samples):
            # Target window indices: [i, i + input_len - 1]
            # Context window starts at max(0, i - context_len)
            start_idx = max(0, i - context_len)
            end_idx = i + input_len
            window_chunk = raw_data[start_idx:end_idx]
            
            futures[executor.submit(decompose_single_window, window_chunk, target_len=input_len, K=vmd_k)] = i
        
        for future in tqdm(as_completed(futures), total=num_samples, desc="    Causal VMD"):
            idx = futures[future]
            try:
                vmd_storage[idx] = future.result()
            except Exception as e:
                vmd_storage[idx, 0, :, :, 0] = raw_data[idx:idx+input_len, :, 0]

    return vmd_storage