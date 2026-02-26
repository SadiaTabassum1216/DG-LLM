"""
Data Preprocessing Script for DG-LLM

This script converts raw traffic data into the required format for training
with configurable input and output sequence lengths.

Usage:
    python preprocess_data.py --data PEMSD04 --input_len 12 --output_len 24
    python preprocess_data.py --data taxi_drop --input_len 24 --output_len 12
    
Raw data format expected:
    - .npz file with 'data' key: [total_timesteps, num_nodes, features]
    - or .npy file: [total_timesteps, num_nodes, features]
"""

import numpy as np
import os
import argparse
from typing import Tuple


def create_sliding_windows(
    data: np.ndarray, 
    input_len: int = 12, 
    output_len: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window samples from time series data.
    
    Args:
        data: Raw traffic data [total_timesteps, num_nodes, features]
        input_len: Number of historical timesteps for input
        output_len: Number of future timesteps to predict
        
    Returns:
        x: Input sequences [num_samples, input_len, num_nodes, features]
        y: Target sequences [num_samples, output_len, num_nodes, 1]
    """
    total_len = input_len + output_len
    num_samples = data.shape[0] - total_len + 1
    
    if num_samples <= 0:
        raise ValueError(
            f"Data has {data.shape[0]} timesteps, but need at least {total_len} "
            f"(input_len={input_len} + output_len={output_len})"
        )
    
    x_list = []
    y_list = []
    
    for i in range(num_samples):
        # Input: all features
        x_list.append(data[i : i + input_len])
        # Target: only the first feature (traffic flow/demand)
        y_list.append(data[i + input_len : i + total_len, :, 0:1])
    
    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    return x, y


def split_data(
    x: np.ndarray, 
    y: np.ndarray, 
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> dict:
    """
    Chronologically split data into train/val/test sets.
    
    Args:
        x: Input sequences [num_samples, T_in, N, F]
        y: Target sequences [num_samples, T_out, N, 1]
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        
    Returns:
        Dictionary with train/val/test splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    num_samples = x.shape[0]
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    splits = {
        'train': {
            'x': x[:train_end],
            'y': y[:train_end]
        },
        'val': {
            'x': x[train_end:val_end],
            'y': y[train_end:val_end]
        },
        'test': {
            'x': x[val_end:],
            'y': y[val_end:]
        }
    }
    
    return splits


def load_raw_data(data_path: str) -> np.ndarray:
    """
    Load raw traffic data from various formats.
    
    Supports:
        - .npz files (looks for 'data' or first available key)
        - .npy files
        - .h5 files (looks for 'data' key)
        
    Returns:
        data: [total_timesteps, num_nodes, features]
    """
    ext = os.path.splitext(data_path)[1].lower()
    
    if ext == '.npz':
        loaded = np.load(data_path)
        # Try common key names
        for key in ['data', 'raw', 'traffic', 'flow']:
            if key in loaded:
                data = loaded[key]
                break
        else:
            # Use first available key
            data = loaded[list(loaded.keys())[0]]
            
    elif ext == '.npy':
        data = np.load(data_path)
        
    elif ext in ['.h5', '.hdf5']:
        import h5py
        with h5py.File(data_path, 'r') as f:
            for key in ['data', 'raw', 'traffic', 'flow']:
                if key in f:
                    data = f[key][:]
                    break
            else:
                data = f[list(f.keys())[0]][:]
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Ensure 3D: [T, N, F]
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    
    return data.astype(np.float32)


def preprocess_dataset(
    raw_data_path: str,
    output_dir: str,
    input_len: int = 12,
    output_len: int = 12,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
):
    """
    Full preprocessing pipeline.
    
    Args:
        raw_data_path: Path to raw data file
        output_dir: Directory to save processed files
        input_len: Input sequence length
        output_len: Output sequence length
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
    """
    print("=" * 60)
    print("DG-LLM Data Preprocessing")
    print("=" * 60)
    
    # Load raw data
    print(f"\n1. Loading raw data from: {raw_data_path}")
    data = load_raw_data(raw_data_path)
    print(f"   Raw data shape: {data.shape}")
    print(f"   - Total timesteps: {data.shape[0]}")
    print(f"   - Number of nodes: {data.shape[1]}")
    print(f"   - Features: {data.shape[2]}")
    
    # Create sliding windows
    print(f"\n2. Creating sliding windows...")
    print(f"   - Input length: {input_len}")
    print(f"   - Output length: {output_len}")
    x, y = create_sliding_windows(data, input_len, output_len)
    print(f"   Generated {x.shape[0]} samples")
    print(f"   - x shape: {x.shape}")
    print(f"   - y shape: {y.shape}")
    
    # Split data
    print(f"\n3. Splitting data chronologically...")
    print(f"   - Train: {train_ratio*100:.0f}%")
    print(f"   - Val: {val_ratio*100:.0f}%")
    print(f"   - Test: {test_ratio*100:.0f}%")
    splits = split_data(x, y, train_ratio, val_ratio, test_ratio)
    
    for split_name, split_data in splits.items():
        print(f"   {split_name}: {split_data['x'].shape[0]} samples")
    
    # Save processed data
    print(f"\n4. Saving to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        save_path = os.path.join(output_dir, f"{split_name}.npz")
        np.savez(save_path, x=split_data['x'], y=split_data['y'])
        print(f"   Saved: {save_path}")
    
    # Save config for reference
    config = {
        'input_len': input_len,
        'output_len': output_len,
        'num_nodes': data.shape[1],
        'num_features': data.shape[2],
        'train_samples': splits['train']['x'].shape[0],
        'val_samples': splits['val']['x'].shape[0],
        'test_samples': splits['test']['x'].shape[0],
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }
    config_path = os.path.join(output_dir, 'config.npz')
    np.savez(config_path, **config)
    print(f"   Saved config: {config_path}")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    
    return splits


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess traffic data for DG-LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard 12->12 preprocessing
    python preprocess_data.py --raw_path data/PEMSD04.npz --output_dir Dataset/PEMSD04/processed

    # Long-term forecasting: 12 input → 60 output
    python preprocess_data.py --raw_path data/PEMSD04.npz --output_dir Dataset/PEMSD04_12_60/processed --input_len 12 --output_len 60

    # More history: 24 input → 12 output  
    python preprocess_data.py --raw_path data/taxi.npy --output_dir Dataset/taxi_24_12/processed --input_len 24 --output_len 12

    # Custom split ratios (70/15/15)
    python preprocess_data.py --raw_path data/bike.npz --output_dir Dataset/bike/processed --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
        """
    )
    
    parser.add_argument('--raw_path', type=str, required=True,
                        help='Path to raw data file (.npz, .npy, or .h5)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--input_len', type=int, default=12,
                        help='Input sequence length (default: 12)')
    parser.add_argument('--output_len', type=int, default=12,
                        help='Output sequence length (default: 12)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Training set ratio (default: 0.6)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        parser.error(f"Ratios must sum to 1.0, got {total_ratio}")
    
    preprocess_dataset(
        raw_data_path=args.raw_path,
        output_dir=args.output_dir,
        input_len=args.input_len,
        output_len=args.output_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == '__main__':
    main()
