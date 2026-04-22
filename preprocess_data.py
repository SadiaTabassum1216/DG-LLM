"""
Data Preprocessing Script for DG-LLM

This script converts raw traffic data into the required format for training
with configurable input and output sequence lengths.

Usage:
    python preprocess_data.py --raw_path data/PEMSD04.npz --output_dir Dataset/PEMSD04/processed --input_len 12 --output_len 24
    python preprocess_data.py --raw_path data/taxi.npy --output_dir Dataset/taxi_drop/processed --input_len 24 --output_len 12
    
Raw data format expected:
    - .npz file with 'data' key: [total_timesteps, num_nodes, features]
    - or .npy file: [total_timesteps, num_nodes, features]
"""

import numpy as np
import os
import argparse
from utils import create_sliding_windows


def add_temporal_features(data: np.ndarray, steps_per_day: int = 288) -> np.ndarray:
    """
    Add time-of-day and day-of-week channels when only flow is present.
    
    Args:
        data: Raw traffic data [total_timesteps, num_nodes, features]
        steps_per_day: Number of timesteps in a single day
        
    Returns:
        data with temporal features added (if not already present)
    """
    if data.shape[-1] >= 3:
        return data

    total_timesteps, num_nodes, _ = data.shape
    step_indices = np.arange(total_timesteps)
    
    time_of_day = ((step_indices % steps_per_day) / float(steps_per_day)).astype(np.float32)
    time_of_day = time_of_day[:, np.newaxis, np.newaxis]
    time_of_day = np.tile(time_of_day, (1, num_nodes, 1))
    
    day_of_week = ((step_indices // steps_per_day) % 7).astype(np.float32)
    day_of_week = day_of_week[:, np.newaxis, np.newaxis]
    day_of_week = np.tile(day_of_week, (1, num_nodes, 1))
    
    data_with_features = np.concatenate([data[..., 0:1], time_of_day, day_of_week], axis=-1)
    return data_with_features


def split_data_chronologically(
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


def load_raw_traffic_data(data_path: str) -> np.ndarray:
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
    steps_per_day: int = 288,
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
    data = load_raw_traffic_data(raw_data_path)
    print(f"   Raw data shape: {data.shape}")
    print(f"   - Total timesteps: {data.shape[0]}")
    print(f"   - Number of nodes: {data.shape[1]}")
    print(f"   - Features: {data.shape[2]}")
    
    # Add temporal features
    print(f"\n2. Adding temporal features (steps_per_day={steps_per_day})...")
    data = add_temporal_features(data, steps_per_day)
    print(f"   Features updated to: {data.shape[2]}")
    
    # Create sliding windows
    print(f"\n3. Creating sliding windows...")
    print(f"   - Input length: {input_len}")
    print(f"   - Output length: {output_len}")
    x, y = create_sliding_windows(data, input_len, output_len)
    print(f"   Generated {x.shape[0]} samples")
    print(f"   - x shape: {x.shape}")
    print(f"   - y shape: {y.shape}")
    
    # Split data
    print(f"\n4. Splitting data chronologically...")
    print(f"   - Train: {train_ratio*100:.0f}%")
    print(f"   - Val: {val_ratio*100:.0f}%")
    print(f"   - Test: {test_ratio*100:.0f}%")
    splits = split_data_chronologically(x, y, train_ratio, val_ratio, test_ratio)
    
    for split_name, split_data in splits.items():
        print(f"   {split_name}: {split_data['x'].shape[0]} samples")
    
    # Save processed data
    print(f"\n5. Saving to: {output_dir}")
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
    parser.add_argument('--steps_per_day', type=int, default=288,
                        help='Number of timesteps in a single day (default: 288 for 5-min intervals)')
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
        steps_per_day=args.steps_per_day,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == '__main__':
    main()
