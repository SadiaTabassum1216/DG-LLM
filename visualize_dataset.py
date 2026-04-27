import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

# Configure standard output to use UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # In case of older Python versions

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def get_file_info(path):
    size_str = format_size(path.stat().st_size)
    struct_str = ""

    ext = path.suffix.lower()
    try:
        if ext == '.npz':
            with np.load(path, allow_pickle=True) as data:
                shapes = []
                for k in data.files:
                    try:
                        arr = data[k]
                        if hasattr(arr, 'shape'):
                            shapes.append(f"'{k}': {arr.shape}")
                    except Exception:
                        pass
                if shapes:
                    struct_str = f" [npz | {', '.join(shapes)}]"
        elif ext == '.npy':
            data = np.load(path, allow_pickle=True)
            if hasattr(data, 'shape'):
                struct_str = f" [npy | shape: {data.shape}]"
        elif ext == '.pkl':
            with open(path, 'rb') as f:
                # Try latin1 for older pickle versions (often used in these datasets)
                try:
                    data = pickle.load(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    data = pickle.load(f, encoding='latin1')
            
            if isinstance(data, list):
                # Try to print shape if list elements are arrays
                elem_info = ""
                if len(data) > 0 and hasattr(data[-1], 'shape'):
                    elem_info = f", e.g. array shapes: {data[-1].shape}"
                struct_str = f" [pkl | list of len {len(data)}{elem_info}]"
            elif isinstance(data, dict):
                struct_str = f" [pkl | dict with keys: {list(data.keys())[:5]}{'...' if len(data)>5 else ''}]"
            elif hasattr(data, 'shape'):
                struct_str = f" [pkl | array shape: {data.shape}]"
            else:
                struct_str = f" [pkl | type: {type(data).__name__}]"
                
    except Exception as e:
        struct_str = f" [error reading structure: {e}]"
        
    return f" ({size_str}){struct_str}"

def preview_data(path, num_rows=5):
    """
    Prints a preview of the data in the file.
    """
    ext = path.suffix.lower()
    print(f"\n      >>> Data Preview ({path.name}):")
    try:
        if ext == '.npz':
            with np.load(path, allow_pickle=True) as data:
                for k in data.files:
                    arr = data[k]
                    print(f"      - Key '{k}':")
                    if hasattr(arr, 'shape'):
                        snippet = arr[:num_rows]
                        print(f"        {str(snippet).replace('\n', '\n        ')}")
                    else:
                        print(f"        {arr}")
        elif ext == '.npy':
            data = np.load(path, allow_pickle=True)
            snippet = data[:num_rows]
            print(f"      {str(snippet).replace('\n', '\n      ')}")
        elif ext == '.pkl':
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f)
                except UnicodeDecodeError:
                    f.seek(0)
                    data = pickle.load(f, encoding='latin1')
            
            if isinstance(data, (np.ndarray, list)):
                snippet = data[:num_rows]
                print(f"      {str(snippet).replace('\n', '\n      ')}")
            elif isinstance(data, pd.DataFrame):
                snippet = data.head(num_rows)
                print(f"      {str(snippet).replace('\n', '\n      ')}")
            elif isinstance(data, dict):
                print("      {")
                for i, (k, v) in enumerate(data.items()):
                    if i >= num_rows:
                        print("        ...")
                        break
                    val_str = str(v)[:100] + ("..." if len(str(v)) > 100 else "")
                    print(f"        '{k}': {val_str}")
                print("      }")
            else:
                print(f"      {str(data)[:200]}")
        else:
            print("      (Preview not supported for this file type)")
    except Exception as e:
        print(f"      (Error previewing data: {e})")
    print("")

def print_tree(directory, prefix="", preview=False, num_rows=5):
    """
    Recursively prints the directory tree.
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory '{directory}' does not exist or is not a directory.")
        return

    # Sort contents, putting directories first
    contents = list(directory.iterdir())
    contents.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

    pointers = ["├── "] * (len(contents) - 1) + ["└── "] if contents else []

    for pointer, path in zip(pointers, contents):
        if path.is_dir():
            print(prefix + pointer + path.name)
            extension = "│   " if pointer == "├── " else "    "
            print_tree(path, prefix=prefix + extension, preview=preview, num_rows=num_rows)
        else:
            info = get_file_info(path)
            print(prefix + pointer + path.name + info)
            if preview:
                preview_data(path, num_rows)

def main():
    parser = argparse.ArgumentParser(description="Visualize the folder structure and data of a dataset.")
    parser.add_argument("name", type=str, help="Name of the dataset (e.g., PEMSD04)")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to preview (default: 5)")
    parser.add_argument("--no-preview", action="store_true", help="Disable data preview")
    args = parser.parse_args()

    base_path = Path("Dataset") # Changed to Dataset to match actual folder name
    if not base_path.exists():
        base_path = Path("dataset")
    dataset_path = base_path / args.name

    if not base_path.exists():
        print(f"Error: 'dataset' folder not found in the current directory ({os.getcwd()}).")
        return

    if not dataset_path.exists():
        print(f"Error: Dataset '{args.name}' not found inside '{base_path}' folder.")
        print("Available datasets:")
        for d in base_path.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        return

    print(f"\nStructure for dataset: {args.name}")
    print(f"{dataset_path.name}")
    print_tree(dataset_path, preview=not args.no_preview, num_rows=args.rows)

if __name__ == "__main__":
    main()
