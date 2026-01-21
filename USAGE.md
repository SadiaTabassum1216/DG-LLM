# DG-LLM Usage Guide

## Quick Start

```bash
# Default training (PEMSD04, 50 epochs)
python main.py

# Specify dataset and epochs
python main.py --data PEMSD08 --epochs 100

# Full customization
python main.py --data taxi_drop --epochs 100 --batch_size 16 --lrate 5e-4
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `PEMSD04` | Dataset name |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `8` | Batch size |
| `--lrate` | `1e-3` | Learning rate |
| `--wdecay` | `1e-5` | Weight decay |
| `--llm_layer` | `6` | Number of GPT-2 layers |
| `--U` | `1` | Top U trainable layers |
| `--vmd_k` | `3` | VMD modes |
| `--log_dir` | `./logs` | Checkpoint directory |
| `--seed` | `42` | Random seed |
| `--test_only` | flag | Skip training, test only |
| `--visualize` | flag | Generate plots after testing |

## Available Datasets

| Dataset | Nodes | Type |
|---------|-------|------|
| `PEMSD04` | 307 | Traffic flow |
| `PEMSD08` | 170 | Traffic flow |
| `bike_drop` | 250 | Bike sharing |
| `bike_pick` | 250 | Bike sharing |
| `taxi_drop` | 266 | Taxi trips |
| `taxi_pick` | 266 | Taxi trips |

## Examples

### Train on Different Datasets

```bash
python main.py --data PEMSD04 --epochs 50 --batch_size 8
python main.py --data taxi_drop --epochs 100 --batch_size 16
python main.py --data bike_pick --epochs 50 --batch_size 32
```

### Test & Visualize (After Training)

```bash
python main.py --data PEMSD04 --test_only --visualize
```

### Resume Training from Checkpoint

```bash
# Just run again - it auto-loads from ./logs/latest_checkpoint.pth
python main.py --data PEMSD04 --epochs 100
```

### Custom Hyperparameters

```bash
python main.py --data PEMSD08 --lrate 5e-4 --wdecay 1e-4 --llm_layer 8 --U 2
```

## Output Files

After training, check `./logs/`:

- `best_model.pth` - Best model weights
- `latest_checkpoint.pth` - Resume checkpoint
- `predictions_node0_h0.png` - Prediction plot
- `diagnostics_node0.png` - Error analysis
- `weekly_node0.png` - Weekly patterns
