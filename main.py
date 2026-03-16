import torch
import os
import random
import numpy as np
import argparse
import json
import time
from tqdm import tqdm
from data_loader import load_dataset_optimized
from trainer import VMD_Trainer, test_model
from visualization import visualize_model_predictions, verify_temporal_features, visualize_advanced_diagnostics, visualize_weekly_horizon1
from utils import load_pickle
from experiment_utils import seed_everything, compute_statistics, save_statistical_results
from evaluate import evaluate_model_statistical


def parse_args():
    """Parse command-line arguments for DG-LLM training."""
    parser = argparse.ArgumentParser(description='DG-LLM: Dynamic Graph LLM for Traffic Forecasting')
    
    # Dataset
    parser.add_argument('--data', type=str, default='PEMSD04',
                        choices=['PEMSD04', 'PEMSD08', 'bike_drop', 'bike_pick', 'taxi_drop', 'taxi_pick'],
                        help='Dataset name (default: PEMSD04)')
    parser.add_argument('--root_path', type=str, default='./Dataset/',
                        help='Root path for datasets (default: ./Dataset/)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--wdecay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    
    # Model
    parser.add_argument('--llm_layer', type=int, default=6,
                        help='Number of GPT-2 layers to use (default: 6)')
    parser.add_argument('--U', type=int, default=1,
                        help='Top U layers are fully trainable (default: 1)')
    parser.add_argument('--vmd_k', type=int, default=3,
                        help='Number of VMD modes (default: 3)')
    parser.add_argument('--lora_r', type=int, default=16, dest='lora_rank',
                        help='LoRA rank r (default: 16)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha scaling factor (default: 32)')
    
    # I/O dimensions
    parser.add_argument('--input_dim', type=int, default=3,
                        help='Input dimension (Flow, ToD, DoW) (default: 3)')
    parser.add_argument('--input_len', type=int, default=12,
                        help='Input sequence length (default: 12)')
    parser.add_argument('--output_len', type=int, default=12,
                        help='Output/prediction length (default: 12)')
    
    # Misc
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for saving logs and checkpoints (default: ./logs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training, only run testing and visualization')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations after testing')
    
    # Multi-seed experiments for statistical rigor
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Number of random seeds to run (default: 1)')
    parser.add_argument('--seed_start', type=int, default=42,
                        help='Starting seed value for multi-seed experiments (default: 42)')
    parser.add_argument('--save_stats', action='store_true',
                        help='Save statistical results to JSON file')
    
    # Efficiency optimizations
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps for larger effective batch size (default: 1)')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validate every N epochs (default: 1, set to 5 for faster training)')
    parser.add_argument('--enable_compile', action='store_true',
                        help='Enable torch.compile() for ~30%% speedup (PyTorch 2.0+ required)')
    
    parser.add_argument('--use_amp', '--use_bf16', action='store_true',
                        help='Enable mixed precision (auto-detects FP16 on T4/Turing, BF16 on Ampere+)')
    
    args = parser.parse_args()
    
    # Derived attributes
    args.data_path = os.path.join(args.root_path, args.data, 'processed')
    
    # Dataset-specific node counts
    if 'PEMSD04' in args.data:
        args.num_nodes = 307
    elif 'PEMSD08' in args.data:
        args.num_nodes = 170
    elif 'bike' in args.data:
        args.num_nodes = 250
    elif 'taxi' in args.data:
        args.num_nodes = 266
    else:
        args.num_nodes = 307  # Default
    
    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args


def train_single_seed(seed, args, data, adj_mx):
    """
    Train model with a single seed and return test metrics.
    This function is used by the multi-seed experiment framework.
    
    Args:
        seed: Random seed
        args: Parsed arguments
        data: Loaded dataset dictionary
        adj_mx: Adjacency matrix
    
    Returns:
        Dictionary with test metrics (mae, rmse, mape)
    """
    # Seed-specific log directory
    seed_log_dir = os.path.join(args.log_dir, f"seed_{seed}")
    os.makedirs(seed_log_dir, exist_ok=True)
    
    # Initialize Trainer
    trainer = VMD_Trainer(args, data['scaler'], adj_mx, args.device)
    
    # Training loop
    best_val_loss = float('inf')
    training_log = {"epochs": [], "train_loss": [], "val_loss": [], "val_mae": [], "best_epoch": 0}
    
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        epoch_start = time.time()
        # Train
        epoch_loss = []
        epoch_metrics = []
        
        accumulation_counter = 0
        for x, y, vmd in data['train_loader'].get_iterator():
            tx = x.to(args.device, non_blocking=True).transpose(1, 3)
            ty = y.to(args.device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
            tvmd = vmd.to(args.device, non_blocking=True)
            
            # Pass accumulation step for gradient accumulation
            loss, metrics = trainer.train_step(tx, ty, tvmd, accumulation_step=accumulation_counter)
            accumulation_counter += 1
            
            epoch_loss.append(loss)
            epoch_metrics.append(metrics)
        
        avg_train_loss = np.mean(epoch_loss)
        avg_train_mae = np.mean([m[0] for m in epoch_metrics])
        
        # Validation (only every val_interval epochs)
        val_interval = getattr(args, 'val_interval', 1)
        if epoch % val_interval == 0 or epoch == args.epochs:
            # Evaluate
            val_loss = []
            val_metrics = []
            
            for x, y, vmd in data['val_loader'].get_iterator():
                tx = x.to(args.device, non_blocking=True).transpose(1, 3)
                ty = y.to(args.device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
                tvmd = vmd.to(args.device, non_blocking=True)
                
                loss, metrics = trainer.eval_step(tx, ty, tvmd)
                val_loss.append(loss)
                val_metrics.append(metrics)
            
            avg_val_loss = np.mean(val_loss)
            avg_val_mae = np.mean([m[0] for m in val_metrics])
            avg_val_rmse = np.mean([m[2] for m in val_metrics])
            
            epoch_time = time.time() - epoch_start
            epoch_pbar.set_postfix(loss=f"{avg_train_loss:.4f}", val_mae=f"{avg_val_mae:.4f}", time=f"{epoch_time:.1f}s")
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.4f} | Val RMSE: {avg_val_rmse:.4f} | Time: {epoch_time:.1f}s")
            
            # Log training metrics
            training_log["epochs"].append(epoch)
            training_log["train_loss"].append(float(avg_train_loss))
            training_log["val_loss"].append(float(avg_val_loss))
            training_log["val_mae"].append(float(avg_val_mae))
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(seed_log_dir, 'best_model.pth')
                torch.save(trainer.model.state_dict(), best_model_path)
                training_log["best_epoch"] = epoch
        else:
            # Skip validation, just print training status
            epoch_time = time.time() - epoch_start
            epoch_pbar.set_postfix(loss=f"{avg_train_loss:.4f}", time=f"{epoch_time:.1f}s")
            if epoch % 10 == 0:
                print(f"    Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Time: {epoch_time:.1f}s (validation skipped)")
    
    # Save training log
    log_path = os.path.join(seed_log_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"    ✓ Training log saved to: {log_path}")
    
    # Test on best model
    best_model_path = os.path.join(seed_log_dir, 'best_model.pth')
    test_results = evaluate_model_statistical(
        trainer,
        data['test_loader'],
        args.device,
        data['scaler'],
        args.output_len,
        current_seed=seed
    )
    
    # Save test results
    results_path = os.path.join(seed_log_dir, 'results.json')
    results_to_save = {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in test_results.items()}
    results_to_save['seed'] = seed
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"    ✓ Test results saved to: {results_path}")
    
    return test_results


def main():
    args = parse_args()
    
    print(f"{'='*60}")
    print(f"  DG-LLM Training - {args.data}")
    print(f"  Device: {args.device}")
    print(f"  Nodes: {args.num_nodes}")
    if args.num_seeds > 1:
        print(f"  Multi-seed mode: {args.num_seeds} seeds")
    print(f"{'='*60}\n")

    # 1. Load Data (once, shared across seeds)
    print(">> Loading Dataset...")
    seed_everything(args.seed_start)  # Use consistent seed for data loading
    data = load_dataset_optimized(args.data_path, args.batch_size, args)
    
    # Sanity check
    verify_temporal_features(data['train_loader'])

    # 2. Load Adjacency Matrix
    adj_path = os.path.join(args.root_path, args.data, 'adj_mx.pkl')
    adj_mx = None
    if os.path.exists(adj_path):
        print(f">> Loading adjacency matrix from {adj_path}...")
        adj_data = load_pickle(adj_path)
        if isinstance(adj_data, list):
            adj_mx = adj_data[2]
        else:
            adj_mx = adj_data
        print(f"   Adjacency matrix shape: {adj_mx.shape}")
    else:
        print(f">> Warning: No adjacency matrix found at {adj_path}. Using identity.")
        adj_mx = np.eye(args.num_nodes)

    # 3. Run experiments
    if args.num_seeds > 1:
        # Multi-seed experiment mode
        print(f"\n{'='*70}")
        print(f"  MULTI-SEED EXPERIMENT MODE")
        print(f"  Running {args.num_seeds} independent experiments")
        print(f"{'='*70}")
        
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
        
        # Run multi-seed experiments
        all_results = {}
        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n{'─'*70}")
            print(f"  Experiment {seed_idx}/{args.num_seeds} - Seed: {seed}")
            print(f"{'─'*70}")
            
            seed_everything(seed)
            seed_results = train_single_seed(seed, args, data, adj_mx)
            
            # Accumulate results (both scalars and per-horizon lists)
            for metric, value in seed_results.items():
                if isinstance(value, (int, float)):
                    # Scalar metrics
                    if metric not in all_results:
                        all_results[metric] = []
                    all_results[metric].append(value)
                elif isinstance(value, list):
                    # Per-horizon metrics
                    if metric not in all_results:
                        all_results[metric] = []
                    all_results[metric].append(value)
        
        # Compute and display overall statistics
        print(f"\n{'='*70}")
        print(f"  OVERALL RESULTS ACROSS {args.num_seeds} SEEDS")
        print(f"{'='*70}")
        
        stats = compute_statistics(all_results, confidence_level=0.95)
        
        # Print formatted results
        print(f"\n{'Metric':<15} | {'Mean':<12} | {'Std':<12} | {'95% CI':<25}")
        print(f"{'-'*70}")
        for metric in ['mae', 'rmse', 'mape']:
            if metric in stats:
                s = stats[metric]
                ci_str = f"[{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]"
                print(f"{metric.upper():<15} | {s['mean']:<12.4f} | {s['std']:<12.4f} | {ci_str:<25}")
        print(f"{'='*70}")
        
        # Compute and display per-horizon statistics
        from horizon_stats import aggregate_per_horizon_metrics, print_per_horizon_statistics, save_per_horizon_statistics
        
        horizon_metrics = {k: v for k, v in all_results.items() if k.startswith('horizon_')}
        if horizon_metrics:
            horizon_stats = aggregate_per_horizon_metrics(horizon_metrics, confidence_level=0.95)
            print_per_horizon_statistics(horizon_stats, args.num_seeds)
            
            # Save per-horizon stats if requested
            if args.save_stats:
                horizon_save_path = os.path.join(args.log_dir, f'{args.data}_horizon_stats.json')
                save_per_horizon_statistics(horizon_stats, horizon_save_path)
        
        
        # Save statistical results
        if args.save_stats:
            stats_path = os.path.join(args.log_dir, f'{args.data}_multiseed_stats.json')
            save_statistical_results(stats, stats_path, format='json')
        
    else:
        # Single-seed mode (original behavior)
        seed_everything(args.seed)
        
        print("\n>> Initializing Model...")
        trainer = VMD_Trainer(args, data['scaler'], adj_mx, args.device)
        print(f"   Total parameters: {trainer.model.param_num():,}")

        # Check for existing checkpoint
        latest_ckpt = os.path.join(args.log_dir, 'latest_checkpoint.pth')
        start_epoch = 1
        best_val_loss = float('inf')
        
        if os.path.exists(latest_ckpt):
            print(f"\n>> Found existing checkpoint at {latest_ckpt}")
            start_epoch, best_val_loss = trainer.load_checkpoint(latest_ckpt)
            start_epoch += 1
            print(f"   Resuming from epoch {start_epoch}")

        # Training Loop (skip if --test_only)
        if not args.test_only:
            print(f"\n>> Starting Training from Epoch {start_epoch}...")
            training_log = {"epochs": [], "train_loss": [], "val_loss": [], "val_mae": [], "best_epoch": 0}
            
            epoch_pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Training", unit="epoch")
            for epoch in epoch_pbar:
                epoch_start = time.time()
                # Train
                epoch_loss = []
                epoch_metrics = []
                
                for x, y, vmd in data['train_loader'].get_iterator():
                    tx = x.to(args.device, non_blocking=True).transpose(1, 3)
                    ty = y.to(args.device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
                    tvmd = vmd.to(args.device, non_blocking=True)
                    
                    loss, metrics = trainer.train_step(tx, ty, tvmd)
                    epoch_loss.append(loss)
                    epoch_metrics.append(metrics)
                
                avg_train_loss = np.mean(epoch_loss)
                avg_train_mae = np.mean([m[0] for m in epoch_metrics])
                
                # Evaluate
                val_loss = []
                val_metrics = []
                
                for x, y, vmd in data['val_loader'].get_iterator():
                    tx = x.to(args.device, non_blocking=True).transpose(1, 3)
                    ty = y.to(args.device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
                    tvmd = vmd.to(args.device, non_blocking=True)
                    
                    loss, metrics = trainer.eval_step(tx, ty, tvmd)
                    val_loss.append(loss)
                    val_metrics.append(metrics)
                
                avg_val_loss = np.mean(val_loss)
                avg_val_mae = np.mean([m[0] for m in val_metrics])
                avg_val_rmse = np.mean([m[2] for m in val_metrics])
                
                epoch_time = time.time() - epoch_start
                epoch_pbar.set_postfix(loss=f"{avg_train_loss:.4f}", val_mae=f"{avg_val_mae:.4f}", time=f"{epoch_time:.1f}s")
                print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.4f} | Val RMSE: {avg_val_rmse:.4f} | Time: {epoch_time:.1f}s")
                
                # Log training metrics
                training_log["epochs"].append(epoch)
                training_log["train_loss"].append(float(avg_train_loss))
                training_log["val_loss"].append(float(avg_val_loss))
                training_log["val_mae"].append(float(avg_val_mae))
                
                # Save checkpoint
                trainer.save_checkpoint(epoch, avg_val_loss, os.path.join(args.log_dir, 'latest_checkpoint.pth'))
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(trainer.model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
                    print(f"  >> New Best Model Saved (Val Loss: {avg_val_loss:.4f})")
                    training_log["best_epoch"] = epoch
            
            # Save training log
            log_path = os.path.join(args.log_dir, 'training_log.json')
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)
            print(f"\n✓ Training log saved to: {log_path}")
        else:
            print("\n>> Skipping training (--test_only mode)")

        # Check if model checkpoint exists
        best_model_path = os.path.join(args.log_dir, 'best_model.pth')
        latest_ckpt = os.path.join(args.log_dir, 'latest_checkpoint.pth')
        
        model_path = None
        if os.path.exists(best_model_path):
            model_path = best_model_path
            print(f"\n✓ Found best model at: {best_model_path}")
        elif os.path.exists(latest_ckpt):
            model_path = latest_ckpt
            print(f"\n✓ Found latest checkpoint at: {latest_ckpt}")
        else:
            print(f"\n✗ ERROR: No trained model found!")
            print(f"  Searched for:")
            print(f"    - {best_model_path}")
            print(f"    - {latest_ckpt}")
            print(f"\n  Please train a model first by running without --test_only")
            return

        # Load model once
        print(f"\n>> Loading model from {model_path}...")
        trainer.model.load_state_dict(torch.load(model_path, weights_only=False))
        trainer.model.eval()

        # Testing and Visualization (combined to avoid redundant passes)
        if args.visualize:
            print("\n" + "="*60)
            print("  TESTING MODEL & GENERATING VISUALIZATIONS")
            print("="*60)
            
            # Single pass through test data for both testing and visualization
            all_preds = []
            all_reals = []
            
            with torch.no_grad():
                for x, y, vmd in tqdm(data['test_loader'].get_iterator(), desc="Evaluating"):
                    tx = x.to(args.device, non_blocking=True).transpose(1, 3)
                    ty = y.to(args.device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
                    tvmd = vmd.to(args.device, non_blocking=True)
                    x_in = tx.permute(0, 3, 2, 1)  # [B, T, N, F]
                    
                    pred, _ = trainer.model(tvmd, x_in)
                    preds_unscaled = data['scaler'].inverse_transform(pred)
                    reals_unscaled = data['scaler'].inverse_transform(ty.permute(0, 2, 1).unsqueeze(-1))
                    
                    all_preds.append(preds_unscaled)
                    all_reals.append(reals_unscaled)
            
            preds = torch.cat(all_preds, dim=0)
            reals = torch.cat(all_reals, dim=0)
            
            # Compute and print test metrics
            from utils import MAE_torch, RMSE_torch, MAPE_torch
            test_mae = MAE_torch(preds, reals, 0).item()
            test_rmse = RMSE_torch(preds, reals, 0).item()
            test_mape = MAPE_torch(preds, reals, 0).item()
            
            print(f"\n{'='*60}")
            print(f"  TEST RESULTS")
            print(f"{'='*60}")
            print(f"  MAE:  {test_mae:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
            print(f"  MAPE: {test_mape:.4f}")
            print(f"{'='*60}")
            
            # Generate visualizations
            print(f"\n>> Generating visualizations...")
            print(f"   Predictions shape: {preds.shape}")
            print(f"   Ground truth shape: {reals.shape}")
            
            visualize_model_predictions(preds, reals, node_idx=0, horizon_idx=0)
            visualize_advanced_diagnostics(preds, reals, node_idx=0)
            visualize_weekly_horizon1(preds, reals, node_idx=0)
            
            print("\n✓ Visualizations saved to ./logs/")
            print("  - predictions_node0_h0.png")
            print("  - diagnostics_node0.png")
            print("  - weekly_node0.png")
        else:
            # Just testing, no visualization
            print("\n" + "="*60)
            print("  TESTING BEST MODEL")
            print("="*60)
            test_results = test_model(trainer, data, args.device, model_path)
            test_mae = test_results['mae']
            test_rmse = test_results['rmse']
            test_mape = test_results['mape']
        
        # Save test results to JSON for multi-seed aggregation
        results_path = os.path.join(args.log_dir, 'results.json')
        results_to_save = {
            'mae': test_mae,
            'rmse': test_rmse,
            'mape': test_mape,
            'seed': args.seed,
            'dataset': args.data,
            'epochs': args.epochs
        }
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()

