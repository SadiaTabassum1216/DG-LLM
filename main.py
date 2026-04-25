import torch
import os
import numpy as np
import argparse
import json
import time
from tqdm import tqdm

from utils import load_pickle, seed_everything
from data_loader import load_dataset
from trainer import Trainer
from visualization import plot_predictions_vs_ground_truth, plot_error_diagnostics, plot_weekly_1step_ahead_predictions
from paths import DATASET_DIR, RESULTS_LOGS_DIR


def parse_args():
    """Parse command-line arguments for DG-LLM training."""
    parser = argparse.ArgumentParser(description='DG-LLM: Dynamic Graph LLM for Traffic Forecasting')
    
    # Dataset
    parser.add_argument('--data', type=str, default='PEMSD04',
                        choices=['PEMSD04', 'PEMSD08', 'bike_drop', 'bike_pick', 'taxi_drop', 'taxi_pick'],
                        help='Dataset name (default: PEMSD04)')
    parser.add_argument('--root_path', type=str, default=str(DATASET_DIR),
                        help='Root path for datasets')
    
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
    parser.add_argument('--log_dir', type=str, default=str(RESULTS_LOGS_DIR),
                        help='Directory for saving logs and checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training, only run testing and visualization')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations after testing')
    
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


def main():
    args = parse_args()
    
    print(f"{'='*60}")
    print(f"  DG-LLM Training - {args.data}")
    print(f"  Device: {args.device}")
    print(f"  Nodes: {args.num_nodes}")
    print(f"{'='*60}\n")

    # 1. Load Data
    print(">> Loading Dataset...")
    seed_everything(args.seed)
    data = load_dataset(args.data_path, args.batch_size, args)
    
    # Sanity check
    # validate_temporal_features(data['train_loader'])

    # 2. Load Adjacency Matrix
    adj_path = os.path.join(args.root_path, args.data, 'adj_mx.pkl')
    adj_mx = None
    if os.path.exists(adj_path):
        print(f">> Loading adjacency matrix from {adj_path}...")
        adj_data = load_pickle(adj_path)
        if isinstance(adj_data, list):
            adj_mx = adj_data[2]    # For PEMS datasets, the 3rd element is the normalized adjacency matrix
        else:
            adj_mx = adj_data
        print(f"   Adjacency matrix shape: {adj_mx.shape}")
    else:
        print(f">> Warning: No adjacency matrix found at {adj_path}. Using identity.")
        adj_mx = np.eye(args.num_nodes) # Fallback to identity if no adjacency matrix is provided

    # 3. Train model
    print("\n>> Initializing Model...")
    trainer = Trainer(args, data['scaler'], adj_mx, args.device)
    print(f"   Total parameters: {trainer.model.param_num():,}")
    
    # training
    # validation
    # logging
    # checkpoint saving
    # best model saving

    # Check for existing checkpoint
    latest_ckpt = os.path.join(args.log_dir, 'latest_checkpoint.pth')
    start_epoch = 1
    best_val_loss = float('inf')
    
    if os.path.exists(latest_ckpt):
        print(f"\n>> Found existing checkpoint at {latest_ckpt}")
        start_epoch, best_val_loss = trainer.load_checkpoint(latest_ckpt)
        start_epoch += 1
        print(f"   Resuming from epoch {start_epoch}")

    # Training Loop
    if not args.test_only:
        print(f"\n>> Starting Training from Epoch {start_epoch}...")
        training_log = {"epochs": [], "train_loss": [], "val_loss": [], "val_mae": [], "best_epoch": 0}
        
        epoch_pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Training", unit="epoch")
        for epoch in epoch_pbar:
            epoch_start = time.time()
            # Train
            epoch_loss = []
            epoch_metrics = []
            
            train_loader = data['train_loader']
            num_train_batches = len(train_loader)

            for batch_idx, (x, y, vmd) in enumerate(train_loader.get_iterator()):
                tx = x.to(args.device, non_blocking=True)
                ty = y.to(args.device, non_blocking=True)
                tvmd = vmd.to(args.device, non_blocking=True)

                # Handle gradient accumulation by using minibatches
                accumulation_step = batch_idx % trainer.grad_accum_steps
                is_last_batch = batch_idx == num_train_batches - 1
                
                
                loss, metrics = trainer.train(
                    tx,
                    ty,
                    tvmd,
                    accumulation_step=accumulation_step,
                    is_last_batch=is_last_batch,
                )
                epoch_loss.append(loss)
                epoch_metrics.append(metrics)
            
            avg_train_loss = np.mean(epoch_loss)
            # avg_train_mae = np.mean([m[0] for m in epoch_metrics])
            
            # Evaluate
            val_loss = []
            val_metrics = []
            
            for x, y, vmd in data['val_loader'].get_iterator():
                tx = x.to(args.device, non_blocking=True)
                ty = y.to(args.device, non_blocking=True)
                tvmd = vmd.to(args.device, non_blocking=True)
                
                loss, metrics = trainer.eval(tx, ty, tvmd)
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
        print(f"\n Training log saved to: {log_path}")
    else:
        print("\n>> Skipping training (--test_only mode)")




    # Evaluate
    # Check if model checkpoint exists
    best_model_path = os.path.join(args.log_dir, 'best_model.pth')
    latest_ckpt = os.path.join(args.log_dir, 'latest_checkpoint.pth')
    
    model_path = None
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"\n Found best model at: {best_model_path}")
    elif os.path.exists(latest_ckpt):
        model_path = latest_ckpt
        print(f"\n Found latest checkpoint at: {latest_ckpt}")
    else:
        print(f"\n ERROR: No trained model found!")
        print(f"  Searched for:")
        print(f"    - {best_model_path}")
        print(f"    - {latest_ckpt}")
        print(f"\n  Please train a model first by running without --test_only")
        return

    # Load model once
    print(f"\n>> Loading model from {model_path}...")
    trainer.load_model(model_path, strict=False)
    trainer.model.eval()

    # Testing and Visualization
    if args.visualize:
        print("\n" + "="*60)
        print("  TESTING MODEL & GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Single pass through test data for both testing and visualization
        all_preds = []
        all_reals = []
        
        with torch.no_grad():
            for x, y, vmd in tqdm(data['test_loader'].get_iterator(), desc="Evaluating"):
                tx = x.to(args.device, non_blocking=True)
                ty = y.to(args.device, non_blocking=True)
                tvmd = vmd.to(args.device, non_blocking=True)
                x_in = tx  # BTNF
                
                pred, _ = trainer.model(tvmd, x_in)
                preds_unscaled = data['scaler'].inverse_transform(pred)
                reals_unscaled = data['scaler'].inverse_transform(ty)
                
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
        
        plot_predictions_vs_ground_truth(preds, reals, node_idx=0, horizon_idx=0, save_dir=args.log_dir)
        plot_error_diagnostics(preds, reals, node_idx=0, save_dir=args.log_dir)
        plot_weekly_1step_ahead_predictions(preds, reals, node_idx=0, save_dir=args.log_dir)
    else:
        # Just testing, no visualization
        print("\n" + "="*60)
        print("  TESTING BEST MODEL")
        print("="*60)
        test_results = trainer.test(data['test_loader'])
        test_mae = test_results['mae']
        test_rmse = test_results['rmse']
        test_mape = test_results['mape']
    
    # Save test results to JSON
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
    print(f"\n Results saved to: {results_path}")


if __name__ == "__main__":
    main()
