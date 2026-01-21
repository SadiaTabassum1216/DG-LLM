import torch
import os
import random
import numpy as np
import argparse
from data_loader import load_dataset_optimized
from trainer import VMD_Trainer, test_model
from visualization import visualize_model_predictions, verify_temporal_features, visualize_advanced_diagnostics, visualize_weekly_horizon1
from utils import load_pickle


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


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    seed_everything(args.seed)
    
    print(f"{'='*60}")
    print(f"  DG-LLM Training - {args.data}")
    print(f"  Device: {args.device}")
    print(f"  Nodes: {args.num_nodes}")
    print(f"{'='*60}\n")

    # 1. Load Data
    print(">> Loading Dataset...")
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

    # 3. Initialize Trainer
    print("\n>> Initializing Model...")
    trainer = VMD_Trainer(args, data['scaler'], adj_mx, args.device)
    print(f"   Total parameters: {trainer.model.param_num():,}")

    # 4. Check for existing checkpoint
    latest_ckpt = os.path.join(args.log_dir, 'latest_checkpoint.pth')
    start_epoch = 1
    best_val_loss = float('inf')
    
    if os.path.exists(latest_ckpt):
        print(f"\n>> Found existing checkpoint at {latest_ckpt}")
        start_epoch, best_val_loss = trainer.load_checkpoint(latest_ckpt)
        start_epoch += 1
        print(f"   Resuming from epoch {start_epoch}")


    # 5. Training Loop (skip if --test_only)
    if not args.test_only:
        print(f"\n>> Starting Training from Epoch {start_epoch}...")
        for epoch in range(start_epoch, args.epochs + 1):
            # Train
            epoch_loss = []
            epoch_metrics = []
            
            for x, y, vmd in data['train_loader'].get_iterator():
                tx = torch.Tensor(x).to(args.device).transpose(1, 3)
                ty = torch.Tensor(y).to(args.device).transpose(1, 3)[:, 0, :, :]
                tvmd = torch.Tensor(vmd).to(args.device)
                
                loss, metrics = trainer.train_step(tx, ty, tvmd)
                epoch_loss.append(loss)
                epoch_metrics.append(metrics)
            
            avg_train_loss = np.mean(epoch_loss)
            avg_train_mae = np.mean([m[0] for m in epoch_metrics])
            
            # Evaluate
            val_loss = []
            val_metrics = []
            
            for x, y, vmd in data['val_loader'].get_iterator():
                tx = torch.Tensor(x).to(args.device).transpose(1, 3)
                ty = torch.Tensor(y).to(args.device).transpose(1, 3)[:, 0, :, :]
                tvmd = torch.Tensor(vmd).to(args.device)
                
                loss, metrics = trainer.eval_step(tx, ty, tvmd)
                val_loss.append(loss)
                val_metrics.append(metrics)
            
            avg_val_loss = np.mean(val_loss)
            avg_val_mae = np.mean([m[0] for m in val_metrics])
            avg_val_rmse = np.mean([m[2] for m in val_metrics])
            
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.4f} | Val RMSE: {avg_val_rmse:.4f}")
            
            # Save checkpoint
            trainer.save_checkpoint(epoch, avg_val_loss, os.path.join(args.log_dir, 'latest_checkpoint.pth'))
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(trainer.model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))
                print(f"  >> New Best Model Saved (Val Loss: {avg_val_loss:.4f})")
    else:
        print("\n>> Skipping training (--test_only mode)")

    # 7. Testing
    print("\n" + "="*60)
    print("  TESTING BEST MODEL")
    print("="*60)
    best_model_path = os.path.join(args.log_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        test_model(trainer, data, args.device, best_model_path)
    else:
        print("No best model found. Testing with latest checkpoint...")
        test_model(trainer, data, args.device, latest_ckpt)

    # 8. Visualization (if --visualize flag is set)
    if args.visualize:
        print("\n" + "="*60)
        print("  GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Load best model for visualization
        if os.path.exists(best_model_path):
            trainer.model.load_state_dict(torch.load(best_model_path, weights_only=False))
        trainer.model.eval()
        
        # Collect predictions
        all_preds = []
        all_reals = []
        
        with torch.no_grad():
            for x, y, vmd in data['test_loader'].get_iterator():
                tx = torch.Tensor(x).to(args.device).transpose(1, 3)
                ty = torch.Tensor(y).to(args.device).transpose(1, 3)[:, 0, :, :]
                tvmd = torch.Tensor(vmd).to(args.device)
                x_in = tx.permute(0, 3, 2, 1)  # [B, T, N, F]
                
                pred, _ = trainer.model(tvmd, x_in)
                preds_unscaled = data['scaler'].inverse_transform(pred)
                reals_unscaled = data['scaler'].inverse_transform(ty.permute(0, 2, 1).unsqueeze(-1))
                
                all_preds.append(preds_unscaled)
                all_reals.append(reals_unscaled)
        
        preds = torch.cat(all_preds, dim=0)
        reals = torch.cat(all_reals, dim=0)
        
        print(f"Predictions shape: {preds.shape}")
        print(f"Reals shape: {reals.shape}")
        
        # Generate plots
        visualize_model_predictions(preds, reals, node_idx=0, horizon_idx=0)
        visualize_advanced_diagnostics(preds, reals, node_idx=0)
        visualize_weekly_horizon1(preds, reals, node_idx=0)
        
        print("\n>> Visualizations complete!")


if __name__ == "__main__":
    main()
