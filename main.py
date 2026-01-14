import torch
import os
import random
import numpy as np
from data_loader import load_dataset_optimized
from trainer import VMD_Trainer, test_model
from visualization import visualize_model_predictions, verify_temporal_features
from utils import load_pickle


class Args:
    """Configurable arguments for DG-LLM - matching notebook configuration."""
    def __init__(self):
        # Dataset
        self.data = 'PEMSD04'  # Options: 'PEMSD04', 'PEMSD08', 'bike_drop', 'bike_pick', 'taxi_drop', 'taxi_pick'
        self.root_path = './Dataset/'
        self.data_path = os.path.join(self.root_path, self.data, 'processed')
        
        # Dataset-specific node counts
        if 'PEMSD04' in self.data:
            self.num_nodes = 307
        elif 'PEMSD08' in self.data:
            self.num_nodes = 170
        elif 'bike' in self.data:
            self.num_nodes = 250
        elif 'taxi' in self.data:
            self.num_nodes = 266
        else:
            self.num_nodes = 307  # Default
        
        # Input/Output dimensions (matching notebook)
        self.input_dim = 3  # Flow, ToD, DoW
        self.input_len = 12
        self.output_len = 12
        
        # GPT-2 / Model settings (matching notebook)
        self.llm_layer = 6  # Number of GPT-2 layers
        self.U = 1  # Top U layers are fully trainable
        
        # VMD
        self.vmd_k = 3  # Number of VMD modes
        
        # Training (matching notebook)
        self.lrate = 1e-3  # Learning rate
        self.wdecay = 1e-5  # Weight decay
        self.batch_size = 32
        self.epochs = 50
        self.log_dir = './logs'
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = Args()
    seed_everything()
    
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

    # 5. Training Loop
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

    # 6. Testing
    print("\n" + "="*60)
    print("  TESTING BEST MODEL")
    print("="*60)
    best_model_path = os.path.join(args.log_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        test_model(trainer, data, args.device, best_model_path)
    else:
        print("No best model found. Testing with latest checkpoint...")
        test_model(trainer, data, args.device, latest_ckpt)


if __name__ == "__main__":
    main()
