import torch
import os
import random
import numpy as np
from data_loader import load_dataset_optimized
from model import DGLLM
from trainer import DGLLM_Trainer, test_model
from visualization import visualize_model_predictions, verify_temporal_features
from utils import load_pickle

class Args:
    """Configurable arguments for DG-LLM."""
    def __init__(self):
        # Data
        self.data = 'PEMSD04' # Options: 'PEMSD04', 'PEMSD08'
        self.root_path = './Dataset/'
        self.data_path = os.path.join(self.root_path, self.data, 'processed')
        
        # Model
        self.input_len = 12
        self.output_len = 12
        
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
            self.num_nodes = 307 # Default
            
        self.vmd_k = 3
        self.d_model = 64
        self.patch_size = 1
        self.lora_r = 8
        self.llm_layers = 6 # Added from notebook
        self.U = 1 # Added from notebook (number of layers to keep trainable)
        self.fusion = 'attention' # Updated default to 'attention'
        self.use_checkpoint = True
        
        # Training
        self.batch_size = 32
        self.epochs = 50
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.log_dir = './log'
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
    
    print(f"--- Running DG-LLM on {args.data} ---")
    print(f"Device: {args.device}")

    # 1. Load Data
    data = load_dataset_optimized(args.data_path, args.batch_size, args)
    
    # Sanity check
    verify_temporal_features(data['train_loader'])

    # 1.5 Load Adjacency
    adj_path = os.path.join(args.root_path, args.data, 'adj_mx.pkl')
    adj_mx = None
    if os.path.exists(adj_path):
        print(f"Loading adjacency matrix from {adj_path}...")
        adj_data = load_pickle(adj_path)
        # Extract matrix if it's a list (common in traffic datasets)
        if isinstance(adj_data, list):
            adj_mx = adj_data[2]
        else:
            adj_mx = adj_data
        print(f"Adjacency matrix loaded with shape: {adj_mx.shape}")

    # 2. Initialize Model
    model = DGLLM(args, adj_mx=adj_mx)
    trainer = DGLLM_Trainer(model, data['scaler'], args, args.device)

    # 3. Training Loop
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(data['train_loader'])
        val_loss, mae, mape, rmse, wmape = trainer.eval_epoch(data['val_loader'])
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val MAE: {mae:.4f} | RMSE: {rmse:.4f}")
        
        trainer.save_checkpoint(epoch, val_loss)

    # 4. Testing
    print("\nLoading best model for testing...")
    best_path = os.path.join(args.log_dir, 'best_model.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state'])
        
    preds, reals = test_model(model, data['test_loader'], data['scaler'], args.device)
    
    # 5. Visualization
    visualize_model_predictions(preds, reals)

if __name__ == "__main__":
    main()
