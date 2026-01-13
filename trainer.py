import torch
import torch.nn as nn
import os
import time
from utils import Ranger, metric, metric_per_horizon

class DGLLM_Trainer:
    """Trainer class for managing the DG-LLM training process."""
    def __init__(self, model, scaler, args, device):
        self.model = model.to(device)
        self.scaler = scaler
        self.args = args
        self.device = device
        
        self.optimizer = Ranger(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss_fn = nn.L1Loss() # MAE Loss
        
        self.log_dir = args.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y, vmd in loader:
            x, y, vmd = x.to(self.device), y.to(self.device), vmd.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x, vmd)
            
            # Inverse scale for metric but use scaled for loss?
            # Notebook uses scaled loss for stability
            loss = self.loss_fn(pred, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        preds, reals = [], []
        
        for x, y, vmd in loader:
            x, y, vmd = x.to(self.device), y.to(self.device), vmd.to(self.device)
            pred = self.model(x, vmd)
            
            loss = self.loss_fn(pred, y)
            total_loss += loss.item()
            
            # Denormalize for real-world metrics
            preds.append(self.scaler.inverse_transform(pred))
            reals.append(self.scaler.inverse_transform(y))
            
        avg_loss = total_loss / len(loader)
        
        preds = torch.cat(preds, dim=0)
        reals = torch.cat(reals, dim=0)
        
        mae, mape, rmse, wmape = metric(preds, reals)
        return avg_loss, mae, mape, rmse, wmape

    def save_checkpoint(self, epoch, val_loss):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        torch.save(state, os.path.join(self.log_dir, 'latest_checkpoint.pth'))
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(state, os.path.join(self.log_dir, 'best_model.pth'))
            print(f"  [Save] New Best Model (Loss: {val_loss:.4f})")

@torch.no_grad()
def test_model(model, loader, scaler, device):
    """Full test cycle with per-horizon metrics."""
    model.eval()
    preds, reals = [], []
    
    for x, y, vmd in loader:
        x, vmd = x.to(device), vmd.to(device)
        pred = model(x, vmd)
        preds.append(scaler.inverse_transform(pred))
        reals.append(scaler.inverse_transform(y.to(device)))
        
    preds = torch.cat(preds, dim=0)
    reals = torch.cat(reals, dim=0)
    
    maes, mapes, rmses = metric_per_horizon(preds, reals)
    
    print("\n" + "="*30)
    print("      TEST RESULTS")
    print("="*30)
    for i in range(len(maes)):
        print(f"Horizon {i+1:02d} | MAE: {maes[i]:.4f} | MAPE: {mapes[i]:.4f} | RMSE: {rmses[i]:.4f}")
    
    avg_mae, avg_mape, avg_rmse, _ = metric(preds, reals)
    print("-"*30)
    print(f"OVERALL    | MAE: {avg_mae:.4f} | MAPE: {avg_mape:.4f} | RMSE: {avg_rmse:.4f}")
    print("="*30)
    
    return preds, reals
