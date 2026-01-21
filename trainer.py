import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from utils import Ranger, MAE_torch, MAPE_torch, RMSE_torch, metric


class VMD_Trainer:
    def __init__(self, args, scaler, adj_mx, device, lightweight=False):
        self.args = args
        self.device = device
        self.scaler = scaler
        self.lightweight = lightweight
        
        from model import DGLLM, LightweightDGLLM
        
        # Use lightweight model if enabled
        ModelClass = LightweightDGLLM if lightweight else DGLLM
        self.model = ModelClass(
            device, adj_mx, args.input_dim, args.num_nodes, 
            args.input_len, args.output_len, args.llm_layer, args.U,
            vmd_K=args.vmd_k
        ).to(device)
        
        self.optimizer = Ranger(self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        self.loss_fn = MAE_torch
        
        # Mixed precision training (FP16) - major speedup on modern GPUs
        self.use_amp = lightweight and device.type == 'cuda'
        if self.use_amp:
            self.grad_scaler = torch.amp.GradScaler('cuda')
            print("  >> Mixed Precision (FP16) ENABLED")
        
        self.log_dir = args.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.best_val_loss = float('inf')

    def save_checkpoint(self, epoch, val_loss, path):
        """Saves everything needed to resume training."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': val_loss,
            'lightweight': self.lightweight,
        }
        torch.save(state, path)
        print(f"--- Checkpoint saved to {path} (Epoch {epoch}) ---")

    def load_checkpoint(self, path):
        print(f"--- Loading checkpoint from {path} ---")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Warning: Optimizer state could not be fully loaded. Resetting optimizer.")
            
        return checkpoint['epoch'], checkpoint['best_val_loss']

    def train_step(self, x, y_real, vmd_data): 
        self.model.train()
        self.optimizer.zero_grad()
        
        x_in = x.permute(0, 3, 2, 1)
        
        if self.use_amp:
            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                preds, _ = self.model(vmd_data, x_in)
                preds = preds.transpose(1, 3)
                preds_scaled = self.scaler.inverse_transform(preds)
                real_scaled = torch.unsqueeze(y_real, 1)
                loss = self.loss_fn(preds_scaled, real_scaled, 0.0)
            
            # Scaled backward pass
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            preds, _ = self.model(vmd_data, x_in)
            preds = preds.transpose(1, 3)
            preds_scaled = self.scaler.inverse_transform(preds)
            real_scaled = torch.unsqueeze(y_real, 1)
            loss = self.loss_fn(preds_scaled, real_scaled, 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
        
        return loss.item(), metric(preds_scaled, real_scaled)

    def eval_step(self, x, y_real, vmd_data):
        """
        Fixed eval_step that matches train_step's shape handling.
        Returns aggregated metrics (not per-horizon).
        """
        self.model.eval()
        x_in = x.permute(0, 3, 2, 1)
        
        with torch.no_grad():
            preds, _ = self.model(vmd_data, x_in)
            
        preds = preds.transpose(1, 3)
        preds_scaled = self.scaler.inverse_transform(preds)
        
        real_scaled = torch.unsqueeze(y_real, 1)
        
        loss = self.loss_fn(preds_scaled, real_scaled, 0.0).item()
        metrics = metric(preds_scaled, real_scaled)
        
        return loss, metrics



def test_model(trainer, dataloader, device, model_path):
    """
    Fixed test_model that works with the simplified eval_step.
    Computes per-horizon metrics by slicing predictions.
    """
    print(f"\n>> Loading best model from {model_path} ...")
    trainer.model.load_state_dict(torch.load(model_path, weights_only=False))
    trainer.model.eval()
    
    horizon_mae = [[] for _ in range(trainer.args.output_len)]
    horizon_mape = [[] for _ in range(trainer.args.output_len)]
    horizon_rmse = [[] for _ in range(trainer.args.output_len)]
    
    print(">> Starting Detailed Horizon Evaluation...")
    
    for x, y, vmd in tqdm(dataloader["test_loader"].get_iterator(), desc="Testing"):
        tx = torch.Tensor(x).to(device).transpose(1, 3)
        ty = torch.Tensor(y).to(device).transpose(1, 3)[:, 0, :, :]
        tvmd = torch.Tensor(vmd).to(device)
        
        x_in = tx.permute(0, 3, 2, 1)
        with torch.no_grad():
            preds, _ = trainer.model(tvmd, x_in)
        
        preds_scaled = trainer.scaler.inverse_transform(preds)
        real_scaled = ty.permute(0, 2, 1).unsqueeze(-1)
        
        for t in range(trainer.args.output_len):
            p = preds_scaled[:, t, ...]
            r = real_scaled[:, t, ...]
            
            horizon_mae[t].append(MAE_torch(p, r, 0).item())
            horizon_mape[t].append(MAPE_torch(p, r, 0).item())
            horizon_rmse[t].append(RMSE_torch(p, r, 0).item())

    print("\n" + "="*50)
    print(f"{'Horizon':<10} | {'MAE':<10} | {'MAPE':<10} | {'RMSE':<10}")
    print("-" * 50)
    
    total_mae, total_mape, total_rmse = [], [], []
    
    for i in range(trainer.args.output_len):
        m_mae = np.mean(horizon_mae[i])
        m_mape = np.mean(horizon_mape[i])
        m_rmse = np.mean(horizon_rmse[i])
        
        total_mae.append(m_mae)
        total_mape.append(m_mape)
        total_rmse.append(m_rmse)
        
        print(f"Step {i+1:02d}    | {m_mae:<10.4f} | {m_mape:<10.4f} | {m_rmse:<10.4f}")
    
    print("-" * 50)
    print(f"AVERAGE    | {np.mean(total_mae):<10.4f} | {np.mean(total_mape):<10.4f} | {np.mean(total_rmse):<10.4f}")
    print("="*50)
