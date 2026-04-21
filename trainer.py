import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from utils import Ranger, MAE_torch, MAPE_torch, RMSE_torch, compute_metrics


class VMD_Trainer:
    def __init__(self, args, scaler, adj_mx, device):
        self.args = args
        self.device = device
        self.scaler = scaler
        
        from model import DGLLM
        
        self.model = DGLLM(
            device, adj_mx, args.input_dim, args.num_nodes, 
            args.input_len, args.output_len, args.llm_layer, args.U,
            vmd_K=args.vmd_k
        ).to(device)
        
        # Enable torch.compile for ~30% speedup (PyTorch 2.0+)
        if hasattr(args, 'enable_compile') and args.enable_compile:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("  >> PyTorch compilation ENABLED (~30% speedup)")
            else:
                print("  >> WARNING: torch.compile not available (requires PyTorch 2.0+)")
        
        self.optimizer = Ranger(self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
        self.loss_fn = MAE_torch
        
        # Gradient accumulation for larger effective batch sizes
        self.grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
        if self.grad_accum_steps > 1:
            print(f"  >> Gradient Accumulation: {self.grad_accum_steps} steps (Effective batch = {args.batch_size * self.grad_accum_steps})")
        
        # Mixed precision — auto-detect best dtype for GPU
        self.use_amp = getattr(args, 'use_amp', False) or getattr(args, 'use_bf16', False)
        self.amp_dtype = None
        self.grad_scaler = None
        
        if self.use_amp and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 8:  # Ampere+ (A100, L4, etc.) → BF16, no scaler needed
                self.amp_dtype = torch.bfloat16
                print(f"  >> AMP ENABLED: BF16 (GPU compute {cc[0]}.{cc[1]}, no overflow risk)")
            else:  # Turing/Volta (T4, V100, etc.) → FP16 with GradScaler
                self.amp_dtype = torch.float16
                self.grad_scaler = torch.amp.GradScaler('cuda')
                print(f"  >> AMP ENABLED: FP16 + GradScaler (GPU compute {cc[0]}.{cc[1]})")
        else:
            print("  >> Mixed Precision DISABLED")
        
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

    def train_step(self, x, y_real, vmd_data, accumulation_step=0): 
        self.model.train()
        
        # Only zero gradients at the start of accumulation cycle
        if accumulation_step == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        x_in = x.permute(0, 3, 2, 1)
        
        # Forward pass (with or without AMP)
        ctx = torch.amp.autocast('cuda', dtype=self.amp_dtype) if self.use_amp else torch.cuda.amp.autocast(enabled=False)
        with ctx:
            preds, _ = self.model(vmd_data, x_in)
            preds = preds.transpose(1, 3)
            preds_scaled = self.scaler.inverse_transform(preds)
            real_scaled = torch.unsqueeze(y_real, 1)
            loss = self.loss_fn(preds_scaled, real_scaled, 0.0)
        
        # Normalize loss for gradient accumulation
        loss = loss / self.grad_accum_steps
        
        # Backward pass
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step at end of accumulation cycle
        if (accumulation_step + 1) % self.grad_accum_steps == 0:
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
                # Check for NaN/Inf grads — skip this step if found
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if torch.isfinite(grad_norm):
                    self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
        
        # Return denormalized loss for logging
        return (loss.item() * self.grad_accum_steps), compute_metrics(preds_scaled, real_scaled)

    def eval_step(self, x, y_real, vmd_data):
        """
        Fixed eval_step that matches train_step's shape handling.
        Returns aggregated metrics (not per-horizon).
        """
        self.model.eval()
        x_in = x.permute(0, 3, 2, 1)
        
        ctx = torch.amp.autocast('cuda', dtype=self.amp_dtype) if self.use_amp else torch.cuda.amp.autocast(enabled=False)
        with torch.no_grad(), ctx:
            preds, _ = self.model(vmd_data, x_in)
            
        preds = preds.transpose(1, 3)
        preds_scaled = self.scaler.inverse_transform(preds)
        
        real_scaled = torch.unsqueeze(y_real, 1)
        
        loss = self.loss_fn(preds_scaled, real_scaled, 0.0).item()
        metrics = compute_metrics(preds_scaled, real_scaled)
        
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
        tx = x.to(device, non_blocking=True).transpose(1, 3)
        ty = y.to(device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
        tvmd = vmd.to(device, non_blocking=True)
        
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
    
    return {
        'mae': float(np.mean(total_mae)),
        'rmse': float(np.mean(total_rmse)),
        'mape': float(np.mean(total_mape))
    }
