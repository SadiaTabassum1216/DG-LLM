import os
import torch
import optuna
import argparse
import numpy as np

from data_loader import load_dataset
from model import DGLLM
from utils import seed_everything, load_pickle, Ranger

def create_model(trial, args, device, adj_mx):
    # --- Consolidated Hyperparameter Sampling ---
    
    # 1. Graph Pruning & Stability
    p_keep = trial.suggest_float("p_keep", 0.01, 0.2)
    hysteresis_ratio = trial.suggest_float("hysteresis_ratio", 0.5, 0.95)
    edge_dropout = trial.suggest_float("edge_dropout", 0.0, 0.4)
    
    # 2. Graph Attention Scoring
    gat_tau = trial.suggest_float("gat_tau", 0.5, 1.5)
    head_dropout = trial.suggest_float("head_dropout", 0.0, 0.5)
    leaky_slope = trial.suggest_float("leaky_slope", 0.1, 0.4)
    
    # 3. Adaptive Graph Blending & EMA
    mix_hi = trial.suggest_float("mix_hi", 0.7, 1.0)
    mix_lo = trial.suggest_float("mix_lo", 0.3, 0.7)
    ema_m = trial.suggest_float("ema_m", 0.8, 0.99)
    warmup_steps = trial.suggest_int("warmup_steps", 100, 2000)
    
    # 4. Node Importance & Priors
    degree_prior_base = trial.suggest_float("degree_prior_base", 0.1, 0.8)
    degree_prior_scale = trial.suggest_float("degree_prior_scale", 0.1, 1.0)
    
    # 5. Global Residual Scale
    residual_scale = trial.suggest_float("RESIDUAL_SCALE", 0.01, 0.2)

    model = DGLLM(
        device=device,
        static_road_network=adj_mx,
        input_dim=args.input_dim,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        llm_layer=args.llm_layer,
        U=args.U,
        vmd_K=args.vmd_k,
        use_attention_fusion=True,
    ).to(device)
    
    # Apply Global Residual Scale
    model.RESIDUAL_SCALE = residual_scale
    
    # Inject spatial and temporal hyperparameters into each mode processor
    for mode_proc in model.mode_processors:
        mode_proc.p_keep = p_keep
        mode_proc.hysteresis_ratio = hysteresis_ratio
        mode_proc.edge_dropout = edge_dropout
        mode_proc.gat_tau = gat_tau
        mode_proc.head_dropout = head_dropout
        mode_proc.leaky_slope = leaky_slope
        mode_proc.mix_hi = mix_hi
        mode_proc.mix_lo = mix_lo
        mode_proc.ema_m = ema_m
        mode_proc.warmup_steps = warmup_steps
        mode_proc.DEGREE_PRIOR_BASE = degree_prior_base
        mode_proc.DEGREE_PRIOR_SCALE = degree_prior_scale
    
    return model

def objective(trial, args, device, train_loader, val_loader, scaler, adj_mx):
    print(f"\n--- Starting Trial {trial.number} ---")
    model = create_model(trial, args, device, adj_mx)
    optimizer = Ranger(model.parameters(), lr=1e-3)
    
    max_train_batches = 5
    max_val_batches = 2
    
    # Quick Training (1 epoch, subset of data)
    model.train()
    train_loss = 0.0
    batches_processed = 0
    
    for batch_x, batch_y, batch_vmd in train_loader.get_iterator():
        tx = batch_x.to(device, non_blocking=True)
        ty = batch_y.to(device, non_blocking=True)
        tvmd = batch_vmd.to(device, non_blocking=True)
        
        x_in = tx
        
        optimizer.zero_grad()
        prediction, _ = model(tvmd, x_in)
        
        # Scale back to calculate loss
        pred_scaled = scaler.inverse_transform(prediction)
        real_scaled = ty
        loss = torch.nn.functional.l1_loss(pred_scaled, real_scaled)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        batches_processed += 1
        
        if batches_processed >= max_train_batches:
            break
            
    print(f"Trial {trial.number} - Train Loss: {train_loss / max(1, batches_processed):.4f}")
    
    # Quick Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_vmd in val_loader.get_iterator():
            tx = batch_x.to(device, non_blocking=True)
            ty = batch_y.to(device, non_blocking=True)
            tvmd = batch_vmd.to(device, non_blocking=True)
            
            x_in = tx
            prediction, _ = model(tvmd, x_in)
            
            pred_scaled = scaler.inverse_transform(prediction)
            real_scaled = ty
            loss = torch.nn.functional.l1_loss(pred_scaled, real_scaled)
            
            val_loss += loss.item()
            val_batches += 1
            
            if val_batches >= max_val_batches:
                break
                
    avg_val_loss = val_loss / max(1, val_batches)
    print(f"Trial {trial.number} - Val Loss: {avg_val_loss:.4f}")
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    return avg_val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='taxi_drop')
    parser.add_argument('--root_path', type=str, default='./Dataset')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=15)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args.data_path = os.path.join(args.root_path, args.data, 'processed')
    args.num_nodes = 266
    args.input_dim = 3
    args.input_len = 12
    args.output_len = 12
    args.llm_layer = 6
    args.U = 1
    args.vmd_k = 3
    
    print("Loading dataset...")
    data = load_dataset(args.data_path, args.batch_size, args)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    scaler = data['scaler']
    
    adj_path = os.path.join(args.data_path, 'adj_mx.pkl')
    adj_mx = load_pickle(adj_path)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args, device, train_loader, val_loader, scaler, adj_mx), n_trials=args.trials)
    
    print("\n" + "="*50)
    print("Bayesian Optimization Completed!")
    print("="*50)
    print("Best Trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)
