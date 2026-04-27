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
    p_keep = trial.suggest_float("p_keep", 0.02, 0.15)
    hysteresis_ratio = trial.suggest_float("hysteresis_ratio", 0.3, 0.8)
    edge_dropout = trial.suggest_float("edge_dropout", 0.05, 0.3)
    
    # 2. Graph Attention Scoring
    gat_tau = trial.suggest_float("gat_tau", 0.1, 1.0)
    head_dropout = trial.suggest_float("head_dropout", 0.1, 0.5)
    leaky_slope = trial.suggest_float("leaky_slope", 0.1, 0.3)
    
    # 3. Adaptive Graph Blending & EMA
    mix_hi = trial.suggest_float("mix_hi", 0.8, 1.0)
    mix_lo = trial.suggest_float("mix_lo", 0.1, 0.5)
    ema_m = trial.suggest_float("ema_m", 0.9, 0.999)
    warmup_steps = trial.suggest_int("warmup_steps", 200, 1000)
    
    # 4. Node Importance & Priors
    degree_prior_base = trial.suggest_float("degree_prior_base", 0.05, 0.5)
    degree_prior_scale = trial.suggest_float("degree_prior_scale", 0.05, 0.5)
    
    # 5. Global Residual Scale
    residual_scale = trial.suggest_float("RESIDUAL_SCALE", 0.05, 0.3)

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
    model.GLOBAL_FLOW_RESIDUAL_SCALE = residual_scale
    
    # Inject spatial and temporal hyperparameters into each mode processor
    for mode_proc in model.mode_processors:
        mode_proc.pruning_keep_ratio = p_keep
        mode_proc.stability_hysteresis_ratio = hysteresis_ratio
        mode_proc.graph_edge_dropout = edge_dropout
        mode_proc.gat_temperature = gat_tau
        mode_proc.attention_head_dropout = head_dropout
        mode_proc.gat_leaky_slope = leaky_slope
        mode_proc.initial_static_weight = mix_hi
        mode_proc.final_static_weight = mix_lo
        mode_proc.graph_ema_momentum = ema_m
        mode_proc.graph_learning_warmup = warmup_steps
        mode_proc.NODE_DEGREE_BASE_PRIOR = degree_prior_base
        mode_proc.NODE_DEGREE_IMPORTANCE_SCALE = degree_prior_scale
    
    return model

def objective(trial, args, device, train_loader, val_loader, scaler, adj_mx):
    print(f"\n--- Starting Trial {trial.number} ---")
    model = create_model(trial, args, device, adj_mx)
    optimizer = Ranger(model.parameters(), lr=1e-3)
    
    # Significant increase in data coverage for reliable metrics
    max_train_batches = args.train_batches
    max_val_batches = args.val_batches
    
    # Training Loop
    model.train()
    train_loss = 0.0
    batches_processed = 0
    
    for batch_x, batch_y, batch_vmd in train_loader.get_iterator():
        tx = batch_x.to(device, non_blocking=True)
        ty = batch_y.to(device, non_blocking=True)
        tvmd = batch_vmd.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        prediction, _ = model(tvmd, tx)
        
        pred_scaled = scaler.inverse_transform(prediction)
        loss = torch.nn.functional.l1_loss(pred_scaled, ty)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        batches_processed += 1
        
        if batches_processed >= max_train_batches:
            break
            
    avg_train_loss = train_loss / max(1, batches_processed)
    print(f"Trial {trial.number} - Train MAE: {avg_train_loss:.4f}")
    
    # Validation Loop
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_vmd in val_loader.get_iterator():
            tx = batch_x.to(device, non_blocking=True)
            ty = batch_y.to(device, non_blocking=True)
            tvmd = batch_vmd.to(device, non_blocking=True)
            
            prediction, _ = model(tvmd, tx)
            pred_scaled = scaler.inverse_transform(prediction)
            loss = torch.nn.functional.l1_loss(pred_scaled, ty)
            
            val_loss += loss.item()
            val_batches += 1
            
            # Intermediate reporting for pruning
            if val_batches % 10 == 0:
                intermediate_loss = val_loss / val_batches
                trial.report(intermediate_loss, val_batches)
                if trial.should_prune():
                    print(f"Trial {trial.number} PRUNED at step {val_batches}")
                    raise optuna.exceptions.TrialPruned()

            if val_batches >= max_val_batches:
                break
                
    avg_val_loss = val_loss / max(1, val_batches)
    print(f"Trial {trial.number} - Val MAE: {avg_val_loss:.4f}")
    
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
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--train_batches', type=int, default=100, help='Batches per trial training')
    parser.add_argument('--val_batches', type=int, default=50, help='Batches per trial validation')
    parser.add_argument('--db', type=str, default='optuna_study.db', help='SQLite DB for persistence')
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args.data_path = os.path.join(args.root_path, args.data, 'processed')
    
    # Dataset-specific node counts
    if 'PEMSD04' in args.data: args.num_nodes = 307
    elif 'PEMSD08' in args.data: args.num_nodes = 170
    elif 'bike' in args.data: args.num_nodes = 250
    elif 'taxi' in args.data: args.num_nodes = 266
    else: args.num_nodes = 307
    
    args.input_dim = 3
    args.input_len = 12
    args.output_len = 12
    args.llm_layer = 6
    args.U = 1
    args.vmd_k = 3
    
    print(f"Loading {args.data} dataset...")
    data = load_dataset(args.data_path, args.batch_size, args)
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    scaler = data['scaler']
    
    adj_path = os.path.join(args.root_path, args.data, 'adj_mx.pkl')
    adj_mx = load_pickle(adj_path)
    if isinstance(adj_mx, list): adj_mx = adj_mx[2]
    
    storage = f"sqlite:///{args.db}"
    study = optuna.create_study(
        study_name=f"dgllm_tuning_{args.data}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, args, device, train_loader, val_loader, scaler, adj_mx), 
        n_trials=args.trials
    )
    
    print("\n" + "="*50)
    print("Optimization Completed!")
    print(f"Study saved to: {args.db}")
    print("="*50)
    print("Best Trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("="*50)
