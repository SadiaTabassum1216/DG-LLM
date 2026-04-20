"""
Enhanced evaluation module with statistical rigor.
Provides comprehensive evaluation with per-horizon metrics.
"""

from typing import Dict

import numpy as np
import torch

from utils import MAE_torch, MAPE_torch, RMSE_torch


def evaluate_model_statistical(
    trainer,
    dataloader,
    device,
    scaler,
    output_len: int,
    num_seeds: int = 1,
    current_seed: int = None,
) -> Dict[str, float]:
    """
    Evaluate model and return aggregate plus per-horizon metrics.
    """
    del num_seeds  # preserved for backward-compatible call sites
    trainer.model.eval()

    horizon_mae = [[] for _ in range(output_len)]
    horizon_mape = [[] for _ in range(output_len)]
    horizon_rmse = [[] for _ in range(output_len)]

    all_maes = []
    all_mapes = []
    all_rmses = []

    with torch.no_grad():
        for x, y, vmd in dataloader.get_iterator():
            tx = x.to(device, non_blocking=True).transpose(1, 3)
            ty = y.to(device, non_blocking=True).transpose(1, 3)[:, 0, :, :]
            tvmd = vmd.to(device, non_blocking=True)

            x_in = tx.permute(0, 3, 2, 1)
            preds, _ = trainer.model(tvmd, x_in)

            preds_scaled = scaler.inverse_transform(preds)
            real_scaled = ty.permute(0, 2, 1).unsqueeze(-1)

            for t in range(output_len):
                p = preds_scaled[:, t, ...]
                r = real_scaled[:, t, ...]
                horizon_mae[t].append(MAE_torch(p, r, 0).item())
                horizon_mape[t].append(MAPE_torch(p, r, 0).item())
                horizon_rmse[t].append(RMSE_torch(p, r, 0).item())

            all_maes.append(MAE_torch(preds_scaled, real_scaled, 0).item())
            all_mapes.append(MAPE_torch(preds_scaled, real_scaled, 0).item())
            all_rmses.append(RMSE_torch(preds_scaled, real_scaled, 0).item())

    results = {
        "mae": float(np.mean(all_maes)),
        "rmse": float(np.mean(all_rmses)),
        "mape": float(np.mean(all_mapes)),
        "horizon_mae": [float(np.mean(h)) for h in horizon_mae],
        "horizon_mape": [float(np.mean(h)) for h in horizon_mape],
        "horizon_rmse": [float(np.mean(h)) for h in horizon_rmse],
    }

    label = f"Seed {current_seed}" if current_seed is not None else "Evaluation"
    print(f"\n  {label} Results:")
    print(f"    Overall MAE:  {results['mae']:.4f}")
    print(f"    Overall RMSE: {results['rmse']:.4f}")
    print(f"    Overall MAPE: {results['mape']:.4f}")

    return results
