import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalEmbedding(nn.Module):
    """
    Learnable temporal embeddings for time-of-day and day-of-week.
    
    Args:
        time: Number of time slots per day (288 for 5-min, 48 for 30-min)
        features: Embedding dimension
    """
    def __init__(self, time, features):
        super().__init__()
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_day)
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        B, T, N, _ = x.shape
        
        # Use LAST timestep's temporal features (correct for prediction)
        day_emb = x[:, -1, :, 1]   # [B, N]
        week_emb = x[:, -1, :, 2]  # [B, N]
        
        # Index with clamping
        day_idx = (day_emb * self.time).long().clamp(0, self.time - 1)
        week_idx = week_emb.long().clamp(0, 6)
        
        time_day_emb = self.time_day[day_idx]    # [B, N, D]
        time_week_emb = self.time_week[week_idx] # [B, N, D]
        
        combined = time_day_emb + time_week_emb
        return combined.transpose(1, 2).unsqueeze(-1)  # [B, D, N, 1]
