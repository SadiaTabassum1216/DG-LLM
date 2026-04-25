import torch
import torch.nn as nn

class TemporalEmbedding(nn.Module):
    """
    Provides learnable embeddings for periodic temporal patterns (Time-of-Day and Day-of-Week).
    
    Args:
        daily_intervals: Number of time slots in a single day (e.g., 288 for 5-min intervals).
        embedding_dim: Dimension of the resulting feature vector.
    """
    def __init__(self, daily_intervals, embedding_dim):
        super().__init__()
        self.daily_intervals = daily_intervals
        self.time_day_emb = nn.Parameter(torch.empty(daily_intervals, embedding_dim))
        self.time_week_emb = nn.Parameter(torch.empty(7, embedding_dim))
        
        nn.init.xavier_uniform_(self.time_day_emb)
        nn.init.xavier_uniform_(self.time_week_emb)

    def forward(self, input_tensor):
        """
        Extracts temporal context from the last timestep of the input sequence.
        
        Args:
            input_tensor: [Batch, Time, Nodes, Features]
            
        Returns:
            Combined temporal embeddings: [Batch, Dimension, Nodes, 1]
        """
        # Extract features for the most recent timestep
        # Feature 1: Time of Day (normalized 0-1)
        # Feature 2: Day of Week (integer 0-6)
        tod_raw = input_tensor[:, -1, :, 1]   
        dow_raw = input_tensor[:, -1, :, 2]  
        
        # Convert floating point features to discrete embedding indices
        tod_idx = (tod_raw * self.daily_intervals).long().clamp(0, self.daily_intervals - 1)
        dow_idx = dow_raw.long().clamp(0, 6)
        
        # Look up embeddings
        tod_embedding = self.time_day_emb[tod_idx]    # [Batch, Nodes, Dimension]
        dow_embedding = self.time_week_emb[dow_idx]   # [Batch, Nodes, Dimension]
        
        # Fuse the two temporal contexts
        fused_temporal_context = tod_embedding + dow_embedding
        
        # Reshape to match the spatial feature layout: [Batch, Dimension, Nodes, 1]
        return fused_temporal_context.transpose(1, 2).unsqueeze(-1)
