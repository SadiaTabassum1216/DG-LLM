import torch
import torch.nn as nn

class TemporalEmbedding(nn.Module):
    """
    Standard temporal embedding for traffic forecasting.
    Expects TOD (Time-of-Day) and DOW (Day-of-Week) as input.
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        # TOD (288 steps/day)
        self.tod_embed = nn.Embedding(288, d_model)
        # DOW (7 days/week)
        self.dow_embed = nn.Embedding( 7, d_model)

    def forward(self, x):
        # x is assumed to be [B, T, N, 2] 
        # where x[..., 0] is TOD (normalized 0-1) and x[..., 1] is DOW (0-6)
        
        # Denormalize TOD: 0-1 -> 0-287
        tod = (x[..., 0] * 288).long()
        dow = x[..., 1].long()

        return self.tod_embed(tod) + self.dow_embed(dow)
