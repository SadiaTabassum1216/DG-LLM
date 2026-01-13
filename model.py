import torch
import torch.nn as nn
from temporal_embedding import TemporalEmbedding
from backbone import PFA, adjacency_to_pairwise_bias

class TemporalMSFeats(nn.Module):
    """
    Temporal Multi-scale Feature extraction using 1D convolutions.
    Kernels: 1, 3, 5.
    """
    def __init__(self, input_dim, gpt_channel, to_gpt_channel):
        super(TemporalMSFeats, self).__init__()
        self.tconv1 = nn.Conv1d(input_dim, gpt_channel, 1, padding=0, bias=False)
        self.tconv3 = nn.Conv1d(input_dim, gpt_channel, 3, padding=1, bias=False)
        self.tconv5 = nn.Conv1d(input_dim, gpt_channel, 5, padding=2, bias=False)
        self.tproj  = nn.Linear(3 * gpt_channel, to_gpt_channel, bias=False)

    def forward(self, x):
        # x: [B*N, T, F]
        B_N, T, F = x.shape
        x_in = x.transpose(1, 2) # [B*N, F, T]
        
        b1 = F.adaptive_avg_pool1d(F.relu(self.tconv1(x_in)), 1).squeeze(-1)
        b3 = F.adaptive_avg_pool1d(F.relu(self.tconv3(x_in)), 1).squeeze(-1)
        b5 = F.adaptive_avg_pool1d(F.relu(self.tconv5(x_in)), 1).squeeze(-1)
        
        feats = torch.cat([b1, b3, b5], dim=-1) # [B*N, 3*C]
        return self.tproj(feats).unsqueeze(1) # [B*N, 1, D]

class DG_Mode_Processor(nn.Module):
    """
    Dynamic components for a single VMD Mode.
    Combines: Dynamic Graph Learning + Temporal MS Feats + PFA Backbone.
    """
    def __init__(self, d_model, args):
        super(DG_Mode_Processor, self).__init__()
        self.d_model = d_model
        self.num_nodes = args.num_nodes
        self.input_dim = 3 # Flow, ToD, DoW
        
        # 1. Temporal MS Features
        self.tms = TemporalMSFeats(self.input_dim, d_model, d_model)
        self.tgate = nn.Linear(d_model, d_model)
        
        # 2. Dynamic GAT (Simplified)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        
        # 3. Backbone
        self.pfa = PFA(d_model, args.patch_size, args)
        
        # 4. Output Projection
        self.output_fc = nn.Linear(d_model, args.output_len)

    def forward(self, x_in, adjacency_matrix=None):
        # x_in: [B, T, N, 3] (Mode Flow, ToD, DoW)
        B, T, N, F = x_in.shape
        
        # Reshape for temporal backbone
        # [B*N, T, F]
        h_nodes = x_in.transpose(1, 2).reshape(B*N, T, F)
        
        # Temporal Guidance
        t_feats = self.tms(h_nodes) # [B*N, 1, D]
        gate = torch.sigmoid(self.tgate(t_feats))
        
        # Backbone processing per node
        out = self.pfa(h_nodes, adjacency_matrix=adjacency_matrix) # [B*N, T, D]
        
        # Apply gate (Trend/Periodic guidance)
        out = out + gate * t_feats
        
        # Reshape back [B, N, T, D]
        out = out.view(B, N, T, -1)
        
        # Project to Forecast [B, N, H]
        out_last = out[:, :, -1, :] # Last token per node
        forecast = self.output_fc(out_last) # [B, N, H]
        
        return forecast.unsqueeze(-1).transpose(1, 2) # [B, H, N, 1]

class DGLLM(nn.Module):
    """
    Multi-Mode Dynamic ST-LLM (DGLLM).
    Orchestrates multiple DG_Mode_Processor instances for VMD modes.
    """
    def __init__(self, args, adj_mx=None):
        super(DGLLM, self).__init__()
        self.d_model = args.d_model
        self.K = args.vmd_k
        self.adj_mx = adj_mx # Static adjacency if available
        
        # Independent Processors for each VMD Mode
        self.mode_processors = nn.ModuleList([
            DG_Mode_Processor(self.d_model, args) for _ in range(self.K)
        ])
        
        # Fusion Layers (Attention-based from notebook)
        self.fusion_query = nn.Linear(args.output_len, args.output_len)
        self.fusion_key = nn.Linear(args.output_len, args.output_len)
        
        # Residual projection for trend preservation
        self.residual_proj = nn.Sequential(
            nn.Linear(args.input_len, args.output_len),
            nn.LayerNorm(args.output_len),
            nn.ReLU()
        )

    def attention_fusion(self, mode_preds):
        # mode_preds: List of [B, H, N, 1]
        stacked = torch.stack(mode_preds, dim=0) # [K, B, H, N, 1]
        K, B, H, N, _ = stacked.shape
        
        # Reshape for score calculation
        x = stacked.squeeze(-1).permute(0, 1, 3, 2).reshape(K, B * N, H) # [K, B*N, H]
        
        q = self.fusion_query(x)
        k = self.fusion_key(x)
        
        # Simplified Attention Fusion across modes
        scores = torch.mean(torch.matmul(q, k.transpose(1, 2)), dim=-1) # [K, B*N]
        weights = torch.softmax(scores, dim=0).unsqueeze(-1).unsqueeze(1).view(K, B, 1, N, 1)
        
        fused = torch.sum(weights * stacked, dim=0)
        return fused

    def forward(self, x, vmd):
        # x: [B, T, N, 3] (Original Flow, TOD, DOW)
        # vmd: [B, K, T, N, 1]
        
        B, T, N, _ = x.shape
        time_feats = x[..., 1:] # [B, T, N, 2]
        
        # 1. Process each mode
        mode_outputs = []
        for k in range(self.K):
            # Concat mode flow with original time features
            mode_in = torch.cat([vmd[:, k], time_feats], dim=-1) # [B, T, N, 3]
            out_k = self.mode_processors[k](mode_in, self.adj_mx)
            mode_outputs.append(out_k)
            
        # 2. Attention Fusion
        final = self.attention_fusion(mode_outputs)
        
        # 3. Residual Trend Preservation (0.1 weight from notebook)
        res = x[..., 0].transpose(1, 2) # [B, N, T]
        res = self.residual_proj(res).transpose(1, 2).unsqueeze(-1) # [B, H, N, 1]
        final = final + 0.1 * res
            
        return final
