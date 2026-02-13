import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from temporal_embedding import TemporalEmbedding
from backbone import PFA


class ModeProcessor(nn.Module):
    """
    Spatio-Temporal LLM for a single VMD mode.
    
    Combines:
    - Temporal multi-scale convolutions
    - Dynamic graph learning via GAT
    - GPT-2 backbone with graph attention
    """
    def __init__(
        self, 
        device, 
        adj_mx, 
        input_dim=3, 
        num_nodes=266,      
        input_len=12, 
        output_len=12, 
        llm_layer=6, 
        U=1,   
        # Dynamic Graph Hyperparams
        use_dynamic_graph=True,
        heads=4, 
        tau_hi=0.8, 
        tau_lo=0.5, 
        p_keep=0.15, 
        min_neighbors=20,
        mix_hi=0.6, 
        mix_lo=0.2, 
        k_min=16, 
        k_max=28
    ):
        super().__init__()
        self.device = device
        self.adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(device)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.use_dynamic_graph = use_dynamic_graph
        
        # Dynamic Graph Params
        self.heads = heads
        self.tau_hi, self.tau_lo = tau_hi, tau_lo
        self.p_keep = p_keep
        self.min_neighbors = int(min_neighbors)

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_A", torch.zeros((num_nodes, num_nodes)))
        self.register_buffer("prev_A", torch.zeros((num_nodes, num_nodes)))
        
        self.head_dropout = 0.1
        self.leaky_slope = 0.2
        self.gat_tau = 1.0
        self.ema_m = 0.99
        self.eps = 1e-6
        self.edge_dropout = 0.1
        self.symmetrize = True
        self.hysteresis_ratio = 0.8
        self.warmup_steps = 500

        self.k_min, self.k_max = k_min, k_max
        self.mix_hi, self.mix_lo = mix_hi, mix_lo

        # Dimensions - 768 for GPT-2 compatibility
        time_steps = 288  # PEMS (5-min intervals)
        gpt_channel, to_gpt_channel = 256, 768
        
        # Front-end
        self.start_conv = nn.Conv2d(input_dim * input_len, gpt_channel, kernel_size=(1, 1))
        self.Temb = TemporalEmbedding(time_steps, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)
        
        self.in_layer = nn.Conv2d(gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1))
        self.feat_norm = nn.LayerNorm(to_gpt_channel)
        
        # Graph Learning Parts
        self.gat_q = nn.Linear(to_gpt_channel, to_gpt_channel, bias=False)
        self.gat_k = nn.Linear(to_gpt_channel, to_gpt_channel, bias=False)
        self.gat_a = nn.Parameter(torch.randn(heads, 2 * to_gpt_channel, 1) * (1.0 / math.sqrt(to_gpt_channel)))
        self.bilin_W = nn.Parameter(torch.randn(to_gpt_channel, to_gpt_channel) * (1.0 / math.sqrt(to_gpt_channel)))
        
        # Temporal Multi-scale
        self.tconv1 = nn.Conv1d(input_dim, gpt_channel, 1, padding=0, bias=False)
        self.tconv3 = nn.Conv1d(input_dim, gpt_channel, 3, padding=1, bias=False)
        self.tconv5 = nn.Conv1d(input_dim, gpt_channel, 5, padding=2, bias=False)
        self.tproj = nn.Linear(3 * gpt_channel, to_gpt_channel, bias=False)
        self.tgate = nn.Linear(to_gpt_channel, to_gpt_channel)
        
        # Backbone
        self.gpt = PFA(device, gpt_layers=llm_layer, U=U, dropout_rate=0.1)
        self.regression_layer = nn.Conv2d(to_gpt_channel, output_len, kernel_size=(1, 1))

    def _schedule(self):
        t = float(self.global_step.item())
        T = max(self.warmup_steps, 1)
        k = int(round(self.k_min + (self.k_max - self.k_min) * min(t / T, 1.0)))
        mix = self.mix_hi + (self.mix_lo - self.mix_hi) * min(t / T, 1.0)
        return k, mix

    def _degree_prior(self, A):
        degree = A.sum(dim=-1, keepdim=True)
        max_deg = degree.max() + 1e-6
        return degree / max_deg

    def _to_BTSF(self, x):
        if x.dim() == 4 and x.shape[1] == self.input_dim: 
            return x.permute(0, 3, 2, 1).contiguous()
        return x

    def _temporal_ms_feats(self, history_data):
        B, T, S, Fdim = history_data.shape
        x = history_data.permute(0, 2, 3, 1).reshape(B * S, Fdim, T)
        b1 = F.adaptive_avg_pool1d(F.relu(self.tconv1(x)), 1).squeeze(-1)
        b3 = F.adaptive_avg_pool1d(F.relu(self.tconv3(x)), 1).squeeze(-1)
        b5 = F.adaptive_avg_pool1d(F.relu(self.tconv5(x)), 1).squeeze(-1)
        feats = torch.cat([b1, b3, b5], dim=-1)
        feats = self.tproj(feats).view(B, S, -1)
        return feats

    def _build_adjacency(self, feats) -> torch.Tensor:
        """
        feats: [B, S, D] -> returns binary [S, S] adjacency with
        temperature, EMA smoothing, edge/head dropout, quantile selection,
        degree prior, hysteresis, and warmup union.
        """
        B, S, D = feats.shape
        _, mix_alpha_curr = self._schedule()

        # 1) GAT logits (multi-head) — VECTORIZED over all heads
        h = F.normalize(feats, dim=-1)
        q = self.gat_q(h)
        k = self.gat_k(h)
        qi = q.unsqueeze(2).expand(-1, -1, S, -1)
        kj = k.unsqueeze(1).expand(-1, S, -1, -1)
        pair = torch.cat([qi, kj], dim=-1)  # [B, S, S, 2D]

        # Batched attention: einsum over all H heads at once
        # gat_a: [H, 2D, 1] → squeeze to [H, 2D]
        # pair: [B, S, S, 2D]
        # result: [B, H, S, S]
        logits_all = torch.einsum('bijd,hd->bhij', pair, self.gat_a.squeeze(-1))
        logits_all = F.leaky_relu(logits_all, self.leaky_slope)

        # Head dropout: mask out entire heads during training
        if self.training and self.head_dropout > 0:
            H = logits_all.size(1)
            head_mask = (torch.rand(H, device=logits_all.device) >= self.head_dropout).float()
            # Ensure at least one head survives
            if head_mask.sum() == 0:
                head_mask[0] = 1.0
            head_mask = head_mask.view(1, H, 1, 1)
            logits_all = logits_all * head_mask
            logits = logits_all.sum(dim=1) / head_mask.sum()
        else:
            logits = logits_all.mean(dim=1)

        logits = logits / max(self.gat_tau, 1e-6)
        A_prob = torch.softmax(logits, dim=-1)

        # 2) EMA smoothing
        A_mean = A_prob.mean(dim=0).detach()
        self.ema_A = self.ema_m * self.ema_A + (1.0 - self.ema_m) * A_mean

        fixed = (self.adj_mx > 0).float()
        fixed = fixed / (fixed.sum(-1, keepdim=True) + self.eps)
        A_blend = (1.0 - mix_alpha_curr) * self.ema_A + mix_alpha_curr * fixed

        # 4) edge dropout
        if self.training and self.edge_dropout > 0:
            keep = (torch.rand_like(A_blend) > self.edge_dropout).float()
            A_blend = A_blend * keep

        # 5) degree prior
        prior = self._degree_prior(A_blend)
        A_blend = A_blend * (0.8 + 0.2 * prior)

        # 6) ADAPTIVE DENSITY (quantile)
        A_work = A_blend.clone()
        A_work.fill_diagonal_(0.0)
        p = float(self.p_keep)
        p = min(max(p, 1e-3), 0.99)
        thresh = torch.quantile(A_work, 1.0 - p, dim=-1, keepdim=True)
        A_bin = (A_work >= thresh).float()

        # 7) self-loops; symmetrize
        A_bin.fill_diagonal_(1.0)
        if self.symmetrize:
            A_bin = torch.maximum(A_bin, A_bin.t())

        # 8) HYSTERESIS
        row_mean = A_work.mean(dim=-1, keepdim=True)
        low_mask = (A_work < (row_mean * self.hysteresis_ratio)).float()
        keep_prev = self.prev_A * (1.0 - low_mask)
        A_bin = torch.clamp(A_bin + keep_prev, 0.0, 1.0)
        self.prev_A = A_bin.detach()

        # 9) warmup: union with fixed graph
        self.global_step += 1
        if self.global_step.item() < self.warmup_steps:
            A_bin = torch.maximum(A_bin, (self.adj_mx > 0).float())

        return A_bin

    def forward(self, x_in):
        # x_in: [B, T, N, F] (Mode data + Time features)
        x_in = self._to_BTSF(x_in)
        B, T, S, Fdim = x_in.shape
        data = x_in.permute(0, 3, 2, 1)  # [B, F, S, T]
        
        # Embeddings
        tem_emb = self.Temb(x_in)
        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
        
        input_data = data.transpose(1, 2).contiguous().view(B, S, -1).transpose(1, 2).unsqueeze(-1)
        input_data = self.start_conv(input_data)
        
        data_st = torch.cat([input_data, tem_emb, node_emb], dim=1)
        data_st = self.in_layer(data_st)
        data_st = F.leaky_relu(data_st).permute(0, 2, 1, 3).squeeze(-1)
        data_st = self.feat_norm(data_st)
        
        # Temporal Guidance
        t_feats = self._temporal_ms_feats(x_in)
        gate = torch.sigmoid(self.tgate(self.feat_norm(t_feats)))
        data_st_fused = data_st + gate * t_feats
        
        # Graph
        if self.use_dynamic_graph:
            adj = self._build_adjacency(data_st_fused)
        else:
            adj = self.adj_mx
            
        # GPT
        out = self.gpt(data_st_fused, adj)
        
        # Project
        out = out.permute(0, 2, 1).unsqueeze(-1)
        pred = self.regression_layer(out)
        
        return pred, adj


class DGLLM(nn.Module):
    def __init__(self, device, adj_mx, input_dim=3, num_nodes=266, 
                 input_len=12, output_len=12, llm_layer=6, U=1, 
                 vmd_K=3, use_attention_fusion=True):
        super().__init__()
        self.vmd_K = vmd_K
        self.use_attention_fusion = use_attention_fusion
        self.output_len = output_len
        
        # Create a Dynamic Graph ST-LLM for each mode
        self.mode_models = nn.ModuleList([
            ModeProcessor(
                device, adj_mx, input_dim, num_nodes, input_len, output_len, 
                llm_layer, U, use_dynamic_graph=True
            ) for _ in range(vmd_K)
        ])
        
        # Fusion Layers
        if use_attention_fusion:
            self.fusion_query = nn.Linear(output_len, output_len)
            self.fusion_key = nn.Linear(output_len, output_len)
            self.fusion_value = nn.Linear(output_len, output_len)
        else:
            self.mode_weights = nn.Parameter(torch.ones(vmd_K) / vmd_K)
            
        # Residual connection from original input (trend preservation)
        self.residual_proj = nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.LayerNorm(output_len),
            nn.ReLU()
        )

    def attention_fusion(self, mode_preds):
        # mode_preds: List of [B, Out_T, N, 1]
        stacked = torch.stack(mode_preds, dim=0)  # [K, B, T, N, 1]
        K, B, T, N, _ = stacked.shape
        x = stacked.squeeze(-1).permute(0, 1, 3, 2).reshape(K, B * N, T)
        
        q = self.fusion_query(x)
        k = self.fusion_key(x)
        v = self.fusion_value(x)
        
        attn = F.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(T), dim=0)
        scores = torch.mean(torch.matmul(q, k.transpose(1, 2)), dim=-1)
        weights = F.softmax(scores, dim=0).unsqueeze(-1).unsqueeze(1).view(K, B, 1, N, 1)
        
        fused = torch.sum(weights * stacked, dim=0)
        return fused

    def forward(self, vmd_data, original_input):
        """
        vmd_data: [B, K, T, N, 1] (Just the flow)
        original_input: [B, T, N, F] (Flow, Day, Week)
        """
        B, K, T, N, _ = vmd_data.shape
        time_feats = original_input[..., 1:]
        
        preds = []
        graphs = []
        
        for k in range(K):
            mode_flow = vmd_data[:, k, ...]
            mode_in = torch.cat([mode_flow, time_feats], dim=-1)
            
            p, g = self.mode_models[k](mode_in)
            preds.append(p)
            graphs.append(g)
            
        if self.use_attention_fusion:
            final = self.attention_fusion(preds)
        else:
            w = F.softmax(self.mode_weights, dim=0)
            final = sum(preds[i] * w[i] for i in range(K))
            
        # Residual
        res = original_input[..., 0].permute(0, 2, 1)
        res = self.residual_proj(res).permute(0, 2, 1).unsqueeze(-1)
        final = final + 0.1 * res
        
        return final, graphs

    def param_num(self):
        return sum(p.numel() for p in self.parameters())