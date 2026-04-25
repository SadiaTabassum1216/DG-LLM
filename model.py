import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import SpatialGPTBackbone
from temporal_embedding import TemporalEmbedding

class ModeProcessor(nn.Module):
    """
    Spatio-temporal processor for a single VMD mode.
    """
    
    # Configuration constants for channel dimensions
    DEFAULT_TIME_STEPS = 288  # PEMS uses 5-minute intervals
    DEFAULT_GPT_CHANNEL = 256
    DEFAULT_BACKBONE_CHANNEL = 768  # 3 * gpt_channel
    
    # Final Optimized Hyperparameters (from Consolidated Bayesian Run)
    DEFAULT_HEAD_DROPOUT = 0.34
    DEFAULT_LEAKY_SLOPE = 0.20
    DEFAULT_GAT_TAU = 1.47
    DEFAULT_EMA_M = 0.81
    DEFAULT_EPSILON = 1e-6
    DEFAULT_EDGE_DROPOUT = 0.12
    DEFAULT_SYMMETRIZE = True
    DEFAULT_HYSTERESIS_RATIO = 0.66
    DEFAULT_WARMUP_STEPS = 732
    DEFAULT_P_KEEP = 0.13
    DEFAULT_MIX_HI = 0.77
    DEFAULT_MIX_LO = 0.57
    
    # Node Importance (Degree Prior) parameters
    DEGREE_PRIOR_BASE = 0.37  
    DEGREE_PRIOR_SCALE = 0.42 

    # --- Spatio-Temporal Hyperparameters ---
    # These control how nodes interact and how the graph structure evolves over time.
    # Optuna Bayesian optimization has been used to find the best default values.
    
    # Pruning: Only the top p_keep percentage of learned edges are kept to maintain sparsity.
    # Blending (mix_hi/mix_lo): Controls the transition from the physical road network (high mix) 
    # to the learned adaptive graph (low mix) during training.
    # Hysteresis: Prevents rapid oscillation of graph edges by penalizing sudden removals.
    # Warmup: The number of steps over which the blending ratio and degree prior are annealed.

    def __init__(
        self,
        device,
        static_road_network,
        input_dim,
        num_nodes,
        input_len,
        output_len,
        llm_layer,
        U,
        backbone_channel=DEFAULT_BACKBONE_CHANNEL,
        gpt_channel=DEFAULT_GPT_CHANNEL,
        use_dynamic_graph=True,
        heads=4,
        p_keep=DEFAULT_P_KEEP,
        mix_hi=DEFAULT_MIX_HI,
        mix_lo=DEFAULT_MIX_LO,
    ):
        """
        Initializes a mode-specific processor that combines spatial and temporal patterns.
        """
        super().__init__()
        self.adj_mx = torch.tensor(static_road_network, dtype=torch.float32).to(device)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.use_dynamic_graph = use_dynamic_graph

        self.heads = heads
        self.p_keep = p_keep
        self.mix_hi = mix_hi
        self.mix_lo = mix_lo
        
        self.input_len = input_len
        self.gpt_channel = gpt_channel
        self.backbone_channel = backbone_channel
        self.time_steps = self.DEFAULT_TIME_STEPS

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_A", torch.zeros((num_nodes, num_nodes)))
        self.register_buffer("prev_A", torch.zeros((num_nodes, num_nodes)))
        
        binary_adj = (self.adj_mx > 0).float()
        self.register_buffer(
            "norm_adj_mx",
            binary_adj / (binary_adj.sum(-1, keepdim=True) + self.DEFAULT_EPSILON)
        )
        self.register_buffer("binary_adj_mx", binary_adj)

        # Graph attention hyperparameters
        self.head_dropout = self.DEFAULT_HEAD_DROPOUT
        self.leaky_slope = self.DEFAULT_LEAKY_SLOPE
        self.gat_tau = self.DEFAULT_GAT_TAU
        self.ema_m = self.DEFAULT_EMA_M
        self.eps = self.DEFAULT_EPSILON
        self.edge_dropout = self.DEFAULT_EDGE_DROPOUT
        self.symmetrize = self.DEFAULT_SYMMETRIZE
        self.hysteresis_ratio = self.DEFAULT_HYSTERESIS_RATIO
        self.warmup_steps = self.DEFAULT_WARMUP_STEPS

        self.feature_encoder = nn.Conv2d(input_dim * input_len, gpt_channel, kernel_size=(1, 1))
        self.temporal_embedding = TemporalEmbedding(self.time_steps, gpt_channel)
        self.node_identity_emb = nn.Parameter(torch.empty(num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_identity_emb)

        self.input_projection = nn.Conv2d(gpt_channel * 3, backbone_channel, kernel_size=(1, 1))
        self.feature_norm = nn.LayerNorm(backbone_channel)
        self.temporal_feature_norm = nn.LayerNorm(backbone_channel)

        self.gat_q = nn.Linear(backbone_channel, backbone_channel, bias=False)
        self.gat_k = nn.Linear(backbone_channel, backbone_channel, bias=False)
        self.gat_a = nn.Parameter(
            torch.randn(heads, 2 * backbone_channel, 1) * (1.0 / math.sqrt(backbone_channel))
        )

        # Multi-scale temporal convolutions
        self.temporal_conv_1 = nn.Conv1d(input_dim, gpt_channel, 1, padding=0, bias=False)
        self.temporal_conv_3 = nn.Conv1d(input_dim, gpt_channel, 3, padding=1, bias=False)
        self.temporal_conv_5 = nn.Conv1d(input_dim, gpt_channel, 5, padding=2, bias=False)
        self.temporal_projection = nn.Linear(3 * gpt_channel, backbone_channel, bias=False)
        self.temporal_gate = nn.Linear(backbone_channel, backbone_channel)

        self.backbone = SpatialGPTBackbone(
            device,
            gpt_layers=llm_layer,
            U=U,
            dropout_rate=0.1,
        )
        self.regression_layer = nn.Conv2d(backbone_channel, output_len, kernel_size=(1, 1))

    def _standardize_layout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures the input tensor follows the standard [Batch, Time, Nodes, Features] layout.
        """
        # Convert legacy [Batch, Features, Nodes, Time] to standard layout
        if x.dim() == 4 and x.shape[1] == self.input_dim:
            return x.permute(0, 3, 2, 1).contiguous()
        return x

    def _extract_temporal_patterns(self, history_data: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-scale temporal features (short-term and long-term trends) for each node.

        Returns:
            A tensor of shape [Batch, Nodes, Features].
        """
        batch_size, time_steps, num_nodes, feature_dim = history_data.shape
        temporal_input = history_data.permute(0, 2, 3, 1).reshape(
            batch_size * num_nodes, feature_dim, time_steps
        )

        branch_1 = F.adaptive_avg_pool1d(F.relu(self.temporal_conv_1(temporal_input)), 1).squeeze(-1)
        branch_3 = F.adaptive_avg_pool1d(F.relu(self.temporal_conv_3(temporal_input)), 1).squeeze(-1)
        branch_5 = F.adaptive_avg_pool1d(F.relu(self.temporal_conv_5(temporal_input)), 1).squeeze(-1)

        temporal_features = torch.cat([branch_1, branch_3, branch_5], dim=-1)
        temporal_features = self.temporal_projection(temporal_features)
        return temporal_features.view(batch_size, num_nodes, -1)

    def _get_blending_ratio(self) -> float:
        """
        Calculates the current blending ratio between the fixed road network and the learned graph.
        
        The ratio shifts from 'mix_hi' (mostly fixed) to 'mix_lo' (mostly learned) 
        during the initial training warmup.
        """
        step = float(self.global_step.item())
        warmup = max(self.warmup_steps, 1)
        progress = min(step / warmup, 1.0)
        return self.mix_hi + (self.mix_lo - self.mix_hi) * progress

    def _calculate_node_importance(self, graph_scores: torch.Tensor) -> torch.Tensor:
        importance = graph_scores.sum(dim=-1, keepdim=True)
        max_importance = importance.max() + 1e-6
        return importance / max_importance

    def _generate_adaptive_graph(self, features: torch.Tensor) -> torch.Tensor:
        _, num_nodes, _ = features.shape
        mix_alpha = self._get_blending_ratio()

        normalized_features = F.normalize(features, dim=-1)
        query = self.gat_q(normalized_features)
        key = self.gat_k(normalized_features)

        query_i = query.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        key_j = key.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        pair_features = torch.cat([query_i, key_j], dim=-1)

        logits_all = torch.einsum("bijd,hd->bhij", pair_features, self.gat_a.squeeze(-1))
        logits_all = F.leaky_relu(logits_all, self.leaky_slope)

        if self.training and self.head_dropout > 0:
            num_heads = logits_all.size(1)
            head_mask = (torch.rand(num_heads, device=logits_all.device) >= self.head_dropout).float()
            if head_mask.sum() == 0:
                head_mask[0] = 1.0
            head_mask = head_mask.view(1, num_heads, 1, 1)
            logits_all = logits_all * head_mask
            logits = logits_all.sum(dim=1) / head_mask.sum()
        else:
            logits = logits_all.mean(dim=1)

        logits = logits / max(self.gat_tau, 1e-6)
        graph_prob = torch.softmax(logits, dim=-1)

        mean_graph = graph_prob.mean(dim=0).detach()
        self.ema_A = self.ema_m * self.ema_A + (1.0 - self.ema_m) * mean_graph

        # Mix current learned weights with the pre-computed static normalized adjacency
        graph_scores = (1.0 - mix_alpha) * self.ema_A + mix_alpha * self.norm_adj_mx

        if self.training and self.edge_dropout > 0:
            keep_mask = (torch.rand_like(graph_scores) > self.edge_dropout).float()
            graph_scores = graph_scores * keep_mask

        importance_weight = self._calculate_node_importance(graph_scores)
        # Apply connectivity-based weighting
        graph_scores = graph_scores * (
            self.DEGREE_PRIOR_BASE + self.DEGREE_PRIOR_SCALE * importance_weight
        )

        working_scores = graph_scores.clone()
        working_scores.fill_diagonal_(0.0)

        # Retain only the strongest connections (top-K pruning)
        keep_ratio = min(max(float(self.p_keep), 1e-3), 0.99)
        threshold = torch.quantile(working_scores, 1.0 - keep_ratio, dim=-1, keepdim=True)
        graph_binary = (working_scores >= threshold).float()

        # Nodes are always connected to themselves
        graph_binary.fill_diagonal_(1.0)
        if self.symmetrize:
            graph_binary = torch.maximum(graph_binary, graph_binary.t())

        row_mean = working_scores.mean(dim=-1, keepdim=True)
        low_confidence_mask = (working_scores < (row_mean * self.hysteresis_ratio)).float()
        keep_previous_edges = self.prev_A * (1.0 - low_confidence_mask)
        graph_binary = torch.clamp(graph_binary + keep_previous_edges, 0.0, 1.0)
        self.prev_A = graph_binary.detach()

        self.global_step += 1
        if self.global_step.item() < self.warmup_steps:
            graph_binary = torch.maximum(graph_binary, self.binary_adj_mx)

        return graph_binary

    def forward(self, input_tensor: torch.Tensor):
        """Processes one VMD mode through the spatio-temporal layers."""
        input_tensor = self._standardize_layout(input_tensor)
        batch_size, _, num_nodes, _ = input_tensor.shape

        temporal_embedding = self.temporal_embedding(input_tensor)
        node_embedding = (
            self.node_identity_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        )

        value_features = input_tensor.permute(0, 3, 2, 1).reshape(
            batch_size, self.input_dim * self.input_len, num_nodes, 1
        )
        value_features = self.feature_encoder(value_features)

        fused_features = torch.cat(
            [value_features, temporal_embedding, node_embedding], dim=1
        )
        spatiotemporal_features = self.input_projection(fused_features)
        spatiotemporal_features = F.leaky_relu(spatiotemporal_features).permute(0, 2, 1, 3).squeeze(-1)
        spatiotemporal_features = self.feature_norm(spatiotemporal_features)

        temporal_features = self._extract_temporal_patterns(input_tensor)
        temporal_gate = torch.sigmoid(self.temporal_gate(self.temporal_feature_norm(temporal_features)))
        fused_features = spatiotemporal_features + temporal_gate * temporal_features

        if self.use_dynamic_graph:
            adjacency = self._generate_adaptive_graph(fused_features)
        else:
            adjacency = self.adj_mx

        backbone_output = self.backbone(fused_features, adjacency)
        prediction = self.regression_layer(backbone_output.permute(0, 2, 1).unsqueeze(-1))
        return prediction, adjacency


class DGLLM(nn.Module):
    # Final Optimized Residual Connection Scaling
    RESIDUAL_SCALE = 0.13
    
    def __init__(
        self,
        device,
        static_road_network,
        input_dim,
        num_nodes,
        input_len,
        output_len,
        llm_layer,
        U,
        vmd_K,
        use_attention_fusion=True,
    ):
        """
        Initializes the multi-mode DG-LLM forecasting model.
        
        Args:
            device: Target torch device.
            static_road_network: Static graph prior (physical road network).
            input_dim: Number of input features per node.
            num_nodes: Number of nodes in the graph.
            input_len: Historical window size.
            output_len: Prediction horizon.
            llm_layer: Total transformer layers per mode.
            U: Number of top layers to remain trainable in the backbone.
            vmd_K: Number of VMD frequency modes to process.
            use_attention_fusion: If True, uses attention to blend modes.
        """
        super().__init__()
        self.use_attention_fusion = use_attention_fusion

        self.mode_processors = nn.ModuleList(
            [
                ModeProcessor(
                    device,
                    static_road_network,
                    input_dim,
                    num_nodes,
                    input_len,
                    output_len,
                    llm_layer,
                    U,
                    use_dynamic_graph=True,
                )
                for _ in range(vmd_K)
            ]
        )

        if use_attention_fusion:
            # Multi-mode attention fusion parameters
            self.fusion_query = nn.Linear(output_len, output_len)
            self.fusion_key = nn.Linear(output_len, output_len)
        else:
            # Simple weighted average parameters
            self.mode_blending_weights = nn.Parameter(torch.ones(vmd_K) / vmd_K)

        self.flow_residual_projection = nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.LayerNorm(output_len),
            nn.ReLU(),
        )

    def forward(self, vmd_data, original_input):
        """
        Runs the end-to-end prediction by processing and fusing all traffic modes.

        Args:
            vmd_data: Frequency-decomposed traffic modes [Batch, Modes, Time, Nodes, 1].
            original_input: Original traffic data [Batch, Time, Nodes, Features].

        Returns:
            A tuple of (final_forecast, learned_graph_structures).
        """
        _, vmd_K, _, _, _ = vmd_data.shape
        time_features = original_input[..., 1:]

        mode_predictions = []
        learned_graphs = []
        for mode_index in range(vmd_K):
            mode_flow = vmd_data[:, mode_index, ...]
            mode_input = torch.cat([mode_flow, time_features], dim=-1)

            prediction, learned_graph = self.mode_processors[mode_index](mode_input)
            mode_predictions.append(prediction)
            learned_graphs.append(learned_graph)

        if self.use_attention_fusion:
            final_prediction = self._blend_modes(mode_predictions)
        else:
            weights = F.softmax(self.mode_blending_weights, dim=0)
            final_prediction = sum(
                mode_predictions[mode_index] * weights[mode_index]
                for mode_index in range(vmd_K)
            )

        # Baseline Flow Residual Connection
        flow_residual = original_input[..., 0].permute(0, 2, 1)
        flow_residual = self.flow_residual_projection(flow_residual).permute(0, 2, 1).unsqueeze(-1)
        final_prediction = final_prediction + self.RESIDUAL_SCALE * flow_residual

        return final_prediction, learned_graphs

    def _blend_modes(self, mode_predictions):
        """Blends predictions from different VMD modes using an attention mechanism."""
        stacked_predictions = torch.stack(mode_predictions, dim=0)  # [K, B, T, N, 1]
        num_modes, batch_size, output_len, num_nodes, _ = stacked_predictions.shape

        mode_features = stacked_predictions.squeeze(-1).permute(0, 1, 3, 2).reshape(
            num_modes, batch_size * num_nodes, output_len
        )
        query = self.fusion_query(mode_features)
        key = self.fusion_key(mode_features)

        scores = torch.matmul(query, key.transpose(1, 2)).mean(dim=-1)
        weights = F.softmax(scores, dim=0).unsqueeze(-1).unsqueeze(1).view(
            num_modes, batch_size, 1, num_nodes, 1
        )
        return torch.sum(weights * stacked_predictions, dim=0)

    def param_num(self):
        """Return the total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())
