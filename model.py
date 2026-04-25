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
    
    # Hyperparameters for graph attention and adjacency learning
    DEFAULT_HEAD_DROPOUT = 0.31  # Optuna Bayesian optimized
    DEFAULT_LEAKY_SLOPE = 0.30  # Optuna Bayesian optimized 
    DEFAULT_GAT_TAU = 1.03  # Optuna Bayesian optimized
    DEFAULT_EMA_M = 0.90  # Optuna Bayesian optimized 
    DEFAULT_EPSILON = 1e-6
    DEFAULT_EDGE_DROPOUT = 0.11  # Optuna Bayesian optimized
    DEFAULT_SYMMETRIZE = True
    DEFAULT_HYSTERESIS_RATIO = 0.85  # Optuna Bayesian optimized
    DEFAULT_WARMUP_STEPS = 850  # Optuna Bayesian optimized (was 500)
    
    # Degree prior scaling coefficients
    DEGREE_PRIOR_BASE = 0.29  # Optuna Bayesian optimized (was 0.8)
    DEGREE_PRIOR_SCALE = 0.94  # Optuna Bayesian optimized (was 0.2)

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
        use_dynamic_graph=True,
        heads=4,
        gpt_channel=DEFAULT_GPT_CHANNEL,
        backbone_channel=DEFAULT_BACKBONE_CHANNEL,
        time_steps=DEFAULT_TIME_STEPS,
        p_keep=0.06,  # Optuna Bayesian optimized 
        mix_hi=0.89,  # Optuna Bayesian optimized
        mix_lo=0.50,  # Optuna Bayesian optimized 
    ):
        """Initialize a single-mode spatio-temporal processing block.

        Args:
            device: Target torch device.
            adj_mx: Static adjacency matrix used as graph prior.
            input_dim: Number of input features per node and time step.
            num_nodes: Number of graph nodes.
            input_len: Number of historical time steps.
            output_len: Forecast horizon.
            llm_layer: Number of backbone transformer-like layers.
            U: Backbone expansion factor.
            use_dynamic_graph: Whether to build a learned adjacency per step.
            heads: Number of heads for graph attention scoring.
            gpt_channel: Intermediate channel dimension.
            backbone_channel: Backbone feature channel dimension.
            time_steps: Number of time steps for temporal embedding.
            p_keep: Fraction of edges to retain after thresholding.
            mix_hi: Initial blend weight for fixed adjacency during warmup.
            mix_lo: Final blend weight for fixed adjacency after warmup.
        """
        super().__init__()

        self.adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(device)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.use_dynamic_graph = use_dynamic_graph

        self.heads = heads
        self.p_keep = p_keep
        self.mix_hi = mix_hi
        self.mix_lo = mix_lo
        
        # Store temporal and channel dimensions as instance attributes
        self.input_len = input_len
        self.gpt_channel = gpt_channel
        self.backbone_channel = backbone_channel
        self.time_steps = time_steps

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_A", torch.zeros((num_nodes, num_nodes)))
        self.register_buffer("prev_A", torch.zeros((num_nodes, num_nodes)))
        
        # Pre-compute normalized adjacency matrix to avoid redundant computation
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

        self.start_conv = nn.Conv2d(input_dim * input_len, gpt_channel, kernel_size=(1, 1))
        self.temporal_embedding = TemporalEmbedding(time_steps, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        # Verify channel scaling consistency
        assert backbone_channel == gpt_channel * 3, (
            f"backbone_channel ({backbone_channel}) must equal 3 * gpt_channel ({gpt_channel * 3})"
        )
        
        self.input_projection = nn.Conv2d(gpt_channel * 3, backbone_channel, kernel_size=(1, 1))
        self.feature_norm = nn.LayerNorm(backbone_channel)
        # Separate LayerNorm for temporal features to avoid interference
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
        """Weights nodes based on their relative connectivity (degree) in the graph."""
        importance = graph_scores.sum(dim=-1, keepdim=True)
        max_importance = importance.max() + 1e-6
        return importance / max_importance

    def _generate_adaptive_graph(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generates a custom, dynamic graph structure based on current traffic features.

        Args:
            features: Current node states in [Batch, Nodes, Dimension].

        Returns:
            A binary connectivity matrix [Nodes, Nodes].
        """
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

        # Confidence-based hysteresis to maintain graph stability over time
        row_mean = working_scores.mean(dim=-1, keepdim=True)
        low_confidence_mask = (working_scores < (row_mean * self.hysteresis_ratio)).float()
        keep_previous_edges = self.prev_A * (1.0 - low_confidence_mask)
        graph_binary = torch.clamp(graph_binary + keep_previous_edges, 0.0, 1.0)
        self.prev_A = graph_binary.detach()

        self.global_step += 1
        # During warmup, strictly enforce the static physical road network
        if self.global_step.item() < self.warmup_steps:
            graph_binary = torch.maximum(graph_binary, self.binary_adj_mx)

        return graph_binary

    def forward(self, input_tensor: torch.Tensor):
        """Processes one VMD mode through the spatio-temporal layers."""
        input_tensor = self._standardize_layout(input_tensor)
        batch_size, _, num_nodes, _ = input_tensor.shape

        temporal_embedding = self.temporal_embedding(input_tensor)
        node_embedding = (
            self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        )

        # Reshape input to [B, T*F, N, 1] for conv2d
        # Permute: (B, T, N, F) -> (B, F, N, T), then reshape to (B, F*T, N, 1)
        flattened_input = input_tensor.permute(0, 3, 2, 1).reshape(batch_size, self.input_dim * self.input_len, num_nodes, 1)
        flattened_input = self.start_conv(flattened_input)

        spatiotemporal_features = torch.cat([flattened_input, temporal_embedding, node_embedding], dim=1)
        spatiotemporal_features = self.input_projection(spatiotemporal_features)
        spatiotemporal_features = F.leaky_relu(spatiotemporal_features).permute(0, 2, 1, 3).squeeze(-1)
        spatiotemporal_features = self.feature_norm(spatiotemporal_features)

        temporal_features = self._extract_temporal_patterns(input_tensor)
        # Use separate LayerNorm for temporal features to avoid interference
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
    # Residual connection scaling coefficient
    RESIDUAL_SCALE = 0.06  # Optuna Bayesian optimized
    
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
        vmd_K=3,
        use_attention_fusion=True,
    ):
        """Initialize the multi-mode DG-LLM forecasting model.

        Args:
            device: Target torch device.
            adj_mx: Static adjacency matrix prior.
            input_dim: Number of input features per node and time step.
            num_nodes: Number of graph nodes.
            input_len: Number of historical time steps.
            output_len: Forecast horizon.
            llm_layer: Number of backbone layers in each mode processor.
            U: Backbone expansion factor.
            vmd_K: Number of VMD modes.
            use_attention_fusion: Whether to fuse modes with attention.
        """
        super().__init__()
        self.use_attention_fusion = use_attention_fusion

        self.mode_models = nn.ModuleList(
            [
                ModeProcessor(
                    device,
                    adj_mx,
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
            self.fusion_query = nn.Linear(output_len, output_len)
            self.fusion_key = nn.Linear(output_len, output_len)
        else:
            self.mode_weights = nn.Parameter(torch.ones(vmd_K) / vmd_K)

        self.residual_proj = nn.Sequential(
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
        _, num_modes, _, _, _ = vmd_data.shape
        time_features = original_input[..., 1:]

        mode_predictions = []
        learned_graphs = []
        for mode_index in range(num_modes):
            mode_flow = vmd_data[:, mode_index, ...]
            mode_input = torch.cat([mode_flow, time_features], dim=-1)

            prediction, adjacency = self.mode_models[mode_index](mode_input)
            mode_predictions.append(prediction)
            learned_graphs.append(adjacency)

        if self.use_attention_fusion:
            final_prediction = self._blend_modes(mode_predictions)
        else:
            mode_weights = F.softmax(self.mode_weights, dim=0)
            final_prediction = sum(
                mode_predictions[mode_index] * mode_weights[mode_index]
                for mode_index in range(num_modes)
            )

        residual = original_input[..., 0].permute(0, 2, 1)
        residual = self.residual_proj(residual).permute(0, 2, 1).unsqueeze(-1)
        final_prediction = final_prediction + self.RESIDUAL_SCALE * residual

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
