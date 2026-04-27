import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import SpatialGPTBackbone
from temporal_embedding import TemporalEmbedding

class ModeProcessor(nn.Module):
    """Spatio-temporal processor for a single VMD mode."""
    
    # Configuration constants for channel dimensions
    DEFAULT_TIME_STEPS = 288        # PEMS uses 5-minute intervals
    DEFAULT_GPT_CHANNEL = 256
    DEFAULT_BACKBONE_CHANNEL = 768  # 3 * gpt_channel
    
    # 1. Graph Attention Scoring
    ATTENTION_HEAD_DROPOUT = 0.3        # Probability of dropping an entire attention head
    GAT_LEAKY_SLOPE = 0.25              # Slope for LeakyReLU in GAT scoring
    DEFAULT_GAT_TEMPERATURE = 0.5       # [LEARNABLE] Softmax temperature for graph attention scores
    
    # 2. Graph Evolution & Stability
    GRAPH_EMA_MOMENTUM = 0.95           # Momentum for exponential moving average of learned graph
    GRAPH_EDGE_DROPOUT = 0.1            # Probability of dropping individual edges during training
    
    # 3. Graph Pruning & Blending
    GRAPH_LEARNING_WARMUP = 500         # Steps to transition from static road network to learned graph
    GRAPH_PRUNING_KEEP_RATIO = 0.05     # Top % of learned edges to retain
    
    INITIAL_STATIC_GRAPH_WEIGHT = 0.8   # Initial weighting of the physical road network
    FINAL_STATIC_GRAPH_WEIGHT = 0.2     # [LEARNABLE] Final weighting of the physical road network after warmup
    
    # 4. Node Importance (Degree Prior) Default Values
    DEFAULT_NODE_DEGREE_BASE_PRIOR = 0.35         # [LEARNABLE] Base importance for all nodes regardless of connectivity
    DEFAULT_NODE_DEGREE_IMPORTANCE_SCALE = 0.4    # [LEARNABLE] Scaling factor for node connectivity importance
    
    DEFAULT_EPSILON = 1e-6
    DEFAULT_SYMMETRIZE = True

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
        use_dynamic_graph=True, # FOR ABALATION
        heads=4,
        pruning_keep_ratio=GRAPH_PRUNING_KEEP_RATIO,
        initial_static_weight=INITIAL_STATIC_GRAPH_WEIGHT,
        final_static_weight=FINAL_STATIC_GRAPH_WEIGHT,
    ):
        """Initializes a mode-specific processor that combines spatial and temporal patterns."""
        super().__init__()
        
        # --- 1. Basic Configuration ---
        self.adj_mx = torch.tensor(static_road_network, dtype=torch.float32).to(device)
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.input_len = input_len
        self.use_dynamic_graph = use_dynamic_graph
        self.heads = heads
        self.gpt_channel = gpt_channel
        self.backbone_channel = backbone_channel
        self.time_steps = self.DEFAULT_TIME_STEPS

        # --- 2. Memory & State (Buffers) ---
        self.register_buffer("total_training_steps", torch.zeros((), dtype=torch.long))
        self.register_buffer("purely_dynamic_graph", torch.zeros((num_nodes, num_nodes)))
        
        binary_adj = (self.adj_mx > 0).float()
        self.register_buffer("norm_adj_mx", binary_adj / (binary_adj.sum(-1, keepdim=True) + self.DEFAULT_EPSILON))
        self.register_buffer("binary_adj_mx", binary_adj)

        # --- 3. Static Hyperparameters ---
        self.attention_head_dropout = self.ATTENTION_HEAD_DROPOUT
        self.gat_leaky_slope = self.GAT_LEAKY_SLOPE
        self.graph_ema_momentum = self.GRAPH_EMA_MOMENTUM
        self.eps = self.DEFAULT_EPSILON
        self.graph_edge_dropout = self.GRAPH_EDGE_DROPOUT
        self.symmetrize = self.DEFAULT_SYMMETRIZE
        self.graph_learning_warmup = self.GRAPH_LEARNING_WARMUP
        self.pruning_keep_ratio = pruning_keep_ratio
        self.initial_static_weight = initial_static_weight
        self.final_static_weight = final_static_weight

        # --- 4. Learnable Parameters ---
        # [LEARNABLE] Graph Attention Temperature
        self.gat_temp_raw = nn.Parameter(torch.tensor(self.DEFAULT_GAT_TEMPERATURE))
        # [LEARNABLE] Node Importance Scaling
        self.node_degree_base_prior = nn.Parameter(torch.tensor(self.DEFAULT_NODE_DEGREE_BASE_PRIOR))
        self.node_degree_importance_scale = nn.Parameter(torch.tensor(self.DEFAULT_NODE_DEGREE_IMPORTANCE_SCALE))
        # [LEARNABLE] Adaptive Gating (Static vs Dynamic balance)
        initial_logit = math.log(self.FINAL_STATIC_GRAPH_WEIGHT / (1.0 - self.FINAL_STATIC_GRAPH_WEIGHT + self.eps))
        self.learnable_static_weight_logit = nn.Parameter(torch.tensor(initial_logit))

        # --- 5. Input Embedding Layers ---
        self.feature_encoder = nn.Conv2d(input_dim * input_len, gpt_channel, kernel_size=(1, 1))
        self.temporal_embedding = TemporalEmbedding(self.time_steps, gpt_channel)
        self.node_identity_emb = nn.Parameter(torch.empty(num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_identity_emb)

        # --- 6. Spatio-Temporal Interaction Layers ---
        self.input_projection = nn.Conv2d(gpt_channel * 3, backbone_channel, kernel_size=(1, 1))
        self.feature_norm = nn.LayerNorm(backbone_channel)
        self.temporal_feature_norm = nn.LayerNorm(backbone_channel)
        
        # Graph Attention scoring components
        self.gat_q = nn.Linear(backbone_channel, backbone_channel, bias=False)
        self.gat_k = nn.Linear(backbone_channel, backbone_channel, bias=False)
        self.gat_a = nn.Parameter(torch.randn(heads, 2 * backbone_channel, 1) * (1.0 / math.sqrt(backbone_channel)))

        # Multi-scale temporal convolutions (TCN)
        self.temporal_conv_1 = nn.Conv1d(input_dim, gpt_channel, 1, padding=0, bias=False)
        self.temporal_conv_3 = nn.Conv1d(input_dim, gpt_channel, 3, padding=1, bias=False)
        self.temporal_conv_5 = nn.Conv1d(input_dim, gpt_channel, 5, padding=2, bias=False)
        self.temporal_projection = nn.Linear(3 * gpt_channel, backbone_channel, bias=False)
        self.temporal_gate = nn.Linear(backbone_channel, backbone_channel)

        # Core Spatial-Temporal Backbone (GPT-based)
        self.backbone = SpatialGPTBackbone(device, gpt_layers=llm_layer, U=U, dropout_rate=0.1)
        self.regression_layer = nn.Conv2d(backbone_channel, output_len, kernel_size=(1, 1))


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

    def _standardize_layout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensures the input tensor follows the standard [Batch, Time, Nodes, Features] layout.
        """
        # Convert legacy [Batch, Features, Nodes, Time] to standard layout
        if x.dim() == 4 and x.shape[1] == self.input_dim:
            return x.permute(0, 3, 2, 1).contiguous()
        return x


    def _get_blending_ratio(self) -> torch.Tensor:
        """
        Calculates the current blending ratio between the fixed road network and the learned graph.
        
        1. Warmup Phase (Curriculum): Linear transition from INITIAL_STATIC_GRAPH_WEIGHT 
           to FINAL_STATIC_GRAPH_WEIGHT.
        2. Post-Warmup (Adaptive Gating): Uses a learnable parameter to find the optimal ratio.
        """
        step = float(self.total_training_steps.item())
        warmup = max(self.graph_learning_warmup, 1)
        
        if step < warmup:
            # Phase 1: Fixed Curriculum Warmup
            progress = step / warmup
            return self.initial_static_weight + (self.final_static_weight - self.initial_static_weight) * progress
        else:
            # Phase 2: Learnable Adaptive Gating
            return torch.sigmoid(self.learnable_static_weight_logit)


    def _calculate_node_importance(self, graph_scores: torch.Tensor) -> torch.Tensor:
        importance = graph_scores.sum(dim=-1, keepdim=True)
        max_importance = importance.max() + 1e-6
        return importance / max_importance


    def _apply_graph_attention(self, features: torch.Tensor) -> torch.Tensor:
        """Core GAT mechanism: Discovers relationships between nodes based on current features."""
        _, num_nodes, _ = features.shape
        normalized_features = F.normalize(features, dim=-1)
        
        # Query/Key mapping
        query = self.gat_q(normalized_features)
        key = self.gat_k(normalized_features)

        # Calculate all-pairs similarity scores
        query_i = query.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        key_j = key.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        pair_features = torch.cat([query_i, key_j], dim=-1)

        # Multi-head scoring
        logits_all = torch.einsum("bijd,hd->bhij", pair_features, self.gat_a.squeeze(-1))
        logits_all = F.leaky_relu(logits_all, self.gat_leaky_slope)

        # Optional: Head Dropout
        if self.training and self.attention_head_dropout > 0:
            num_heads = logits_all.size(1)
            head_mask = (torch.rand(num_heads, device=logits_all.device) >= self.attention_head_dropout).float()
            if head_mask.sum() == 0: head_mask[0] = 1.0
            logits = (logits_all * head_mask.view(1, num_heads, 1, 1)).sum(dim=1) / head_mask.sum()
        else:
            logits = logits_all.mean(dim=1)

        # Learnable Temperature scaling
        gat_temperature = F.softplus(self.gat_temp_raw) + self.eps
        return torch.softmax(logits / gat_temperature, dim=-1)

    def _update_dynamic_memory(self, graph_prob: torch.Tensor):
        """Updates the dynamic graph using Exponential Moving Average (EMA)."""
        mean_graph = graph_prob.mean(dim=0).detach()
        self.purely_dynamic_graph = (
            self.graph_ema_momentum * self.purely_dynamic_graph + 
            (1.0 - self.graph_ema_momentum) * mean_graph
        )

    def _fuse_with_spatial_prior(self, dynamic_graph: torch.Tensor, mix_alpha: float) -> torch.Tensor:
        """Blends the dynamic graph with the physical road network and applies priors."""
        # 1. Hybrid Blending
        fused = (1.0 - mix_alpha) * dynamic_graph + mix_alpha * self.norm_adj_mx

        # 2. Regularization (Edge Dropout)
        if self.training and self.graph_edge_dropout > 0:
            keep_mask = (torch.rand_like(fused) > self.graph_edge_dropout).float()
            fused = fused * keep_mask

        # 3. Connectivity-based Prior (Importance Scaling)
        importance_weight = self._calculate_node_importance(fused)
        return fused * (self.node_degree_base_prior + self.node_degree_importance_scale * importance_weight)

    def _prune_and_stabilize(self, scores: torch.Tensor) -> torch.Tensor:
        """Optimizes the graph by retaining only strong connections and ensuring connectivity."""
        working_scores = scores.clone()
        working_scores.fill_diagonal_(0.0)

        # Top-K Pruning
        keep_ratio = min(max(float(self.pruning_keep_ratio), 1e-3), 0.99)
        threshold = torch.quantile(working_scores, 1.0 - keep_ratio, dim=-1, keepdim=True)
        final_graph = (working_scores >= threshold).float()

        # Final structural cleanup
        final_graph.fill_diagonal_(1.0)
        if self.symmetrize:
            final_graph = torch.maximum(final_graph, final_graph.t())
            
        # Global Step and Warmup Logic
        self.total_training_steps += 1
        if self.total_training_steps.item() < self.graph_learning_warmup:
            final_graph = torch.maximum(final_graph, self.binary_adj_mx)
            
        return final_graph



    def _generate_adaptive_graph(self, features: torch.Tensor) -> torch.Tensor:
        """Main entry point for building the adaptive graph through a 4-step pipeline."""
        # 1. Dynamic Graph Generation (GAT)
        graph_prob = self._apply_graph_attention(features)
        
        # 2. Dynamic Graph Memory Update (EMA)
        self._update_dynamic_memory(graph_prob)
        
        # 3. Integration (Blend with Road Network)
        mix_alpha = self._get_blending_ratio()
        fused_scores = self._fuse_with_spatial_prior(self.purely_dynamic_graph, mix_alpha)
        
        # 4. Optimization (Pruning & Stability)
        return self._prune_and_stabilize(fused_scores)



    def forward(self, input_tensor: torch.Tensor):
        """Processes one VMD mode through the spatio-temporal layers."""
        # 1. Standardize and get dimensions
        x = self._standardize_layout(input_tensor)
        batch_size, time_steps, num_nodes, feature_dim = x.shape

        # 2. Prepare Embeddings (Time + Space + Value)
        time_emb = self.temporal_embedding(x)
        space_emb = self.node_identity_emb.t().view(1, -1, num_nodes, 1).expand(batch_size, -1, -1, -1)
        
        # Reshape for Conv2d encoder: [B, T, N, F] -> [B, F*T, N, 1]
        val_input = x.permute(0, 3, 2, 1).reshape(batch_size, -1, num_nodes, 1)
        val_emb = self.feature_encoder(val_input)
        
        # 3. Fuse and Project to Backbone
        fused = torch.cat([val_emb, time_emb, space_emb], dim=1)
        spatiotemporal_features = self.input_projection(fused)
        spatiotemporal_features = F.leaky_relu(spatiotemporal_features).permute(0, 2, 1, 3).squeeze(-1)
        spatiotemporal_features = self.feature_norm(spatiotemporal_features)

        # 4. Temporal Gating (TCN local features)
        temporal_features = self._extract_temporal_patterns(x)
        temporal_gate = torch.sigmoid(self.temporal_gate(self.temporal_feature_norm(temporal_features)))
        fused_features = spatiotemporal_features + temporal_gate * temporal_features

        # 5. Dynamic Graph Generation
        if self.use_dynamic_graph:
            adjacency = self._generate_adaptive_graph(fused_features)
        else:
            adjacency = self.adj_mx

        # 6. Backbone Processing & Prediction
        backbone_output = self.backbone(fused_features, adjacency)
        prediction = self.regression_layer(backbone_output.permute(0, 2, 1).unsqueeze(-1))
        return prediction, adjacency


class DGLLM(nn.Module):
    # Final Optimized Global Flow Residual Scaling Default Value
    DEFAULT_GLOBAL_FLOW_RESIDUAL_SCALE = 0.1    # [LEARNABLE]
    
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
        """Initializes the master model that coordinates all VMD modes."""
        super().__init__()
        self.use_attention_fusion = use_attention_fusion

        # --- 1. Mode Processing Components ---
        self.mode_processors = nn.ModuleList(
            [
                ModeProcessor(
                    device, static_road_network, input_dim, num_nodes,
                    input_len, output_len, llm_layer, U, use_dynamic_graph=True
                )
                for _ in range(vmd_K)
            ]
        )

        # --- 2. Multi-Mode Fusion Logic ---
        if use_attention_fusion:
            # Dynamic Cross-Mode Attention
            self.fusion_query = nn.Linear(output_len, output_len)
            self.fusion_key = nn.Linear(output_len, output_len)
        else:
            # Fixed Weighted Average
            self.mode_blending_weights = nn.Parameter(torch.ones(vmd_K) / vmd_K)

        # --- 3. Global Residual Path ---
        # [LEARNABLE] Global Flow Scaling
        self.global_flow_residual_scale = nn.Parameter(torch.tensor(self.DEFAULT_GLOBAL_FLOW_RESIDUAL_SCALE))
        
        # Projection to match input history to output horizon
        self.flow_residual_projection = nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.LayerNorm(output_len),
            nn.ReLU(),
        )

    def forward(self, vmd_data: torch.Tensor, original_input: torch.Tensor):
        """
        Runs the end-to-end prediction by processing and fusing all traffic modes.

        Args:
            vmd_data: Frequency-decomposed traffic modes [Batch, Modes, Time, Nodes, 1].
            original_input: Original traffic data [Batch, Time, Nodes, Features].

        Returns:
            A tuple of (final_forecast, learned_graph_structures).
        """
        _, vmd_K, _, _, _ = vmd_data.shape
        time_features = original_input[..., 1:]  # Time-of-day and Day-of-week info

        # 1. Individual Mode Processing
        # We pass each frequency mode through its own ModeProcessor
        mode_predictions = []
        learned_graphs = []
        for i in range(vmd_K):
            mode_flow = vmd_data[:, i, ...]
            mode_input = torch.cat([mode_flow, time_features], dim=-1)

            pred, graph = self.mode_processors[i](mode_input)
            mode_predictions.append(pred)
            learned_graphs.append(graph)

        # 2. Mode Fusion (Combining the predictions)
        if self.use_attention_fusion:
            # Dynamic Attention-based Fusion
            final_prediction = self._blend_modes(mode_predictions)
        else:
            # Simple Weighted Average Fusion
            weights = F.softmax(self.mode_blending_weights, dim=0)
            final_prediction = sum(mode_predictions[i] * weights[i] for i in range(vmd_K))

        # 3. Global Flow Residual (Final Scaling)
        # We add a shortcut from the current raw flow to the prediction
        raw_flow = original_input[..., 0].permute(0, 2, 1) # [B, N, T]
        flow_path = self.flow_residual_projection(raw_flow).permute(0, 2, 1).unsqueeze(-1)
        final_prediction = final_prediction + self.global_flow_residual_scale * flow_path

        return final_prediction, learned_graphs

    def _blend_modes(self, mode_predictions):
        """Fuses mode predictions using a cross-mode attention mechanism."""
        stacked = torch.stack(mode_predictions, dim=0)  # [K, B, T, N, 1]
        K, B, T, N, _ = stacked.shape
        
        # Prepare features for attention: [K, B*N, T]
        feat = stacked.squeeze(-1).permute(0, 1, 3, 2).reshape(K, B * N, T)
        
        # Calculate cross-mode attention scores
        query = self.fusion_query(feat)
        key = self.fusion_key(feat)
        scores = torch.matmul(query, key.transpose(1, 2)).mean(dim=-1) # [K, B*N]
        
        # Apply weights and sum
        weights = F.softmax(scores, dim=0).view(K, B, 1, N, 1)
        return (stacked * weights).sum(dim=0)

    # UTIL
    def param_num(self):
        """Return the total number of trainable and non-trainable parameters."""
        return sum(p.numel() for p in self.parameters())