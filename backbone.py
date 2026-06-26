from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model


def _create_spatial_attention_mask(
    connectivity_matrix: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Converts connectivity matrix to additive attention mask with -inf for disconnected nodes.
    """
    Converts a road-network connectivity matrix into an additive attention mask.
    - Connected nodes: 0 bias
    - Disconnected nodes: -inf bias
    """
    seq_len = connectivity_matrix.size(0)
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    mask = mask.masked_fill(~connectivity_matrix.bool(), float("-inf"))
    return mask.view(1, 1, seq_len, seq_len)


class SpatialAwareGPT2Attention(GPT2Attention):
    """GPT-2 attention layer modified to support spatial layout biases."""

    def _get_cached_causal_bias(
        self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Generates and caches the standard causal mask for autoregressive attention.
        """Returns a cached causal additive bias [1, 1, Tq, Tk]."""
        cache_key = (query_len, key_len, device)
        if not hasattr(self, "_cached_causal_bias") or self._cached_causal_bias_key != cache_key:
            disallowed = torch.triu(
                torch.ones(query_len, key_len, device=device, dtype=torch.bool), diagonal=1
            )
            bias = torch.zeros((query_len, key_len), device=device, dtype=dtype)
            self._cached_causal_bias = bias.masked_fill(disallowed, float("-inf"))
            self._cached_causal_bias_key = cache_key

        return self._cached_causal_bias.to(dtype=dtype).view(1, 1, query_len, key_len)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        # Reshapes tensor to split embedding dimension into multiple attention heads.
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        return tensor.view(*new_shape).permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        # Combines multiple attention heads back into a single hidden dimension.
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(*new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        attn_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Computes attention scores using both causal and optional spatial masks.
        """Self-attention with optional spatial/graph bias."""
        # Unused GPT-2 arguments
        del attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask, output_attentions

        if layer_past is None and "past_key_value" in kwargs:
            layer_past = kwargs["past_key_value"]

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        present = (key, value) if use_cache else None
        batch_size, num_heads, query_len, _ = query.size()
        key_len = key.size(-2)

        # 1. Start with Causal Bias
        attention_bias = self._get_cached_causal_bias(
            query_len=query_len, key_len=key_len, device=hidden_states.device, dtype=query.dtype
        )

        # --------------------------------------------------------
        # 2. Integrate Spatial/Graph Bias
        if attn_bias is not None:
            if attn_bias.dim() != 4 or attn_bias.size(2) != query_len or attn_bias.size(3) != key_len:
                raise ValueError(f"attn_bias shape mismatch: expected [B|1, H|1, {query_len}, {key_len}], got {list(attn_bias.size())}")
            
            if attn_bias.size(0) == 1 and batch_size > 1:
                attn_bias = attn_bias.expand(batch_size, -1, -1, -1)
            if attn_bias.size(1) == 1 and num_heads > 1:
                attn_bias = attn_bias.expand(-1, num_heads, -1, -1)
            
            attention_bias = attention_bias + attn_bias.to(dtype=query.dtype)
        # elif not hasattr(self, "_logged_causal_only"):
        #     print(f"[SpatialAwareGPT2Attention] Causal-only mode (no spatial bias). T={query_len}")
        #     self._logged_causal_only = True

        # 3. Scaled Dot Product Attention
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present


class SpatialAwareGPT2Block(GPT2Block):
    """GPT-2 block that passes spatial connectivity info to its attention layer."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        attn_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Executes one transformer block pass including layer norm, attention, and MLP.
        """Run one GPT-2 block with spatial attention bias."""
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attn_bias=attn_bias,
            **kwargs,
        )
        attn_output, present = attn_outputs
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return (hidden_states, present) if use_cache else (hidden_states,)


def _inject_spatial_awareness(gpt2_model: GPT2Model) -> None:
    # Runtime patching to replace standard GPT2 layers with spatial-aware versions.
    """
    Swaps standard GPT-2 layers with spatial-aware versions while preserving weights.
    """
    for block in gpt2_model.h:
        block.attn.__class__ = SpatialAwareGPT2Attention
        block.__class__ = SpatialAwareGPT2Block


class SpatialGPTBackbone(nn.Module):
    # Transformer backbone with spatial awareness for traffic forecasting.

    def __init__(
        self,
        device: str = "cuda:0",
        gpt_layers: int = 6,
        U: int = 1,
        middle_lora_layers: Optional[int] = None,
        dropout_rate: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        # Initializes the backbone by loading GPT2, injecting spatial layers, and applying LoRA.
        """
        Initialize GPT-2 backbone with LoRA and optional graph-biased attention.
        
        Args:
            device: Kept for API compatibility.
            gpt_layers: Number of GPT-2 blocks to retain.
            U: Number of top layers receiving graph bias and remaining trainable.
            dropout_rate: Dropout probability.
            use_gradient_checkpointing: Enable memory-efficient gradients.
        """
        super().__init__()
        self.num_backbone_layers = gpt_layers
        self.unfrozen_top_layers = max(1, min(U, gpt_layers))
        self.middle_lora_layers = (
            self.unfrozen_top_layers if middle_lora_layers is None else max(0, min(middle_lora_layers, gpt_layers))
        )
        self.dropout_rate = dropout_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.lora_rank = 16

        self.gpt2 = self._init_base_model()
        self._freeze_lower_layers()
        self.dropout = nn.Dropout(p=dropout_rate)



    def _init_base_model(self):
        # Loads pre-trained GPT2 and applies spatial awareness and PEFT/LoRA adapters.
        gpt2 = GPT2Model.from_pretrained(
            "gpt2",
            attn_implementation="eager",
            output_attentions=False,
            output_hidden_states=False,
        )
        gpt2.h = gpt2.h[: self.num_backbone_layers]

        _inject_spatial_awareness(gpt2)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
            target_modules=["c_attn"],
            bias="none",
        )
        return get_peft_model(gpt2, lora_config)

    def _get_layer_training_stage(self, layer_idx: int, total_layers: int) -> str:
        """Return the adaptation stage for a block: bottom, middle, or top."""
        top_start = max(0, total_layers - self.unfrozen_top_layers)
        middle_start = max(0, top_start - self.middle_lora_layers)

        if layer_idx >= top_start:
            return "top"
        if layer_idx >= middle_start:
            return "middle"
        return "bottom"

    def _freeze_lower_layers(self) -> None:
        """Freezes lower layers and configures a three-tier adaptation policy."""
        blocks = self.gpt2.base_model.model.h

        for i, layer in enumerate(blocks):
            for name, param in layer.named_parameters():
                param.requires_grad = True

    def _resolve_runtime_flags(
        self,
        gpt2_model: GPT2Model,
        use_cache: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
    ):
        # Determines execution flags based on provided inputs or model defaults.
        return (
            use_cache if use_cache is not None else gpt2_model.config.use_cache,
            output_hidden_states if output_hidden_states is not None else gpt2_model.config.output_hidden_states,
            return_dict if return_dict is not None else gpt2_model.config.use_return_dict,
        )

    def _prepare_inputs(
        self,
        gpt2_model: GPT2Model,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
        position_ids: Optional[torch.LongTensor],
    ):
        # Prepares final hidden states by combining token/input embeddings with positional embeddings.
        """Resolves inputs into hidden states and position IDs."""
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
            inputs_embeds = gpt2_model.wte(input_ids)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        else:
            raise ValueError("You must specify input_ids or inputs_embeds.")

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(gpt2_model.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        hidden_states = inputs_embeds + gpt2_model.wpe(position_ids)
        return hidden_states, device, past_key_values

    def _build_spatial_mask(
        self,
        connectivity_matrix: Optional[torch.FloatTensor],
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        # Validates dimensions and generates the spatial attention mask for the current sequence.
        """Validates and creates the spatial attention mask."""
        if connectivity_matrix is None:
            return None

        if connectivity_matrix.dim() != 2 or connectivity_matrix.size(0) != sequence_length:
            raise ValueError(
                f"connectivity_matrix must be [T, T] matching sequence length {sequence_length}. "
                f"Got {list(connectivity_matrix.size())}"
            )

        return _create_spatial_attention_mask(
            connectivity_matrix=connectivity_matrix.to(device), device=device, dtype=dtype
        )

    def _block_forward_with_checkpoint(
        self,
        block: GPT2Block,
        hidden_states: torch.Tensor,
        layer_head_mask: Optional[torch.Tensor],
        layer_attention_bias: Optional[torch.Tensor],
    ):
        # Wraps a block execution in a checkpointing function to save memory during training.
        """Executes a block forward pass using gradient checkpointing."""
        def checkpointed_fn(h):
            return block(
                h,
                layer_past=None,
                attention_mask=None,
                head_mask=layer_head_mask,
                use_cache=False,
                output_attentions=False,
                attn_bias=layer_attention_bias,
            )[0]

        return checkpoint(checkpointed_fn, hidden_states, use_reentrant=False)

    def _process_sequence(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        connectivity_matrix: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, dict]:
        # Orchestrates the full forward pass through all transformer layers with spatial biasing.
        """Processes the input sequence through spatial-aware transformer layers."""
        gpt2_model = self.gpt2.base_model.model
        use_cache, output_hidden_states, return_dict = self._resolve_runtime_flags(
            gpt2_model, use_cache, output_hidden_states, return_dict
        )

        hidden_states, device, past_key_values = self._prepare_inputs(
            gpt2_model, input_ids, inputs_embeds, past_key_values, position_ids
        )

# --------------------------------------------------------
        spatial_mask = self._build_spatial_mask(
            connectivity_matrix, hidden_states.size(1), device, hidden_states.dtype
        )

        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None
        top_layer_start = len(gpt2_model.h) - self.unfrozen_top_layers

        for i, (block, layer_past) in enumerate(zip(gpt2_model.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Apply spatial bias only to the top trainable layers.
            layer_attn_bias = spatial_mask if i >= top_layer_start else None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.training and self.use_gradient_checkpointing and not use_cache:
                hidden_states = self._block_forward_with_checkpoint(
                    block, hidden_states, layer_head_mask, layer_attn_bias
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    head_mask=layer_head_mask,
                    use_cache=use_cache,
                    attn_bias=layer_attn_bias,
                )
                hidden_states = outputs[0]
                if use_cache:
                    presents += (outputs[1],)

        hidden_states = gpt2_model.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )

    def forward(self, input_embeddings: torch.Tensor, connectivity_matrix: torch.Tensor):
        # Entry point for processing embeddings through the complete spatial transformer stack.
        """Processes input embeddings using the spatial-aware transformer backbone."""
        outputs = self._process_sequence(
            inputs_embeds=input_embeddings,
            connectivity_matrix=connectivity_matrix,
            use_cache=False,
        )
        return self.dropout(outputs.last_hidden_state)
