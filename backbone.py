from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model


def build_pairwise_attention_bias(
    adjacency_matrix: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert a binary adjacency matrix [T, T] into an additive attention bias.
    Allowed edges receive 0, blocked edges receive -inf.
    """
    assert (
        adjacency_matrix.dim() == 2
        and adjacency_matrix.size(0) == adjacency_matrix.size(1)
    ), "adjacency_matrix must be [T, T]"

    seq_len = adjacency_matrix.size(0)
    bias = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    bias = bias.masked_fill(~adjacency_matrix.to(torch.bool), float("-inf"))
    return bias.view(1, 1, seq_len, seq_len)


class GraphBiasGPT2Attention(GPT2Attention):
    """
    GPT-2 attention layer extended with an additive graph bias.
    """

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
        # This project routes masking through attn_bias rather than the stock
        # Hugging Face mask arguments. The extra parameters remain here so the
        # patched layer is still callable through the usual GPT-2 API surface.
        del attention_mask, encoder_hidden_states, encoder_attention_mask, head_mask
        del output_attentions

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
        query_len = query.size(-2)
        key_len = key.size(-2)
        batch_size = query.size(0)
        num_heads = query.size(1)

        causal_bias = self._get_cached_causal_bias(
            query_len=query_len,
            key_len=key_len,
            device=hidden_states.device,
            dtype=query.dtype,
        )

        if attn_bias is not None:
            if attn_bias.dim() != 4 or attn_bias.size(2) != query_len or attn_bias.size(3) != key_len:
                raise ValueError(
                    f"attn_bias must be [B|1, H|1, Tq, Tk]; got {tuple(attn_bias.size())}"
                )
            if attn_bias.size(0) == 1 and batch_size > 1:
                attn_bias = attn_bias.expand(batch_size, -1, -1, -1)
            if attn_bias.size(1) == 1 and num_heads > 1:
                attn_bias = attn_bias.expand(-1, num_heads, -1, -1)
            attention_bias = causal_bias + attn_bias.to(dtype=query.dtype)
        else:
            attention_bias = causal_bias

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, present

    def _get_cached_causal_bias(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Cache one broadcastable causal mask per (Tq, Tk, device) tuple so each
        # layer reuse avoids rebuilding the same upper-triangular mask.
        cache_key = (query_len, key_len, device)
        if not hasattr(self, "_cached_causal_bias") or self._cached_causal_bias_key != cache_key:
            disallowed = torch.triu(
                torch.ones(query_len, key_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
            bias = torch.zeros((query_len, key_len), device=device, dtype=dtype)
            bias = bias.masked_fill(disallowed, float("-inf"))
            self._cached_causal_bias = bias
            self._cached_causal_bias_key = cache_key

        return self._cached_causal_bias.to(device=device, dtype=dtype).view(1, 1, query_len, key_len)


class GraphBiasGPT2Block(GPT2Block):
    """
    GPT-2 block variant that forwards attn_bias into the attention layer.
    """

    def forward(
        self,
        hidden_states: torch.FloatTensor,
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
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output, present = self.attn(
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
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        if use_cache:
            return hidden_states, present
        return (hidden_states,)


def patch_gpt2_with_graph_bias(gpt2_model: GPT2Model) -> None:
    """
    Swap GPT-2 blocks and attention modules in place so pretrained weights remain intact.
    """
    for block in gpt2_model.h:
        block.attn.__class__ = GraphBiasGPT2Attention
        block.__class__ = GraphBiasGPT2Block


class GraphAwareGPTBackbone(nn.Module):
    """
    GPT-2 backbone with LoRA adaptation and graph-aware attention bias on the top layers.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        gpt_layers: int = 6,
        U: int = 1,
        dropout_rate: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        del device
        self.num_backbone_layers = gpt_layers
        self.unfrozen_top_layers = U
        self.dropout_rate = dropout_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.lora_rank = 16
        self.dropout = nn.Dropout(p=dropout_rate)

        self.gpt2 = self._build_gpt2_backbone()
        self._configure_trainable_layers()

    def _build_gpt2_backbone(self):
        gpt2 = GPT2Model.from_pretrained(
            "gpt2",
            attn_implementation="eager",
            output_attentions=False,
            output_hidden_states=False,
        )
        gpt2.h = gpt2.h[: self.num_backbone_layers]

        patch_gpt2_with_graph_bias(gpt2)

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
            target_modules=["c_attn"],
            bias="none",
        )
        return get_peft_model(gpt2, lora_config)

    def _configure_trainable_layers(self) -> None:
        total_layers = len(self.gpt2.base_model.model.h)
        top_layer_start = total_layers - self.unfrozen_top_layers

        for layer_index, layer in enumerate(self.gpt2.base_model.model.h):
            for name, param in layer.named_parameters():
                if layer_index < top_layer_start:
                    param.requires_grad = "ln" in name or "wpe" in name
                else:
                    param.requires_grad = "mlp" not in name

    def _resolve_runtime_flags(
        self,
        gpt2_model: GPT2Model,
        use_cache: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
    ):
        return (
            use_cache if use_cache is not None else gpt2_model.config.use_cache,
            output_hidden_states
            if output_hidden_states is not None
            else gpt2_model.config.output_hidden_states,
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
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds.")

        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(gpt2_model.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = gpt2_model.wte(input_ids)

        hidden_states = inputs_embeds + gpt2_model.wpe(position_ids)
        return hidden_states, device, past_key_values

    def _build_graph_attention_bias(
        self,
        adjacency_matrix: Optional[torch.FloatTensor],
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if adjacency_matrix is None:
            return None

        if (
            adjacency_matrix.dim() != 2
            or adjacency_matrix.size(0) != sequence_length
            or adjacency_matrix.size(1) != sequence_length
        ):
            raise ValueError(
                f"adjacency_matrix must be [T, T] matching sequence length {sequence_length}"
            )

        return build_pairwise_attention_bias(
            adjacency_matrix=adjacency_matrix.to(device),
            device=device,
            dtype=dtype,
        )

    def _build_checkpointed_block_forward(
        self,
        block: GPT2Block,
        layer_head_mask: Optional[torch.Tensor],
        layer_attention_bias: Optional[torch.Tensor],
    ):
        def checkpointed_forward(hidden_states_input):
            outputs = block(
                hidden_states_input,
                layer_past=None,
                attention_mask=None,
                head_mask=layer_head_mask,
                use_cache=False,
                output_attentions=False,
                attn_bias=layer_attention_bias,
            )
            return outputs[0]

        return checkpointed_forward

    def _run_gpt2_backbone(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        adjacency_matrix: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, dict]:
        # DG-LLM uses GPT-2 only as an embedding-driven decoder with graph bias.
        # The extra GPT-2 arguments stay here for compatibility with standard GPT-2 calls.
        del attention_mask, token_type_ids, encoder_hidden_states, encoder_attention_mask

        gpt2_model = self.gpt2.base_model.model
        use_cache, output_hidden_states, return_dict = self._resolve_runtime_flags(
            gpt2_model,
            use_cache,
            output_hidden_states,
            return_dict,
        )
        del output_attentions

        hidden_states, device, past_key_values = self._prepare_inputs(
            gpt2_model,
            input_ids,
            inputs_embeds,
            past_key_values,
            position_ids,
        )

        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        graph_attention_bias = self._build_graph_attention_bias(
            adjacency_matrix=adjacency_matrix,
            sequence_length=hidden_states.size(1),
            device=device,
            dtype=hidden_states.dtype,
        )

        total_layers = len(gpt2_model.h)
        top_layer_start = total_layers - self.unfrozen_top_layers

        for layer_index, (block, layer_past) in enumerate(zip(gpt2_model.h, past_key_values)):
            # Only the top unfrozen layers consume the graph bias. Lower layers remain
            # plain GPT-2 blocks so local language-model structure is preserved.
            layer_attention_bias = (
                graph_attention_bias
                if layer_index >= top_layer_start and graph_attention_bias is not None
                else None
            )
            layer_head_mask = head_mask[layer_index] if head_mask is not None else None

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.training and self.use_gradient_checkpointing and not use_cache:
                hidden_states = checkpoint(
                    self._build_checkpointed_block_forward(
                        block=block,
                        layer_head_mask=layer_head_mask,
                        layer_attention_bias=layer_attention_bias,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
                continue

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=None,
                head_mask=layer_head_mask,
                use_cache=use_cache,
                output_attentions=False,
                attn_bias=layer_attention_bias,
            )
            hidden_states = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)

        hidden_states = gpt2_model.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in [hidden_states, presents, all_hidden_states]
                if value is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor):
        """
        x: [B, T, D] input embeddings
        adjacency_matrix: [T, T] with 1 for allowed edges and 0 for blocked edges
        """
        outputs = self._run_gpt2_backbone(
            inputs_embeds=x,
            adjacency_matrix=adjacency_matrix,
            use_cache=False,
        )
        return self.dropout(outputs.last_hidden_state)
