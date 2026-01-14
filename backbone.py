import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from peft import get_peft_model, LoraConfig
from torch.utils.checkpoint import checkpoint

# ============================================================================
# 1. CUSTOM GPT-2 ATTENTION WITH PAIRWISE BIAS
# ============================================================================
class CustomGPT2Attention(GPT2Attention):
    """
    Extends HF GPT2Attention to accept attn_bias: [B, H, Tq, Tk] (additive; 0 allow, -inf block).
    Keeps standard causal masking; final mask = causal + attn_bias.
    Uses torch.nn.functional.scaled_dot_product_attention.
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
        # Handle both layer_past and past_key_value
        if layer_past is None and 'past_key_value' in kwargs:
            layer_past = kwargs['past_key_value']

        bsz, seq_len, _ = hidden_states.size()
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        def shape(x):
            new_x_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        def unshape(x):
            x = x.permute(0, 2, 1, 3).contiguous()
            new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
            return x.view(*new_x_shape)

        query = shape(query)
        key = shape(key)
        value = shape(value)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        present = (key, value) if use_cache else None

        Tq = query.size(-2)
        Tk = key.size(-2)
        H = query.size(1)

        # Build causal mask (additive; -inf on future)
        causal_disallow = torch.triu(
            torch.ones(Tq, Tk, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        causal_add = torch.zeros((Tq, Tk), device=hidden_states.device, dtype=query.dtype)
        causal_add = causal_add.masked_fill(causal_disallow, float("-inf"))
        causal_add = causal_add.view(1, 1, Tq, Tk).expand(bsz, H, Tq, Tk)

        # Combine with user-provided attn_bias if present
        if attn_bias is not None:
            if attn_bias.dim() != 4 or attn_bias.size(2) != Tq or attn_bias.size(3) != Tk:
                raise ValueError(f"attn_bias must be [B|1, H|1, Tq, Tk]; got {tuple(attn_bias.size())}")
            if attn_bias.size(0) == 1 and bsz > 1:
                attn_bias = attn_bias.expand(bsz, -1, -1, -1)
            if attn_bias.size(1) == 1 and H > 1:
                attn_bias = attn_bias.expand(-1, H, -1, -1)
            additive_mask = causal_add + attn_bias.to(query.dtype)
        else:
            additive_mask = causal_add

        # SDPA
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=additive_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )

        attn_output = unshape(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (None,)
        return outputs


# ============================================================================
# 2. CUSTOM GPT-2 BLOCK
# ============================================================================
class CustomGPT2Block(GPT2Block):
    """Same as GPT2Block, but passes attn_bias through to attention."""

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

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attn_bias=attn_bias,
            **kwargs,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


# ============================================================================
# 3. PATCHING FUNCTION
# ============================================================================
def patch_gpt2_for_pairwise_mask(gpt2_model):
    """
    Safely patch in-place without calling __init__ on GPT-2 internals.
    We keep the same block/attn instances and just swap their classes.
    """
    for i, blk in enumerate(gpt2_model.h):
        blk.attn.__class__ = CustomGPT2Attention
        blk.__class__ = CustomGPT2Block


# ============================================================================
# 4. ADJACENCY TO PAIRWISE BIAS
# ============================================================================
def adjacency_to_pairwise_bias(adj, B, H, device, dtype):
    """
    adj: [T, T] with 1 (allow) / 0 (block)
    returns additive bias [B, H, T, T] with 0 for allowed, -inf for blocked.
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj must be [T,T]"
    T = adj.size(0)
    add = torch.zeros((T, T), device=device, dtype=dtype)
    add = add.masked_fill(~adj.to(torch.bool), float("-inf"))
    add = add.view(1, 1, T, T).expand(B, H, T, T)
    return add


# ============================================================================
# 5. PFA - PRE-TRAINED FOUNDATION ADAPTER (EXACT NOTEBOOK VERSION)
# ============================================================================
class PFA(nn.Module):
    """
    Pre-trained Foundation Adapter: GPT-2 backbone with LoRA fine-tuning
    and graph-aware attention via pairwise adjacency bias.
    """
    def __init__(
        self, 
        device: str = "cuda:0", 
        gpt_layers: int = 6, 
        U: int = 1, 
        dropout_rate: float = 0.0,
        use_gradient_checkpointing: bool = True
    ):
        super(PFA, self).__init__()

        # Load GPT-2 with pretrained weights
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2",
            attn_implementation="eager",
            output_attentions=True,
            output_hidden_states=True
        )

        # Truncate to first gpt_layers
        self.gpt2.h = self.gpt2.h[:gpt_layers]

        self.U = U
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lora_rank = 16
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Patch BEFORE applying LoRA
        patch_gpt2_for_pairwise_mask(self.gpt2)

        # LoRA
        self.lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
            target_modules=['c_attn'],
            bias="none"
        )
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)

        # Freezing policy
        gpt_layers_count = len(self.gpt2.base_model.model.h)
        for layer_index, layer in enumerate(self.gpt2.base_model.model.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers_count - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def custom_forward(
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
        
        gpt2_model = self.gpt2.base_model.model
        
        output_attentions = output_attentions if output_attentions is not None else gpt2_model.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else gpt2_model.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else gpt2_model.config.use_cache
        return_dict = return_dict if return_dict is not None else gpt2_model.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You must specify input_ids or inputs_embeds.")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(gpt2_model.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = gpt2_model.wte(input_ids)
        position_embeds = gpt2_model.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None

        total_layers = len(gpt2_model.h)
        top_start = total_layers - self.U

        # Precompute pairwise bias if adjacency provided
        pair_bias = None
        if adjacency_matrix is not None:
            H = gpt2_model.config.n_head
            T = hidden_states.size(1)
            if adjacency_matrix.dim() != 2 or adjacency_matrix.size(0) != T or adjacency_matrix.size(1) != T:
                raise ValueError(f"adjacency_matrix must be [T,T] matching sequence length {T}")
            pair_bias = adjacency_to_pairwise_bias(
                adj=adjacency_matrix.to(device),
                B=batch_size,
                H=H,
                device=device,
                dtype=hidden_states.dtype
            )

        # Main layer loop
        for i, (block, layer_past) in enumerate(zip(gpt2_model.h, past_key_values)):
            use_bias = (i >= top_start and pair_bias is not None)
            
            if self.training and self.use_gradient_checkpointing and not use_cache:
                def create_custom_forward(module, current_bias):
                    def custom_forward(hidden_states_input):
                        outputs = module(
                            hidden_states_input,
                            layer_past=None,
                            attention_mask=None,
                            head_mask=head_mask[i] if head_mask is not None else None,
                            use_cache=False,
                            output_attentions=False,
                            attn_bias=current_bias
                        )
                        return outputs[0]
                    return custom_forward
                
                hidden_states = checkpoint(
                    create_custom_forward(block, pair_bias if use_bias else None),
                    hidden_states,
                    use_reentrant=False
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=None,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    attn_bias=pair_bias if use_bias else None
                )
                hidden_states = outputs[0]

                if use_cache:
                    presents = presents + (outputs[1],)
                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2] if len(outputs) > 2 else None,)

        hidden_states = gpt2_model.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor):
        """
        x: [B, T, D] - Input embeddings
        adjacency_matrix: [T, T] with 1 (allow) / 0 (block)
        """
        out = self.custom_forward(
            inputs_embeds=x,
            adjacency_matrix=adjacency_matrix,
            use_cache=False
        ).last_hidden_state
        out = self.dropout(out)
        return out
