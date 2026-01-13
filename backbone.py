import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model
from peft import get_peft_model, LoraConfig, TaskType

class CustomGPT2Attention(GPT2Attention):
    """
    Modified GPT-2 Attention to support pairwise adjacency bias.
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
        attn_bias: Optional[torch.Tensor] = None,           # NEW: [B, H, Tq, Tk] with 0/-inf
        **kwargs,                                           # Robustness
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
        key   = shape(key)
        value = shape(value)

        if layer_past is not None and len(layer_past) == 2:
            past_key, past_value = layer_past
            key   = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        present = (key, value) if use_cache else None

        Tq = query.size(-2)
        Tk = key.size(-2)
        H  = query.size(1)

        # Build causal mask
        causal_disallow = torch.triu(
            torch.ones(Tq, Tk, device=hidden_states.device, dtype=torch.bool),
            diagonal=1
        )
        causal_add = torch.zeros((Tq, Tk), device=hidden_states.device, dtype=query.dtype)
        causal_add = causal_add.masked_fill(causal_disallow, float("-inf"))
        causal_add = causal_add.view(1, 1, Tq, Tk).expand(bsz, H, Tq, Tk)

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

class CustomGPT2Block(GPT2Block):
    """Modified GPT-2 Block to pass attn_bias to attention layer."""
    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        attn_bias: Optional[torch.Tensor] = None, # NEW
        **kwargs, # Robustness
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attn_bias=attn_bias, # PASS BIAS
            **kwargs, # PASS EXTRA
        )
        attn_output = attn_outputs[0]  
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  

def patch_gpt2_for_pairwise_mask(model: GPT2Model):
    """Replaces standard GPT2Attention components with CustomGPT2Attention."""
    for i, block in enumerate(model.h):
        old_attn = block.attn
        new_attn = CustomGPT2Attention(model.config, is_cross_attention=False, layer_idx=i)
        
        new_attn.c_attn = old_attn.c_attn
        new_attn.c_proj = old_attn.c_proj
        new_attn.attn_dropout = old_attn.attn_dropout
        new_attn.resid_dropout = old_attn.resid_dropout
        
        block.attn = new_attn
        
        # Upgrade block to CustomGPT2Block
        new_block = CustomGPT2Block(model.config, layer_idx=i)
        new_block.ln_1 = block.ln_1
        new_block.attn = block.attn
        new_block.ln_2 = block.ln_2
        new_block.mlp  = block.mlp
        model.h[i] = new_block
    return model

def adjacency_to_pairwise_bias(adj: torch.Tensor):
    """
    Converts [B, N, N] adjacency matrix into [B, 1, N, N] attention bias.
    1.0 (connected) -> 0.0 (no mask)
    0.0 (disconnected) -> -inf (masked)
    """
    bias = torch.zeros_like(adj)
    bias = bias.masked_fill(adj == 0, float("-inf"))
    return bias.unsqueeze(1) 

class PFA(nn.Module):
    """
    Pre-trained Foundation Adapter.
    Wraps GPT-2 with LoRA and custom graph attention patching.
    """
    def __init__(self, d_model, patch_size, args):
        super(PFA, self).__init__()
        self.d_model = d_model
        
        # 1. Load Pre-trained GPT-2
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained('gpt2')
        config.n_embd = d_model
        config.n_inner = 4 * d_model
        config.n_head = 8
        self.backbone = GPT2Model(config)
        
        # Truncate layers to match args
        self.backbone.h = self.backbone.h[:args.llm_layers]
        
        # 2. Patch for Graph Awareness
        self.backbone = patch_gpt2_for_pairwise_mask(self.backbone)

        # 3. Apply LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.05,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

        # 4. Freezing Policy (from notebook)
        # U is number of layers from top to keep trainable
        gpt_layers = args.llm_layers
        U = args.U
        for layer_index, layer in enumerate(self.backbone.base_model.model.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - U:
                    if "ln" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        # 5. Gradient Checkpointing
        if args.use_checkpoint:
            self.backbone.gradient_checkpointing_enable()

    def custom_forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        adjacency_matrix: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        model = self.backbone.base_model.model
        device = inputs_embeds.device
        
        # We assume sequence length T matches adjacency matrix size
        T = inputs_embeds.size(1)
        H = model.config.n_head
        B = inputs_embeds.size(0)
        
        pair_bias = None
        if adjacency_matrix is not None:
             pair_bias = adjacency_to_pairwise_bias(adjacency_matrix).expand(B, H, T, T)
        
        # Manual Forward Through Blocks
        # (Simplified version of notebook's custom_forward)
        hidden_states = inputs_embeds # We skip WPE for temporal since we use TemporalEmbedding
        
        for block in model.h:
            outputs = block(
                hidden_states,
                attn_bias=pair_bias
            )
            hidden_states = outputs[0]
            
        return hidden_states

    def forward(self, x, adjacency_matrix=None):
        # x: [B, T, D]
        # adjacency_matrix: [B, N, N]
        
        # Use custom manual loop to ensure attn_bias is used
        out = self.custom_forward(
            inputs_embeds=x,
            adjacency_matrix=adjacency_matrix
        )
        return out
