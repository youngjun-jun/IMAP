import os
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanAttention, _get_added_kv_projections, _get_qkv_projections
from diffusers.models.transformers.transformer_wan import WanAttnProcessor

from imap.imap_utils import select_head, select_visual_token


class ModifiedWanAttnProcessor:
    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )
        self.current_timestep_index = None
        self.block_index = None
        self.is_imap_layer = False
        self.imap_sep_score = None
        self.imap_sep_topk = None
        self.imap_qk_matching_target = None
        
    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # cross-attention
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        
        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)
            
                
        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        if encoder_hidden_states is not None:
            sel_head_idx = select_head(
                hidden_state=hidden_states[0].transpose(0,1),
                text_seq_length=0, 
                F=13,
                H=30,
                W=52,
                sep_score=self.imap_sep_score,
                topk=self.imap_sep_topk,
            )
            assert sel_head_idx.shape[0] > 0, "No head selected in IMAM layer, please check the selection criteria."

            assert self.imap_qk_matching_target == "prompt"
            text_key = key 
            
            sel_vis_idx, scaled_qk_matching = select_visual_token(
                visual_tokens=hidden_states[0].transpose(0,1),
                image_query=query[0].transpose(0,1),
                text_key=text_key[0].transpose(0,1),
                return_logits=True,
            )
            
            ##########################################
            cross_attention_maps = torch.softmax(scaled_qk_matching, dim=1)
            cross_attention_maps = einops.reduce(
                cross_attention_maps,
                "heads concepts patches -> concepts patches",
                "mean"
            )
            ##########################################
                
            F_frames = sel_vis_idx.shape[-1] 
            img_qkv_reshaped = einops.rearrange(
                hidden_states[0], '(F P) h dim -> h F P dim', F=F_frames
            ) 

            Text_tokens = sel_vis_idx.shape[1] 
            img_qkv_5d = img_qkv_reshaped.unsqueeze(1).expand(-1, Text_tokens, -1, -1, -1)
            gather_idx = sel_vis_idx[..., None, None].long().expand(
                -1, -1, -1, 1, img_qkv_5d.size(-1)
            )
            sel_img_hidden_states = img_qkv_5d.gather(3, gather_idx).squeeze(3) 

            imap = einops.einsum(
                sel_img_hidden_states, 
                img_qkv_reshaped,      
                'h texts f d, h f p d -> h texts f p'
            ) 
            
            imap = (imap - imap.mean(dim=1, keepdim=True)) / (imap.std(dim=1, keepdim=True) + 1e-6)
            
            imap = einops.rearrange(imap, 'heads texts F patches -> heads texts (F patches)')

            sel_imap = einops.reduce(
                imap.index_select(0, sel_head_idx),
                "heads texts patches -> texts patches",
                "mean"
            )
            imap = einops.reduce(
                imap,
                "heads texts patches -> texts patches",
                "sum"
            )
            
            ##########################################
            saliency_map_dict = {
                "sel_imap": sel_imap,
                "imap": imap,
                "cross_attention_maps": cross_attention_maps,
            }
            ##########################################

        else:
            saliency_map_dict = None

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states, saliency_map_dict

class ModifiedWanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=ModifiedWanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output, saliency_map_dict = self.attn2(
            norm_hidden_states, 
            encoder_hidden_states, 
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states, saliency_map_dict