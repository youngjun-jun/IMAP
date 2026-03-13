import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import einops

from torch import nn
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.normalization import CogVideoXLayerNormZero

from imap.imap_utils import select_head, select_visual_token

class CustomCogVideoXAttnProcessor2_0:

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # runtime save context (set by outer modules)
        self.current_timestep_index = None
        self.block_index = None
        self.is_imap_layer = False
        self.imap_sep_score = None
        self.imap_sep_topk = None
        self.imap_qk_matching_target = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Apply k, q, v to concept hidden states
        concept_query = attn.to_q(concept_hidden_states)
        concept_key = attn.to_k(concept_hidden_states)
        concept_value = attn.to_v(concept_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Reshape concept query, key, value
        concept_query = concept_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        concept_key = concept_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        concept_value = concept_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            concept_query = attn.norm_q(concept_query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            concept_key = attn.norm_k(concept_key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        sel_head_idx = select_head(
            hidden_state=hidden_states[1],
            text_seq_length=text_seq_length,
            F=13,
            H=30,
            W=45,
            sep_score=self.imap_sep_score,
            topk=self.imap_sep_topk,
        )
        assert sel_head_idx.shape[0] > 0, "No head selected in IMAM layer, please check the selection criteria."

        ###################### Concept Attention Projections ###########################
        # Package together the concept and image embeddings for attention
        image_query = query[:, :, text_seq_length:]
        image_key = key[:, :, text_seq_length:]
        image_value = value[:, :, text_seq_length:]
        concept_image_queries = torch.cat([concept_query, image_query], dim=2)
        concept_image_keys = torch.cat([concept_key, image_key], dim=2)
        concept_image_values = torch.cat([concept_value, image_value], dim=2)
        # Apply the attention to the concept and image embeddings
        concept_attn_hidden_states = F.scaled_dot_product_attention(
            concept_image_queries, 
            concept_image_keys, 
            concept_image_values, 
            dropout_p=0.0
        )
        # Pull out just the concept embedding outputs
        attn_concept_hidden_states = concept_attn_hidden_states[:, :, :concept_hidden_states.size(1)]
        ############# Compute Cross Attention Maps ##############
        cross_attention_maps = einops.einsum(
            image_query[1],
            concept_key[1],
            "heads patches dim, heads concepts dim -> heads concepts patches"
        )
        # Average over the heads
        cross_attention_maps = einops.reduce(
            cross_attention_maps,
            "heads concepts patches -> concepts patches",
            "mean"
        )
        ################################################################################
        # ########## Concept Attention Projection ##########
        # Pull out the hidden states for the image patches 
        image_hidden_states = hidden_states[:, :, text_seq_length:]
        # Do the concept attention projections
        concept_attention_maps = einops.einsum(
            attn_concept_hidden_states[1],
            image_hidden_states[1],
            "heads concepts dim, heads patches dim -> heads concepts patches"
        )
        # Average over the heads            
        concept_attention_maps = einops.reduce(
            concept_attention_maps,
            "heads concepts patches -> concepts patches",
            "sum"
        )
        ################################################################################
        # ########## IMAP ##########
        text_key = key[1, :, :text_seq_length] if self.imap_qk_matching_target == "prompt" else concept_key[1]
        sel_vis_idx = select_visual_token(
            visual_tokens=image_hidden_states[1],
            image_query=image_query[1],
            text_key=text_key,
        ) 
        F_frames = sel_vis_idx.shape[-1]
        img_qkv_reshaped = einops.rearrange(
            image_hidden_states[1], 'h (F P) dim -> h F P dim', F=F_frames
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
    
        concept_attention_dict = {
            "sel_imap": sel_imap,
            "imap": imap,
            "concept_attention_maps": concept_attention_maps,
            "cross_attention_maps": cross_attention_maps,
        }
        # #############################################

        # reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        concept_hidden_states = attn_concept_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim) 
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        concept_hidden_states = attn.to_out[0](concept_hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        concept_hidden_states = attn.to_out[1](concept_hidden_states)
        # Apply 

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states, concept_hidden_states, concept_attention_dict


class ModifiedCogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CustomCogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        ################## Concept Attention ##################

        _, norm_concept_hidden_states, _, concept_gate_msa = self.norm1(
            hidden_states, concept_hidden_states, temb
        )

        # Concept attention
        attn_hidden_states, attn_encoder_hidden_states, attn_concept_hidden_states, concept_attention_dict = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            concept_hidden_states=norm_concept_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        concept_hidden_states = concept_hidden_states + concept_gate_msa * attn_concept_hidden_states

        _, norm_concept_hidden_states, _, concept_gate_ff = self.norm2(
            hidden_states, concept_hidden_states, temb
        )

        concept_ff_output = self.ff(norm_concept_hidden_states)

        concept_hidden_states = concept_hidden_states + concept_gate_ff * concept_ff_output

        ######################################################

        # Now do the normal attention

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )
     
        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

       
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states, concept_hidden_states, concept_attention_dict
