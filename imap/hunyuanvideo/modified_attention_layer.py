import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from diffusers.utils import logging
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoAttnProcessor2_0

from imap.imap_utils import select_head, select_visual_token

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class ModifiedHunyuanVideoAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "ModifiedHunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        concept_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)
            
            if concept_hidden_states is not None:
                concept_query = attn.add_q_proj(concept_hidden_states)
                concept_key = attn.add_k_proj(concept_hidden_states)
                concept_value = attn.add_v_proj(concept_hidden_states)

                concept_query = concept_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                concept_key = concept_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
                concept_value = concept_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

                if attn.norm_added_q is not None:
                    concept_query = attn.norm_added_q(concept_query)
                if attn.norm_added_k is not None:
                    concept_key = attn.norm_added_k(concept_key)
                    
                text_seq_length = encoder_query.shape[2]
                
        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        saliency_map_dict = None
        if concept_hidden_states is not None:
            sel_head_idx = select_head(
                hidden_state=hidden_states[0],
                text_seq_length=text_seq_length,
                F=13,
                H=30,
                W=45,
                sep_score=self.imap_sep_score,
                topk=self.imap_sep_topk,
                text_seq_back=True,
            )
            assert sel_head_idx.shape[0] > 0, "No head selected in IMAM layer, please check the selection criteria."
            
            ###################### Concept Attention Projections ###########################
            # Package together the concept and image embeddings for attention
            image_query = query[:, :, :-text_seq_length]
            image_key = key[:, :, :-text_seq_length]
            image_value = value[:, :, :-text_seq_length]
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
                image_query[0],
                concept_key[0],
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
            image_hidden_states = hidden_states[:, :, :-text_seq_length]
            # Do the concept attention projections
            concept_attention_maps = einops.einsum(
                attn_concept_hidden_states[0],
                image_hidden_states[0],
                "heads concepts dim, heads patches dim -> heads concepts patches"
            )
            # Average over the heads            
            concept_attention_maps = einops.reduce(
                concept_attention_maps,
                "heads concepts patches -> concepts patches",
                "sum"
            )
            ################################################################################
            # ########## IMAM ##########
            text_key = key[0, :, -text_seq_length:] if self.imap_qk_matching_target == "prompt" else concept_key[0]
            sel_vis_idx = select_visual_token(
                visual_tokens=image_hidden_states[0],
                image_query=image_query[0],
                text_key=text_key,
            ) 
            F_frames = sel_vis_idx.shape[-1]
            img_qkv_reshaped = einops.rearrange(
                image_hidden_states[0], 'h (F P) dim -> h F P dim', F=F_frames
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
           
            saliency_map_dict = {
                "sel_imap": sel_imap,
                "imap": imap,
                "concept_attention_maps": concept_attention_maps,
                "cross_attention_maps": cross_attention_maps,
            }
            # #############################################
        
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states, concept_hidden_states, saliency_map_dict

class ModifiedHunyuanVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=ModifiedHunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        concept_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        if concept_hidden_states is not None:
            norm_concept_hidden_states, _, _, _, _ = self.norm1_context(
                concept_hidden_states, emb=temb
            )

        # 2. Joint attention
        attn_output, context_attn_output, concept_attn_output, saliency_map_dict = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            concept_hidden_states=norm_concept_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)
        if concept_hidden_states is not None:
            concept_hidden_states = concept_hidden_states + concept_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        if concept_hidden_states is not None:
            norm_concept_hidden_states = self.norm2_context(concept_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        if concept_hidden_states is not None:
            norm_concept_hidden_states = norm_concept_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        if concept_hidden_states is not None:
            concept_ff_output = self.ff_context(norm_concept_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if concept_hidden_states is not None:
            concept_hidden_states = concept_hidden_states + c_gate_mlp.unsqueeze(1) * concept_ff_output

        return hidden_states, encoder_hidden_states, concept_hidden_states, saliency_map_dict


class ModifiedHunyuanVideoSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states

