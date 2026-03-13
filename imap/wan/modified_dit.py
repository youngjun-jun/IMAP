import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import AttentionMixin
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding, WanRotaryPosEmbed

from imap.wan.modified_attention_layer import ModifiedWanTransformerBlock

logger = logging.get_logger(__name__)


class ModifiedWanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                ModifiedWanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        saliency_map_kwargs: Optional[Dict[str, Any]] = None,
        imap_layer: Optional[list[int]] = None,
        imap_sep_score: Optional[str] = None,
        imap_sep_topk: Optional[int] = None,
        imap_qk_matching_target: Optional[str] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
    

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        saliency_map_dict = None
        if imap_qk_matching_target is not None:
            saliency_map_dict = {
                "sel_imap": [],
                "imap": [],
                "cross_attention_maps": [],
            }

        # 4. Transformer blocks
        if imap_qk_matching_target is not None: # conditional generation only
            for i, block in enumerate(self.blocks):
            
                # Set up saving context for attention processor if requested
                if saliency_map_kwargs is not None:
                    proc = getattr(block.attn2, "processor", None)
                    if isinstance(proc, AttentionProcessor) or hasattr(proc, "__call__"):
                        proc.current_timestep_index = int(saliency_map_kwargs.get("timestep_index", -1))
                        proc.block_index = i
                                
                if imap_layer is not None:
                    is_imap_layer = True if i in imap_layer else False
                    proc = getattr(block.attn2, "processor", None)
                    if isinstance(proc, AttentionProcessor) or hasattr(proc, "__call__"):
                        # Set IMAM flags on the custom attention processor
                        proc.is_imap_layer = is_imap_layer
                        proc.imap_sep_score = imap_sep_score
                        proc.imap_sep_topk = imap_sep_topk
                        proc.imap_qk_matching_target = imap_qk_matching_target
                
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, current_saliency_map_dict = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                    )
                else:
                    hidden_states, current_saliency_map_dict = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

                for key in current_saliency_map_dict:
                    saliency_map_dict[key].append(current_saliency_map_dict[key]) # concepts, patches

            add_dict = {}
            for key in saliency_map_dict:
                saliency_map_dict[key] = torch.stack(saliency_map_dict[key], dim=0) 

                add_dict[f"{key}_layers"] = saliency_map_dict[key][imap_layer]  
                saliency_map_dict[key] = saliency_map_dict[key][saliency_map_kwargs["layers"]]  

                if not saliency_map_kwargs.get("except_softmax", False):
                    saliency_map_dict[key] = torch.nn.functional.softmax(
                        saliency_map_dict[key], 
                        dim=-2 
                    )
                    add_dict[f"{key}_layers"] = torch.nn.functional.softmax(
                        add_dict[f"{key}_layers"], 
                        dim=-2
                    )

                if imap_qk_matching_target == "prompt":
                    num_text_token = len(saliency_map_kwargs["tokens"])
                else:
                    num_text_token = len(saliency_map_kwargs["concepts"])

                add_dict[f"{key}_layers"] = add_dict[f"{key}_layers"][:, :num_text_token]
                add_dict[f"{key}_layers"] = torch.mean(
                    add_dict[f"{key}_layers"], 
                    dim=0
                ) 
                saliency_map_dict[key] = saliency_map_dict[key][:, :num_text_token]
                saliency_map_dict[key] = torch.mean(
                    saliency_map_dict[key], 
                    dim=0
                )

            saliency_map_dict.update(add_dict)

        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, _ = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                    )
                else:
                    hidden_states, _ = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict and saliency_map_dict is not None:
            return (output, saliency_map_dict)

        if saliency_map_dict is not None:
            return Transformer2DModelOutput(sample=output), saliency_map_dict
        else:
            return Transformer2DModelOutput(sample=output)