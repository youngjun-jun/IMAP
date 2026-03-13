from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.loaders import FromOriginalModelMixin

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoConditionEmbedding,
    HunyuanVideoPatchEmbed,
    HunyuanVideoRotaryPosEmbed,
    HunyuanVideoTokenRefiner,
    HunyuanVideoTokenReplaceSingleTransformerBlock,
    HunyuanVideoTokenReplaceTransformerBlock,
)

from imap.hunyuanvideo.modified_attention_layer import ModifiedHunyuanVideoSingleTransformerBlock, ModifiedHunyuanVideoTransformerBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class ModifiedHunyuanVideoTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]
    _repeated_blocks = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: Tuple[int] = (16, 56, 56),
        image_condition_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        supported_image_condition_types = ["latent_concat", "token_replace"]
        if image_condition_type is not None and image_condition_type not in supported_image_condition_types:
            raise ValueError(
                f"Invalid `image_condition_type` ({image_condition_type}). Supported ones are: {supported_image_condition_types}"
            )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )

        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim, pooled_projection_dim, guidance_embeds, image_condition_type
        )

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        if image_condition_type == "token_replace":
            self.transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    ModifiedHunyuanVideoTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_layers)
                ]
            )

        # 4. Single stream transformer blocks
        if image_condition_type == "token_replace":
            self.single_transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceSingleTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_single_layers)
                ]
            )
        else:
            self.single_transformer_blocks = nn.ModuleList(
                [
                    ModifiedHunyuanVideoSingleTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_single_layers)
                ]
            )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        concept_hidden_states: Optional[torch.Tensor] = None,
        concept_mask: Optional[torch.Tensor] = None,
        pooled_concept_projections: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
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
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        first_frame_num_tokens = 1 * post_patch_height * post_patch_width

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)
        
        if pooled_concept_projections is not None:
            _, concept_token_replace_emb = self.time_text_embed(timestep, pooled_concept_projections, guidance)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)
        
        if concept_hidden_states is not None:
            concept_hidden_states = self.context_embedder(concept_hidden_states, timestep, concept_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.ones(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )
        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
        indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0) 
        mask_indices = indices >= effective_sequence_length.unsqueeze(1) 
        attention_mask = attention_mask.masked_fill(mask_indices, False)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) 

        if concept_hidden_states is not None:
            saliency_map_dict = {
                "sel_imap": [],
                "imap": [],
                "concept_attention_maps": [],
                "cross_attention_maps": [],
            }
        else:
            saliency_map_dict = None

        # 4. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            
            if concept_hidden_states is not None:
                # Set up saving context for attention processor if requested
                if saliency_map_kwargs is not None:
                    proc = getattr(block.attn, "processor", None)
                    if isinstance(proc, AttentionProcessor) or hasattr(proc, "__call__"):
                        # enable only when timestep index matches and flags present
                        proc.current_timestep_index = int(saliency_map_kwargs.get("timestep_index", -1))
                        proc.block_index = i
                        
                if imap_layer is not None:
                    is_imap_layer = True if i in imap_layer else False
                    proc = getattr(block.attn, "processor", None)
                    if isinstance(proc, AttentionProcessor) or hasattr(proc, "__call__"):
                        # Set IMAM flags on the custom attention processor
                        proc.is_imap_layer = is_imap_layer
                        proc.imap_sep_score = imap_sep_score
                        proc.imap_sep_topk = imap_sep_topk
                        proc.imap_qk_matching_target = imap_qk_matching_target
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states, concept_hidden_states, current_saliency_map_dict = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    concept_hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    concept_token_replace_emb,
                    first_frame_num_tokens,
                )
            else:
                hidden_states, encoder_hidden_states, concept_hidden_states, current_saliency_map_dict = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    concept_hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    concept_token_replace_emb,
                    first_frame_num_tokens,
                )

            if concept_hidden_states is not None:
                for key in current_saliency_map_dict:
                    saliency_map_dict[key].append(current_saliency_map_dict[key]) 

        for block in self.single_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )


        if concept_hidden_states is not None:
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

                if "imap" in key and imap_qk_matching_target == "prompt":
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

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states, saliency_map_dict)

        return Transformer2DModelOutput(sample=hidden_states), saliency_map_dict
