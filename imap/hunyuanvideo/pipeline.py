from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
import einops

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers import HunyuanVideoPipeline
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE, retrieve_timesteps

class ModifiedHunyuanVideoPipeline(HunyuanVideoPipeline):

    def encode_concepts(
        self,
        concepts: list[str],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        average_until_eos: bool = False,
    ):
        concept_embeds, concept_mask = self._get_llama_prompt_embeds(
            prompt=concepts,
            prompt_template=prompt_template,
            device=device,
            dtype=dtype,
            max_sequence_length=16,
        )

        concept_embeds = concept_embeds[:, 0, :].unsqueeze(0) 
        concept_embeds = torch.cat([concept_embeds, torch.zeros(1, max_sequence_length - len(concepts), concept_embeds.shape[-1], device=concept_embeds.device, dtype=concept_embeds.dtype)], dim=1)
        
        concept_mask = concept_mask[:, 0].unsqueeze(0) 
        concept_mask = torch.cat([concept_mask, torch.zeros(1, max_sequence_length - len(concepts), device=concept_mask.device, dtype=concept_mask.dtype)], dim=1)

        prompt = " ".join(concepts)
        pooled_concept_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=77,
        )
        pooled_concept_embeds = torch.zeros_like(pooled_concept_embeds).to(pooled_concept_embeds.device)

        return concept_embeds, pooled_concept_embeds, concept_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Union[str, List[str]] = None,
        concepts: Optional[List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        saliency_map_kwargs: Optional[Dict[str, Any]] = None,
        # IMAM-related optional arguments (mirrors Renoise pipeline; ignored if not used)
        imap_layer: Optional[list[int]] = None,
        imap_sep_score: Optional[str] = None,
        imap_sep_topk: Optional[int] = None,
        imap_qk_matching_target: Optional[str] = None,
    ):

        if saliency_map_kwargs is None:
            saliency_map_kwargs = {}
        if "concepts" not in saliency_map_kwargs:
            saliency_map_kwargs["concepts"] = concepts

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        transformer_dtype = self.transformer.dtype
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        ############# Encode the concepts #############
        concept_embeds, pooled_concept_embeds, concept_mask = self.encode_concepts(
            concepts=concepts,
            prompt_template=prompt_template,
            device=device,
            dtype=transformer_dtype,
            max_sequence_length=max_sequence_length,
        )
        concept_embeds = concept_embeds.to(transformer_dtype)
        concept_mask = concept_mask.to(transformer_dtype)
        pooled_concept_embeds = pooled_concept_embeds.to(transformer_dtype)
        ###############################################

        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prompt_attention_mask=negative_prompt_attention_mask,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(transformer_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        saliency_map_dict = {
            "cross_attention_maps": [],
        }

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                ca_kwargs = dict(saliency_map_kwargs or {})
                ca_kwargs["timestep_index"] = int(i)
                with self.transformer.cache_context("cond"):
                    transformer_output = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        pooled_projections=pooled_prompt_embeds,
                        concept_hidden_states=concept_embeds,
                        concept_mask=concept_mask,
                        pooled_concept_projections=pooled_concept_embeds,
                        guidance=guidance,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        saliency_map_kwargs=ca_kwargs,
                        imap_layer=imap_layer,
                        imap_sep_score=imap_sep_score,
                        imap_sep_topk=imap_sep_topk,
                        imap_qk_matching_target=imap_qk_matching_target,
                    )
                noise_pred = transformer_output[0]
                current_saliency_map_dict = transformer_output[1]
                for key in current_saliency_map_dict:
                    if key not in saliency_map_dict:
                        saliency_map_dict[key] = []
                    saliency_map_dict[key].append(current_saliency_map_dict[key])
                noise_pred = noise_pred.float()
                
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_attention_mask=negative_prompt_attention_mask,
                            pooled_projections=negative_pooled_prompt_embeds,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Generic processing for all collected attention-like maps
        # 1) stack per-step lists
        for k, v in list(saliency_map_dict.items()):
            if k.startswith("_"):
                continue
            if isinstance(v, list):
                if len(v) == 0:
                    continue
                saliency_map_dict[k] = torch.stack(v, dim=0)
        # 2) Compute reshape parameters
        grid_height = 30
        grid_width = 45
        attn_frames = 13
        # 3) For each key, select requested timesteps, optional softmax, reshape and reduce
        for k, tensor in list(saliency_map_dict.items()):
            if k.startswith("_"):
                continue
            if not torch.is_tensor(tensor) or tensor.ndim == 0:
                continue
            # Select timesteps of interest
            device_sel = tensor.device
            idx_tensor = torch.tensor(saliency_map_kwargs.get("timesteps", []), dtype=torch.long, device=device_sel)
            if idx_tensor.numel() == 0:
                # If none requested, keep all
                sel = tensor
            else:
                sel = tensor.index_select(0, idx_tensor)

            if not saliency_map_kwargs.get("except_softmax", False) and sel.shape[-2] > 1:
                sel = torch.nn.functional.softmax(sel, dim=-2)
            elif not saliency_map_kwargs.get("except_softmax", False) and sel.shape[-2] == 1:
                sel = (sel - sel.min()) / (sel.max() - sel.min() + 1e-5)
            # Reshape patches -> (frames, H, W)
            sel = einops.rearrange(
                sel,
                "steps concepts (frames height width) -> steps concepts frames height width",
                frames=attn_frames,
                width=grid_width,
                height=grid_height,
            )
            # Reduce over steps
            sel = einops.reduce(
                sel,
                "steps concepts frames width height -> concepts frames width height",
                reduction="mean",
            )
            saliency_map_dict[k] = sel

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video), saliency_map_dict
    

class RenoiseHunyuanVideoPipeline(HunyuanVideoPipeline):

    def encode_concepts(
        self,
        concepts: list[str],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        average_until_eos: bool = False,
    ):
        concept_embeds, concept_mask = self._get_llama_prompt_embeds(
            prompt=concepts,
            prompt_template=prompt_template,
            device=device,
            dtype=dtype,
            max_sequence_length=16,
        )

        concept_embeds = concept_embeds[:, 0, :].unsqueeze(0) # 1, len(concepts), 4096
        concept_embeds = torch.cat([concept_embeds, torch.zeros(1, max_sequence_length - len(concepts), concept_embeds.shape[-1], device=concept_embeds.device, dtype=concept_embeds.dtype)], dim=1)
        
        concept_mask = concept_mask[:, 0].unsqueeze(0)  # 1, len(concepts)
        concept_mask = torch.cat([concept_mask, torch.zeros(1, max_sequence_length - len(concepts), device=concept_mask.device, dtype=concept_mask.dtype)], dim=1)

        prompt = " ".join(concepts)
        pooled_concept_embeds = self._get_clip_prompt_embeds(
            prompt=prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=77,
        )
        pooled_concept_embeds = torch.zeros_like(pooled_concept_embeds).to(pooled_concept_embeds.device)

        return concept_embeds, pooled_concept_embeds, concept_mask

    @torch.no_grad()
    def __call__(
        self,
        encoded_video: torch.FloatTensor = None, 
        renoise_timestep: Union[int, List[int]] = None,
        test_full_denoise: bool = False,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Union[str, List[str]] = None,
        concepts: Optional[List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        saliency_map_kwargs: Optional[Dict[str, Any]] = None,
        imap_layer: Optional[list[int]] = None,
        imap_sep_score: Optional[str] = None,
        imap_sep_topk: Optional[int] = None,
        imap_qk_matching_target: Optional[str] = None,
    ):

        if saliency_map_kwargs is None:
            saliency_map_kwargs = {}
        if "concepts" not in saliency_map_kwargs:
            saliency_map_kwargs["concepts"] = concepts

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
        )

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        transformer_dtype = self.transformer.dtype
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        ############# Encode the concepts #############
        concept_embeds, pooled_concept_embeds, concept_mask = self.encode_concepts(
            concepts=concepts,
            prompt_template=prompt_template,
            device=device,
            dtype=transformer_dtype,
            max_sequence_length=max_sequence_length,
        )
        concept_embeds = concept_embeds.to(transformer_dtype)
        concept_mask = concept_mask.to(transformer_dtype)
        pooled_concept_embeds = pooled_concept_embeds.to(transformer_dtype)
        ###############################################

        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prompt_attention_mask=negative_prompt_attention_mask,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(transformer_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare guidance condition
        guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        saliency_map_dict = {
            "cross_attention_maps": [],
        }

        # --- Renoise/Denoise control ---
        # Normalize renoise_timestep to a list of indices for convenience and keep only valid ones
        assert renoise_timestep is not None, "Please provide renoise_timestep (int or list[int])"
        if isinstance(renoise_timestep, (list, tuple)):
            raw_indices = [int(x) for x in renoise_timestep]
        else:
            raw_indices = [int(renoise_timestep)]
        indices = [idx for idx in raw_indices if 0 <= idx < len(timesteps)]
        if len(indices) == 0:
            raise ValueError(
                f"No valid renoise_timestep indices among {raw_indices}; valid range is [0, {len(timesteps)-1}]"
            )

        if test_full_denoise:
            # Full resume: start at single index (or first in list) and run to the end
            start_idx = indices[0]
            renoise_t = timesteps[start_idx]
            # Prepare per-sample timestep tensor matching scheduler.set_timesteps output
            renoise_tensor = renoise_t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
            latents = self.scheduler.scale_noise(
                encoded_video.to(device=latents.device, dtype=latents.dtype),
                renoise_tensor,
                latents,
            )
            run_timesteps = timesteps[start_idx:]

            with self.progress_bar(total=len(run_timesteps)) as progress_bar:
                for i, t in enumerate(run_timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    latent_model_input = latents.to(transformer_dtype)
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    ca_kwargs = dict(saliency_map_kwargs or {})
                    # keep global index for logging/analysis
                    ca_kwargs["timestep_index"] = int(start_idx) + int(i)
                    with self.transformer.cache_context("cond"):
                        transformer_output = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            pooled_projections=pooled_prompt_embeds,
                            concept_hidden_states=concept_embeds,
                            concept_mask=concept_mask,
                            pooled_concept_projections=pooled_concept_embeds,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            saliency_map_kwargs=ca_kwargs,
                            imap_layer=imap_layer,
                            imap_sep_score=imap_sep_score,
                            imap_sep_topk=imap_sep_topk,
                            imap_qk_matching_target=imap_qk_matching_target,
                        )
                    noise_pred = transformer_output[0]
                    current_saliency_map_dict = transformer_output[1]
                    for key in current_saliency_map_dict:
                        if key not in saliency_map_dict:
                            saliency_map_dict[key] = []
                        saliency_map_dict[key].append(current_saliency_map_dict[key])
                    # record collected global timestep index
                    if "_step_indices" not in saliency_map_dict:
                        saliency_map_dict["_step_indices"] = []
                    saliency_map_dict["_step_indices"].append(int(ca_kwargs["timestep_index"]))
                    noise_pred = noise_pred.float()
                    
                    if do_true_cfg:
                        with self.transformer.cache_context("uncond"):
                            neg_noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_attention_mask=negative_prompt_attention_mask,
                                pooled_projections=negative_pooled_prompt_embeds,
                                guidance=guidance,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        else:
            
            noise_epsilon = latents.clone()
            with self.progress_bar(total=len(indices)) as progress_bar:
                for i, idx in enumerate(indices):
                    t = timesteps[int(idx)]
                    # Per-sample timestep (match scheduler.set_timesteps dtype/device)
                    renoise_tensor = t.expand(noise_epsilon.shape[0]).to(device=noise_epsilon.device, dtype=noise_epsilon.dtype)
                    # x_t = sigma(t) * noise + (1 - sigma) * x0
                    latents = self.scheduler.scale_noise(
                        encoded_video.to(device=noise_epsilon.device, dtype=noise_epsilon.dtype),
                        renoise_tensor,
                        noise_epsilon,
                    )

                    if self.interrupt:
                        continue

                    self._current_timestep = t
                    latent_model_input = latents.to(transformer_dtype)
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    ca_kwargs = dict(saliency_map_kwargs or {})
                    ca_kwargs["timestep_index"] = int(idx)
                    with self.transformer.cache_context("cond"):
                        transformer_output = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            pooled_projections=pooled_prompt_embeds,
                            concept_hidden_states=concept_embeds,
                            concept_mask=concept_mask,
                            pooled_concept_projections=pooled_concept_embeds,
                            guidance=guidance,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            saliency_map_kwargs=ca_kwargs,
                            imap_layer=imap_layer,
                            imap_sep_score=imap_sep_score,
                            imap_sep_topk=imap_sep_topk,
                            imap_qk_matching_target=imap_qk_matching_target,
                        )
                    noise_pred = transformer_output[0]
                    current_saliency_map_dict = transformer_output[1]
                    for key in current_saliency_map_dict:
                        if key not in saliency_map_dict:
                            saliency_map_dict[key] = []
                        saliency_map_dict[key].append(current_saliency_map_dict[key])
                    # record collected global timestep index
                    if "_step_indices" not in saliency_map_dict:
                        saliency_map_dict["_step_indices"] = []
                    saliency_map_dict["_step_indices"].append(int(ca_kwargs["timestep_index"]))
                    noise_pred = noise_pred.float()
                    
                    if do_true_cfg:
                        with self.transformer.cache_context("uncond"):
                            neg_noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                encoder_attention_mask=negative_prompt_attention_mask,
                                pooled_projections=negative_pooled_prompt_embeds,
                                guidance=guidance,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # progress update per index in single-step renoise mode
                    progress_bar.update()


        # Generic processing for all collected attention-like maps
        # 1) stack per-step lists
        for k, v in list(saliency_map_dict.items()):
            if k.startswith("_"):
                continue
            if isinstance(v, list):
                if len(v) == 0:
                    continue
                saliency_map_dict[k] = torch.stack(v, dim=0)

        # Build position selector once
        any_key = next((k for k in saliency_map_dict.keys() if not k.startswith("_")), None)
        if any_key is not None:
            collected_indices = saliency_map_dict.get("_step_indices", list(range(saliency_map_dict[any_key].shape[0])))
        else:
            collected_indices = []

        saliency_map_yesno = [True if i in saliency_map_kwargs.get("timesteps", []) else False for i in collected_indices]
        saliency_map_yes = [idx for idx, yesno in enumerate(saliency_map_yesno) if yesno]

        # 2) Compute reshape parameters
        grid_height = 30
        grid_width = 45
        attn_frames = 13
        # 3) For each key, select requested timesteps, optional softmax, reshape and reduce
        for k, tensor in list(saliency_map_dict.items()):
            if k.startswith("_"):
                continue
            if tensor.ndim == 0:
                continue
            if len(saliency_map_yes) == 0:
                raise ValueError(f"No matching collected indices for the requested timesteps {saliency_map_kwargs.get('timesteps', [])}")
            device_sel = tensor.device
            idx_tensor = torch.tensor(saliency_map_yes, dtype=torch.long, device=device_sel)
            sel = tensor.index_select(0, idx_tensor)
            if not saliency_map_kwargs.get("except_softmax", False) and sel.shape[-2] > 1:
                sel = torch.nn.functional.softmax(sel, dim=-2)
            elif not saliency_map_kwargs.get("except_softmax", False) and sel.shape[-2] == 1:
                sel = (sel - sel.min()) / (sel.max() - sel.min() + 1e-5)
                
            # Reshape patches -> (frames, H, W)
            sel = einops.rearrange(
                sel,
                "steps concepts (frames height width) -> steps concepts frames height width",
                frames=attn_frames,
                width=grid_width,
                height=grid_height,
            )
            # Reduce over steps
            sel = einops.reduce(
                sel,
                "steps concepts frames width height -> concepts frames width height",
                reduction="mean",
            )
            saliency_map_dict[k] = sel

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video), saliency_map_dict