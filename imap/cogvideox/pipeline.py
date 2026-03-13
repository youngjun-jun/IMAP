import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import einops
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers import CogVideoXDPMScheduler
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers import CogVideoXPipeline

class ModifiedCogVideoXPipeline(CogVideoXPipeline):

    def encode_concepts(
        self,
        concepts: list[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seq_length: int = 226,
        average_until_eos: bool = False,
    ):

        device = device or self._execution_device

        concept_embeds = self._get_t5_prompt_embeds(
            prompt=concepts,
            num_videos_per_prompt=1,
            max_sequence_length=16, 
            device=device,
            dtype=dtype,
        )   
        if not average_until_eos:
            concept_embeds = concept_embeds[:, 0, :]
        else:
            tok = self.tokenizer(
                concepts,
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tok.input_ids.to(device)
            attn_mask = tok.attention_mask.to(device)
            eos_id = self.tokenizer.eos_token_id
            L = input_ids.size(1)
            default_end = (attn_mask.sum(dim=-1) - 1).clamp(min=1)
            if eos_id is None:
                end_idx = default_end
            else:
                has_eos = (input_ids == eos_id).any(dim=-1)
                eos_pos = torch.argmax((input_ids == eos_id).to(torch.int64), dim=-1)
                end_idx = torch.where(has_eos, eos_pos, default_end)
                end_idx = end_idx.clamp(min=1)
            positions = torch.arange(L, device=device)[None, :]
            mask = (positions < end_idx[:, None]).to(concept_embeds.dtype)
            mask = mask.unsqueeze(-1) 
            summed = (concept_embeds * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            concept_embeds = summed / counts
        concept_embeds = torch.stack([concept_embeds] * 2)
        concept_embeds = torch.cat([concept_embeds, torch.zeros(2, seq_length - len(concepts), concept_embeds.size(-1), device=device, dtype=dtype)], dim=1)

        return concept_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        concepts: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        saliency_map_kwargs: Optional[Dict[str, Any]] = None,
        imap_layer: Optional[list[int]] = None,
        imap_sep_score: Optional[str] = None,
        imap_sep_topk: Optional[int] = None,
        imap_qk_matching_target: Optional[str] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:

        if saliency_map_kwargs is None:
            saliency_map_kwargs = {}
        if "concepts" not in saliency_map_kwargs:
            saliency_map_kwargs["concepts"] = concepts

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        concept_embeds = self.encode_concepts(
            concepts, 
            device=device, 
            dtype=prompt_embeds.dtype,
            average_until_eos=bool((saliency_map_kwargs or {}).get("concept_avg_until_eos", False)),  
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal
            # Recompute latent_frames after padding num_frames to ensure correct unpatching later
            latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. 
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        saliency_map_dict = {
            "cross_attention_maps": [],
            "_step_indices": [],
        }

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                ca_kwargs = dict(saliency_map_kwargs or {})
                ca_kwargs["timestep_index"] = int(i)
                transformer_output = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    concept_hidden_states=concept_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
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

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

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
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        attn_frames = (
            latent_frames
            if self.transformer.config.patch_size_t is None
            else (latent_frames + self.transformer.config.patch_size_t - 1) // self.transformer.config.patch_size_t
        )
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

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video), saliency_map_dict
    

class RenoiseCogVideoXPipeline(CogVideoXPipeline):

    def encode_concepts(
        self,
        concepts: list[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        seq_length: int = 226,
        average_until_eos: bool = False,
    ):

        device = device or self._execution_device

        concept_embeds = self._get_t5_prompt_embeds(
            prompt=concepts,
            num_videos_per_prompt=1,
            max_sequence_length=16, 
            device=device,
            dtype=dtype,
        )   
        if not average_until_eos:
            concept_embeds = concept_embeds[:, 0, :]
        else:
            tok = self.tokenizer(
                concepts,
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tok.input_ids.to(device)
            attn_mask = tok.attention_mask.to(device)
            eos_id = self.tokenizer.eos_token_id
            L = input_ids.size(1)
            default_end = (attn_mask.sum(dim=-1) - 1).clamp(min=1)
            if eos_id is None:
                end_idx = default_end
            else:
                has_eos = (input_ids == eos_id).any(dim=-1)
                eos_pos = torch.argmax((input_ids == eos_id).to(torch.int64), dim=-1)
                end_idx = torch.where(has_eos, eos_pos, default_end)
                end_idx = end_idx.clamp(min=1)
            positions = torch.arange(L, device=device)[None, :]
            mask = (positions < end_idx[:, None]).to(concept_embeds.dtype)
            mask = mask.unsqueeze(-1) 
            summed = (concept_embeds * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            concept_embeds = summed / counts
        concept_embeds = torch.stack([concept_embeds] * 2)
        concept_embeds = torch.cat([concept_embeds, torch.zeros(2, seq_length - len(concepts), concept_embeds.size(-1), device=device, dtype=dtype)], dim=1)

        return concept_embeds

    @torch.no_grad()
    def __call__(
        self,
        encoded_video: torch.FloatTensor = None,
        renoise_timestep: Union[int, List[int]] = None,
        test_full_denoise: bool = False,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        concepts: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        saliency_map_kwargs: Optional[Dict[str, Any]] = None,
        imap_layer: Optional[list[int]] = None,
        imap_sep_score: Optional[str] = None,
        imap_sep_topk: Optional[int] = None,
        imap_qk_matching_target: Optional[str] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:

        if "concepts" not in saliency_map_kwargs:
            saliency_map_kwargs["concepts"] = concepts

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        assert encoded_video is not None, "Please provide the encoded_video"
        assert encoded_video.shape[-4] == 16, "The encoded video must have 16 channels"
        assert encoded_video.shape[-3] == (num_frames - 1) // self.vae_scale_factor_temporal + 1, f"The encoded video must have {(num_frames - 1) // self.vae_scale_factor_temporal + 1} frames"
        assert encoded_video.shape[-2] == height // self.vae_scale_factor_spatial, f"The encoded video must have {height // self.vae_scale_factor_spatial} height"
        assert encoded_video.shape[-1] == width // self.vae_scale_factor_spatial, f"The encoded video must have {width // self.vae_scale_factor_spatial} width"

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        concept_embeds = self.encode_concepts(
            concepts, 
            device=device, 
            dtype=prompt_embeds.dtype,
            average_until_eos=bool((saliency_map_kwargs or {}).get("concept_avg_until_eos", False)),  
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal
            # Recompute latent_frames after padding num_frames to ensure correct unpatching later
            latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        saliency_map_dict = {
            "concept_attention_maps": [],
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
            renoise_tensor = torch.full((latents.shape[0],), int(renoise_t.item()), device=latents.device, dtype=torch.long)
            latents = self.scheduler.add_noise(
                encoded_video.to(device=latents.device, dtype=latents.dtype).permute(0, 2, 1, 3, 4),
                latents,
                renoise_tensor,
            )
            run_timesteps = timesteps[start_idx:]

            with self.progress_bar(total=len(run_timesteps)) as progress_bar:
                # for DPM-solver++
                old_pred_original_sample = None
                for i, t in enumerate(run_timesteps):

                    if self.interrupt:
                        continue

                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    # predict noise model_output
                    ca_kwargs = dict(saliency_map_kwargs or {})
                    # keep global index for logging/analysis
                    ca_kwargs["timestep_index"] = int(start_idx) + int(i)
                    transformer_output = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        concept_hidden_states=concept_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
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

                    # perform guidance
                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            run_timesteps[i - 1] if i > 0 else None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    latents = latents.to(prompt_embeds.dtype)

                    # call the callback, if provided
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # update once per iteration
                    progress_bar.update()
        else:
            # Single-step mode: for each index, renoise and denoise exactly one step independently
            noise_epsilon = latents.clone()
            with self.progress_bar(total=len(indices)) as progress_bar:
                old_pred_original_sample = None
                for i, idx in enumerate(indices):
                    t = timesteps[int(idx)]
                    renoise_tensor = torch.full((noise_epsilon.shape[0],), int(t.item()), device=noise_epsilon.device, dtype=torch.long)
                    latents = self.scheduler.add_noise(
                        encoded_video.to(device=noise_epsilon.device, dtype=noise_epsilon.dtype).permute(0, 2, 1, 3, 4),
                        noise_epsilon,
                        renoise_tensor,
                    )

                    if self.interrupt:
                        continue

                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    timestep = t.expand(latent_model_input.shape[0])

                    ca_kwargs = dict(saliency_map_kwargs or {})
                    ca_kwargs["timestep_index"] = int(idx)
                    transformer_output = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        concept_hidden_states=concept_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
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
                    if "_step_indices" not in saliency_map_dict:
                        saliency_map_dict["_step_indices"] = []
                    saliency_map_dict["_step_indices"].append(int(ca_kwargs["timestep_index"]))
                    noise_pred = noise_pred.float()

                    if use_dynamic_cfg:
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            None,
                            t,
                            None,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    latents = latents.to(prompt_embeds.dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    progress_bar.update()

        # Process and reshape all collected attention-like maps
        # Determine which step indices to keep based on mode
        # We rely on a common _step_indices list gathered during the denoising loop
        # and select those that match the requested timesteps or used indices.
        # First, ensure all list entries are stacked into tensors
        for k, v in list(saliency_map_dict.items()):
            if k.startswith("_"):
                continue
            if isinstance(v, list):
                saliency_map_dict[k] = torch.stack(v, dim=0)

        # Build position selector once
        any_key = next((k for k in saliency_map_dict.keys() if not k.startswith("_")), None)
        if any_key is not None:
            collected_indices = saliency_map_dict.get("_step_indices", list(range(saliency_map_dict[any_key].shape[0])))
        else:
            collected_indices = []

        saliency_map_yesno = [True if i in saliency_map_kwargs.get("timesteps", []) else False for i in collected_indices]
        saliency_map_yes = [idx for idx, yesno in enumerate(saliency_map_yesno) if yesno]

        # Precompute reshape parameters
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        # Number of temporal tokens produced by patch embedding (accounts for patch_size_t)
        attn_frames = (
            latent_frames
            if self.transformer.config.patch_size_t is None
            else (latent_frames + self.transformer.config.patch_size_t - 1) // self.transformer.config.patch_size_t
        )
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
            # Mean over steps
            sel = einops.reduce(
                sel,
                "steps concepts frames width height -> concepts frames width height",
                reduction="mean",
            )
            saliency_map_dict[k] = sel

        if not output_type == "latent":
            # Discard any padding frames that were added for CogVideoX 1.5
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video), saliency_map_dict