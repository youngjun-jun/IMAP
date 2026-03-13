from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import einops

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

class ModifiedWanPipeline(WanPipeline):

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        concepts: Optional[List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
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
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
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
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        
        
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
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

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        saliency_map_dict = {
            "cross_attention_maps": [],
        }

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    # seq_len: num_num_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                ca_kwargs = dict(saliency_map_kwargs or {})
                ca_kwargs["timestep_index"] = int(i)

                with current_model.cache_context("cond"):
                    transformer_output = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
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

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

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
        grid_width = 52
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
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video), saliency_map_dict


class RenoiseWanPipeline(WanPipeline):
    
    def encode_concepts(
        self,
        concepts: list[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 226,
        average_until_eos: bool = False,
    ):

        device = device or self._execution_device

        concept_embeds = self._get_t5_prompt_embeds(
            prompt=concepts,
            num_videos_per_prompt=1,
            max_sequence_length=16, ### Max 16 tokens for concepts
            device=device,
            dtype=dtype,
        )   
        if not average_until_eos:
            # Use only the first token [C, 16, D] -> [C, D]
            concept_embeds = concept_embeds[:, 0, :]
        else:
            # Average tokens from 0 up to (but not including) eos token
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
            # default end: last non-pad token index (exclude it)
            default_end = (attn_mask.sum(dim=-1) - 1).clamp(min=1)
            if eos_id is None:
                end_idx = default_end
            else:
                has_eos = (input_ids == eos_id).any(dim=-1)
                eos_pos = torch.argmax((input_ids == eos_id).to(torch.int64), dim=-1)
                end_idx = torch.where(has_eos, eos_pos, default_end)
                end_idx = end_idx.clamp(min=1)
            positions = torch.arange(L, device=device)[None, :]
            mask = (positions < end_idx[:, None]).to(concept_embeds.dtype)  # [C, L]
            mask = mask.unsqueeze(-1)  # [C, L, 1]
            summed = (concept_embeds * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            concept_embeds = summed / counts
        # Duplicate the concept embeddings to match the batch size
        concept_embeds = torch.stack([concept_embeds] * 2)
        # Pad to the sequence length
        concept_embeds = torch.cat([concept_embeds, torch.zeros(2, max_sequence_length - len(concepts), concept_embeds.size(-1), device=device, dtype=dtype)], dim=1)

        return concept_embeds

    @torch.no_grad()
    def __call__(
        self,
        encoded_video: torch.FloatTensor = None, # torch.Size([1, 16, 13, 60, 90]), dtype: torch.bfloat16
        renoise_timestep: Union[int, List[int]] = None,
        test_full_denoise: bool = False,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        concepts: Optional[List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
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
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
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
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        
        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
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

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        saliency_map_dict = {
            "cross_attention_maps": [],
        }

        # --- Renoise/Denoise control ---
        # Normalize renoise_timestep to a list of indices for convenience and keep only valid ones
        if encoded_video is None:
            raise ValueError("encoded_video must be provided for renoise mode.")
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
                encoded_video.to(device=latents.device, dtype=latents.dtype),
                latents,
                renoise_tensor,
            )
            run_timesteps = timesteps[start_idx:]

            with self.progress_bar(total=len(run_timesteps)) as progress_bar:
                for i, t in enumerate(run_timesteps):
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    if boundary_timestep is None or t >= boundary_timestep:
                        # wan2.1 or high-noise stage in wan2.2
                        current_model = self.transformer
                        current_guidance_scale = guidance_scale
                    else:
                        # low-noise stage in wan2.2
                        current_model = self.transformer_2
                        current_guidance_scale = guidance_scale_2

                    latent_model_input = latents.to(transformer_dtype)
                    if self.config.expand_timesteps:
                        # seq_len: num_num_frames * latent_height//2 * latent_width//2
                        temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                        # batch_size, seq_len
                        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    else:
                        timestep = t.expand(latents.shape[0])

                    ca_kwargs = dict(saliency_map_kwargs or {})
                    ca_kwargs["timestep_index"] = int(start_idx) + int(i)

                    with current_model.cache_context("cond"):
                        transformer_output = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
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

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # progress update per iteration in resume mode
                    progress_bar.update()
                        
        else:
            # Single-step mode: for each index, renoise and denoise exactly one step independently
            noise_epsilon = latents.clone()
            with self.progress_bar(total=len(indices)) as progress_bar:
                for i, idx in enumerate(indices):
                    t = timesteps[int(idx)]
                    renoise_tensor = torch.full((noise_epsilon.shape[0],), int(t.item()), device=noise_epsilon.device, dtype=torch.long)
                    latents = self.scheduler.add_noise(
                        encoded_video.to(device=noise_epsilon.device, dtype=noise_epsilon.dtype),
                        noise_epsilon,
                        renoise_tensor,
                    )
                    
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    if boundary_timestep is None or t >= boundary_timestep:
                        # wan2.1 or high-noise stage in wan2.2
                        current_model = self.transformer
                        current_guidance_scale = guidance_scale
                    else:
                        # low-noise stage in wan2.2
                        current_model = self.transformer_2
                        current_guidance_scale = guidance_scale_2

                    latent_model_input = latents.to(transformer_dtype)
                    if self.config.expand_timesteps:
                        # seq_len: num_num_frames * latent_height//2 * latent_width//2
                        temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                        # batch_size, seq_len
                        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                    else:
                        timestep = t.expand(latents.shape[0])

                    ca_kwargs = dict(saliency_map_kwargs or {})
                    ca_kwargs["timestep_index"] = int(idx)

                    with current_model.cache_context("cond"):
                        transformer_output = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
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

                    if self.do_classifier_free_guidance:
                        with current_model.cache_context("uncond"):
                            noise_uncond = current_model(
                                hidden_states=latent_model_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                            )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    # progress update per iteration in single-step mode
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
        concept_attention_yesno = [True if i in saliency_map_kwargs.get("timesteps", []) else False for i in collected_indices]
        concept_attention_yes = [idx for idx, yesno in enumerate(concept_attention_yesno) if yesno]                

        # 2) Compute reshape parameters
        grid_height = 30
        grid_width = 52
        attn_frames = 13
        
        # 3) For each key, select requested timesteps, optional softmax, reshape and reduce
        for k, tensor in list(saliency_map_dict.items()):
            if k.startswith("_"):
                continue
            if tensor.ndim == 0:
                continue
            if len(concept_attention_yes) == 0:
                raise ValueError(f"No matching collected indices for the requested timesteps {saliency_map_kwargs.get('timesteps', [])}")
            device_sel = tensor.device
            idx_tensor = torch.tensor(concept_attention_yes, dtype=torch.long, device=device_sel)
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
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, saliency_map_dict)

        return WanPipelineOutput(frames=video), saliency_map_dict