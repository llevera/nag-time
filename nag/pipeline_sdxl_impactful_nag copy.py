from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate, is_torch_xla_available
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    retrieve_timesteps,
    rescale_noise_cfg,
)

from .pipeline_sdxl_nag import NAGStableDiffusionXLPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class ImpactfulNAGStableDiffusionXLPipeline(NAGStableDiffusionXLPipeline):
    """
    NAG with timing from 'Impact of Negative Prompts':
    - Delay enabling NAG until nag_start fraction of the trajectory
    - Optional linear ramp of nag_scale for nag_ramp_steps after start
    - Keep existing nag_end behaviour

    Backwards-compatible cool-down (time-based):
    - Optional post-NAG cool-down that temporarily reduces CFG by a fraction.
    - Disabled by default (nag_cooldown=0.0) to preserve existing behaviour.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],

        # NAG controls
        nag_scale: float = 1.0,
        nag_tau: float = 2.5,
        nag_alpha: float = 0.5,
        nag_negative_prompt: str = None,
        nag_negative_prompt_embeds: Optional[torch.Tensor] = None,
        nag_end: float = 1.0,

        # Timing knobs from Impact of Negative Prompts
        nag_start: float = 0.2,      # start NAG after 20% of the trajectory
        nag_ramp_steps: int = 0,     # optional soft-start; 0 disables ramp

        # time-based cool-down (backward compatible when 0.0)
        nag_cooldown: float = 0.0,          # fraction after nag_end for cool-down (0.0 disables)
        nag_cooldown_cfg_drop: float = 0.2, # relative CFG drop in cool-down (e.g., 0.2 = -20%)

        **kwargs,
    ):
        # ---- legacy callback kwargs deprecation ----
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` to `__call__` is deprecated. Use `callback_on_step_end`.",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` to `__call__` is deprecated. Use `callback_on_step_end`.",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # ---- defaults ----
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # ---- inputs check ----
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False
        self._nag_scale = nag_scale

        # ---- batch size ----
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # ---- encode prompts ----
        lora_scale = self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        (
            prompt_embeds,
            neg_prompt_embeds,
            pooled_prompt_embeds,
            neg_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance or self.do_normalized_attention_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # Prepare NAG negative embeddings (NOT appended yet; we append only when NAG actually turns on AND CFG is on)
        if self.do_normalized_attention_guidance:
            if nag_negative_prompt_embeds is None:
                if nag_negative_prompt is None:
                    if negative_prompt is not None:
                        if self.do_classifier_free_guidance:
                            nag_negative_prompt_embeds = neg_prompt_embeds
                        else:
                            nag_negative_prompt = negative_prompt
                    else:
                        nag_negative_prompt = ""
                if nag_negative_prompt is not None and nag_negative_prompt_embeds is None:
                    nag_negative_prompt_embeds = self.encode_prompt(
                        prompt=nag_negative_prompt,
                        device=device,
                        num_images_per_prompt=num_images_per_prompt,
                        do_classifier_free_guidance=False,
                        lora_scale=lora_scale,
                        clip_skip=self.clip_skip,
                    )[0]

        # ---- timesteps ----
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # ---- latents ----
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # ---- extra step kwargs ----
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ---- added time ids & embeds ----
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        # Base CFG packing: only when CFG is actually on
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([neg_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        # DO NOT append NAG negative branch yet (we’ll do it at nag_start if CFG is on)
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # IP-Adapter
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # ---- denoising setup ----
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and 0 < self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len([ts for ts in timesteps if ts >= discrete_timestep_cutoff])
            timesteps = timesteps[:num_inference_steps]

        # ---- guidance scale embedding (base/default) ----
        base_timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            base_gs_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            base_timestep_cond = self.get_guidance_scale_embedding(
                base_gs_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # ---- timing helpers ----
        def _to_ddpm(frac: float) -> int:
            frac = max(0.0, min(1.0, float(frac)))
            return math.floor((1 - frac) * 999)

        nag_start_t = _to_ddpm(nag_start)
        nag_end_t = _to_ddpm(nag_end)

        cooldown_active = float(nag_cooldown) > 0.0
        if cooldown_active:
            cooldown_end_frac = min(1.0, float(nag_end) + float(nag_cooldown))
            cooldown_end_t = _to_ddpm(cooldown_end_frac)
        else:
            cooldown_end_t = None

        origin_attn_procs = self.unet.attn_processors
        attn_procs_applied = False
        attn_procs_recovered = False
        i_start = None

        # ---- loop ----
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Effective CFG state for this step (dup count matches encoder batch)
                do_cfg_for_batch = self.do_classifier_free_guidance
                branch_count = 2 if do_cfg_for_batch else 1
                latent_model_input = torch.cat([latents] * branch_count) if do_cfg_for_batch else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Turn NAG on at nag_start (and optionally ramp)
                if (
                    self.do_normalized_attention_guidance
                    and not attn_procs_applied
                    and t <= nag_start_t
                ):
                    current_scale = 1.0 if nag_ramp_steps > 0 else nag_scale
                    self._set_nag_attn_processor(current_scale, nag_tau, nag_alpha)

                    # Append NAG negative branch ONLY if CFG is on,
                    # so encoder_hidden_states batch matches latent duplication.
                    if do_cfg_for_batch and (nag_negative_prompt_embeds is not None):
                        prompt_embeds = torch.cat([prompt_embeds, nag_negative_prompt_embeds], dim=0)

                    attn_procs_applied = True
                    i_start = i

                # Optional ramp of nag_scale
                if (
                    self.do_normalized_attention_guidance
                    and attn_procs_applied
                    and not attn_procs_recovered
                    and nag_ramp_steps > 0
                    and i_start is not None
                ):
                    steps_since_on = max(0, i - i_start)
                    if steps_since_on <= nag_ramp_steps:
                        # safe denominator: max(1, nag_ramp_steps) avoids div-by-zero
                        ramped = 1.0 + (nag_scale - 1.0) * (steps_since_on / max(1, nag_ramp_steps))
                        self._set_nag_attn_processor(ramped, nag_tau, nag_alpha)
                    else:
                        if abs(self._nag_scale - nag_scale) > 1e-6:
                            self._set_nag_attn_processor(nag_scale, nag_tau, nag_alpha)

                # Turn NAG off after nag_end and restore shapes
                if (
                    self.do_normalized_attention_guidance
                    and attn_procs_applied
                    and not attn_procs_recovered
                    and t < nag_end_t
                ):
                    self.unet.set_attn_processor(origin_attn_procs)
                    # slice prompt_embeds back to match latent_model_input size
                    prompt_embeds = prompt_embeds[: len(latent_model_input)]
                    attn_procs_recovered = True

                # Time-based cool-down of effective CFG
                if cooldown_active:
                    in_cooldown = (t <= nag_end_t) and (t >= cooldown_end_t)
                else:
                    in_cooldown = False

                if in_cooldown:
                    current_guidance_scale = max(1.0, self.guidance_scale * (1.0 - float(nag_cooldown_cfg_drop)))
                else:
                    current_guidance_scale = self.guidance_scale

                # timestep_cond: reuse base embedding unless cool-down changes CFG
                if self.unet.config.time_cond_proj_dim is not None:
                    if cooldown_active and (current_guidance_scale != self.guidance_scale):
                        gs_tensor = torch.tensor(current_guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                        timestep_cond = self.get_guidance_scale_embedding(
                            gs_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                        ).to(device=device, dtype=latents.dtype)
                    else:
                        timestep_cond = base_timestep_cond
                else:
                    timestep_cond = None

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # Standard 2-branch CFG math when CFG is enabled
                if do_cfg_for_batch:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                # callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", neg_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", neg_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and callback_steps is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        # ---- decode ----
        if output_type != "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype and torch.backends.mps.is_available():
                self.vae = self.vae.to(latents.dtype)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if output_type != "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        # ensure processors are restored in any branch
        if self.do_normalized_attention_guidance and not attn_procs_recovered:
            self.unet.set_attn_processor(origin_attn_procs)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return StableDiffusionXLPipelineOutput(images=image)
