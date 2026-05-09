"""Per-frame inference core for the StreamDiffusion pipeline.

Owns the timestep schedule, noise buffers, seed transitions, VAE
encode/decode, and UNet/scheduler step math. The pipeline holds an
instance as ``self.inference`` and delegates the per-frame denoising
loop to ``predict_x0`` / ``encode_image`` / ``decode_image``.

Lifecycle: ``attach(pipe)`` rebinds the helper to the live pipeline
on each load/swap. The helper holds tensor state directly (alphas,
betas, c_skip, c_out, init_noise, stock_noise, x_t_latent_buffer,
seed-transition fields); other pipeline state (scheduler, vae, unet,
controlnet, prompts, generator, batch shape) is read through the
back-reference.

Compromise (per REFACTOR_PLAN.md): ``_last_seed`` lives on the
pipeline because ``_prepare_runtime_state`` reads it for change
detection — the helper writes through via the back-ref.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)


class InferenceCore:
    """Timestep schedule, noise buffers, and the UNet/VAE step math."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

        # Live pipe back-reference — set via ``attach()``.
        self.pipe: Any = None

        # --- Timestep schedule (rebuilt by ``set_timesteps``) ---
        self.timesteps: torch.Tensor | None = None
        self.sub_timesteps: list = []
        self.sub_timesteps_tensor: torch.Tensor | None = None
        self.c_skip: torch.Tensor | None = None
        self.c_out: torch.Tensor | None = None
        self.alpha_prod_t_sqrt: torch.Tensor | None = None
        self.beta_prod_t_sqrt: torch.Tensor | None = None

        # --- Noise buffers ---
        self.init_noise: torch.Tensor | None = None
        self.stock_noise: torch.Tensor | None = None
        self.x_t_latent_buffer: torch.Tensor | None = None

        # --- Seed transition state ---
        # When seed_transition_steps > 0, ``init_noise`` is slerped from the
        # source seed's tensor toward the target across N frames instead of
        # hard-swapping. SDXL-Turbo / DMD2-1step have weaker stock_noise
        # feedback than SD-Turbo, so without this seed changes read as cuts.
        self._seed_transition_source: torch.Tensor | None = None
        self._seed_transition_target: torch.Tensor | None = None
        self._seed_transition_progress: int = 0
        self._seed_transition_total: int = 0

    def attach(self, pipe: Any) -> None:
        """Rebind to the live pipeline. Called from ``_ensure_pipe_loaded``
        and ``ModelLoader.swap`` after the diffusers pipe is loaded.
        """
        self.pipe = pipe

    def reset_buffers(self) -> None:
        """Drop noise buffers + seed transition. Called on model swap."""
        self.init_noise = None
        self.stock_noise = None
        self.x_t_latent_buffer = None
        self.cancel_seed_transition()

    # ------------------------------------------------------------------
    # Timestep schedule
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int, strength: float) -> None:
        """Build the LCM timestep schedule + per-step alpha/beta/c_skip/c_out.

        Honors ``pipe._timesteps_override`` when present. Distilled 1-step
        models (DMD2, Hyper-SD, Lightning) are trained at a specific
        timestep and produce garbage at any other one — letting LCMScheduler
        pick the default would feed them ~t=979 (near max noise) where they
        were never trained.
        """
        p = self.pipe
        scheduler = p.scheduler
        t_list = p.t_list
        repeats = p.frame_bff_size if p.use_denoising_batch else 1

        if p._timesteps_override is not None:
            scheduler.set_timesteps(num_inference_steps, self.device, strength=strength)
            self.timesteps = torch.tensor(
                p._timesteps_override, device=self.device, dtype=torch.long
            )
        else:
            scheduler.set_timesteps(num_inference_steps, self.device, strength=strength)
            self.timesteps = scheduler.timesteps.to(self.device)

        self.sub_timesteps = [self.timesteps[t] for t in t_list]

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor, repeats=repeats, dim=0
        )

        c_skip_list, c_out_list = [], []
        for timestep in self.sub_timesteps:
            c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        alpha_list, beta_list = [], []
        ac = scheduler.alphas_cumprod
        last_idx = len(ac) - 1
        for timestep in self.sub_timesteps:
            # Clamp into range — skipping would break the downstream
            # ``.view(len(t_list), 1, 1, 1)`` reshape if a timestep happened
            # to land out of range for this scheduler.
            idx = min(int(timestep), last_idx)
            alpha_list.append(ac[idx].sqrt())
            beta_list.append((1 - ac[idx]).sqrt())

        alpha_prod_t_sqrt = (
            torch.stack(alpha_list)
            .view(len(t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_list)
            .view(len(t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt, repeats=repeats, dim=0
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt, repeats=repeats, dim=0
        )

    # ------------------------------------------------------------------
    # Noise buffers + seed transitions
    # ------------------------------------------------------------------

    def initialize_noise(self) -> None:
        """Allocate the seeded ``init_noise`` and matching ``stock_noise``."""
        p = self.pipe
        self.init_noise = torch.randn(
            (p.batch_size, 4, p.latent_height, p.latent_width),
            generator=p.generator,
            device=self.device,
            dtype=self.dtype,
        )
        self.stock_noise = torch.zeros_like(self.init_noise)

    def setup_seed_transition(self, new_seed: int, total_steps: int) -> None:
        """Begin a multi-frame slerp from current ``init_noise`` to a new seed.

        Falls back to a hard cut (re-seed + regenerate immediately) when
        ``total_steps <= 0`` or no prior ``init_noise`` exists. The first
        frame after this runs at the source noise; subsequent frames slerp
        toward the target via :meth:`advance_seed_transition`.
        """
        p = self.pipe
        self.cancel_seed_transition()
        if total_steps <= 0 or self.init_noise is None:
            p.generator.manual_seed(new_seed)
            p._last_seed = new_seed
            self.x_t_latent_buffer = None
            self.initialize_noise()
            return

        self._seed_transition_source = self.init_noise.detach().clone()
        p.generator.manual_seed(new_seed)
        self._seed_transition_target = torch.randn(
            self.init_noise.shape,
            generator=p.generator,
            device=self.device,
            dtype=self.dtype,
        )
        self._seed_transition_progress = 0
        self._seed_transition_total = total_steps
        p._last_seed = new_seed
        # Match the hard-cut path's stock_noise reset so the StreamDiffusion
        # feedback term doesn't carry the previous seed's accumulator.
        self.stock_noise = torch.zeros_like(self.init_noise)
        self.x_t_latent_buffer = None

    @staticmethod
    def _slerp_noise(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        """Spherical interpolation between two noise tensors.

        Linear interpolation drops the variance of standard-normal noise to
        ``(1-t)² + t²`` mid-blend (0.5 at t=0.5), which the diffusion model
        renders as washed-out / blurry output. Slerp keeps the result on the
        same hypersphere as the endpoints, preserving variance and producing
        a perceptually smooth crossfade between scenes.
        """
        a_flat = a.flatten().float()
        b_flat = b.flatten().float()
        a_norm = a_flat.norm()
        b_norm = b_flat.norm()
        cos_omega = (a_flat @ b_flat) / (a_norm * b_norm + 1e-8)
        cos_omega = cos_omega.clamp(-1.0, 1.0)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        # Collinear endpoints — degenerate to lerp to avoid divide-by-zero.
        if sin_omega.abs() < 1e-6:
            return torch.lerp(a, b, t)
        w_a = torch.sin((1.0 - t) * omega) / sin_omega
        w_b = torch.sin(t * omega) / sin_omega
        return (w_a * a + w_b * b).to(dtype=a.dtype)

    def advance_seed_transition(self) -> None:
        """Slerp ``init_noise`` one step toward the target. No-op when idle."""
        if self._seed_transition_total <= 0:
            return
        self._seed_transition_progress += 1
        if self._seed_transition_progress >= self._seed_transition_total:
            self.init_noise = self._seed_transition_target.clone()
            self.cancel_seed_transition()
            return
        t = self._seed_transition_progress / self._seed_transition_total
        self.init_noise = self._slerp_noise(
            self._seed_transition_source,
            self._seed_transition_target,
            t,
        )

    def cancel_seed_transition(self) -> None:
        """Drop any in-flight seed transition without snapping init_noise."""
        self._seed_transition_source = None
        self._seed_transition_target = None
        self._seed_transition_progress = 0
        self._seed_transition_total = 0

    # ------------------------------------------------------------------
    # VAE encode / decode + noise injection
    # ------------------------------------------------------------------

    def encode_image(
        self, image_tensors: torch.Tensor, add_noise: bool = True
    ) -> torch.Tensor:
        """Encode image to latent space."""
        vae = self.pipe.vae
        # Convert from [0, 1] to [-1, 1] range as expected by VAE
        image_tensors = image_tensors * 2.0 - 1.0
        image_tensors = image_tensors.to(device=self.device, dtype=vae.dtype)
        img_latent = retrieve_latents(vae.encode(image_tensors), None)
        img_latent = img_latent * vae.config.scaling_factor
        if add_noise:
            img_latent = self.add_noise(
                img_latent, self.init_noise[0], 0, strength=1.0
            )
        return img_latent

    def decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        vae = self.pipe.vae
        return vae.decode(x_0_pred_out / vae.config.scaling_factor, return_dict=False)[0]

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
        strength: Optional[float] = None,
    ) -> torch.Tensor:
        """Add noise to samples."""
        if strength is None:
            strength = self.pipe.strength
        return self.alpha_prod_t_sqrt[t_index] * original_samples + (
            self.beta_prod_t_sqrt[t_index] * noise * strength
        )

    # ------------------------------------------------------------------
    # UNet + scheduler step
    # ------------------------------------------------------------------

    def _scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Perform a batch step in the scheduler."""
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            return self.c_out * F_theta + self.c_skip * x_t_latent_batch
        F_theta = (
            x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
        ) / self.alpha_prod_t_sqrt[idx]
        return self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch

    def _unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, List[int]],
        added_cond_kwargs,
        idx: Optional[int] = None,
    ):
        """Perform a single UNet denoising step."""
        p = self.pipe
        if p.guidance_scale > 1.0 and (p.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif p.guidance_scale > 1.0 and (p.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        # Compute ControlNet residuals if conditioning is available. Same
        # signature for eager and TRT ControlNet adapters.
        down_block_res_samples = None
        mid_block_res_sample = None
        if p.controlnet is not None and p.controlnet_input is not None:
            batch_size = x_t_latent_plus_uc.shape[0]
            cond_image = p.controlnet_input.expand(batch_size, -1, -1, -1)
            down_block_res_samples, mid_block_res_sample = p.controlnet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=p.prompts.prompt_embeds,
                controlnet_cond=cond_image,
                conditioning_scale=p.controlnet_conditioning_scale,
                return_dict=False,
            )

        model_pred = p.unet(
            x_t_latent_plus_uc,
            t_list,
            encoder_hidden_states=p.prompts.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        if p.use_denoising_batch:
            denoised_batch = self._scheduler_step_batch(model_pred, x_t_latent, idx)
            if p.cfg_type == "self" or p.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self._scheduler_step_batch(model_pred, scaled_noise, idx)
                alpha_next = torch.concat(
                    [
                        self.alpha_prod_t_sqrt[1:],
                        torch.ones_like(self.alpha_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = alpha_next * delta_x
                beta_next = torch.concat(
                    [
                        self.beta_prod_t_sqrt[1:],
                        torch.ones_like(self.beta_prod_t_sqrt[0:1]),
                    ],
                    dim=0,
                )
                delta_x = delta_x / beta_next
                init_noise = torch.concat(
                    [self.init_noise[1:], self.init_noise[0:1]], dim=0
                )
                self.stock_noise = init_noise + delta_x
        else:
            denoised_batch = self._scheduler_step_batch(model_pred, x_t_latent, idx)

        return denoised_batch, model_pred

    def predict_x0(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        """Predict denoised latent from noisy latent."""
        p = self.pipe
        added_cond_kwargs = {}

        if p.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if p.sdxl:
                batch = x_t_latent.shape[0]
                te = p.prompts.add_text_embeds.to(self.device)
                ti = p.prompts.add_time_ids.to(self.device)
                if te.shape[0] != batch:
                    te = te[:1].expand(batch, -1)
                if ti.shape[0] != batch:
                    ti = ti[:1].expand(batch, -1)
                added_cond_kwargs = {"text_embeds": te, "time_ids": ti}

            x_t_latent = x_t_latent.to(self.device)
            t_list = t_list.to(self.device)
            x_0_pred_batch, _ = self._unet_step(
                x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs
            )
            x_0_pred_out = x_0_pred_batch
            self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(1).repeat(p.frame_bff_size)
                if p.sdxl:
                    added_cond_kwargs = {
                        "text_embeds": p.prompts.add_text_embeds.to(self.device),
                        "time_ids": p.prompts.add_time_ids.to(self.device),
                    }
                x_0_pred, _ = self._unet_step(
                    x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs
                )
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if p.do_add_noise:
                        x_t_latent = self.alpha_prod_t_sqrt[
                            idx + 1
                        ] * x_0_pred + self.beta_prod_t_sqrt[
                            idx + 1
                        ] * torch.randn_like(
                            x_0_pred, device=self.device, dtype=self.dtype
                        )
                    else:
                        x_t_latent = self.alpha_prod_t_sqrt[idx + 1] * x_0_pred
            x_0_pred_out = x_0_pred

        return x_0_pred_out
