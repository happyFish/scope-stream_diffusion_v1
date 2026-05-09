"""StreamDiffusion pipeline implementation for Scope."""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import torch
import numpy as np
import PIL.Image
from diffusers import (
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from scope.core.pipelines.interface import Pipeline, Requirements

from .controlnet import ControlNetHandler
from .model_loader import MODEL_PRESETS, ModelLoader
from .prompt_encoder import PromptEncoder, normalize_prompts
from .schema import StreamDiffusionConfig
from .trt_lifecycle import TRTLifecycle

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


# Import or inline the helper utilities
class SimilarImageFilter:
    """Simple similar image filter implementation."""

    def __init__(self):
        self.threshold = 0.98
        self.max_skip_frame = 10
        self.skip_count = 0

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def set_max_skip_frame(self, max_skip_frame: int):
        self.max_skip_frame = max_skip_frame

    def __call__(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        # Simplified - always return the image
        # TODO: Implement actual similarity checking
        return image_tensor


class StreamDiffusionPipeline(Pipeline):
    """StreamDiffusion pipeline for real-time Stable Diffusion generation."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        """Return the configuration class for this pipeline."""
        return StreamDiffusionConfig

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_id: Optional[str] = None,
        model_id_or_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Initialize the StreamDiffusion pipeline.

        Args:
            device: Torch device to use
            model_id / model_id_or_path: Model ID or path to load. The schema
                field is ``model_id_or_path``; ``model_id`` is accepted as an
                alias so older callers keep working.
            torch_dtype: Data type for tensors
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = torch_dtype

        # Store config if Scope passes it
        self.config = kwargs.get("config") or kwargs.get("pipeline_config")
        print(f"Init - Config object: {self.config}")

        # The schema's field is ``model_id_or_path``. Scope's pipeline_manager
        # merges schema defaults into the init kwargs by their declared name,
        # so what we see at __init__ is the *schema default*, not the user's
        # UI selection — that only arrives via runtime kwargs/config on the
        # first __call__. To avoid a spurious "load SD-Turbo, then immediately
        # swap to the user's pick" on every startup, defer the actual model
        # load to ``_ensure_pipe_loaded`` (called from __call__ once we have
        # the runtime selection). The init-time arg is just a tentative
        # default in case nothing more authoritative shows up at runtime.
        config_model = getattr(self.config, "model_id_or_path", None) if self.config else None
        model_id = model_id or config_model or model_id_or_path or "stabilityai/sd-turbo"
        self.model_id = model_id
        preset = MODEL_PRESETS.get(model_id, {})
        self._timesteps_override = preset.get("timesteps_override")
        # CFG-distilled models (DMD2, future Hyper-SD / Lightning) explode in
        # the implicit txt2img→img2img loopback because each iteration re-
        # applies the model's baked-in guidance shaping. Default True for
        # everything else (SD-Turbo, SDXL-Turbo) since the iterative
        # refinement is what gives those models their polished t2i look.
        self._implicit_loopback: bool = preset.get("implicit_loopback", True)
        print(f"[StreamDiffusion] Tentative model: {model_id} (load deferred to first __call__)")

        # Model-dependent attrs are populated by ``_ensure_pipe_loaded``.
        self.pipe = None
        self.sdxl: bool = False
        self.text_encoder = None
        self.unet = None
        self.vae = None
        self._full_vae = None  # populated on load
        self._taesd_vae = None
        self._using_taesd = False
        self.scheduler = None
        self.image_processor = None

        # legacy torch.compile flag — kept so other code paths that read
        # `_unet_compiled` (e.g. TRTLifecycle._ensure_unet's "restore eager" branch)
        # continue to work. compile_unet schema field is gone in Phase 5.
        self._unet_compiled: bool = False

        # Read acceleration_mode at init from schema defaults / load_params.
        # The runtime kwargs path is unreliable because moth's 30fps param flood
        # overflows scope's parameter_queue (maxsize=8) and most updates drop.
        # An init-time read is deterministic.
        acceleration_mode = kwargs.get("acceleration_mode", "none")
        if acceleration_mode not in ("none", "trt"):
            acceleration_mode = "none"
        if acceleration_mode == "trt":
            print(f"[TRT] acceleration_mode='trt' detected at init")

        # Identify this pipeline instance for the cross-instance TRT adapter
        # cache. node_id is the user-supplied graph node id from Scope and is
        # stable across graph edits; until upstream Scope passes it through,
        # the helper falls back to a model-scoped anon key (correct for a
        # single SD node, the common case).
        node_id: str | None = kwargs.get("node_id")

        # TRT engine lifecycle (build / restore / swap / teardown of UNet,
        # ControlNet, and TAESD adapters; runtime acceleration-mode toggle).
        # ``attach()`` wires the helper to the live pipe at load time and on
        # every model swap.
        self.trt = TRTLifecycle(
            self.device,
            self.dtype,
            node_id=node_id,
            acceleration_mode=acceleration_mode,
        )

        # Scheduler / image_processor are model-dependent — populated by
        # ``_ensure_pipe_loaded`` on the first __call__.

        # Prompt encoding (text-encode, blending, transitions, negative
        # subtraction) lives on its own helper. ``attach()`` wires it to
        # the live pipe at load time and on every model swap. Inference
        # reads ``self.prompts.prompt_embeds`` / ``add_text_embeds`` /
        # ``add_time_ids`` directly.
        self.prompts = PromptEncoder(self.device, self.dtype)

        # Model loading lifecycle (HF/preset load, SDXL fp16 VAE swap,
        # TAESD swap, LoRA stubs, GPU teardown). Single back-reference set
        # via ``attach()``; subsequent ``swap()`` calls reuse it.
        self.model_loader = ModelLoader(self.device, self.dtype)
        self.model_loader.attach(self)

        # State that will be set during runtime
        self.generator = torch.Generator(device=self.device)
        self.similar_filter = SimilarImageFilter()
        self.prev_image_result = None
        self.inference_time_ema = 0

        # ControlNet support
        self._cn = ControlNetHandler(self.device, self.dtype)
        self.controlnet = None
        self.controlnet_input = None
        self.controlnet_conditioning_scale = 1.0

        # Runtime state (will be set from kwargs in __call__)
        self.width = 512
        self.height = 512
        self.latent_height = 64
        self.latent_width = 64
        self.frame_bff_size = 1
        self.denoising_steps_num = 1
        self.batch_size = 1
        self.cfg_type = "self"
        self.use_denoising_batch = True
        self.do_add_noise = False
        self.strength = 0.9
        self.guidance_scale = 0.0
        self.delta = 1.0
        self.t_list = [0]
        self.similar_image_filter = False

        # Cache keys for _prepare_runtime_state — None forces full recompute on first call
        self._schedule_key: tuple | None = (
            None  # (num_inference_steps, strength, t_index_list)
        )
        self._last_seed: int | None = None
        self._noise_shape: tuple | None = None  # (batch_size, latent_h, latent_w)

        # Seed transition state — when seed_transition_steps > 0, lerp
        # `init_noise` from the previous seed's tensor to the new seed's
        # tensor over N frames instead of hard-swapping. SDXL-Turbo /
        # DMD2-1step have weaker stock_noise feedback than SD-Turbo, so
        # without this seed changes read as hard cuts.
        self._seed_transition_source: torch.Tensor | None = None
        self._seed_transition_target: torch.Tensor | None = None
        self._seed_transition_progress: int = 0
        self._seed_transition_total: int = 0

        # Mode-transition tracking — detect video↔text switches without a pipeline reload
        self._last_mode: str | None = None

        # TRT setup is deferred along with the model load — engines need
        # ``self.pipe.unet`` to exist. ``_ensure_pipe_loaded`` runs
        # ``self.trt.setup`` immediately after loading when acceleration_mode
        # is 'trt', so the first frame still pays the build cost up-front
        # rather than mid-stream.

        print("StreamDiffusion pipeline initialized (model load deferred)")

    def _ensure_pipe_loaded(self, model_id: str) -> None:
        """Load the diffusion model and populate model-dependent state.

        Called once from the first ``__call__`` with the user's actual
        ``model_id_or_path`` from runtime kwargs/config. Doing the load here
        instead of in ``__init__`` avoids a wasted "load schema default,
        immediately swap to user's pick" cycle, since Scope's
        pipeline_manager only forwards schema defaults at __init__ time.
        Subsequent runtime model changes go through ``_swap_model``.

        Helper attach order: PromptEncoder, ControlNetHandler, TRTLifecycle.
        TRT runs last because its engine builds need ``self.pipe.unet`` and
        the freshly-loaded text-encoder / VAE state to be in place.
        """
        if self.pipe is not None:
            return
        print(f"[StreamDiffusion] Loading model: {model_id}")
        self.model_id = model_id
        preset = MODEL_PRESETS.get(model_id, {})
        self._timesteps_override = preset.get("timesteps_override")
        self._implicit_loopback = preset.get("implicit_loopback", True)
        self.pipe = self.model_loader.load(model_id)
        print(f"[StreamDiffusion] Model loaded: {self.pipe.__class__.__name__}")

        self.sdxl = type(self.pipe) is StableDiffusionXLPipeline
        if self.sdxl and self.dtype == torch.float16:
            self.model_loader.install_sdxl_fp16_vae()

        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self._full_vae = self.vae
        self._using_taesd = False
        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.image_processor = VaeImageProcessor(self.pipe.vae_scale_factor)
        self.prompts.attach(self.pipe, self.sdxl)
        self.trt.attach(self, self.sdxl)

        if self.trt.acceleration_mode == "trt":
            self.trt.setup(**self.trt.setup_args_from_config())

    def _swap_model(self, new_model_id: str) -> None:
        """Thin entry point. Mechanics live in :meth:`ModelLoader.swap`."""
        self.model_loader.swap(new_model_id)

    def prepare(self, **kwargs) -> "Requirements | None":
        """Specify pipeline requirements based on current mode.

        Scope calls this with video=True sentinel when in video mode, and
        without 'video' (or video=None) in text mode. Returns Requirements
        with input_size=1 for video mode, None for text/generator mode.
        """
        from scope.core.pipelines.defaults import prepare_for_mode
        return prepare_for_mode(self.__class__, {}, kwargs, video_input_size=1)

    def _prepare_runtime_state(
        self,
        prompts: list[dict],
        prompt_interpolation_method: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
        delta: float,
        width: int,
        height: int,
        use_denoising_batch: bool,
        do_add_noise: bool,
        transition: Optional[dict] = None,
        transition_steps: int = 0,
        seed_transition_steps: int = 0,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        t_index_list: Optional[List[int]] = None,
    ):
        """Prepare runtime state from parameters.

        Expensive operations (timestep schedule, noise buffers, prompt encoding) are
        gated behind change-detection so they only run when the relevant parameters
        actually change.  On a steady-state stream with fixed parameters only the
        transition/embedding-blender path executes per frame.
        """
        # --- Dimensions ---
        dims_changed = width != self.width or height != self.height
        self.width = width
        self.height = height
        if dims_changed:
            self.latent_height = int(height // self.pipe.vae_scale_factor)
            self.latent_width = int(width // self.pipe.vae_scale_factor)

        # --- Scheduler defaults ---
        # `num_inference_steps` is the user-facing sharpness lever: more steps =
        # sharper detail (SD-Turbo proper is the exception — it's distilled for
        # 1 step). When the caller didn't pin a `t_index_list`, walk every step
        # in the schedule so the UNet sees the full LCM timestep range.
        if t_index_list is None:
            t_index_list = list(range(num_inference_steps))

        # --- Cheap scalar assignments ---
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.delta = delta
        self.cfg_type = cfg_type
        self.use_denoising_batch = use_denoising_batch
        self.do_add_noise = do_add_noise
        self.do_classifier_free_guidance = guidance_scale > 1.0

        # --- Batch size ---
        if t_index_list is None:
            t_index_list = [0]
        self.t_list = t_index_list
        self.denoising_steps_num = len(t_index_list)
        self.frame_bff_size = 1
        self.batch_size = (
            self.denoising_steps_num * self.frame_bff_size
            if use_denoising_batch
            else self.frame_bff_size
        )

        # --- Timestep schedule: only recompute when schedule params change ---
        schedule_key = (num_inference_steps, strength, tuple(t_index_list))
        if schedule_key != self._schedule_key:
            self._set_timesteps(num_inference_steps, strength)
            self._schedule_key = schedule_key

        # --- Seed + noise buffers: only reset when seed or spatial shape changes ---
        noise_shape = (self.batch_size, self.latent_height, self.latent_width)
        seed_changed = seed != self._last_seed
        shape_changed = noise_shape != self._noise_shape or dims_changed

        if shape_changed:
            # Different latent shape can't be lerped against the old buffer;
            # hard-reset and cancel any in-flight seed transition.
            self.generator.manual_seed(seed)
            self._last_seed = seed
            self._cancel_seed_transition()
            self.x_t_latent_buffer = None
            self._initialize_noise()
            self._noise_shape = noise_shape
        elif seed_changed:
            # Hard cut when seed_transition_steps == 0; multi-frame lerp otherwise.
            self._setup_seed_transition(seed, seed_transition_steps)

        # Advance any in-flight seed transition by one frame. No-op when idle.
        self._advance_seed_transition()

        # --- Prompt embeddings & transitions ---
        # All prompt encoding, blending, transition handling, and SDXL aug-
        # conditioning lives in the PromptEncoder helper. After this call,
        # ``self.prompts.prompt_embeds`` (and add_text_embeds / add_time_ids
        # for SDXL) holds the conditioning for this frame.
        self.prompts.encode_for_frame(
            prompts=prompts,
            interpolation_method=prompt_interpolation_method,
            width=width,
            height=height,
            batch_size=self.batch_size,
            transition=transition,
            transition_steps=transition_steps,
        )

    def _set_timesteps(self, num_inference_steps: int, strength: float):
        """Set the timesteps for the diffusion process.

        Honors `MODEL_PRESETS[...]["timesteps_override"]` when present.
        Distilled 1-step models (DMD2, Hyper-SD, Lightning) are trained at
        a specific timestep and produce garbage at any other one — letting
        LCMScheduler pick the default would feed them ~t=979 (near max
        noise) where they were never trained.
        """
        if self._timesteps_override is not None:
            # Pin the override; still call set_timesteps so the scheduler
            # internals (timestep_scaling, etc.) are populated for any
            # downstream lookups.
            self.scheduler.set_timesteps(
                num_inference_steps, self.device, strength=strength
            )
            self.timesteps = torch.tensor(
                self._timesteps_override, device=self.device, dtype=torch.long
            )
        else:
            self.scheduler.set_timesteps(
                num_inference_steps, self.device, strength=strength
            )
            self.timesteps = self.scheduler.timesteps.to(self.device)

        # Make sub timesteps list
        self.sub_timesteps = []
        for t in self.t_list:
            self.sub_timesteps.append(self.timesteps[t])

        sub_timesteps_tensor = torch.tensor(
            self.sub_timesteps, dtype=torch.long, device=self.device
        )
        self.sub_timesteps_tensor = torch.repeat_interleave(
            sub_timesteps_tensor,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

        # Calculate scaling factors
        c_skip_list = []
        c_out_list = []
        for timestep in self.sub_timesteps:
            c_skip, c_out = self.scheduler.get_scalings_for_boundary_condition_discrete(
                timestep
            )
            c_skip_list.append(c_skip)
            c_out_list.append(c_out)

        self.c_skip = (
            torch.stack(c_skip_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.c_out = (
            torch.stack(c_out_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )

        # Calculate alpha/beta values
        alpha_prod_t_sqrt_list = []
        beta_prod_t_sqrt_list = []
        ac = self.scheduler.alphas_cumprod
        last_idx = len(ac) - 1
        for timestep in self.sub_timesteps:
            # Clamp into range instead of skipping — skipping would make the
            # downstream .view(len(t_list), 1, 1, 1) reshape fail when any
            # timestep happened to land out of range for this scheduler.
            idx = min(int(timestep), last_idx)
            alpha_prod_t_sqrt_list.append(ac[idx].sqrt())
            beta_prod_t_sqrt_list.append((1 - ac[idx]).sqrt())

        alpha_prod_t_sqrt = (
            torch.stack(alpha_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        beta_prod_t_sqrt = (
            torch.stack(beta_prod_t_sqrt_list)
            .view(len(self.t_list), 1, 1, 1)
            .to(dtype=self.dtype, device=self.device)
        )
        self.alpha_prod_t_sqrt = torch.repeat_interleave(
            alpha_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )
        self.beta_prod_t_sqrt = torch.repeat_interleave(
            beta_prod_t_sqrt,
            repeats=self.frame_bff_size if self.use_denoising_batch else 1,
            dim=0,
        )

    def _initialize_noise(self):
        """Initialize noise tensors."""
        self.init_noise = torch.randn(
            (self.batch_size, 4, self.latent_height, self.latent_width),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )

        self.stock_noise = torch.zeros_like(self.init_noise)

    def _setup_seed_transition(self, new_seed: int, total_steps: int) -> None:
        """Begin a multi-frame lerp from the current init_noise to the new seed.

        Falls back to a hard cut (re-seed + regenerate immediately) when
        ``total_steps <= 0`` or no prior ``init_noise`` exists. The first
        frame after this runs at the source noise; subsequent frames lerp
        toward the target via :meth:`_advance_seed_transition`.
        """
        self._cancel_seed_transition()
        if total_steps <= 0 or self.init_noise is None:
            self.generator.manual_seed(new_seed)
            self._last_seed = new_seed
            self.x_t_latent_buffer = None
            self._initialize_noise()
            return

        self._seed_transition_source = self.init_noise.detach().clone()
        self.generator.manual_seed(new_seed)
        self._seed_transition_target = torch.randn(
            self.init_noise.shape,
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        self._seed_transition_progress = 0
        self._seed_transition_total = total_steps
        self._last_seed = new_seed
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

    def _advance_seed_transition(self) -> None:
        """Slerp ``init_noise`` one step toward the target. No-op when idle."""
        if self._seed_transition_total <= 0:
            return
        self._seed_transition_progress += 1
        if self._seed_transition_progress >= self._seed_transition_total:
            self.init_noise = self._seed_transition_target.clone()
            self._cancel_seed_transition()
            return
        t = self._seed_transition_progress / self._seed_transition_total
        self.init_noise = self._slerp_noise(
            self._seed_transition_source,
            self._seed_transition_target,
            t,
        )

    def _cancel_seed_transition(self) -> None:
        """Drop any in-flight seed transition without snapping init_noise."""
        self._seed_transition_source = None
        self._seed_transition_target = None
        self._seed_transition_progress = 0
        self._seed_transition_total = 0


    def _encode_image(
        self, image_tensors: torch.Tensor, add_noise: bool = True
    ) -> torch.Tensor:
        """Encode image to latent space."""
        # Convert from [0, 1] to [-1, 1] range as expected by VAE
        image_tensors = image_tensors * 2.0 - 1.0
        image_tensors = image_tensors.to(device=self.device, dtype=self.vae.dtype)
        img_latent = retrieve_latents(self.vae.encode(image_tensors), None)
        img_latent = img_latent * self.vae.config.scaling_factor
        if add_noise:
            img_latent = self._add_noise(
                img_latent, self.init_noise[0], 0, strength=1.0
            )
        return img_latent

    def _decode_image(self, x_0_pred_out: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        output_latent = self.vae.decode(
            x_0_pred_out / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return output_latent

    def _add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        t_index: int,
        strength: float = None,
    ) -> torch.Tensor:
        """Add noise to samples."""
        if strength is None:
            strength = self.strength

        noisy_samples = self.alpha_prod_t_sqrt[t_index] * original_samples + (
            self.beta_prod_t_sqrt[t_index] * noise * strength
        )
        return noisy_samples

    def _scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        x_t_latent_batch: torch.Tensor,
        added_cond_kwargs,  # noqa: ARG002
        idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Perform a batch step in the scheduler."""
        if idx is None:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt * model_pred_batch
            ) / self.alpha_prod_t_sqrt
            denoised_batch = self.c_out * F_theta + self.c_skip * x_t_latent_batch
        else:
            F_theta = (
                x_t_latent_batch - self.beta_prod_t_sqrt[idx] * model_pred_batch
            ) / self.alpha_prod_t_sqrt[idx]
            denoised_batch = (
                self.c_out[idx] * F_theta + self.c_skip[idx] * x_t_latent_batch
            )

        return denoised_batch

    def _unet_step(
        self,
        x_t_latent: torch.Tensor,
        t_list: Union[torch.Tensor, List[int]],
        added_cond_kwargs,
        idx: Optional[int] = None,
    ):
        """Perform a single UNet denoising step."""
        if self.guidance_scale > 1.0 and (self.cfg_type == "initialize"):
            x_t_latent_plus_uc = torch.concat([x_t_latent[0:1], x_t_latent], dim=0)
            t_list = torch.concat([t_list[0:1], t_list], dim=0)
        elif self.guidance_scale > 1.0 and (self.cfg_type == "full"):
            x_t_latent_plus_uc = torch.concat([x_t_latent, x_t_latent], dim=0)
            t_list = torch.concat([t_list, t_list], dim=0)
        else:
            x_t_latent_plus_uc = x_t_latent

        # Compute ControlNet residuals if conditioning is available.
        # This works for all paths — eager ControlNet, TRT ControlNet — they
        # all expose the diffusers ControlNetModel signature.
        down_block_res_samples = None
        mid_block_res_sample = None
        if self.controlnet is not None and self.controlnet_input is not None:
            batch_size = x_t_latent_plus_uc.shape[0]
            cond_image = self.controlnet_input.expand(batch_size, -1, -1, -1)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompts.prompt_embeds,
                controlnet_cond=cond_image,
                conditioning_scale=self.controlnet_conditioning_scale,
                return_dict=False,
            )

        model_pred = self.unet(
            x_t_latent_plus_uc,
            t_list,
            encoder_hidden_states=self.prompts.prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        # Compute denoised sample
        if self.use_denoising_batch:
            denoised_batch = self._scheduler_step_batch(
                model_pred, x_t_latent, added_cond_kwargs, idx
            )
            if self.cfg_type == "self" or self.cfg_type == "initialize":
                scaled_noise = self.beta_prod_t_sqrt * self.stock_noise
                delta_x = self._scheduler_step_batch(
                    model_pred, scaled_noise, added_cond_kwargs, idx
                )
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
            denoised_batch = self._scheduler_step_batch(
                model_pred, x_t_latent, added_cond_kwargs, idx
            )

        return denoised_batch, model_pred

    def _predict_x0_batch(self, x_t_latent: torch.Tensor) -> torch.Tensor:
        """Predict denoised latent from noisy latent."""
        added_cond_kwargs = {}
        prev_latent_batch = self.x_t_latent_buffer

        if self.use_denoising_batch:
            t_list = self.sub_timesteps_tensor
            if self.sdxl:
                batch = x_t_latent.shape[0]
                te = self.prompts.add_text_embeds.to(self.device)
                ti = self.prompts.add_time_ids.to(self.device)
                if te.shape[0] != batch:
                    te = te[:1].expand(batch, -1)
                if ti.shape[0] != batch:
                    ti = ti[:1].expand(batch, -1)
                added_cond_kwargs = {"text_embeds": te, "time_ids": ti}

            x_t_latent = x_t_latent.to(self.device)
            t_list = t_list.to(self.device)
            x_0_pred_batch, _model_pred = self._unet_step(
                x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs
            )

            x_0_pred_out = x_0_pred_batch
            self.x_t_latent_buffer = None
        else:
            self.init_noise = x_t_latent
            for idx, t in enumerate(self.sub_timesteps_tensor):
                t = t.view(
                    1,
                ).repeat(
                    self.frame_bff_size,
                )
                if self.sdxl:
                    added_cond_kwargs = {
                        "text_embeds": self.prompts.add_text_embeds.to(self.device),
                        "time_ids": self.prompts.add_time_ids.to(self.device),
                    }
                x_0_pred, _model_pred = self._unet_step(
                    x_t_latent, t, idx=idx, added_cond_kwargs=added_cond_kwargs
                )
                if idx < len(self.sub_timesteps_tensor) - 1:
                    if self.do_add_noise:
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

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        """Process input video frame(s) and return generated output.

        CRITICAL: All runtime parameters come from kwargs, NOT from __init__!

        Args:
            **kwargs: Runtime parameters including:
                - video?: Optional input video frames as torch tensor (T, H, W, C) normalized to [0, 1]
                - prompt: Generation prompt
                - negative_prompt: Negative prompt
                - guidance_scale: CFG scale
                - num_inference_steps: Number of steps
                - strength: Denoising strength
                - seed: Random seed
                - width: Output width
                - height: Output height
                - ... (all other config parameters)

        Returns:
            dict: {"video": output_tensor} where output_tensor is (T, H, W, C) in [0, 1]
        """
        # Extract parameters - handle Scope's parameter format
        video = kwargs.get("video", None)

        # Bypass: pass input through unchanged when disabled
        enabled = kwargs.get("enabled", True)
        if not enabled:
            if video is None or len(video) == 0:
                return {"video": None}
            frame = video[0]
            while frame.ndim > 3:
                frame = frame.squeeze(0)
            if frame.dtype == torch.uint8:
                frame = frame.float() / 255.0
            return {"video": frame.unsqueeze(0)}

        # Detect video↔text mode transitions and self-trigger a reset so stale
        # ControlNet hidden states, EMA bounds, and prev_image_result don't bleed
        # across mode boundaries.
        from scope.core.pipelines.defaults import resolve_input_mode
        current_mode = resolve_input_mode(kwargs)
        if self._last_mode is not None and self._last_mode != current_mode:
            kwargs = {**kwargs, "init_cache": True}
            self.prev_image_result = None
        self._last_mode = current_mode

        # Extract prompts array from Scope
        prompts = kwargs.get("prompts", [])
        # Normalize to list[dict] format
        prompts = (
            normalize_prompts(prompts)
            if prompts
            else [{"text": "", "weight": 1.0}]
        )

        # Get config instance - Scope should pass this
        # Try different ways Scope might pass config
        config = kwargs.get("config") or kwargs.get("pipeline_config")

        # If no config found, try to get it from the pipeline
        if config is None:
            # Check if we stored it during init
            config = getattr(self, "config", None)

        # Helper to get value from config first, then kwargs, then default
        def get_param(key, default):
            # First check config if available (preferred source)
            if config and hasattr(config, key):
                value = getattr(config, key)
                return value
            # Then check kwargs directly
            if key in kwargs:
                value = kwargs[key]
                return value
            # Finally use default
            return default

        # Resolve the user's model selection from runtime kwargs/config.
        # On the first call the pipe isn't loaded yet — __init__ defers the
        # load specifically so we can pick the *real* selection here instead
        # of the schema default that pipeline_manager hands us at __init__.
        # On subsequent calls a runtime change (UI swap) routes through
        # _swap_model.
        requested_model = get_param("model_id_or_path", None) or self.model_id
        if self.pipe is None:
            self._ensure_pipe_loaded(requested_model)
        elif requested_model and requested_model != self.model_id:
            self._swap_model(requested_model)

        # Extract all parameters with config fallback
        prompt_interpolation_method = get_param("prompt_interpolation_method", "linear")
        guidance_scale = get_param("guidance_scale", 0.0)

        # SD-Turbo and SDXL-Turbo are both 1-step distillations.
        num_inference_steps = 1

        # For img2img with SD Turbo, need higher strength for visible changes
        # 0.5-0.7 = moderate, 0.8-0.95 = heavy transformation
        strength = get_param("strength", 0.9)

        seed = get_param("seed", 42)
        seed_transition_steps = get_param("seed_transition_steps", 0)
        delta = get_param("delta", 1.0)
        width = get_param("width", 512)
        height = get_param("height", 512)
        use_denoising_batch = get_param("use_denoising_batch", True)
        do_add_noise = get_param("do_add_noise", True)
        similar_image_filter_enabled = get_param("similar_image_filter_enabled", False)
        image_loopback = get_param("image_loopback", False)
        negative_prompt = get_param("negative_prompt", "")
        negative_prompt_scale = float(get_param("negative_prompt_scale", 1.0))
        controlnet_mode = get_param("controlnet_mode", "none")
        controlnet_scale = get_param("controlnet_scale", 1.0)
        controlnet_temporal_smoothing = get_param("controlnet_temporal_smoothing", 0.5)
        init_cache = kwargs.get("init_cache", False)
        depth_min = get_param("depth_min", 0)
        depth_max = get_param("depth_max", 12)
        depth_skip_interval = get_param("depth_skip_interval", 3)
        depth_input_size = get_param("depth_input_size", 518)
        depth_temporal_cache = get_param("depth_temporal_cache", True)
        # Default True matches schema. With False the full SD VAE decode runs
        # at ~40 ms/call vs TAESD's ~5 ms. Big perf cliff if the param isn't
        # propagated from moth (e.g. queue-drop or absent from project file).
        use_taesd = get_param("use_taesd", True)
        # acceleration_mode is hot-swappable: the engines themselves can't be
        # rebuilt at runtime, but the module references (self.unet etc.) can
        # flip between TRT adapters and eager modules.
        # ``trt.set_acceleration_mode`` swaps; first 'trt' activation builds
        # (slow), subsequent ones hit the cached adapters (instant).
        requested_mode = get_param("acceleration_mode", self.trt.acceleration_mode)
        if requested_mode != self.trt.acceleration_mode:
            self.trt.set_acceleration_mode(requested_mode)
        acceleration_mode = self.trt.acceleration_mode

        # --- Safeguard: prevent invalid strength / num_inference_steps combos ---
        # LCM scheduler requires: floor(original_steps * strength) >= num_inference_steps
        # original_steps defaults to 50 in the scheduler.
        original_steps = 50
        has_video_input = video is not None and len(video) > 0
        uses_video_for_inference = has_video_input and controlnet_mode == "none"

        if not uses_video_for_inference:
            # Text / image_loopback / controlnet-only: cap strength to a floor
            min_strength = (num_inference_steps + 1) / original_steps
            if strength < min_strength:
                strength = min_strength
        else:
            # Video-to-video: user wants low strength to preserve input — reduce steps instead
            max_steps = max(1, int(original_steps * strength))
            if num_inference_steps > max_steps:
                num_inference_steps = max_steps

        # Toggle TAESD/full VAE based on runtime param
        self.model_loader.set_taesd(use_taesd)

        self._cn.update(
            controlnet_mode,
            video,
            width,
            height,
            controlnet_scale,
            init_cache,
            controlnet_temporal_smoothing,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_skip_interval=depth_skip_interval,
            depth_input_size=depth_input_size,
            depth_temporal_cache=depth_temporal_cache,
        )
        self.controlnet = self._cn.model
        self.controlnet_input = self._cn.input

        # TRT engines are normally built at load time (in __init__ /
        # _swap_model). This guard catches the residual cases where runtime
        # values diverge from what was used at load — e.g. the user changes
        # resolution, toggles controlnet on/off, or flips use_taesd in the
        # UI. Fast no-op when nothing changed.
        if acceleration_mode == "trt":
            sig = (int(height), int(width), controlnet_mode, bool(use_taesd))
            if sig != self.trt.setup_signature:
                self.trt.setup(
                    height=int(height),
                    width=int(width),
                    controlnet_mode=controlnet_mode,
                    use_taesd=bool(use_taesd),
                )

        self.controlnet_conditioning_scale = self._cn.scale
        # Extract transition (explicit transition overrides auto-transition)
        transition = kwargs.get("transition", None)
        transition_steps = get_param("transition_steps", 0)

        # Prepare runtime state
        self._prepare_runtime_state(
            prompts=prompts,
            prompt_interpolation_method=prompt_interpolation_method,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
            delta=delta,
            width=width,
            height=height,
            use_denoising_batch=use_denoising_batch,
            do_add_noise=do_add_noise,
            transition=transition,
            transition_steps=transition_steps,
            seed_transition_steps=seed_transition_steps,
        )

        # Apply embedding-space negative subtraction *after* prompt embeds
        # are settled (including any prompt transition / SDXL pooled
        # update). Acts on whatever this frame's conditioning happens to
        # be, which is the right thing during transitions too.
        self.prompts.apply_negative_subtraction(negative_prompt, negative_prompt_scale)

        frame = None

        # Process input. In text-only mode (no video stream) we fall back to
        # the previous frame's output as input — the implicit-loopback path.
        # This is what gives txt2img its iterative refinement: frame 1 is a
        # cold t2i pass and frames 2+ are img2img on the previous output, so
        # SD-Turbo's single-step recovery sharpens detail across frames.
        # Disabled per-model for CFG-distilled checkpoints (DMD2) where the
        # baked-in guidance shaping compounds catastrophically across the
        # feedback loop. Explicit image_loopback=True still wins regardless,
        # so the user can force loopback on DMD2 if they want the stylized
        # divergence (or for testing).
        implicit_ok = self._implicit_loopback and (
            video is None or len(video) == 0
        ) and self.prev_image_result is not None
        if image_loopback or implicit_ok:
            frame = self.prev_image_result
        elif video is not None and len(video) > 0:
            # Convert Scope tensor format to pipeline format
            # Scope: (T, H, W, C) in [0, 1] or [0, 255]
            # Pipeline needs: (B, C, H, W) in [0, 1]
            frame = video[0]  # Take first frame

        if frame is not None:
            # Squeeze any extra dimensions and ensure shape is (H, W, C)
            while frame.ndim > 3:
                frame = frame.squeeze(0)

            # Convert from uint8 [0, 255] to float [0, 1] if needed
            if frame.dtype == torch.uint8:
                frame = frame.float() / 255.0

            # Move to device
            frame = frame.to(device=self.device, dtype=self.dtype)

            # Get actual dimensions after squeezing
            actual_height, actual_width = frame.shape[0], frame.shape[1]

            # Resize if needed — stay on GPU to avoid the CPU↔PIL roundtrip
            if actual_height != height or actual_width != width:
                # (H, W, C) -> (1, C, H, W) for F.interpolate
                frame = frame.permute(2, 0, 1).unsqueeze(0)
                frame = torch.nn.functional.interpolate(
                    frame, size=(height, width), mode="bilinear", align_corners=False
                )
                # (1, C, H, W) -> (H, W, C)
                frame = frame.squeeze(0).permute(1, 2, 0)

            # Convert HWC -> CHW and add batch dimension: (H, W, C) -> (1, C, H, W)
            input_tensor = frame.permute(2, 0, 1).unsqueeze(0)

            # Apply similar image filter if enabled
            if similar_image_filter_enabled:
                filtered = self.similar_filter(input_tensor)
                if filtered is None and self.prev_image_result is not None:
                    # Return previous result
                    output = self.prev_image_result
                    return {"video": output.permute(0, 2, 3, 1).clamp(0, 1)}
                input_tensor = filtered

            input_latent = self._encode_image(input_tensor, add_noise=True)

        else:
            # Text-to-image mode — use the seeded `init_noise` instead of a
            # fresh unseeded randn. With a fresh randn per call, every frame
            # would generate a different scene; the seeded buffer keeps the
            # output stable across frames for the same seed (and lets the
            # user reseed deterministically by changing `seed`).
            input_latent = self.init_noise[0:1].clone()

        x_0_pred_out = self._predict_x0_batch(input_latent)
        # Decode to image space
        x_output = self._decode_image(x_0_pred_out).detach().clone()
        # Normalize from [-1, 1] to [0, 1] (VAE outputs in range [-1, 1])
        x_output = (x_output / 2 + 0.5).clamp(0, 1)
        # Convert back to Scope format: (B, C, H, W) -> (T, H, W, C)
        output = x_output.permute(0, 2, 3, 1)

        # ── Mask compositing ──────────────────────────────────────────
        # Drop-in compatible with vace_input_masks from yolo_mask / scope-sam3
        # (shape (1, 1, F, H, W), binary). SD output where mask=1, original
        # where mask=0; flip via the upstream segmenter's Invert Mask. Skip
        # in pure text mode where there's no original frame to blend with.
        mask_compositing = bool(kwargs.get("mask_compositing", False))
        mask_strength = float(kwargs.get("mask_strength", 1.0))
        masks_in = kwargs.get("vace_input_masks")
        if (
            mask_compositing
            and mask_strength > 0
            and masks_in is not None
            and frame is not None
        ):
            m = masks_in[:, :, 0].to(device=output.device, dtype=output.dtype)
            if m.shape[-2:] != (height, width):
                m = torch.nn.functional.interpolate(
                    m, size=(height, width), mode="bilinear", align_corners=False
                )
            mask_feather = float(kwargs.get("mask_feather", 0.0))
            if mask_feather > 0:
                k = max(1, int(mask_feather) * 2 + 1)
                m = torch.nn.functional.avg_pool2d(m, k, stride=1, padding=k // 2)
            m = (m * mask_strength).clamp(0, 1).permute(0, 2, 3, 1)  # (1,H,W,1)
            orig = frame.unsqueeze(0).to(device=output.device, dtype=output.dtype)
            output = m * output + (1.0 - m) * orig

        # Cache result
        self.prev_image_result = output

        return {"video": output}


def main():
    """Test function that runs the pipeline 10 times."""
    import time

    print("Initializing StreamDiffusion pipeline...")

    # Initialize pipeline
    pipeline = StreamDiffusionPipeline(
        model_id="stabilityai/sd-turbo",
        torch_dtype=torch.float16,
    )

    # Test parameters
    test_params = {
        "prompt": "A beautiful sunset over mountains",
        "negative_prompt": "ugly, blurry, low quality",
        "guidance_scale": 0.0,
        "strength": 0.99,
        "seed": 42,
        "width": 512,
        "height": 512,
        "use_denoising_batch": True,
        "do_add_noise": True,
        "similar_image_filter_enabled": False,
    }

    print("\nTest parameters:")
    print(f"  Prompt: {test_params['prompt']}")
    print(f"  Size: {test_params['width']}x{test_params['height']}")
    print("\nRunning pipeline 10 times...\n")

    # Run 10 times
    inference_times = []
    for i in range(10):
        start_time = time.time()

        # Call the pipeline (text-to-image mode - no video input)
        result = pipeline(**test_params)

        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)

        output = result["video"]
        print(f"  Run {i + 1}/10: {inference_time:.3f}s - Output shape: {output.shape}")

        # Optionally save the first output
        if i == 0:
            try:
                output_np = (output[0].cpu().numpy() * 255).astype(np.uint8)
                img = PIL.Image.fromarray(output_np)
                img.save("streamdiffusion_test_output.png")
                print("    → Saved first output to streamdiffusion_test_output.png")
            except Exception as e:
                print(f"    → Could not save image: {e}")

    # Print statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)

    print(f"\n{'=' * 50}")
    print("Performance Statistics:")
    print(f"  Average: {avg_time:.3f}s ({1 / avg_time:.2f} FPS)")
    print(f"  Min:     {min_time:.3f}s ({1 / min_time:.2f} FPS)")
    print(f"  Max:     {max_time:.3f}s ({1 / max_time:.2f} FPS)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
