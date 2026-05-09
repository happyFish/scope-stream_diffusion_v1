"""StreamDiffusion pipeline implementation for Scope."""

from typing import TYPE_CHECKING, List, Literal, Optional

import torch
import numpy as np
import PIL.Image
from diffusers import (
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from scope.core.pipelines.interface import Pipeline, Requirements

from .controlnet import ControlNetHandler
from .inference_core import InferenceCore
from .model_loader import MODEL_PRESETS, ModelLoader
from .prompt_encoder import PromptEncoder, normalize_prompts
from .schema import StreamDiffusionConfig
from .trt_lifecycle import TRTLifecycle

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


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

        # Per-frame inference: timestep schedule, noise buffers, seed
        # transitions, VAE encode/decode, UNet/scheduler step math. Holds
        # tensor state directly; reads scheduler/vae/unet/controlnet/prompts
        # via the back-ref set in ``attach()``.
        self.inference = InferenceCore(self.device, self.dtype)
        self.inference.attach(self)

        # State that will be set during runtime
        self.generator = torch.Generator(device=self.device)
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

        # Cache keys for _prepare_runtime_state — None forces full recompute on first call
        self._schedule_key: tuple | None = (
            None  # (num_inference_steps, strength, t_index_list)
        )
        self._last_seed: int | None = None
        self._noise_shape: tuple | None = None  # (batch_size, latent_h, latent_w)

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

        Helper attach order: PromptEncoder, InferenceCore, TRTLifecycle.
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
        # ControlNet handler routes to SDXL-specific weights when the host
        # pipeline is SDXL (depth/scribble names map to different repos).
        self._cn.set_sdxl(self.sdxl)

        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self._full_vae = self.vae
        self._using_taesd = False
        self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.image_processor = VaeImageProcessor(self.pipe.vae_scale_factor)
        self.prompts.attach(self.pipe, self.sdxl)
        self.inference.attach(self)
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
            self.inference.set_timesteps(num_inference_steps, strength)
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
            self.inference.cancel_seed_transition()
            self.inference.x_t_latent_buffer = None
            self.inference.initialize_noise()
            self._noise_shape = noise_shape
        elif seed_changed:
            # Hard cut when seed_transition_steps == 0; multi-frame lerp otherwise.
            self.inference.setup_seed_transition(seed, seed_transition_steps)

        # Advance any in-flight seed transition by one frame. No-op when idle.
        self.inference.advance_seed_transition()

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

            input_latent = self.inference.encode_image(input_tensor, add_noise=True)

        else:
            # Text-to-image mode — use the seeded `init_noise` instead of a
            # fresh unseeded randn. With a fresh randn per call, every frame
            # would generate a different scene; the seeded buffer keeps the
            # output stable across frames for the same seed (and lets the
            # user reseed deterministically by changing `seed`).
            input_latent = self.inference.init_noise[0:1].clone()

        x_0_pred_out = self.inference.predict_x0(input_latent)
        # Decode to image space
        x_output = self.inference.decode_image(x_0_pred_out).detach().clone()
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
