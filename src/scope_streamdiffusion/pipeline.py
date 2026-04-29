"""StreamDiffusion pipeline implementation for Scope."""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import torch
import numpy as np
import PIL.Image
from diffusers import (
    DiffusionPipeline,
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.blending import EmbeddingBlender, parse_transition_config

from .controlnet import ControlNetHandler
from .schema import StreamDiffusionConfig

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
        model_id: str = "stabilityai/sd-turbo",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Initialize the StreamDiffusion pipeline.

        Args:
            device: Torch device to use
            model_id: Model ID or path to load
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

        # Load the base model
        print(f"Loading model: {model_id}")
        self.pipe = self._load_model(model_id)
        print(f"Model loaded: {self.pipe.__class__.__name__}")

        # Model components
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self._full_vae = self.vae  # keep reference for toggling
        self._taesd_vae = None
        self._using_taesd = False

        # Check if SDXL
        self.sdxl: bool = type(self.pipe) is StableDiffusionXLPipeline

        # Setup scheduler
        self.scheduler: LCMScheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config
        )

        # Setup image processor
        self.image_processor: VaeImageProcessor = VaeImageProcessor(
            self.pipe.vae_scale_factor
        )

        # Setup embedding blender for prompt weighting and interpolation
        self.embedding_blender = EmbeddingBlender(
            device=self.device,
            dtype=self.dtype,
        )

        # State that will be set during runtime
        self.generator = torch.Generator(device=self.device)
        self._previous_prompt_embeddings = None
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
        self._prompts_key: tuple | None = None
        self._cached_base_embed: torch.Tensor | None = None  # (1, seq_len, hidden_dim)

        # Transition state — the main embedding queue lives inside
        # EmbeddingBlender; the pooled embedding (SDXL only) is interpolated
        # linearly in lockstep here so `add_text_embeds` tracks the morph.
        self._last_transition_id: str | None = None
        self._pooled_source: torch.Tensor | None = None
        self._pooled_target: torch.Tensor | None = None
        self._transition_total_steps: int = 0

        # Mode-transition tracking — detect video↔text switches without a pipeline reload
        self._last_mode: str | None = None

        print("StreamDiffusion pipeline initialized")

    def _load_model(self, model_id: str) -> DiffusionPipeline:
        """Load the diffusion model."""
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            pipe = pipe.to(self.device)

            # Enable xformers memory-efficient attention if available.
            # The schema declares acceleration="xformers" but this was never called.
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[StreamDiffusion] xformers memory-efficient attention enabled")
            except Exception as e:
                print(f"[StreamDiffusion] xformers not available, skipping: {e}")

            return pipe
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            raise

    def _set_taesd(self, enabled: bool) -> None:
        """Switch between TAESD (fast) and full VAE decoder."""
        if enabled == self._using_taesd:
            return
        if enabled:
            if self._taesd_vae is None:
                from diffusers import AutoencoderTiny

                taesd_id = "madebyollin/taesdxl" if self.sdxl else "madebyollin/taesd"
                print(f"[StreamDiffusion] Loading TAESD from {taesd_id}")
                self._taesd_vae = AutoencoderTiny.from_pretrained(
                    taesd_id, torch_dtype=self.dtype
                ).to(self.device)
                print("[StreamDiffusion] TAESD loaded")
            self.vae = self._taesd_vae
            self._using_taesd = True
            print("[StreamDiffusion] Switched to TAESD (fast decode)")
        else:
            self.vae = self._full_vae
            self._using_taesd = False
            print("[StreamDiffusion] Switched to full VAE")

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

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
            print(
                f"Using t_index_list: {t_index_list} from {num_inference_steps} total steps"
            )
            self._set_timesteps(num_inference_steps, strength)
            self._schedule_key = schedule_key

        # --- Seed + noise buffers: only reset when seed or spatial shape changes ---
        noise_shape = (self.batch_size, self.latent_height, self.latent_width)
        seed_changed = seed != self._last_seed
        shape_changed = noise_shape != self._noise_shape or dims_changed

        if seed_changed:
            self.generator.manual_seed(seed)
            self._last_seed = seed

        if seed_changed or shape_changed:
            if self.denoising_steps_num > 1:
                self.x_t_latent_buffer = torch.zeros(
                    (
                        (self.denoising_steps_num - 1) * self.frame_bff_size,
                        4,
                        self.latent_height,
                        self.latent_width,
                    ),
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                self.x_t_latent_buffer = None
            self._initialize_noise()
            self._noise_shape = noise_shape

        # --- Prompt embeddings & transitions ---
        # The key includes spatial dims for SDXL because add_time_ids depend on them.
        # When an explicit transition dict is present, its target_prompts is the
        # authoritative destination; keying against the incoming source prompts
        # would make prompts_changed flap during/after the transition and snap
        # steady state back to the source.
        key_prompts = prompts
        if transition is not None:
            target_raw = transition.get("target_prompts")
            if target_raw:
                key_prompts = self._normalize_prompts(target_raw)
        new_prompts_key = self._make_prompts_key(
            key_prompts, prompt_interpolation_method, width, height
        )
        prompts_changed = new_prompts_key != self._prompts_key

        # Hash the explicit transition dict so repeated sends don't restart it.
        transition_id = self._hash_transition(transition) if transition else None
        new_explicit_transition = (
            transition_id is not None and transition_id != self._last_transition_id
        )

        started_transition = False

        # Cancel any in-flight transition if a new target has arrived so we
        # redirect from the current interpolated position rather than snapping
        # after the old transition drains.
        if self.embedding_blender.is_transitioning() and (
            new_explicit_transition
            or (transition is None and transition_steps > 0 and prompts_changed)
        ):
            self.embedding_blender.cancel_transition()
            self._finish_pooled_transition()

        # 1) Explicit transition (transition dict with target_prompts).
        if new_explicit_transition and not self.embedding_blender.is_transitioning():
            transition_config = parse_transition_config(transition)
            target_prompts_raw = transition.get("target_prompts", [])
            if transition_config.num_steps > 0 and target_prompts_raw:
                target_prompts = self._normalize_prompts(target_prompts_raw)
                started_transition = self._begin_transition(
                    target_prompts=target_prompts,
                    interpolation_method=prompt_interpolation_method,
                    num_steps=transition_config.num_steps,
                    temporal_method=transition_config.temporal_interpolation_method,
                    width=width,
                    height=height,
                )
            self._last_transition_id = transition_id

        # 2) Auto-transition when `prompts` changes with transition_steps > 0.
        elif (
            transition is None
            and transition_steps > 0
            and prompts_changed
            and self._previous_prompt_embeddings is not None
            and not self.embedding_blender.is_transitioning()
        ):
            started_transition = self._begin_transition(
                target_prompts=prompts,
                interpolation_method=prompt_interpolation_method,
                num_steps=transition_steps,
                temporal_method=prompt_interpolation_method,
                width=width,
                height=height,
            )

        # --- Produce prompt_embeds for this frame ---
        if self.embedding_blender.is_transitioning():
            next_embedding = self.embedding_blender.get_next_embedding()
            if next_embedding is not None:
                self.prompt_embeds = next_embedding.repeat(self.batch_size, 1, 1)
                self._advance_pooled_transition()
            else:
                self.prompt_embeds = self._cached_base_embed.repeat(
                    self.batch_size, 1, 1
                )
                self._finish_pooled_transition()
        else:
            # Steady state — re-encode if prompts changed and we didn't start a
            # transition for it (hard cut path, e.g. transition_steps == 0).
            if prompts_changed and not started_transition:
                raw_embeds, _ = self._encode_prompts_array(
                    key_prompts, prompt_interpolation_method
                )
                self._cached_base_embed = raw_embeds[0:1]
                self._prompts_key = new_prompts_key
            # Drop the transition-id guard once the explicit dict is gone so a
            # later identical dict is treated as a fresh request.
            if transition is None:
                self._last_transition_id = None
            self._finish_pooled_transition()
            self.prompt_embeds = self._cached_base_embed.repeat(self.batch_size, 1, 1)

        # Cache embedding as source for the next transition.
        self._previous_prompt_embeddings = self.prompt_embeds[0:1].detach()

    def _make_prompts_key(
        self,
        prompts: list[dict],
        interpolation_method: str,
        width: int,
        height: int,
    ) -> tuple:
        """Identity key for a prompts payload; SDXL includes dims for add_time_ids."""
        return (
            tuple((p.get("text", ""), p.get("weight", 1.0)) for p in prompts),
            interpolation_method,
            (width, height) if self.sdxl else (),
        )

    @staticmethod
    def _hash_transition(transition: dict) -> str:
        """Stable identity for a transition dict so repeated sends don't restart it."""
        import hashlib
        import json

        payload = {
            "num_steps": int(transition.get("num_steps", 0) or 0),
            "method": transition.get("temporal_interpolation_method", "linear"),
            "target": [
                {
                    "text": p.get("text", "") if isinstance(p, dict) else str(p),
                    "weight": float(p.get("weight", 1.0)) if isinstance(p, dict) else 1.0,
                }
                for p in (transition.get("target_prompts") or [])
            ],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def _begin_transition(
        self,
        target_prompts: list[dict],
        interpolation_method: str,
        num_steps: int,
        temporal_method: str,
        width: int,
        height: int,
    ) -> bool:
        """Start a temporal transition from the last emitted embedding toward
        the target prompts.  Eagerly advances `_cached_base_embed` and
        `_prompts_key` to the target so steady state lands there when the queue
        drains.  Returns True if a transition was actually started.
        """
        source_embedding = self._previous_prompt_embeddings
        if source_embedding is None:
            return False

        # Encode and blend target in main embedding space + pooled (SDXL).
        target_embed, target_pooled = self._encode_prompts_array(
            target_prompts, interpolation_method, apply_sdxl_conditioning=False
        )
        target_embed_single = target_embed[0:1]

        # Eagerly move the steady-state cache to the target so once the queue
        # drains we land on the target prompts with no bounce-back.
        self._cached_base_embed = target_embed_single
        self._prompts_key = self._make_prompts_key(
            target_prompts, interpolation_method, width, height
        )

        # Slerp is not supported here: upstream EmbeddingBlender.slerp runs
        # torch.acos on the native dtype; at fp16 the [-1, 1] clamp isn't
        # enough to prevent acos(1.0) → NaN at certain token positions, which
        # nukes the whole conditioning tensor. Until that's fixed upstream,
        # fall back to linear and warn once.
        if temporal_method == "slerp":
            if not getattr(self, "_slerp_fallback_warned", False):
                print(
                    "[StreamDiffusion] slerp temporal interpolation is not "
                    "supported (fp16 NaN in upstream blender); falling back "
                    "to linear."
                )
                self._slerp_fallback_warned = True
            temporal_method = "linear"

        self.embedding_blender.start_transition(
            source_embedding=source_embedding,
            target_embedding=target_embed_single,
            num_steps=num_steps,
            temporal_interpolation_method=temporal_method,
        )

        # Pooled interpolation runs in lockstep with the main queue for SDXL.
        if self.sdxl and target_pooled is not None:
            self._pooled_source = (
                self.add_text_embeds.detach().clone()
                if hasattr(self, "add_text_embeds") and self.add_text_embeds is not None
                else target_pooled.clone()
            )
            self._pooled_target = target_pooled.clone()
            self._transition_total_steps = max(1, num_steps)
        else:
            self._pooled_source = None
            self._pooled_target = None
            self._transition_total_steps = 0

        # start_transition short-circuits when source ≈ target
        # (MIN_EMBEDDING_DIFF_THRESHOLD); report accurately so the caller falls
        # to steady state instead of assuming a transition is live.
        if not self.embedding_blender.is_transitioning():
            self._finish_pooled_transition()
            return False
        return True

    def _advance_pooled_transition(self) -> None:
        """Linearly interpolate `add_text_embeds` toward the target pooled.

        Uses the blender's remaining queue length to compute progress so
        pooled and main embeds stay in lockstep even if start_transition
        short-circuited.
        """
        if not self.sdxl or self._pooled_target is None:
            return
        if self._transition_total_steps <= 0:
            return
        remaining = len(self.embedding_blender._transition_queue)
        done_steps = self._transition_total_steps - remaining
        t = min(1.0, max(0.0, done_steps / self._transition_total_steps))
        source = (
            self._pooled_source
            if self._pooled_source is not None
            else self._pooled_target
        )
        self.add_text_embeds = torch.lerp(source, self._pooled_target, t).to(
            dtype=self.dtype, device=self.device
        )

    def _finish_pooled_transition(self) -> None:
        """Snap pooled to the target and clear transition state."""
        if self.sdxl and self._pooled_target is not None:
            self.add_text_embeds = self._pooled_target.to(
                dtype=self.dtype, device=self.device
            )
        self._pooled_source = None
        self._pooled_target = None
        self._transition_total_steps = 0

    @staticmethod
    def _normalize_prompts(prompts: str | list[str] | list[dict]) -> list[dict]:
        """Normalize prompts to list[dict] format."""
        if isinstance(prompts, str):
            return [{"text": prompts, "weight": 1.0}]
        if isinstance(prompts, list):
            if len(prompts) == 0:
                return [{"text": "", "weight": 1.0}]
            # Check if it's a list of strings
            if isinstance(prompts[0], str):
                return [{"text": text, "weight": 1.0} for text in prompts]
            # Already list[dict]
            return prompts
        return [{"text": str(prompts), "weight": 1.0}]

    def _encode_single_prompt(
        self, prompt_text: str
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode a single prompt string to embeddings.

        Returns:
            (prompt_embeds, pooled_embeds) tuple
        """
        # Use diffusers' built-in encoding
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt_text,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )
        prompt_embeds = encoder_output[0]  # [1, seq_len, hidden_dim]
        pooled_embeds = encoder_output[2] if self.sdxl else None

        return prompt_embeds, pooled_embeds

    def _encode_prompts_array(
        self,
        prompt_items: list[dict],
        interpolation_method: str = "linear",
        apply_sdxl_conditioning: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode multiple weighted prompts and blend them.

        Args:
            prompt_items: List of {"text": str, "weight": float}
            interpolation_method: "linear" or "slerp"
            apply_sdxl_conditioning: When True (default, steady-state encode),
                also updates `self.add_text_embeds` and `self.add_time_ids`
                for SDXL. Set False when encoding a transition target so the
                in-flight pooled/time_ids aren't overwritten mid-morph.

        Returns:
            (blended_prompt_embeds, blended_pooled_embeds) tuple
        """
        if not prompt_items:
            prompt_items = [{"text": "", "weight": 1.0}]

        # Extract texts and weights
        texts = [item.get("text", "") for item in prompt_items]
        weights = [item.get("weight", 1.0) for item in prompt_items]

        # Encode each prompt
        all_prompt_embeds = []
        all_pooled_embeds = [] if self.sdxl else None

        for text in texts:
            prompt_embeds, pooled_embeds = self._encode_single_prompt(text)
            all_prompt_embeds.append(prompt_embeds)
            if self.sdxl and pooled_embeds is not None:
                all_pooled_embeds.append(pooled_embeds)

        # Blend embeddings
        blended_prompt_embeds = self.embedding_blender.blend(
            all_prompt_embeds,
            weights,
            interpolation_method,
            cache_result=True,
        )

        blended_pooled_embeds = None
        if self.sdxl and all_pooled_embeds:
            blended_pooled_embeds = self.embedding_blender.blend(
                all_pooled_embeds,
                weights,
                interpolation_method,
                cache_result=False,
            )

        # Handle SDXL additional embeddings (skipped for transition-target
        # encoding so the live pooled/time_ids aren't overwritten mid-morph).
        if apply_sdxl_conditioning and self.sdxl and blended_pooled_embeds is not None:
            self.add_text_embeds = blended_pooled_embeds
            original_size = (self.height, self.width)
            crops_coords_top_left = (0, 0)
            target_size = (self.height, self.width)
            text_encoder_projection_dim = int(self.add_text_embeds.shape[-1])
            self.add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=self.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )

        return blended_prompt_embeds.repeat(
            self.batch_size, 1, 1
        ), blended_pooled_embeds

    def _set_timesteps(self, num_inference_steps: int, strength: float):
        """Set the timesteps for the diffusion process."""
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
        for timestep in self.sub_timesteps:
            if timestep >= len(self.scheduler.alphas_cumprod):
                print(
                    f"Warning: timestep {timestep} is greater than the number of timesteps {len(self.scheduler.alphas_cumprod)}"
                )
                continue
            alpha_prod_t_sqrt = self.scheduler.alphas_cumprod[timestep].sqrt()
            beta_prod_t_sqrt = (1 - self.scheduler.alphas_cumprod[timestep]).sqrt()
            alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
            beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)

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

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        """Get additional time IDs for SDXL."""
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
                f"but a vector of {passed_add_embed_dim} was created."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

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

        # Compute ControlNet residuals if conditioning is available
        down_block_res_samples = None
        mid_block_res_sample = None
        if self.controlnet is not None and self.controlnet_input is not None:
            batch_size = x_t_latent_plus_uc.shape[0]
            cond_image = self.controlnet_input.expand(batch_size, -1, -1, -1)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x_t_latent_plus_uc,
                t_list,
                encoder_hidden_states=self.prompt_embeds,
                controlnet_cond=cond_image,
                conditioning_scale=self.controlnet_conditioning_scale,
                return_dict=False,
            )

        model_pred = self.unet(
            x_t_latent_plus_uc,
            t_list,
            encoder_hidden_states=self.prompt_embeds,
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
            if self.denoising_steps_num > 1:
                x_t_latent = torch.cat((x_t_latent, prev_latent_batch), dim=0)
                self.stock_noise = torch.cat(
                    (self.init_noise[0:1], self.stock_noise[:-1]), dim=0
                )
            if self.sdxl:
                added_cond_kwargs = {
                    "text_embeds": self.add_text_embeds.to(self.device),
                    "time_ids": self.add_time_ids.to(self.device),
                }

            x_t_latent = x_t_latent.to(self.device)
            t_list = t_list.to(self.device)
            x_0_pred_batch, _model_pred = self._unet_step(
                x_t_latent, t_list, added_cond_kwargs=added_cond_kwargs
            )

            if self.denoising_steps_num > 1:
                x_0_pred_out = x_0_pred_batch[-1].unsqueeze(0)
                if self.do_add_noise:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                        + self.beta_prod_t_sqrt[1:] * self.init_noise[1:]
                    )
                else:
                    self.x_t_latent_buffer = (
                        self.alpha_prod_t_sqrt[1:] * x_0_pred_batch[:-1]
                    )
            else:
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
                        "text_embeds": self.add_text_embeds.to(self.device),
                        "time_ids": self.add_time_ids.to(self.device),
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
            self._normalize_prompts(prompts)
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

        # Extract all parameters with config fallback
        prompt_interpolation_method = get_param("prompt_interpolation_method", "linear")
        guidance_scale = get_param("guidance_scale", 0.0)

        # SD Turbo: Use single timestep (t_index_list=[0]) but set schedule length
        # This matches your working project setup
        num_inference_steps = get_param("num_inference_steps", 3)

        # For img2img with SD Turbo, need higher strength for visible changes
        # 0.5-0.7 = moderate, 0.8-0.95 = heavy transformation
        strength = get_param("strength", 0.9)

        seed = get_param("seed", 42)
        delta = get_param("delta", 1.0)
        width = get_param("width", 512)
        height = get_param("height", 512)
        use_denoising_batch = get_param("use_denoising_batch", True)
        do_add_noise = get_param("do_add_noise", True)
        similar_image_filter_enabled = get_param("similar_image_filter_enabled", False)
        image_loopback = get_param("image_loopback", False)
        controlnet_mode = get_param("controlnet_mode", "none")
        controlnet_scale = get_param("controlnet_scale", 1.0)
        controlnet_temporal_smoothing = get_param("controlnet_temporal_smoothing", 0.5)
        init_cache = kwargs.get("init_cache", False)
        depth_min = get_param("depth_min", 0)
        depth_max = get_param("depth_max", 12)
        depth_skip_interval = get_param("depth_skip_interval", 3)
        use_taesd = get_param("use_taesd", False)

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
        self._set_taesd(use_taesd)

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
        )
        self.controlnet = self._cn.model
        self.controlnet_input = self._cn.input
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
        )

        frame = None

        # Process input
        if image_loopback or (
            (video is None or len(video) == 0) and self.prev_image_result is not None
        ):
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

            # Encode to latent space
            input_latent = self._encode_image(input_tensor)

        else:
            # Text-to-image mode
            input_latent = torch.randn(
                (1, 4, self.latent_height, self.latent_width),
                device=self.device,
                dtype=self.dtype,
            )

        # Run diffusion
        x_0_pred_out = self._predict_x0_batch(input_latent)
        # Decode to image space
        x_output = self._decode_image(x_0_pred_out).detach().clone()
        # Normalize from [-1, 1] to [0, 1] (VAE outputs in range [-1, 1])
        x_output = (x_output / 2 + 0.5).clamp(0, 1)
        # Convert back to Scope format: (B, C, H, W) -> (T, H, W, C)
        output = x_output.permute(0, 2, 3, 1)

        # ── Mask compositing ──────────────────────────────────────────
        # Drop-in compatible with vace_input_masks from yolo_mask / scope-sam3
        # (shape (1, 1, F, H, W), binary). Skip in pure text mode where
        # there's no original frame to blend with.
        mask_compositing = kwargs.get("mask_compositing", "none")
        mask_strength = float(kwargs.get("mask_strength", 1.0))
        masks_in = kwargs.get("vace_input_masks")
        if (
            mask_compositing != "none"
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
            if mask_compositing == "keep_sd_outside":
                m = 1.0 - m
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
        "num_inference_steps": 4,
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
    print(f"  Steps: {test_params['num_inference_steps']}")
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
