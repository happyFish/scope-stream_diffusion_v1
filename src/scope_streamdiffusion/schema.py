"""Configuration schema for StreamDiffusion pipeline."""

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import Field, field_validator
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    InputMode,
    ModeDefaults,
    ui_field_config,
)


class ModelId(StrEnum):
    """Supported StreamDiffusion models (all 1-step distillations)."""

    SD_TURBO = "stabilityai/sd-turbo"
    SDXL_TURBO = "stabilityai/sdxl-turbo"
    DMD2_SDXL_1STEP = "dmd2-sdxl-1step"


class Resolution(IntEnum):
    """Allowed pixel dimensions for width/height.

    Multiples of 64 in [256, 1024]. UNet downsamples latents 3x and ControlNet
    residuals land at /8 in latent space, so pixel dims must divide by 64. TRT
    engines are built for this dynamic range.
    """

    R256 = 256
    R320 = 320
    R384 = 384
    R448 = 448
    R512 = 512
    R576 = 576
    R640 = 640
    R704 = 704
    R768 = 768
    R832 = 832
    R896 = 896
    R960 = 960
    R1024 = 1024


class StreamDiffusionConfig(BasePipelineConfig):
    """Configuration for the StreamDiffusion pipeline."""

    # Pipeline metadata
    pipeline_id = "streamdiffusion"
    pipeline_name = "StreamDiffusion"
    pipeline_description = (
        "Real-time Stable Diffusion with Stream Diffusion acceleration"
    )

    # Enable prompt support in Scope UI
    supports_prompts = True

    # Pipeline modes
    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(height=512, width=512),
    }

    supports_lora = True

    # Accept a mask stream from upstream segmenters (YOLO mask, SAM3 mask, etc.)
    # in addition to video. Mask is (1, 1, F, H, W) binary; compositing happens
    # post-SD per the mask_compositing field below.
    inputs = ["video", "vace_input_masks"]

    # ========================================
    # Pipeline Control
    # ========================================

    enabled: bool = Field(
        default=True,
        description="Enable pipeline processing. When disabled, input video is passed through unchanged.",
        #json_schema_extra=ui_field_config(order=0, label="Enabled"),
    )

    input_mode: InputMode = Field(
        default="text",
        description="Input mode: 'text' generates from prompts only, 'video' transforms input frames",
        #json_schema_extra=ui_field_config(order=1, label="Input Mode"),
    )

    # ========================================
    # Model Configuration
    # ========================================

    model_id_or_path: ModelId = Field(
        default=ModelId.SD_TURBO,
        description=(
            "Model selection. All entries are 1-step distillations. "
            "'dmd2-sdxl-1step' is SDXL-base with the DMD2 distilled UNet "
            "(tianweiy/DMD2) swapped in — quality bump over SDXL-Turbo per "
            "the DMD2 paper. SDXL-derived entries auto-install the fp16-fix VAE."
        ),
        json_schema_extra=ui_field_config(order=8, label="Model"),
    )

    acceleration_mode: Literal["none", "trt"] = Field(
        default="trt",
        description=(
            "TRT-compile UNet (and ControlNet on SD 1.5) for 2-8x denoising "
            "speedup. First build per model takes 5-10 min and caches to "
            "~/.cache/scope-streamdiffusion-trt/. Hot-swappable at runtime: "
            "toggling restores cached engines from process-scope cache "
            "(instant) or builds them on first activation (stalls the "
            "stream). SD 1.5 engines support dynamic resolution 256-1024 "
            "and batch 1-4. SDXL engines (sdxl-turbo, dmd2-sdxl-1step) "
            "support dynamic resolution 512-1024 with static batch=1 — "
            "different envelope to fit a 24 GB VRAM build budget. SDXL "
            "ControlNet works in eager mode (acceleration_mode='none'); "
            "SDXL + ControlNet + TRT is not yet supported (raises "
            "NotImplementedError) — flip acceleration off when using "
            "ControlNet on SDXL until that lands."
        ),
        json_schema_extra=ui_field_config(order=2, label="Acceleration"),
    )

    use_taesd: bool = Field(
        default=True,
        description="Use Tiny AutoEncoder (TAESD) for ~10x faster VAE decoding at slight quality cost",
        #json_schema_extra=ui_field_config(order=2, label="Use TAESD"),
    )

    controlnet_mode: Literal["none", "depth", "scribble"] = Field(
        default="none",
        description="ControlNet conditioning mode. 'depth' runs Video Depth Anything internally and routes the depth map to ControlNet. First switch to a new mode stalls while the model loads.",
        json_schema_extra=ui_field_config(order=3, label="ControlNet Mode"),
    )

    controlnet_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="ControlNet conditioning scale (0.0 = no effect, 2.0 = maximum)",
        json_schema_extra=ui_field_config(order=4, label="ControlNet Scale"),
    )

    depth_min: float = Field(
        default=0,
        ge=0.0,
        le=1,
        description="Minimum depth value for ControlNet",
        #json_schema_extra=ui_field_config(order=5, label="Depth Min"),
    )

    depth_max: float = Field(
        default=1,
        ge=0.0,
        le=1,
        description="Maximum depth value for ControlNet",
        #json_schema_extra=ui_field_config(order=6, label="Depth Max"),
    )

    depth_skip_interval: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Run depth model every Nth frame; reuse cached depth map on intermediate frames. Higher = less GPU cost, more temporal lag.",
        #json_schema_extra=ui_field_config(order=7, label="Depth Skip Interval"),
    )

    depth_input_size: Literal[252, 364, 518] = Field(
        default=252,
        description="Resolution the depth model runs at (must be multiple of 14). Lower = faster but coarser depth. 252 ≈ 4× faster than 518; the depth map is bilinear-upsampled to controlnet resolution either way.",
        #json_schema_extra=ui_field_config(order=8, label="Depth Input Size"),
    )

    depth_temporal_cache: bool = Field(
        default=True,
        description="Use the video model's temporal hidden-state cache for inter-frame consistency. Disabling skips the temporal motion modules entirely (faster, slightly more flicker). Combined with skip interval > 1 the cache buys little, so toggle off for speed.",
        #json_schema_extra=ui_field_config(order=9, label="Depth Temporal Cache"),
    )

    controlnet_temporal_smoothing: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Temporal blending of the ControlNet conditioning map. 0.0 = fully smoothed (previous frame only), 1.0 = no smoothing (current frame only). Lower values reduce flicker; higher values reduce latency.",
        #json_schema_extra=ui_field_config(order=5, label="ControlNet Smoothing"),
    )

    # ========================================
    # Generation Parameters
    # ========================================
    # Note: prompts array is handled by Scope's base when supports_prompts = True

    negative_prompt: str = Field(
        default="",
        description="Negative prompt — what to avoid in the generated image",
        #json_schema_extra=ui_field_config(order=11, label="Negative Prompt"),
    )

    negative_prompt_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description=(
            "Strength of embedding-space negative guidance for single-pass "
            "models (Turbo, DMD2) that can't use standard CFG. The negative "
            "embedding is subtracted from the positive then rescaled to "
            "preserve magnitude — direction shifts but the result stays in "
            "the UNet's training distribution. 0 = disabled. 0.3-0.7 is "
            "typical; >1.0 starts to push out of distribution and outputs "
            "may degrade or go to noise."
        ),
        json_schema_extra=ui_field_config(order=12, label="Negative Scale"),
    )

    prompt_interpolation_method: Literal["linear", "slerp"] = Field(
        default="linear",
        description="Method for blending multiple prompts spatially (and temporally when transition_steps > 0)",
    )

    transition_steps: int = Field(
        default=0,
        ge=0,
        le=240,
        description=(
            "Auto-transition over this many frames whenever prompts change. "
            "0 = hard cut (can cause garbage frames); 8-30 is typical for smooth "
            "prompt morphs. Ignored when an explicit transition dict is sent."
        ),
        #json_schema_extra=ui_field_config(order=10, label="Transition Steps"),
    )

    seed: int = Field(
        default=42,
        ge=0,
        le=2147483647,
        description="Random seed for generation",
        json_schema_extra=ui_field_config(order=13, label="Seed"),
    )

    seed_transition_steps: int = Field(
        default=0,
        ge=0,
        le=240,
        description=(
            "Lerp the seed noise toward the new seed over N frames on each "
            "seed change. 0 = hard cut. SDXL-Turbo and DMD2-1step have less "
            "natural frame-to-frame correlation than SD-Turbo; this gives a "
            "deterministic settle independent of the model."
        ),
        json_schema_extra=ui_field_config(order=14, label="Seed Transition Steps"),
    )

    # ========================================
    # Diffusion Parameters
    # ========================================

    guidance_scale: float = Field(
        default=0.0,
        ge=0.0,
        le=8.0,
        description="Classifier-free guidance scale (0 = none, higher = more prompt adherence)",
        # json_schema_extra=ui_field_config(order=20, label="Guidance Scale"),
    )

    strength: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Denoising strength (how much to transform input)",
        json_schema_extra=ui_field_config(order=22, label="Strength"),
    )

    # ========================================
    # StreamDiffusion Specific
    # ========================================

    delta: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="StreamDiffusion delta parameter",
        # json_schema_extra=ui_field_config(order=30, label="Delta"),
    )

    do_add_noise: bool = Field(
        default=True,
        description="Add noise between denoising steps",
        # json_schema_extra=ui_field_config(order=31, label="Add Noise"),
    )

    use_denoising_batch: bool = Field(
        default=True,
        description="Use batch denoising for better performance",
        # json_schema_extra=ui_field_config(order=32, label="Batch Denoising"),
    )

    use_lcm_lora: bool = Field(
        default=True,
        description="Use LCM LoRA for faster inference",
        # json_schema_extra=ui_field_config(order=33, label="Use LCM LoRA"),
    )

    # ========================================
    # Additional Options
    # ========================================

    image_loopback: bool = Field(
        default=False,
        description="Use last frame as input for the next generation",
        #json_schema_extra=ui_field_config(order=49, label="Image Loopback"),
    )

    # ========================================
    # Mask Compositing (consumes vace_input_masks from upstream segmenter)
    # ========================================

    mask_compositing: bool = Field(
        default=False,
        description=(
            "Composite SD output with the original frame using the incoming "
            "mask. SD output goes where mask=1, original goes where mask=0. "
            "Flip directions by toggling the upstream segmenter's Invert Mask."
        ),
        #json_schema_extra=ui_field_config(order=55, label="Mask Compositing"),
    )

    mask_feather: float = Field(
        default=0.0,
        ge=0.0,
        le=32.0,
        description=(
            "Soft mask edges (pixels). 0 = hard edge. Cheap box-blur applied "
            "to the mask before compositing."
        ),
        #json_schema_extra=ui_field_config(order=56, label="Mask Feather"),
    )

    mask_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Overall mask blend strength. 0 disables compositing, 1 is full effect. "
            "Use intermediate values to ghost the original through the SD output."
        ),
        #json_schema_extra=ui_field_config(order=57, label="Mask Strength"),
    )

    width: Resolution = Field(
        default=Resolution.R512,
        description="Output width (multiple of 64, 256-1024)",
        #json_schema_extra=ui_field_config(order=60, label="Width"),
    )

    height: Resolution = Field(
        default=Resolution.R512,
        description="Output height (multiple of 64, 256-1024)",
        #json_schema_extra=ui_field_config(order=61, label="Height"),
    )

    @field_validator("width", "height", mode="before")
    @classmethod
    def _validate_resolution(cls, v: object) -> Resolution:
        try:
            return Resolution(int(v))  # type: ignore[arg-type]
        except (ValueError, TypeError) as e:
            allowed = ", ".join(str(r.value) for r in Resolution)
            raise ValueError(
                f"Resolution must be one of: {allowed} (multiples of 64 in [256, 1024]); got {v!r}"
            ) from e

