"""Configuration schema for StreamDiffusion pipeline."""

from typing import Literal

from pydantic import Field
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


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
    modes = {"video": ModeDefaults(default=True)}

    supports_lora = True

    # ========================================
    # Model Configuration
    # ========================================

    model_id_or_path: str = Field(
        default="stabilityai/sd-turbo",
        description="Model ID from HuggingFace or local path to model",
        json_schema_extra=ui_field_config(
            order=1,
            label="Model",
        ),
    )

    acceleration: Literal["none", "xformers", "tensorrt"] = Field(
        default="xformers",
        description="Hardware acceleration method",
        json_schema_extra=ui_field_config(order=2, label="Acceleration"),
    )

    # ========================================
    # Generation Parameters
    # ========================================

    prompt: str = Field(
        default="Laid down before the gates of Heaven",
        max_length=5000,
        description="Generation prompt",
        json_schema_extra=ui_field_config(order=10, label="Prompt"),
    )

    negative_prompt: str = Field(
        default="Deformed, ugly, bad anatomy",
        max_length=5000,
        description="Negative prompt",
        json_schema_extra=ui_field_config(order=11, label="Negative Prompt"),
    )

    seed: int = Field(
        default=42,
        ge=0,
        le=2147483647,
        description="Random seed for generation",
        json_schema_extra=ui_field_config(order=12, label="Seed"),
    )

    # ========================================
    # Diffusion Parameters
    # ========================================

    guidance_scale: float = Field(
        default=0.0,
        ge=0.0,
        le=8.0,
        description="Classifier-free guidance scale (0 = none, higher = more prompt adherence)",
        json_schema_extra=ui_field_config(order=20, label="Guidance Scale"),
    )

    num_inference_steps: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Number of denoising steps",
        json_schema_extra=ui_field_config(order=21, label="Inference Steps"),
    )

    strength: float = Field(
        default=0.99,
        ge=0.0,
        le=1.2,
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
        json_schema_extra=ui_field_config(order=30, label="Delta"),
    )

    do_add_noise: bool = Field(
        default=True,
        description="Add noise between denoising steps",
        json_schema_extra=ui_field_config(order=31, label="Add Noise"),
    )

    use_denoising_batch: bool = Field(
        default=True,
        description="Use batch denoising for better performance",
        json_schema_extra=ui_field_config(order=32, label="Batch Denoising"),
    )

    use_lcm_lora: bool = Field(
        default=True,
        description="Use LCM LoRA for faster inference",
        json_schema_extra=ui_field_config(order=33, label="Use LCM LoRA"),
    )

    # ========================================
    # Image Filtering
    # ========================================

    similar_image_filter_enabled: bool = Field(
        default=False,
        description="Enable similar image filter to skip redundant frames",
        json_schema_extra=ui_field_config(order=40, label="Similar Image Filter"),
    )

    similar_image_filter_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Similarity threshold (higher = more strict filtering)",
        json_schema_extra=ui_field_config(order=41, label="Filter Threshold"),
    )

    # ========================================
    # Additional Options
    # ========================================

    image_loopback: bool = Field(
        default=False,
        description="Use last frame as input for the next generation",
        json_schema_extra=ui_field_config(order=49, label="Image Loopback"),
    )

    prompt_weighting: bool = Field(
        default=True,
        description="Enable advanced prompt weighting with Compel",
        json_schema_extra=ui_field_config(order=50, label="Prompt Weighting"),
    )

    # Resolution settings (can be overridden at runtime)
    width: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Output width",
        json_schema_extra=ui_field_config(order=60, label="Width"),
    )

    height: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Output height",
        json_schema_extra=ui_field_config(order=61, label="Height"),
    )
