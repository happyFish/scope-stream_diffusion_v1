"""StreamDiffusion pipeline implementation for Scope."""

import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import torch
import numpy as np
import PIL.Image
from diffusers import (
    ControlNetModel,
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
from scope.core.pipelines.wan2_1.vace import VACEEnabledPipeline

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


class StreamDiffusionPipeline(Pipeline, VACEEnabledPipeline):
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
        controlnet_model_id: str = "",
        **kwargs,  # noqa: ARG002
    ) -> None:
        """Initialize the StreamDiffusion pipeline.

        Args:
            device: Torch device to use
            model_id: Model ID or path to load
            torch_dtype: Data type for tensors
            controlnet_model_id: HuggingFace model ID or local path for ControlNet (empty = disabled)
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
        self.controlnet = None
        self.controlnet_input = None
        self.controlnet_conditioning_scale = 1.0

        if controlnet_model_id:
            print(f"Loading ControlNet: {controlnet_model_id}")
            self.controlnet = self._load_controlnet(controlnet_model_id).to(self.device)
            self.controlnet.eval()
            print("ControlNet loaded")

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
        self.strength = 1.0
        self.guidance_scale = 0.0
        self.delta = 1.0
        self.t_list = [0]
        self.similar_image_filter = False

        print("StreamDiffusion pipeline initialized")

    def _load_controlnet(self, model_id: str) -> ControlNetModel:
        """Load a ControlNet model, downloading via Scope's infrastructure if needed.

        Accepts:
          - HuggingFace URL: https://huggingface.co/{repo}/{resolve|blob}/{rev}/{file}
          - Local .safetensors / .ckpt / .bin file path
          - Full diffusers repo ID: org/repo (must have config.json)
        """
        from scope.core.config import get_models_dir
        from scope.server.download_models import download_hf_repo

        hf_url = re.match(
            r"https://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+)",
            model_id,
        )
        if hf_url:
            repo_id, revision, filename = (
                hf_url.group(1),
                hf_url.group(2),
                hf_url.group(3),
            )
            local_dir = get_models_dir() / repo_id.split("/")[-1]
            local_path = local_dir / filename
            if not local_path.exists():
                print(f"  Downloading {filename} from {repo_id}")
                download_hf_repo(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    allow_patterns=[filename],
                    pipeline_id="streamdiffusion",
                )
            print(f"  Loaded from: {local_path}")
            return self._from_single_file_with_config(local_path, filename)

        if model_id.endswith((".safetensors", ".ckpt", ".bin")):
            return self._from_single_file_with_config(
                model_id, os.path.basename(model_id)
            )

        return ControlNetModel.from_pretrained(model_id, torch_dtype=self.dtype)

    def _from_single_file_with_config(
        self, local_path, filename: str
    ) -> ControlNetModel:
        """Load a single-file ControlNet checkpoint, writing a config.json alongside it
        so diffusers doesn't try to fetch one from stable-diffusion-v1-5 on HuggingFace."""
        from pathlib import Path

        local_path = Path(local_path)
        config_path = local_path.parent / "config.json"

        if not config_path.exists():
            # Detect SD version from filename; default to SD 2.1 if ambiguous
            name_lower = filename.lower()
            is_sd2 = "sd21" in name_lower or "sd2" in name_lower or "v2" in name_lower
            is_sd1 = "sd15" in name_lower or "sd1" in name_lower or "v1" in name_lower
            if is_sd1 and not is_sd2:
                cross_attention_dim = 768
                attention_head_dim = 8
                use_linear_projection = False
            else:
                # SD 2.1 architecture
                cross_attention_dim = 1024
                attention_head_dim = [5, 10, 20, 20]
                use_linear_projection = True

            config = {
                "_class_name": "ControlNetModel",
                "act_fn": "silu",
                "attention_head_dim": attention_head_dim,
                "block_out_channels": [320, 640, 1280, 1280],
                "class_embed_type": None,
                "conditioning_embedding_out_channels": [16, 32, 96, 256],
                "controlnet_conditioning_channel_order": "rgb",
                "cross_attention_dim": cross_attention_dim,
                "down_block_types": [
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ],
                "downsample_padding": 1,
                "flip_sin_to_cos": True,
                "freq_shift": 0,
                "in_channels": 4,
                "layers_per_block": 2,
                "mid_block_scale_factor": 1,
                "norm_eps": 1e-05,
                "norm_num_groups": 32,
                "num_class_embeds": None,
                "only_cross_attention": False,
                "projection_class_embeddings_input_dim": None,
                "resnet_time_scale_shift": "default",
                "transformer_layers_per_block": 1,
                "upcast_attention": False,
                "use_linear_projection": use_linear_projection,
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Wrote ControlNet config: {config_path}")

        return ControlNetModel.from_single_file(
            str(local_path),
            config=str(local_path.parent),
            torch_dtype=self.dtype,
        )

    def _load_model(self, model_id: str) -> DiffusionPipeline:
        """Load the diffusion model."""
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None,
            )
            pipe = pipe.to(self.device)
            return pipe
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            raise

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

    def prepare(self, **kwargs) -> Requirements:
        """Specify pipeline requirements.

        Returns:
            Requirements: Pipeline requirements (e.g., input size)
        """
        return Requirements(input_size=1)  # Process 1 frame at a time

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
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        t_index_list: Optional[List[int]] = None,
    ):
        """Prepare runtime state from parameters.

        This should be called at the start of __call__ to set up the pipeline
        for the current generation.
        """
        # Set dimensions
        self.width = width
        self.height = height
        self.latent_height = int(height // self.pipe.vae_scale_factor)
        self.latent_width = int(width // self.pipe.vae_scale_factor)

        # Set parameters
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.delta = delta
        self.cfg_type = cfg_type
        self.use_denoising_batch = use_denoising_batch
        self.do_add_noise = do_add_noise

        # Set timestep indices
        if t_index_list is None:
            # For SD Turbo/LCM models, use single step for best results
            # For other models, can use multiple steps
            t_index_list = [0]  # Single step denoising (like your working project!)
        self.t_list = t_index_list
        self.denoising_steps_num = len(t_index_list)

        print(
            f"Using t_index_list: {t_index_list} from {num_inference_steps} total steps"
        )

        # Calculate batch size
        self.frame_bff_size = 1
        if use_denoising_batch:
            self.batch_size = self.denoising_steps_num * self.frame_bff_size
        else:
            self.batch_size = self.frame_bff_size

        # Set random seed
        self.generator.manual_seed(seed)

        # Initialize latent buffer
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

        # Encode and blend current prompts
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        current_embeds, _ = self._encode_prompts_array(
            prompts, prompt_interpolation_method
        )

        if transition is not None and not self.embedding_blender.is_transitioning():
            # New transition requested — start it toward target_prompts
            transition_config = parse_transition_config(transition)
            target_prompts_raw = transition.get("target_prompts", [])

            if transition_config.num_steps > 0 and target_prompts_raw:
                target_prompts = self._normalize_prompts(target_prompts_raw)

                # Use previous frame's embedding as the transition source
                source_embedding = self._previous_prompt_embeddings
                if source_embedding is None:
                    source_embedding = current_embeds[0:1].detach()

                # Encode target prompts without overwriting the current cache
                target_texts = [p.get("text", "") for p in target_prompts]
                target_weights = [p.get("weight", 1.0) for p in target_prompts]
                target_encoded = [
                    self._encode_single_prompt(t)[0] for t in target_texts
                ]
                target_embedding = self.embedding_blender.blend(
                    target_encoded,
                    target_weights,
                    prompt_interpolation_method,
                    cache_result=False,
                )

                print(
                    f"\nStarting transition ({transition_config.num_steps} steps, method={transition_config.temporal_interpolation_method}):"
                )
                print(
                    f"  Target prompts: {[p.get('text', '') for p in target_prompts]}"
                )

                self.embedding_blender.start_transition(
                    source_embedding=source_embedding,
                    target_embedding=target_embedding,
                    num_steps=transition_config.num_steps,
                    temporal_interpolation_method=transition_config.temporal_interpolation_method,
                )

                first_embedding = self.embedding_blender.get_next_embedding()
                if first_embedding is not None:
                    self.prompt_embeds = first_embedding.repeat(self.batch_size, 1, 1)
                else:
                    self.prompt_embeds = current_embeds
            else:
                self.prompt_embeds = current_embeds

        elif self.embedding_blender.is_transitioning():
            # Continue ongoing transition
            next_embedding = self.embedding_blender.get_next_embedding()
            if next_embedding is not None:
                self.prompt_embeds = next_embedding.repeat(self.batch_size, 1, 1)
                print(
                    f"  Transition continuing, remaining steps: {len(self.embedding_blender._transition_queue)}"
                )
            else:
                # Transition just completed
                self.prompt_embeds = current_embeds

        else:
            self.prompt_embeds = current_embeds

        # Cache embedding as source for the next transition
        self._previous_prompt_embeddings = self.prompt_embeds[0:1].detach()

        print("\nPrompt embeddings:")
        print(f"  Shape: {self.prompt_embeds.shape}")
        print(
            f"  Range: [{self.prompt_embeds.min():.3f}, {self.prompt_embeds.max():.3f}]"
        )
        print(f"  Mean: {self.prompt_embeds.mean():.3f}")
        print(f"  Num prompts: {len(prompts)}, Method: {prompt_interpolation_method}")

        # Set timesteps
        self._set_timesteps(num_inference_steps, strength)

        # Initialize noise
        self._initialize_noise()

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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode multiple weighted prompts and blend them.

        Args:
            prompt_items: List of {"text": str, "weight": float}
            interpolation_method: "linear" or "slerp"

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

        # Handle SDXL additional embeddings
        if self.sdxl and blended_pooled_embeds is not None:
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
        print("\n_predict_x0_batch:")
        print(f"  Input latent shape: {x_t_latent.shape}")
        print(f"  Input latent range: [{x_t_latent.min():.3f}, {x_t_latent.max():.3f}]")
        print(f"  Denoising steps: {self.denoising_steps_num}")
        print(f"  Use denoising batch: {self.use_denoising_batch}")
        print(f"  Timesteps: {self.sub_timesteps}")

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
        print()
        print("------------------")
        print("Raw kwargs:", kwargs)

        # Extract parameters - handle Scope's parameter format
        video = kwargs.get("video", None)

        # Extract ControlNet conditioning frames (routed here by PipelineProcessor when vace_enabled=True)
        vace_input_frames = kwargs.get("vace_input_frames", None)

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

        print(f"Config object: {config}")
        print(f"Config type: {type(config)}")

        if config:
            print(f"Config attributes: {dir(config)}")

        # Helper to get value from config first, then kwargs, then default
        def get_param(key, default):
            # First check config if available (preferred source)
            if config and hasattr(config, key):
                value = getattr(config, key)
                print(f"  {key} from config: {value}")
                return value
            # Then check kwargs directly
            if key in kwargs:
                value = kwargs[key]
                print(f"  {key} from kwargs: {value}")
                return value
            # Finally use default
            print(f"  {key} using default: {default}")
            return default

        # Extract all parameters with config fallback
        prompt_interpolation_method = get_param("prompt_interpolation_method", "linear")
        guidance_scale = get_param("guidance_scale", 0.0)

        # SD Turbo: Use single timestep (t_index_list=[0]) but set schedule length
        # This matches your working project setup
        num_inference_steps = get_param("num_inference_steps", 25)

        # For img2img with SD Turbo, need higher strength for visible changes
        # 0.5-0.7 = moderate, 0.8-0.95 = heavy transformation
        strength = get_param("strength", 0.8)

        seed = get_param("seed", 42)
        delta = get_param("delta", 1.0)
        width = get_param("width", 512)
        height = get_param("height", 512)
        use_denoising_batch = get_param("use_denoising_batch", True)
        do_add_noise = get_param("do_add_noise", True)
        similar_image_filter_enabled = get_param("similar_image_filter_enabled", False)
        image_loopback = get_param("image_loopback", False)
        vace_context_scale = get_param("vace_context_scale", 1.0)

        # ControlNet conditioning diagnostics
        print(f"[ControlNet] controlnet loaded: {self.controlnet is not None}")
        print(
            f"[ControlNet] vace_input_frames: {'present (' + str(len(vace_input_frames)) + ' frames)' if vace_input_frames is not None else 'ABSENT - check PipelineProcessor routing'}"
        )
        print(
            f"[ControlNet] video: {'present (' + str(len(video)) + ' frames)' if video is not None else 'absent'}"
        )
        print(f"[ControlNet] vace_context_scale: {vace_context_scale}")

        # Process ControlNet conditioning frame if available
        self.controlnet_input = None
        if (
            self.controlnet is not None
            and vace_input_frames is not None
            and len(vace_input_frames) > 0
        ):
            cond_frame = vace_input_frames[0]
            while cond_frame.ndim > 3:
                cond_frame = cond_frame.squeeze(0)
            if cond_frame.dtype == torch.uint8:
                cond_frame = cond_frame.float() / 255.0
            cond_frame = cond_frame.to(device=self.device, dtype=self.dtype)
            # Resize if needed
            actual_h, actual_w = cond_frame.shape[0], cond_frame.shape[1]
            if actual_h != height or actual_w != width:
                cond_np = (cond_frame.cpu().numpy() * 255).astype(np.uint8).squeeze()
                cond_pil = PIL.Image.fromarray(cond_np).resize((width, height))
                cond_frame = torch.from_numpy(np.array(cond_pil)).float() / 255.0
                cond_frame = cond_frame.to(device=self.device, dtype=self.dtype)
            # (H, W, C) -> (1, C, H, W)
            self.controlnet_input = cond_frame.permute(2, 0, 1).unsqueeze(0)
            self.controlnet_conditioning_scale = vace_context_scale
            print(
                f"[ControlNet] conditioning input ready: shape={self.controlnet_input.shape}, scale={vace_context_scale}"
            )
        else:
            print(
                f"[ControlNet] NO conditioning — controlnet={self.controlnet is not None}, vace_input_frames={'present' if vace_input_frames is not None else 'ABSENT'}"
            )

        print(f"Extracted prompts: {prompts}")
        print(
            f"Parameters: steps={num_inference_steps}, strength={strength}, guidance={guidance_scale}, blend={prompt_interpolation_method}"
        )

        # Extract transition (explicit transition overrides prompt-change detection)
        transition = kwargs.get("transition", None)

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
        )

        frame = None

        # Process input
        if image_loopback or (
            (video is None or len(video) == 0) and self.prev_image_result is not None
        ):
            print("Using image loopback:")
            frame = self.prev_image_result
        elif video is not None and len(video) > 0:
            print("Processing video input:")
            print(f"  Video shape: {video[0].shape}")
            print(f"  Video dtype: {video[0].dtype}")
            print(f"  Video device: {video[0].device}")

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
                print("  Converting from uint8 [0, 255] to float [0, 1]")
                frame = frame.float() / 255.0

            # Move to device
            frame = frame.to(device=self.device, dtype=self.dtype)

            # Get actual dimensions after squeezing
            actual_height, actual_width = frame.shape[0], frame.shape[1]

            # Resize if needed
            if actual_height != height or actual_width != width:
                # Convert to PIL for resizing
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                # Squeeze any remaining extra dimensions for PIL
                frame_np = frame_np.squeeze()
                frame_pil = PIL.Image.fromarray(frame_np)
                frame_pil = frame_pil.resize((width, height))
                frame = torch.from_numpy(np.array(frame_pil)).float() / 255.0
                frame = frame.to(device=self.device, dtype=self.dtype)

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
        print("\nAfter diffusion:")
        print(f"  Latent shape: {x_0_pred_out.shape}")
        print(f"  Latent range: [{x_0_pred_out.min():.3f}, {x_0_pred_out.max():.3f}]")

        # Decode to image space
        x_output = self._decode_image(x_0_pred_out).detach().clone()
        print("\nAfter VAE decode (before normalization):")
        print(f"  Image shape: {x_output.shape}")
        print(f"  Image range: [{x_output.min():.3f}, {x_output.max():.3f}]")

        # Normalize from [-1, 1] to [0, 1] (VAE outputs in range [-1, 1])
        x_output = (x_output / 2 + 0.5).clamp(0, 1)
        print("\nAfter normalization:")
        print(f"  Image range: [{x_output.min():.3f}, {x_output.max():.3f}]")

        # Convert back to Scope format: (B, C, H, W) -> (T, H, W, C)
        output = x_output.permute(0, 2, 3, 1)

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
