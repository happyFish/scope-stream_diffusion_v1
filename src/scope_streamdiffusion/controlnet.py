"""ControlNet loading, caching, and frame preprocessing for StreamDiffusion."""

import json
import os
import re
from typing import Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel

# ControlNet model IDs keyed by controlnet_mode
MODEL_IDS: dict[str, str] = {
    "depth": "https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_depth.safetensors",
    "scribble": "https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_scribble.safetensors",
}

_SCRIBBLE_CHECKPOINT = "VACE-Annotators/scribble/anime_style/netG_A_latest.pth"

_DEPTH_MODEL_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
_DEPTH_CHECKPOINT = "Video-Depth-Anything-Small/video_depth_anything_vits.pth"


class ControlNetHandler:
    """Manages ControlNet models and frame preprocessing for StreamDiffusion.

    Models are lazy-loaded on first use per mode and cached. Switching modes stalls
    the stream only on the first call to that mode.

    Usage:
        handler = ControlNetHandler(device, dtype)
        handler.update(mode, video, width, height, scale, reset)
        # then read handler.model, handler.input, handler.scale
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._controlnet_cache: dict[str, ControlNetModel] = {}
        self._depth_model = None
        self._depth_hidden_state = None
        self._depth_min_ema: float | None = None
        self._depth_max_ema: float | None = None
        self._prev_depth_input: torch.Tensor | None = None
        self._scribble_model = None
        self._prev_scribble_input: torch.Tensor | None = None

        self.model: Optional[ControlNetModel] = None
        self.input: Optional[torch.Tensor] = None
        self.scale: float = 1.0

    def update(
        self,
        mode: str,
        video: list | None,
        width: int,
        height: int,
        scale: float,
        reset: bool,
        temporal_smoothing: float = 0.5,
    ) -> None:
        """Update ControlNet state for the current frame.

        Sets self.model, self.input, self.scale based on mode. Call this at the
        start of each __call__ before running the UNet step.

        Args:
            temporal_smoothing: Weight of the current frame in the blend with the
                previous conditioning map. 1.0 = no smoothing (lowest latency),
                0.0 = fully smoothed (previous frame only).
        """
        self.model = None
        self.input = None
        self.scale = scale

        if mode == "depth" and video is not None and len(video) > 0:
            self.model = self._get_model_for_mode("depth")

            if reset:
                self._depth_hidden_state = None
                self._depth_min_ema = None
                self._depth_max_ema = None
                self._prev_depth_input = None

            frame_np = video[0].squeeze(0).cpu().numpy()
            if frame_np.dtype != np.uint8:
                frame_np = (np.clip(frame_np, 0.0, 1.0) * 255).astype(np.uint8)

            # return_tensor=True keeps depth on GPU — avoids an extra GPU→CPU→GPU roundtrip
            depth_t, self._depth_hidden_state = self._get_depth_preprocessor().infer_video_depth_one(
                frame_np,
                input_size=518,
                device="cuda" if self.device.type == "cuda" else "cpu",
                fp32=False,
                cached_hidden_state_list=self._depth_hidden_state,
                return_tensor=True,
            )  # depth_t: (H, W) on self.device

            # EMA on normalization bounds — prevents scale/contrast jumps between frames
            # .min()/.max() are GPU scalars; extracting as Python floats for EMA arithmetic
            d_min, d_max = float(depth_t.min()), float(depth_t.max())
            if self._depth_min_ema is None:
                self._depth_min_ema, self._depth_max_ema = d_min, d_max
            else:
                a = 0.1  # lower = smoother bounds, higher = more responsive
                self._depth_min_ema = a * d_min + (1 - a) * self._depth_min_ema
                self._depth_max_ema = a * d_max + (1 - a) * self._depth_max_ema
            rng = self._depth_max_ema - self._depth_min_ema
            depth_norm = ((depth_t - self._depth_min_ema) / rng).clamp(0.0, 1.0) if rng > 0 else torch.zeros_like(depth_t)

            if depth_norm.shape != (height, width):
                depth_norm = F.interpolate(
                    depth_norm.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0).squeeze(0)

            # (H, W) -> (1, 3, H, W), already on GPU
            self.input = depth_norm.to(dtype=self.dtype).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

            # Output blending — catches residual pixel-level flicker after bounds stabilization
            if self._prev_depth_input is not None and temporal_smoothing < 1.0:
                self.input = temporal_smoothing * self.input + (1 - temporal_smoothing) * self._prev_depth_input
            self._prev_depth_input = self.input

        elif mode == "scribble" and video is not None and len(video) > 0:
            self.model = self._get_model_for_mode("scribble")

            if reset:
                self._prev_scribble_input = None

            frame = video[0].squeeze(0)  # (H, W, C)
            if frame.dtype == torch.uint8:
                frame = frame.float() / 255.0

            # (H, W, C) -> (1, C, H, W)
            frame_input = frame.to(device=self.device, dtype=self.dtype).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                scribble = self._get_scribble_preprocessor()(frame_input)  # (1, 1, H, W)

            # Resize to target resolution if needed
            if scribble.shape[2] != height or scribble.shape[3] != width:
                scribble_np = scribble.squeeze().cpu().float().numpy()
                scribble_pil = PIL.Image.fromarray((scribble_np * 255).astype(np.uint8)).resize((width, height))
                scribble_norm = np.array(scribble_pil).astype(np.float32) / 255.0
                scribble = torch.from_numpy(scribble_norm).to(device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(0)

            # (1, 1, H, W) -> (1, 3, H, W)
            self.input = scribble.to(dtype=self.dtype).repeat(1, 3, 1, 1)

            if self._prev_scribble_input is not None and temporal_smoothing < 1.0:
                self.input = temporal_smoothing * self.input + (1 - temporal_smoothing) * self._prev_scribble_input
            self._prev_scribble_input = self.input

    def _get_model_for_mode(self, mode: str) -> ControlNetModel:
        if mode not in self._controlnet_cache:
            model_id = MODEL_IDS[mode]
            print(f"[ControlNet] Loading {mode} model: {model_id}")
            model = self._load_controlnet(model_id).to(self.device)
            model.eval()
            self._controlnet_cache[mode] = model
            print(f"[ControlNet] {mode} model cached")
        return self._controlnet_cache[mode]

    def _get_depth_preprocessor(self):
        if self._depth_model is None:
            from scope.core.config import get_model_file_path
            from scope.core.pipelines.video_depth_anything.modules import VideoDepthAnything

            print("[Depth] Loading VideoDepthAnything model...")
            checkpoint_path = get_model_file_path(_DEPTH_CHECKPOINT)
            model = VideoDepthAnything(**_DEPTH_MODEL_CONFIG, metric=False)
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True),
                strict=True,
            )
            self._depth_model = model.to(device=self.device).half().eval()
            print("[Depth] Model loaded")
        return self._depth_model

    def _get_scribble_preprocessor(self):
        if self._scribble_model is None:
            from scope.core.config import get_model_file_path
            from scope.core.pipelines.scribble.modules import ContourInference

            print("[Scribble] Loading ContourInference model...")
            checkpoint_path = get_model_file_path(_SCRIBBLE_CHECKPOINT)
            model = ContourInference(input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True)
            model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True),
                strict=True,
            )
            self._scribble_model = model.to(device=self.device, dtype=self.dtype).eval()
            print("[Scribble] Model loaded")
        return self._scribble_model

    def _load_controlnet(self, model_id: str) -> ControlNetModel:
        from scope.core.config import get_models_dir
        from scope.server.download_models import download_hf_repo

        hf_url = re.match(
            r"https://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+)",
            model_id,
        )
        if hf_url:
            repo_id, _, filename = hf_url.group(1), hf_url.group(2), hf_url.group(3)
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
            return self._from_single_file_with_config(model_id, os.path.basename(model_id))

        return ControlNetModel.from_pretrained(model_id, torch_dtype=self.dtype)

    def _from_single_file_with_config(self, local_path, filename: str) -> ControlNetModel:
        from pathlib import Path

        local_path = Path(local_path)
        config_path = local_path.parent / "config.json"

        if not config_path.exists():
            name_lower = filename.lower()
            is_sd2 = "sd21" in name_lower or "sd2" in name_lower or "v2" in name_lower
            is_sd1 = "sd15" in name_lower or "sd1" in name_lower or "v1" in name_lower
            if is_sd1 and not is_sd2:
                cross_attention_dim = 768
                attention_head_dim = 8
                use_linear_projection = False
            else:
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
