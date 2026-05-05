"""ControlNet loading, caching, and frame preprocessing for StreamDiffusion."""

import json
import os
import re
import time
from typing import Optional

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers import ControlNetModel

# Whether the Scope VideoDepthAnything has the GPU-native infer_depth_tensor() method.
# Detected at runtime on first depth call.
_HAS_GPU_DEPTH: bool | None = None

# ControlNet model IDs keyed by controlnet_mode
MODEL_IDS: dict[str, str] = {
    "depth": "https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_depth.safetensors",
    "scribble": "https://huggingface.co/thibaud/controlnet-sd21/resolve/main/control_v11p_sd21_scribble.safetensors",
}

_SCRIBBLE_CHECKPOINT = "VACE-Annotators/scribble/anime_style/netG_A_latest.pth"

_DEPTH_MODEL_CONFIG = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
}
_DEPTH_CHECKPOINT = "Video-Depth-Anything-Small/video_depth_anything_vits.pth"

# ImageNet normalization constants used by VideoDepthAnything's preprocessing.
_DEPTH_MEAN = (0.485, 0.456, 0.406)
_DEPTH_STD = (0.229, 0.224, 0.225)


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
        self._last_depth_shape: tuple[int, int] | None = None
        self._depth_min_ema: float | None = None
        self._depth_max_ema: float | None = None
        self._prev_depth_input: torch.Tensor | None = None
        self._scribble_model = None
        self._prev_scribble_input: torch.Tensor | None = None

        # Depth frame-skipping: reuse cached depth every N frames to save 20-40ms/frame
        self._depth_frame_counter: int = 0
        self._depth_skip_interval: int = 3  # run depth every Nth frame
        self._depth_input_size: int = 518
        self._use_depth_temporal_cache: bool = True

        # FPS telemetry for the depth path. Logged ~every 2s so the cost of
        # different input sizes / cache settings is visible without flooding stdout.
        self._depth_log_interval_s: float = 2.0
        self._depth_log_last_time: float = time.monotonic()
        self._depth_log_calls: int = 0   # depth-model invocations since last log
        self._depth_log_skips: int = 0   # cached-reuse frames since last log
        self._depth_log_total_ms: float = 0.0  # sum of per-call durations

        # Lazy-built GPU normalization tensors — created on first depth call so
        # they live on the right device. Shape (1, 1, 3, 1, 1) to broadcast
        # against the (B, T, C, H, W) model input.
        self._depth_norm_mean: torch.Tensor | None = None
        self._depth_norm_std: torch.Tensor | None = None

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
        temporal_smoothing: float = 1.0,
        depth_min: float = 0,
        depth_max: float = 12,
        depth_skip_interval: int = 3,
        depth_input_size: int = 518,
        depth_temporal_cache: bool = True,
    ) -> None:
        """Update ControlNet state for the current frame.

        Sets self.model, self.input, self.scale based on mode. Call this at the
        start of each __call__ before running the UNet step.

        Args:
            temporal_smoothing: Weight of the current frame in the blend with the
                previous conditioning map. 1.0 = no smoothing (lowest latency),
                0.0 = fully smoothed (previous frame only).
            depth_skip_interval: Run depth model every Nth frame; reuse cached
                depth map on intermediate frames to save 20-40ms/frame.
        """
        self.model = None
        self.input = None
        self.scale = scale
        self._depth_skip_interval = max(1, depth_skip_interval)

        # Depth model running params
        new_input_size = max(14, (int(depth_input_size) // 14) * 14)
        if new_input_size != self._depth_input_size:
            # Patch grid will change — drop the temporal cache to avoid the
            # same shape mismatch we guard against on resolution changes.
            self._depth_hidden_state = None
            self._last_depth_shape = None
            self._depth_input_size = new_input_size
        if bool(depth_temporal_cache) != self._use_depth_temporal_cache:
            self._depth_hidden_state = None
            self._use_depth_temporal_cache = bool(depth_temporal_cache)

        if mode == "depth" and video is not None and len(video) > 0:
            self.model = self._get_model_for_mode("depth")

            if reset:
                self._depth_hidden_state = None
                self._depth_min_ema = None
                self._depth_max_ema = None
                self._prev_depth_input = None
                self._depth_frame_counter = 0

            # Frame-skipping: reuse cached depth on non-key frames
            self._depth_frame_counter += 1
            run_depth = (
                self._prev_depth_input is None
                or self._depth_frame_counter >= self._depth_skip_interval
            )

            if run_depth:
                self._depth_frame_counter = 0

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                depth_t = self._run_depth_inference(video[0])
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                self._depth_log_total_ms += (time.perf_counter() - t0) * 1000.0
                self._depth_log_calls += 1

                # EMA on normalization bounds — prevents scale/contrast jumps between frames
                d_min, d_max = float(depth_t.min()), float(depth_t.max())
                if self._depth_min_ema is None:
                    self._depth_min_ema, self._depth_max_ema = d_min, d_max
                else:
                    a = 0.1  # lower = smoother bounds, higher = more responsive
                    self._depth_min_ema = a * d_min + (1 - a) * self._depth_min_ema
                    self._depth_max_ema = a * d_max + (1 - a) * self._depth_max_ema
                rng = self._depth_max_ema - self._depth_min_ema
                depth_norm = (
                    ((depth_t - self._depth_min_ema) / rng).clamp(0.0, 1.0)
                    if rng > 0
                    else torch.zeros_like(depth_t)
                )

                if depth_norm.shape != (height, width):
                    depth_norm = (
                        F.interpolate(
                            depth_norm.unsqueeze(0).unsqueeze(0),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=True,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

                depth_norm = torch.clamp(depth_norm, min=depth_min, max=depth_max)

                # (H, W) -> (1, 3, H, W), already on GPU
                self.input = (
                    depth_norm.to(dtype=self.dtype)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(1, 3, 1, 1)
                )

                # Output blending — catches residual pixel-level flicker after bounds stabilization.
                # Drop the prev buffer if the resolution changed (live width/height swap).
                if (
                    self._prev_depth_input is not None
                    and self._prev_depth_input.shape == self.input.shape
                    and temporal_smoothing < 1.0
                ):
                    self.input = (
                        temporal_smoothing * self.input
                        + (1 - temporal_smoothing) * self._prev_depth_input
                    )
                self._prev_depth_input = self.input
            else:
                # Reuse cached depth from previous key frame
                self.input = self._prev_depth_input
                self._depth_log_skips += 1

            # Periodic FPS log for the depth path. Reports model FPS (only the
            # frames where depth actually ran) and effective FPS (counting the
            # skipped/reused frames). Helps tune input_size / skip_interval.
            now = time.monotonic()
            elapsed = now - self._depth_log_last_time
            if elapsed >= self._depth_log_interval_s and (
                self._depth_log_calls + self._depth_log_skips
            ) > 0:
                model_fps = self._depth_log_calls / elapsed
                eff_fps = (
                    self._depth_log_calls + self._depth_log_skips
                ) / elapsed
                avg_ms = (
                    self._depth_log_total_ms / self._depth_log_calls
                    if self._depth_log_calls
                    else 0.0
                )
                print(
                    f"[Depth] model={model_fps:.1f} fps ({avg_ms:.1f} ms/call), "
                    f"effective={eff_fps:.1f} fps "
                    f"(calls={self._depth_log_calls}, "
                    f"reused={self._depth_log_skips}, "
                    f"size={self._depth_input_size}, "
                    f"cache={'on' if self._use_depth_temporal_cache else 'off'})",
                    flush=True,
                )
                self._depth_log_last_time = now
                self._depth_log_calls = 0
                self._depth_log_skips = 0
                self._depth_log_total_ms = 0.0

        elif mode == "scribble" and video is not None and len(video) > 0:
            self.model = self._get_model_for_mode("scribble")

            if reset:
                self._prev_scribble_input = None

            frame = video[0].squeeze(0)  # (H, W, C)
            if frame.dtype == torch.uint8:
                frame = frame.float() / 255.0

            # (H, W, C) -> (1, C, H, W)
            frame_input = (
                frame.to(device=self.device, dtype=self.dtype)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            with torch.no_grad():
                scribble = self._get_scribble_preprocessor()(
                    frame_input
                )  # (1, 1, H, W)

            # Resize to target resolution if needed
            if scribble.shape[2] != height or scribble.shape[3] != width:
                scribble_np = scribble.squeeze().cpu().float().numpy()
                scribble_pil = PIL.Image.fromarray(
                    (scribble_np * 255).astype(np.uint8)
                ).resize((width, height))
                scribble_norm = np.array(scribble_pil).astype(np.float32) / 255.0
                scribble = (
                    torch.from_numpy(scribble_norm)
                    .to(device=self.device, dtype=self.dtype)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

            # (1, 1, H, W) -> (1, 3, H, W)
            self.input = scribble.to(dtype=self.dtype).repeat(1, 3, 1, 1)

            if (
                self._prev_scribble_input is not None
                and self._prev_scribble_input.shape == self.input.shape
                and temporal_smoothing < 1.0
            ):
                self.input = (
                    temporal_smoothing * self.input
                    + (1 - temporal_smoothing) * self._prev_scribble_input
                )
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
            from scope.core.pipelines.video_depth_anything.modules import (
                VideoDepthAnything,
            )

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

    def _run_depth_inference_direct(
        self, frame_t: torch.Tensor, input_size: int, cache_in
    ) -> tuple[torch.Tensor, object]:
        """GPU-native depth path that bypasses Scope's CPU-roundtrip wrappers.

        Calls model.forward directly with a square (input_size, input_size) tensor
        already on the GPU — no PIL.Resize, no numpy hop. Returns depth at the
        model's input resolution; caller upsamples to controlnet H/W.

        Args:
            frame_t: (H, W, C) tensor on self.device, uint8 [0,255] or float [0,1].
            input_size: Square model input size (must be multiple of 14).
            cache_in: Cached hidden states from previous frame, or None.

        Returns:
            (depth_t, new_cache): depth at (input_size, input_size), and the
            updated hidden-state list.
        """
        # Build (and cache) the broadcastable mean/std on first use.
        if self._depth_norm_mean is None or self._depth_norm_mean.device != self.device:
            self._depth_norm_mean = torch.tensor(
                _DEPTH_MEAN, device=self.device, dtype=torch.float16
            ).view(1, 1, 3, 1, 1)
            self._depth_norm_std = torch.tensor(
                _DEPTH_STD, device=self.device, dtype=torch.float16
            ).view(1, 1, 3, 1, 1)

        # Move to GPU and convert dtype in one shot. video_playlist may emit
        # frames on CPU (its own device defaults to cpu), so we can't assume
        # the input is already on self.device.
        if frame_t.dtype == torch.uint8:
            x = frame_t.to(device=self.device, dtype=torch.float16) / 255.0
        else:
            x = frame_t.to(device=self.device, dtype=torch.float16).clamp(0.0, 1.0)

        # (H, W, C) -> (1, 1, C, H, W) and resize to (input_size, input_size).
        # Square resize is intentional: the depth map gets bilinear-upsampled to
        # controlnet H/W downstream anyway, so aspect-preservation buys nothing
        # for conditioning and a fixed shape avoids motion-cache invalidation.
        x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        x_btchw = x.flatten(0, 1)  # (1, C, H, W) for interpolate
        x_btchw = F.interpolate(
            x_btchw, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        x = x_btchw.unsqueeze(1)  # (1, 1, C, H, W)
        x = (x - self._depth_norm_mean) / self._depth_norm_std

        # forward returns depth (B, T, H, W) and hidden states.
        depth, new_cache = self._depth_model(x, cached_hidden_state_list=cache_in)
        # depth: (1, 1, H, W) — squeeze to (H, W), float32 for downstream EMA math.
        depth_t = depth.squeeze(0).squeeze(0).to(torch.float32)
        return depth_t, new_cache

    def _run_depth_inference(self, frame: torch.Tensor) -> torch.Tensor:
        """Run depth inference on a single frame, preferring the GPU-native path.

        Order of preference: infer_depth_tensor() (Scope shim, if present),
        our own direct GPU forward, then the legacy CPU-roundtrip path.

        Args:
            frame: (T, H, W, C) or (H, W, C) tensor from the video input.

        Returns:
            depth_t: (H, W) float32 tensor on self.device.
        """
        global _HAS_GPU_DEPTH

        frame_t = frame.squeeze(0)  # (H, W, C)

        # Drop stale temporal cache when input resolution/aspect changes — the
        # depth model's patch grid is shape-dependent, so a cache from a square
        # clip can't be concatenated with a widescreen frame's hidden state.
        shape_key = (int(frame_t.shape[0]), int(frame_t.shape[1]))
        if shape_key != self._last_depth_shape:
            self._depth_hidden_state = None
            self._last_depth_shape = shape_key

        depth_model = self._get_depth_preprocessor()
        device_str = "cuda" if self.device.type == "cuda" else "cpu"

        # Detect GPU-native support on first call
        if _HAS_GPU_DEPTH is None:
            _HAS_GPU_DEPTH = hasattr(depth_model, "infer_depth_tensor")
            if _HAS_GPU_DEPTH:
                print("[Depth] Using GPU-native infer_depth_tensor() path")
            else:
                print("[Depth] Using direct-GPU path (bypassing legacy CPU roundtrip)")

        cache_in = self._depth_hidden_state if self._use_depth_temporal_cache else None
        input_size = self._depth_input_size

        if not _HAS_GPU_DEPTH and self.device.type == "cuda":
            # Direct GPU forward — same model, no PIL/numpy detour.
            depth_t, new_cache = self._run_depth_inference_direct(
                frame_t, input_size, cache_in
            )
            self._depth_hidden_state = new_cache if self._use_depth_temporal_cache else None
            return depth_t

        if _HAS_GPU_DEPTH:
            # GPU-native path: keep everything on GPU
            if frame_t.dtype == torch.uint8:
                frame_gpu = frame_t.float() / 255.0
            else:
                frame_gpu = frame_t.clamp(0.0, 1.0)
            # (H, W, C) -> (1, C, H, W)
            frame_bchw = frame_gpu.to(
                device=self.device, dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0)

            depth_t, new_cache = depth_model.infer_depth_tensor(
                frame_bchw,
                input_size=input_size,
                fp32=False,
                cached_hidden_state_list=cache_in,
            )
            self._depth_hidden_state = new_cache if self._use_depth_temporal_cache else None
            return depth_t  # (H, W) on GPU
        else:
            # Legacy path: CPU roundtrip via numpy
            if frame_t.dtype == torch.uint8:
                frame_np = frame_t.cpu().numpy()
            else:
                frame_np = (frame_t.clamp(0.0, 1.0).cpu().numpy() * 255).astype(
                    np.uint8
                )

            # Try return_tensor=True first (available with updated Scope)
            try:
                depth_t, new_cache = depth_model.infer_video_depth_one(
                    frame_np,
                    input_size=input_size,
                    device=device_str,
                    fp32=False,
                    cached_hidden_state_list=cache_in,
                    return_tensor=True,
                )
                self._depth_hidden_state = new_cache if self._use_depth_temporal_cache else None
                if isinstance(depth_t, torch.Tensor):
                    return depth_t.to(device=self.device, dtype=torch.float32)
            except TypeError:
                pass

            # Oldest path: numpy roundtrip
            depth_np, new_cache = depth_model.infer_video_depth_one(
                frame_np,
                input_size=input_size,
                device=device_str,
                fp32=False,
                cached_hidden_state_list=cache_in,
            )
            self._depth_hidden_state = new_cache if self._use_depth_temporal_cache else None
            return torch.from_numpy(depth_np).to(
                device=self.device, dtype=torch.float32
            )

    def _get_scribble_preprocessor(self):
        if self._scribble_model is None:
            from scope.core.config import get_model_file_path
            from scope.core.pipelines.scribble.modules import ContourInference

            print("[Scribble] Loading ContourInference model...")
            checkpoint_path = get_model_file_path(_SCRIBBLE_CHECKPOINT)
            model = ContourInference(
                input_nc=3, output_nc=1, n_residual_blocks=3, sigmoid=True
            )
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
            return self._from_single_file_with_config(
                model_id, os.path.basename(model_id)
            )

        return ControlNetModel.from_pretrained(model_id, torch_dtype=self.dtype)

    def _from_single_file_with_config(
        self, local_path, filename: str
    ) -> ControlNetModel:
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
