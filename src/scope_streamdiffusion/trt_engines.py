"""TensorRT engine builder + adapter classes.

In-place TRT acceleration. Uses our vendored `_trt/` exporter
(originally from prism) to compile UNet to a TensorRT engine, then swaps
in a thin adapter that mimics the diffusers `UNet2DConditionModel` call
signature so the rest of `pipeline.py` is unchanged.

Engines cache to ~/.cache/scope-streamdiffusion-trt/{model_hash}/.
First-run build: 5-10 min. Subsequent runs reuse the engine (~2 s load).

Dynamic shapes: spatial 256-1024, batch min..max. One engine covers the
range — no rebuild on resolution change.

NOTE: ControlNet residuals are NOT yet wired through the UNet engine.
The vendored UNet engine ignores `down_block_additional_residuals` /
`mid_block_additional_residual` (they go into **kwargs and get dropped).
For controlnet=depth/scribble modes, use acceleration_mode='none' until
Phase 3 lands the combined UNet+ControlNet engine.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

logger = logging.getLogger(__name__)


# IMPORTANT — Scope must be launched with LD_LIBRARY_PATH including BOTH:
#   * $VENV/lib/python3.12/site-packages/nvidia/cudnn/lib (cuDNN runtime)
#   * $VENV/lib/python3.12/site-packages/tensorrt_libs    (libnvinfer_builder_resource_sm*.so)
#
# Without the latter, TRT's engine builder fails to dlopen the per-SM kernel
# library at build time and aborts with:
#   Error Code 6: Unable to load library: libnvinfer_builder_resource_sm89.so
# Setting LD_LIBRARY_PATH at runtime via os.environ doesn't help — the dynamic
# linker reads it once at process start, and tensorrt_libs's ctypes preload
# only covers the libs it knows about up front, not the lazy per-SM dlopens.


def _cache_root() -> Path:
    return Path.home() / ".cache" / "scope-streamdiffusion-trt"


def _model_cache_dir(model_id: str, suffix: str = "") -> Path:
    h = hashlib.sha256(model_id.encode()).hexdigest()[:16]
    name = f"{h}_{suffix}" if suffix else h
    d = _cache_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_unet_engine(
    unet: UNet2DConditionModel,
    *,
    model_id: str,
    image_height: int = 512,
    image_width: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
) -> Path:
    """Build (or reuse) a TRT engine for the given UNet. Returns engine path.

    The engine accepts dynamic spatial dims 256-1024 and batch min..max.
    Plain SD UNet only — does not handle ControlNet residuals. See the
    module docstring.
    """
    from ._trt import UNet, compile_unet, create_onnx_path

    suffix = f"unet_b{min_batch_size}-{max_batch_size}_h{image_height}_w{image_width}"
    cache_dir = _model_cache_dir(model_id, suffix)
    onnx_dir = cache_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_path = cache_dir / "unet.engine"

    if engine_path.exists():
        logger.info(f"[TRT] Reusing cached UNet engine: {engine_path}")
        return engine_path

    logger.info(f"[TRT] Building UNet engine -> {engine_path} (5-10 min on first build)")

    # The vendored UNet model class is plain SD — three inputs, one output,
    # no IPAdapter / ControlNet leak. embedding_dim comes from the actual
    # UNet config so SD1.5/2.1/SDXL all work.
    unet_model = UNet(
        fp16=True,
        device=str(unet.device) if unet.device.type != "meta" else "cuda",
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=unet.config.cross_attention_dim,
        unet_dim=unet.config.in_channels,
    )
    compile_unet(
        unet,
        unet_model,
        str(create_onnx_path("unet", str(onnx_dir), opt=False)),
        str(create_onnx_path("unet", str(onnx_dir), opt=True)),
        str(engine_path),
        opt_batch_size=min_batch_size,
        engine_build_options={
            "build_dynamic_shape": True,
            "build_static_batch": False,
        },
    )
    logger.info(f"[TRT] UNet engine built: {engine_path}")
    return engine_path


class TRTUNetAdapter:
    """Thin wrapper for the vendored UNet engine.

    Existing `pipeline.py._unet_step` calls:

        model_pred = self.unet(
            x_t_latent_plus_uc, t_list,
            encoder_hidden_states=embeds,
            added_cond_kwargs=...,
            down_block_additional_residuals=...,
            mid_block_additional_residual=...,
            return_dict=False,
        )[0]

    The vendored engine __call__ returns `UNet2DConditionOutput(sample=...)`
    and ignores everything beyond (sample, timestep, encoder_hidden_states).
    We adapt to support both `return_dict=False` (tuple) and the default
    (UNet2DConditionOutput).
    """

    def __init__(self, engine_path: Path, cuda_stream):
        from ._trt import UNet2DConditionModelEngine
        self.engine = UNet2DConditionModelEngine(str(engine_path), cuda_stream)
        # Surface that pipeline.py reads
        self.config = _ConfigShim()
        self.add_embedding = None  # SD2.1: no add_embedding (SDXL would)

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        out = self.engine(
            latent_model_input=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        # out is UNet2DConditionOutput
        if return_dict:
            return out
        return (out.sample,)

    # diffusers Module surface — pipeline.py never calls these in practice
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class _ConfigShim:
    """Diffusers config object surface — read by pipeline._prepare_runtime_state."""

    def __init__(self):
        self.addition_time_embed_dim = None
        self.in_channels = 4
        self.out_channels = 4
        self.cross_attention_dim = 1024


def make_cuda_stream():
    """Polygraphy CUDA stream wrapper used by the engine classes."""
    from polygraphy import cuda
    return cuda.Stream()
