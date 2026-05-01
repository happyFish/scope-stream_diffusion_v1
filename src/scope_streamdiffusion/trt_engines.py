"""TensorRT engine builder + adapter classes for StreamDiffusion.

In-place TRT acceleration: rather than migrating to StreamDiffusion's
StreamDiffusionWrapper, we use its low-level engine builders to produce
TRT engines for the UNet and ControlNet, then wrap those engines as
drop-in replacements for the diffusers nn.Modules.

Engines cache in ~/.cache/scope-streamdiffusion-trt/{model_hash}/.
First-run build takes 5-10 min per engine. Subsequent runs reuse the
cached engines (~2s load).

Dynamic shapes: spatial dims 256-1024, batch 1-4 (covers CFG on/off
and frame buffering). One engine covers the whole range — no rebuild
on resolution change.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from diffusers import ControlNetModel, UNet2DConditionModel

logger = logging.getLogger(__name__)


# ─── Cache locations ────────────────────────────────────────────────────────


def _cache_root() -> Path:
    """Where engine + onnx artifacts live across runs."""
    return Path.home() / ".cache" / "scope-streamdiffusion-trt"


def _model_cache_dir(model_id: str, suffix: str = "") -> Path:
    """One subdir per model checkpoint; survive across pipeline restarts."""
    h = hashlib.sha256(model_id.encode()).hexdigest()[:16]
    name = f"{h}_{suffix}" if suffix else h
    d = _cache_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── Engine builders ────────────────────────────────────────────────────────


def build_unet_engine(
    unet: UNet2DConditionModel,
    *,
    model_id: str,
    image_height: int = 512,
    image_width: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    use_control: bool = False,
    embedding_dim: int = 1024,  # SD2.1: 1024, SD1.5: 768, SDXL: 2048
    text_maxlen: int = 77,
) -> Path:
    """Build (or reuse) a TRT engine for the given UNet. Returns engine path.

    The engine accepts dynamic spatial dims 256-1024 and batch min..max.
    With use_control=True, ControlNet residuals are first-class engine
    inputs (input_control_00..N) — pass zeros when controlnet is off.
    """
    from streamdiffusion.acceleration.tensorrt.models.models import UNet
    from streamdiffusion.acceleration.tensorrt.utilities import (
        build_engine,
        export_onnx,
        optimize_onnx,
    )

    suffix = f"unet_b{min_batch_size}-{max_batch_size}_cn{int(use_control)}"
    cache_dir = _model_cache_dir(model_id, suffix)
    onnx_path = cache_dir / "unet.onnx"
    onnx_opt_path = cache_dir / "unet_opt.onnx"
    engine_path = cache_dir / "unet.plan"

    if engine_path.exists():
        logger.info(f"[TRT] Reusing cached UNet engine: {engine_path}")
        return engine_path

    logger.info(f"[TRT] Building UNet engine -> {engine_path} (this takes 5-10min)")

    # Read UNet architecture so the model_data spec matches the actual UNet.
    unet_arch = {
        "block_out_channels": tuple(unet.config.block_out_channels),
        "cross_attention_dim": unet.config.cross_attention_dim,
        "in_channels": unet.config.in_channels,
    }

    model_data = UNet(
        unet=unet,
        fp16=True,
        device="cuda",
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=unet.config.cross_attention_dim,
        text_maxlen=text_maxlen,
        unet_dim=unet.config.in_channels,
        use_control=use_control,
        unet_arch=unet_arch,
        image_height=image_height,
        image_width=image_width,
    )

    if not onnx_opt_path.exists():
        if not onnx_path.exists():
            export_onnx(
                unet,
                str(onnx_path),
                model_data,
                opt_image_height=image_height,
                opt_image_width=image_width,
                opt_batch_size=min_batch_size,
                onnx_opset=17,
            )
        optimize_onnx(
            str(onnx_path),
            str(onnx_opt_path),
            model_data,
        )

    build_engine(
        engine_path=str(engine_path),
        onnx_opt_path=str(onnx_opt_path),
        model_data=model_data,
        opt_image_height=image_height,
        opt_image_width=image_width,
        opt_batch_size=min_batch_size,
        build_static_batch=False,
        build_dynamic_shape=True,
    )
    logger.info(f"[TRT] UNet engine built: {engine_path}")
    return engine_path


def build_controlnet_engine(
    controlnet: ControlNetModel,
    *,
    model_id: str,
    image_height: int = 512,
    image_width: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    text_maxlen: int = 77,
) -> Path:
    """Build (or reuse) a TRT engine for a ControlNet. Returns engine path."""
    from streamdiffusion.acceleration.tensorrt.models.controlnet_models import (
        ControlNetWithEmbeds,
    )
    from streamdiffusion.acceleration.tensorrt.utilities import (
        build_engine,
        export_onnx,
        optimize_onnx,
    )

    suffix = f"cn_b{min_batch_size}-{max_batch_size}"
    cache_dir = _model_cache_dir(model_id, suffix)
    onnx_path = cache_dir / "controlnet.onnx"
    onnx_opt_path = cache_dir / "controlnet_opt.onnx"
    engine_path = cache_dir / "controlnet.plan"

    if engine_path.exists():
        logger.info(f"[TRT] Reusing cached ControlNet engine: {engine_path}")
        return engine_path

    logger.info(f"[TRT] Building ControlNet engine -> {engine_path} (this takes 3-5min)")

    model_data = ControlNetWithEmbeds(
        controlnet=controlnet,
        fp16=True,
        device="cuda",
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=controlnet.config.cross_attention_dim,
        text_maxlen=text_maxlen,
        image_height=image_height,
        image_width=image_width,
    )

    if not onnx_opt_path.exists():
        if not onnx_path.exists():
            export_onnx(
                controlnet,
                str(onnx_path),
                model_data,
                opt_image_height=image_height,
                opt_image_width=image_width,
                opt_batch_size=min_batch_size,
                onnx_opset=17,
            )
        optimize_onnx(
            str(onnx_path),
            str(onnx_opt_path),
            model_data,
        )

    build_engine(
        engine_path=str(engine_path),
        onnx_opt_path=str(onnx_opt_path),
        model_data=model_data,
        opt_image_height=image_height,
        opt_image_width=image_width,
        opt_batch_size=min_batch_size,
        build_static_batch=False,
        build_dynamic_shape=True,
    )
    logger.info(f"[TRT] ControlNet engine built: {engine_path}")
    return engine_path


# ─── Drop-in adapters ───────────────────────────────────────────────────────


class TRTUNetAdapter:
    """Wraps UNet2DConditionModelEngine to mimic diffusers UNet2DConditionModel.

    Existing pipeline.py calls:
        model_pred = self.unet(
            x_t_latent, t_list,
            encoder_hidden_states=embeds,
            added_cond_kwargs=...,
            down_block_additional_residuals=...,
            mid_block_additional_residual=...,
            return_dict=False,
        )[0]

    We accept the same signature and return (sample,).
    """

    def __init__(self, engine_path: Path, cuda_stream, *, use_control: bool):
        from streamdiffusion.acceleration.tensorrt.runtime_engines import (
            UNet2DConditionModelEngine,
        )
        self.engine = UNet2DConditionModelEngine(str(engine_path), cuda_stream)
        self.engine.use_control = use_control
        self.use_control = use_control
        # Mimic the surface our pipeline reads off the diffusers UNet
        self.config = _ConfigShim()
        self.add_embedding = None  # SD2.1 doesn't have add_embedding; SDXL would

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict: bool = True,
        **kwargs,
    ):
        # Engine wants timestep as a tensor; pipeline.py passes a list/tensor.
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        out = self.engine(
            latent_model_input=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            kvo_cache=[],
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )

        # Engine returns either a tensor or a UNet2DConditionOutput-like object.
        if hasattr(out, "sample"):
            sample_out = out.sample
        elif isinstance(out, (tuple, list)):
            sample_out = out[0]
        else:
            sample_out = out

        if return_dict:
            from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
            return UNet2DConditionOutput(sample=sample_out)
        return (sample_out,)


class TRTControlNetAdapter:
    """Wraps ControlNetModelEngine to mimic diffusers ControlNetModel.

    Existing pipeline.py calls:
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x_t_latent_plus_uc, t_list,
            encoder_hidden_states=...,
            controlnet_cond=cond_image,
            conditioning_scale=...,
            return_dict=False,
        )
    """

    def __init__(self, engine_path: Path, cuda_stream):
        from streamdiffusion.acceleration.tensorrt.runtime_engines import (
            ControlNetModelEngine,
        )
        self.engine = ControlNetModelEngine(str(engine_path), cuda_stream)
        self.config = _ConfigShim()

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
        **kwargs,
    ):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        out = self.engine(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=torch.tensor([conditioning_scale], device=sample.device),
        )
        # out is a tuple (down_block_res_samples_list, mid_block_res_sample)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            down_block_res_samples, mid_block_res_sample = out
            if return_dict:
                from diffusers.models.controlnets.controlnet import ControlNetOutput
                return ControlNetOutput(
                    down_block_res_samples=down_block_res_samples,
                    mid_block_res_sample=mid_block_res_sample,
                )
            return (down_block_res_samples, mid_block_res_sample)
        raise RuntimeError(f"Unexpected ControlNet engine output shape: {type(out)}")


class _ConfigShim:
    """Diffusers config object surface — pipeline.py reads addition_time_embed_dim, etc."""

    def __init__(self):
        self.addition_time_embed_dim = None
        self.in_channels = 4
        self.out_channels = 4
        self.cross_attention_dim = 1024


# ─── CUDA stream helper ─────────────────────────────────────────────────────


def make_cuda_stream():
    """Polygraphy CUDA stream wrapper used by both engine classes."""
    from polygraphy import cuda
    return cuda.Stream()
