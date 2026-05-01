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


def build_unet_with_controlnet_engine(
    unet: UNet2DConditionModel,
    controlnet,
    *,
    model_id: str,
    controlnet_id: str,
    image_height: int = 512,
    image_width: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
) -> Path:
    """Build (or reuse) a single TRT engine combining UNet + ControlNet.

    Engine accepts dynamic spatial dims 256-1024 and batch min..max. The
    controlnet conditioning image and scale are runtime inputs — one engine
    serves any conditioning_scale without rebuild.

    Engine cache key includes both model_id and controlnet_id so e.g. depth
    vs scribble ControlNets get separate engines.
    """
    from ._trt import (
        UNetWithControlNet,
        compile_unet_with_controlnet,
        create_onnx_path,
    )

    suffix = f"unetcn_b{min_batch_size}-{max_batch_size}_h{image_height}_w{image_width}"
    cache_dir = _model_cache_dir(f"{model_id}::{controlnet_id}", suffix)
    onnx_dir = cache_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_path = cache_dir / "unet_cn.engine"

    if engine_path.exists():
        logger.info(f"[TRT] Reusing cached UNet+CN engine: {engine_path}")
        return engine_path

    logger.info(f"[TRT] Building UNet+CN engine -> {engine_path} (5-10 min on first build)")

    model_data = UNetWithControlNet(
        fp16=True,
        device=str(unet.device) if unet.device.type != "meta" else "cuda",
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=unet.config.cross_attention_dim,
        unet_dim=unet.config.in_channels,
    )
    compile_unet_with_controlnet(
        unet, controlnet, model_data,
        str(create_onnx_path("unet_cn", str(onnx_dir), opt=False)),
        str(create_onnx_path("unet_cn", str(onnx_dir), opt=True)),
        str(engine_path),
        opt_batch_size=min_batch_size,
        engine_build_options={
            "build_dynamic_shape": True,
            "build_static_batch": False,
        },
    )
    logger.info(f"[TRT] UNet+CN engine built: {engine_path}")
    return engine_path


def build_controlnet_engine(
    controlnet,
    *,
    model_id: str,
    controlnet_id: str,
    image_height: int = 512,
    image_width: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
) -> Path:
    """Build (or reuse) a standalone TRT engine for ControlNet.

    Cache key includes both base model_id and controlnet_id so swapping
    between depth/scribble ControlNets uses separate engines. The
    diffusers ControlNetModel is wrapped to expose conditioning_scale
    as a runtime input — one engine serves any scale without rebuild.
    """
    from ._trt import ControlNet, compile_controlnet, create_onnx_path

    suffix = f"cn_b{min_batch_size}-{max_batch_size}_h{image_height}_w{image_width}"
    cache_dir = _model_cache_dir(f"{model_id}::{controlnet_id}", suffix)
    onnx_dir = cache_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_path = cache_dir / "controlnet.engine"

    if engine_path.exists():
        logger.info(f"[TRT] Reusing cached ControlNet engine: {engine_path}")
        return engine_path

    logger.info(f"[TRT] Building ControlNet engine -> {engine_path} (3-5 min on first build)")

    # Determine block_out_channels and num_down_residuals from the actual
    # ControlNet config (must match the runtime inferred shapes).
    block_out_channels = tuple(controlnet.config.block_out_channels)
    # SD1.5/2.1: 12 down residuals. SDXL would be 9 — adjust if/when needed.
    num_down_residuals = 12

    model_data = ControlNet(
        fp16=True,
        device="cuda",
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=controlnet.config.cross_attention_dim,
        unet_dim=4,
        num_down_residuals=num_down_residuals,
        block_out_channels=block_out_channels,
    )
    compile_controlnet(
        controlnet, model_data,
        str(create_onnx_path("controlnet", str(onnx_dir), opt=False)),
        str(create_onnx_path("controlnet", str(onnx_dir), opt=True)),
        str(engine_path),
        opt_batch_size=min_batch_size,
        engine_build_options={
            "build_dynamic_shape": True,
            "build_static_batch": False,
        },
    )
    import shutil
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir, ignore_errors=True)
    logger.info(f"[TRT] ControlNet engine built: {engine_path}")
    return engine_path


class TRTControlNetAdapter:
    """Drop-in for diffusers ControlNetModel.

    pipeline._unet_step calls:
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            x_t_latent_plus_uc, t_list,
            encoder_hidden_states=embeds,
            controlnet_cond=cond_image,
            conditioning_scale=scale,
            return_dict=False,
        )

    This adapter accepts the same call signature, normalises the scale
    arg to a 1-D tensor (engine input expects shape (1,)), and forwards
    to the TRT engine.
    """

    def __init__(self, engine_path: Path, cuda_stream, *, num_down_residuals: int = 12):
        from ._trt import ControlNetEngine
        self.engine = ControlNetEngine(
            str(engine_path), cuda_stream, num_down_residuals=num_down_residuals,
        )
        self.config = _ConfigShim()

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale=1.0,
        return_dict: bool = True,
        **kwargs,
    ):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        if isinstance(conditioning_scale, torch.Tensor):
            scale = conditioning_scale.to(torch.float32)
            if scale.ndim == 0:
                scale = scale.unsqueeze(0)
        else:
            scale = torch.tensor(
                [float(conditioning_scale)], dtype=torch.float32, device=sample.device,
            )

        down, mid = self.engine(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            controlnet_scale=scale,
        )
        if return_dict:
            from diffusers.models.controlnets.controlnet import ControlNetOutput
            return ControlNetOutput(down_block_res_samples=down, mid_block_res_sample=mid)
        return (down, mid)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def build_unet_with_control_engine(
    unet: UNet2DConditionModel,
    *,
    model_id: str,
    image_height: int = 512,
    image_width: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
    num_down_residuals: int = 12,
) -> Path:
    """Build (or reuse) a TRT UNet engine that accepts ControlNet residuals
    as runtime inputs. Pair with `build_controlnet_engine` to get
    end-to-end TRT acceleration for the depth/scribble path.
    """
    from ._trt import (
        UNetWithControlInputs,
        compile_unet_with_control,
        create_onnx_path,
    )

    suffix = f"unet_ctrl_b{min_batch_size}-{max_batch_size}_h{image_height}_w{image_width}"
    cache_dir = _model_cache_dir(model_id, suffix)
    onnx_dir = cache_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_path = cache_dir / "unet_ctrl.engine"

    if engine_path.exists():
        logger.info(f"[TRT] Reusing cached UNet+ctrl engine: {engine_path}")
        return engine_path

    logger.info(f"[TRT] Building UNet+ctrl engine -> {engine_path} (5-10 min)")

    block_out_channels = tuple(unet.config.block_out_channels)
    model_data = UNetWithControlInputs(
        fp16=True,
        device=str(unet.device) if unet.device.type != "meta" else "cuda",
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=unet.config.cross_attention_dim,
        unet_dim=unet.config.in_channels,
        num_down_residuals=num_down_residuals,
        block_out_channels=block_out_channels,
    )
    compile_unet_with_control(
        unet, model_data,
        str(create_onnx_path("unet_ctrl", str(onnx_dir), opt=False)),
        str(create_onnx_path("unet_ctrl", str(onnx_dir), opt=True)),
        str(engine_path),
        opt_batch_size=min_batch_size,
        engine_build_options={
            "build_dynamic_shape": True,
            "build_static_batch": False,
        },
    )
    import shutil
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir, ignore_errors=True)
    logger.info(f"[TRT] UNet+ctrl engine built: {engine_path}")
    return engine_path


class TRTUNetWithControlAdapter:
    """Drop-in for diffusers UNet that forwards ControlNet residuals to engine.

    pipeline._unet_step calls:
        self.unet(sample, t, encoder_hidden_states=...,
                  down_block_additional_residuals=[...],
                  mid_block_additional_residual=mid,
                  return_dict=False)[0]

    This adapter unpacks the residuals into the engine's named slots.
    """

    def __init__(self, engine_path: Path, cuda_stream, *, num_down_residuals: int = 12, use_cuda_graph: bool = False):
        from ._trt import UNet2DConditionModelWithControlEngine
        self.engine = UNet2DConditionModelWithControlEngine(
            str(engine_path), cuda_stream,
            num_down_residuals=num_down_residuals,
            use_cuda_graph=use_cuda_graph,
        )
        self.num_down_residuals = num_down_residuals
        self.config = _ConfigShim()
        self.add_embedding = None

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict: bool = True,
        **kwargs,
    ):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        if down_block_additional_residuals is None or mid_block_additional_residual is None:
            raise RuntimeError(
                "TRTUNetWithControlAdapter requires ControlNet residuals — "
                "down_block_additional_residuals / mid_block_additional_residual."
            )

        out = self.engine(
            latent_model_input=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_residuals=list(down_block_additional_residuals),
            mid_block_residual=mid_block_additional_residual,
        )
        if return_dict:
            return out
        return (out.sample,)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


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
    # ONNX intermediates can be GBs and aren't needed once the engine is built.
    import shutil
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir, ignore_errors=True)
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

    def __init__(self, engine_path: Path, cuda_stream, *, use_cuda_graph: bool = False):
        from ._trt import UNet2DConditionModelEngine
        # use_cuda_graph: capture the engine's kernel sequence on first call
        # and replay on subsequent calls. Eliminates ~3-5ms of TRT launch
        # overhead per call. Shape must stay constant — changing resolution
        # mid-stream invalidates the captured graph (would need recapture).
        self.engine = UNet2DConditionModelEngine(
            str(engine_path), cuda_stream, use_cuda_graph=use_cuda_graph,
        )
        self._use_cuda_graph = use_cuda_graph
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


class TRTUNetControlNetAdapter:
    """Drop-in for diffusers UNet that internally folds in ControlNet.

    The pipeline.py `_unet_step` does:
        cn(...) -> residuals
        unet(..., down_block_additional_residuals=residuals)

    To use this adapter, _unet_step needs to instead:
        unet_cn(latents, t, embeds, controlnet_image=cond, controlnet_scale=scale)

    This adapter exposes that signature. We expose:
      __call__(sample, timestep, encoder_hidden_states, controlnet_image,
               controlnet_scale, return_dict, **kwargs)
    """

    def __init__(self, engine_path: Path, cuda_stream, *, use_cuda_graph: bool = False):
        from ._trt import UNetWithControlNetEngine
        self.engine = UNetWithControlNetEngine(
            str(engine_path), cuda_stream, use_cuda_graph=use_cuda_graph,
        )
        self.config = _ConfigShim()
        self.add_embedding = None

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        controlnet_image: torch.Tensor,
        controlnet_scale,
        return_dict: bool = True,
        **kwargs,
    ):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        if not isinstance(controlnet_scale, torch.Tensor):
            controlnet_scale = torch.tensor(
                [float(controlnet_scale)], dtype=torch.float32, device=sample.device,
            )
        elif controlnet_scale.ndim == 0:
            controlnet_scale = controlnet_scale.unsqueeze(0).to(torch.float32)

        out = self.engine(
            latent_model_input=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_image=controlnet_image,
            controlnet_scale=controlnet_scale,
        )
        if return_dict:
            return out
        return (out.sample,)

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


def build_taesd_engines(
    taesd_vae,  # diffusers AutoencoderTiny instance
    *,
    model_id: str = "madebyollin/taesd",
    image_height: int = 512,
    image_width: int = 512,
    min_batch_size: int = 1,
    max_batch_size: int = 4,
) -> tuple[Path, Path]:
    """Build (or reuse) TRT engines for TAESD encoder + decoder.

    TAESD is small (~10 MB each), so engine builds are fast (30-60s each
    on first run). The encoder and decoder are independent — separate
    engines, separate cache entries.
    """
    from ._trt import (
        VAE,
        VAEEncoder,
        compile_vae_decoder,
        compile_vae_encoder,
        create_onnx_path,
        TorchVAEEncoder,
    )

    suffix = f"taesd_b{min_batch_size}-{max_batch_size}_h{image_height}_w{image_width}"
    cache_dir = _model_cache_dir(model_id, suffix)
    onnx_dir = cache_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    encoder_engine_path = cache_dir / "taesd_encoder.engine"
    decoder_engine_path = cache_dir / "taesd_decoder.engine"

    # Encoder
    if not encoder_engine_path.exists():
        logger.info(f"[TRT] Building TAESD encoder -> {encoder_engine_path}")
        # The prism TorchVAEEncoder runs retrieve_latents over .encode(); for
        # TAESD that returns AutoencoderTinyOutput(latents=...).
        wrapped_encoder = TorchVAEEncoder(taesd_vae).to(torch.device("cuda")).eval()
        encoder_model = VAEEncoder(
            device="cuda", max_batch_size=max_batch_size, min_batch_size=min_batch_size,
        )
        compile_vae_encoder(
            wrapped_encoder, encoder_model,
            str(create_onnx_path("taesd_encoder", str(onnx_dir), opt=False)),
            str(create_onnx_path("taesd_encoder", str(onnx_dir), opt=True)),
            str(encoder_engine_path),
            opt_batch_size=min_batch_size,
            engine_build_options={"build_dynamic_shape": True, "build_static_batch": False},
        )
    else:
        logger.info(f"[TRT] Reusing cached TAESD encoder: {encoder_engine_path}")

    # Decoder — set vae.forward = vae.decode for export, undo after
    if not decoder_engine_path.exists():
        logger.info(f"[TRT] Building TAESD decoder -> {decoder_engine_path}")
        # Prism's pattern: replace .forward with .decode for export only.
        # Note that AutoencoderTiny.decode returns DecoderOutput(sample=...);
        # the prism path expects forward to return the tensor, so we need a
        # thin wrapper that unwraps the DecoderOutput.
        class _DecoderWrap(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            def forward(self, latent):
                return self.vae.decode(latent, return_dict=False)[0]
        wrapped_decoder = _DecoderWrap(taesd_vae).to(torch.device("cuda")).eval()
        decoder_model = VAE(
            device="cuda", max_batch_size=max_batch_size, min_batch_size=min_batch_size,
        )
        compile_vae_decoder(
            wrapped_decoder, decoder_model,
            str(create_onnx_path("taesd_decoder", str(onnx_dir), opt=False)),
            str(create_onnx_path("taesd_decoder", str(onnx_dir), opt=True)),
            str(decoder_engine_path),
            opt_batch_size=min_batch_size,
            engine_build_options={"build_dynamic_shape": True, "build_static_batch": False},
        )
    else:
        logger.info(f"[TRT] Reusing cached TAESD decoder: {decoder_engine_path}")

    import shutil
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir, ignore_errors=True)
    return encoder_engine_path, decoder_engine_path


class TRTTaesdAdapter:
    """Drop-in for diffusers AutoencoderTiny.

    Pipeline.py calls ``self.vae.encode(x)`` (expects ``.latents``) and
    ``self.vae.decode(x, return_dict=False)`` (expects ``(sample,)``).
    Reuses prism's AutoencoderKLEngine — its encode returns
    AutoencoderTinyOutput(latents=...) and decode returns
    DecoderOutput(sample=...), matching what AutoencoderTiny does.
    """

    class _Config:
        def __init__(self, scaling_factor: float):
            self.scaling_factor = scaling_factor

    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        cuda_stream,
        scaling_factor: float = 1.0,
        vae_scale_factor: int = 8,
        dtype: torch.dtype = torch.float16,
    ):
        from ._trt import AutoencoderKLEngine
        self._engine = AutoencoderKLEngine(
            str(encoder_path), str(decoder_path), cuda_stream,
            scaling_factor=vae_scale_factor,
        )
        self.config = self._Config(scaling_factor)
        self.dtype = dtype

    def encode(self, image_tensors: torch.Tensor, **kwargs):
        return self._engine.encode(image_tensors)

    def decode(self, latent: torch.Tensor, return_dict: bool = True, **kwargs):
        out = self._engine.decode(latent)
        if return_dict:
            return out
        return (out.sample,)

    def to(self, *args, **kwargs):
        return self


def make_cuda_stream():
    """Polygraphy CUDA stream wrapper used by the engine classes."""
    from polygraphy import cuda
    return cuda.Stream()
