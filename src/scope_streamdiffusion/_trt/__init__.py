"""Vendored TRT exporter from prism (~/Projects/prism/app/pipelines/acceleration/tensorrt).

Self-contained ONNX export + engine build + runtime engine wrappers for SD
UNet/VAE. Cleaner than the StreamDiffusion library's exporter because:

  * No IPAdapter input slots leaked into UNet sample inputs
    (lib-side caused 17-vs-14 arg mismatch in our earlier attempt)
  * No `diffusers_ipadapter` import that doesn't exist on PyPI
  * Returns proper diffusers Output types so existing pipeline.py code
    using `result.sample` etc. just works

Source: prism, copied verbatim then trimmed to drop the prism-specific
StreamDiffusion bootstrap.
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)

from .builder import EngineBuilder, create_onnx_path
from .engine import (
    AutoencoderKLEngine,
    ControlNetEngine,
    UNet2DConditionModelEngine,
    UNet2DConditionModelWithControlEngine,
)
from .models import (
    VAE,
    BaseModel,
    ControlNet,
    ControlNetExportWrapper,
    UNet,
    UNetExportWrapperWithControl,
    UNetWithControlInputs,
    VAEEncoder,
)
from .utilities import Engine, build_engine, export_onnx, optimize_onnx


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))


def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path, onnx_opt_path, engine_path,
        opt_batch_size=opt_batch_size, **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path, onnx_opt_path, engine_path,
        opt_batch_size=opt_batch_size, **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    unet = unet.to(torch.device("cuda"), dtype=torch.float16)
    builder = EngineBuilder(model_data, unet, device=torch.device("cuda"))
    builder.build(
        onnx_path, onnx_opt_path, engine_path,
        opt_batch_size=opt_batch_size, **engine_build_options,
    )


__all__ = [
    "AutoencoderKLEngine",
    "BaseModel",
    "ControlNet",
    "ControlNetEngine",
    "ControlNetExportWrapper",
    "Engine",
    "EngineBuilder",
    "TorchVAEEncoder",
    "UNet",
    "UNet2DConditionModelEngine",
    "UNet2DConditionModelWithControlEngine",
    "UNetExportWrapperWithControl",
    "UNetWithControlInputs",
    "VAE",
    "VAEEncoder",
    "build_engine",
    "compile_controlnet",
    "compile_unet",
    "compile_unet_with_control",
    "compile_vae_decoder",
    "compile_vae_encoder",
    "create_onnx_path",
    "export_onnx",
    "optimize_onnx",
]


def compile_unet_with_control(
    unet,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    """Build a TRT engine for UNet with ControlNet residual input slots.

    Same UNet weights as the plain `compile_unet`; the difference is the
    forward signature exposes the residuals as named inputs so ControlNet
    output (from a separate engine) can be threaded in at runtime.
    """
    num_down = getattr(model_data, "num_down_residuals", 12)
    wrapped = UNetExportWrapperWithControl(unet, num_down).to(
        torch.device("cuda"), dtype=torch.float16
    ).eval()
    builder = EngineBuilder(model_data, wrapped, device=torch.device("cuda"))
    builder.build(
        onnx_path, onnx_opt_path, engine_path,
        opt_batch_size=opt_batch_size, **engine_build_options,
    )


def compile_controlnet(
    controlnet,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    """Build a standalone TRT engine for a single ControlNet.

    Wraps the diffusers ControlNetModel in `ControlNetExportWrapper`
    (which makes conditioning_scale a runtime input and unpacks the
    residual list into named tensors), then runs the standard
    export → optimize → engine build pipeline. ControlNet is small
    enough (~700MB) that the prism Optimizer's 2GB shape-inference
    limit isn't a problem here.
    """
    num_down = getattr(model_data, "num_down_residuals", 12)
    wrapped = ControlNetExportWrapper(controlnet, num_down).to(
        torch.device("cuda"), dtype=torch.float16
    ).eval()
    builder = EngineBuilder(model_data, wrapped, device=torch.device("cuda"))
    builder.build(
        onnx_path, onnx_opt_path, engine_path,
        opt_batch_size=opt_batch_size, **engine_build_options,
    )


