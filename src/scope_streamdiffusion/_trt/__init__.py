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
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine
from .models import VAE, BaseModel, UNet, VAEEncoder
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
    "Engine",
    "EngineBuilder",
    "TorchVAEEncoder",
    "UNet",
    "UNet2DConditionModelEngine",
    "VAE",
    "VAEEncoder",
    "build_engine",
    "compile_unet",
    "compile_vae_decoder",
    "compile_vae_encoder",
    "create_onnx_path",
    "export_onnx",
    "optimize_onnx",
]
