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
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine, UNetWithControlNetEngine
from .models import (
    VAE,
    BaseModel,
    UNet,
    UNet2DConditionSingleControlNetModel,
    UNetWithControlNet,
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
    "Engine",
    "EngineBuilder",
    "TorchVAEEncoder",
    "UNet",
    "UNet2DConditionModelEngine",
    "UNet2DConditionSingleControlNetModel",
    "UNetWithControlNet",
    "UNetWithControlNetEngine",
    "VAE",
    "VAEEncoder",
    "build_engine",
    "compile_unet",
    "compile_unet_with_controlnet",
    "compile_vae_decoder",
    "compile_vae_encoder",
    "create_onnx_path",
    "export_onnx",
    "optimize_onnx",
]


def compile_unet_with_controlnet(
    unet,
    controlnet,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    engine_build_options: dict = {},
):
    """Build a single TRT engine for UNet + ControlNet combined.

    Bypasses the prism ``Optimizer`` step because the combined-model ONNX
    graph (~1.3 GB UNet + ~700 MB ControlNet) exceeds the 2 GB protobuf
    limit that ``onnx.shape_inference.infer_shapes`` enforces. We export
    ONNX with external data (so weights live in side-files and the main
    graph stays small) and feed the un-optimized ONNX directly into the
    polygraphy build. The TRT builder runs its own optimization passes;
    skipping prism's pre-optimize gives up some const-folding but
    materially reduces complexity.
    """
    import os
    from .utilities import build_engine, export_onnx

    combined = UNet2DConditionSingleControlNetModel(unet, controlnet).to(
        torch.device("cuda"), dtype=torch.float16
    ).eval()

    consolidated_path = onnx_path.replace(".onnx", "_consolidated.onnx")
    if not os.path.exists(consolidated_path):
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        export_onnx(
            combined,
            onnx_path,
            model_data,
            opt_image_height=512,
            opt_image_width=512,
            opt_batch_size=opt_batch_size,
            onnx_opset=engine_build_options.pop("onnx_opset", 17),
        )
        # torch's legacy onnx exporter writes >2GB models as one .onnx + many
        # per-tensor weight files. Polygraphy/TRT loaders expect a single
        # .onnx.data side-file. Consolidate.
        import onnx
        print(f"[TRT] consolidating external data: {onnx_path} -> {consolidated_path}")
        m = onnx.load(onnx_path, load_external_data=True)
        onnx.save_model(
            m,
            consolidated_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(consolidated_path) + ".data",
        )
        print(f"[TRT] consolidation done")
    else:
        print(f"Found cached consolidated ONNX: {consolidated_path}")
    onnx_path = consolidated_path

    if os.path.exists(engine_path):
        print(f"Found cached engine: {engine_path}")
        return

    build_engine(
        engine_path=engine_path,
        onnx_opt_path=onnx_path,  # un-optimized — TRT does its own passes
        model_data=model_data,
        opt_image_height=512,
        opt_image_width=512,
        opt_batch_size=opt_batch_size,
        build_static_batch=engine_build_options.get("build_static_batch", False),
        build_dynamic_shape=engine_build_options.get("build_dynamic_shape", True),
    )
