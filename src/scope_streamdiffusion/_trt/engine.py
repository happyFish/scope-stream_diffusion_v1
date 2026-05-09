from typing import *

import torch
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from polygraphy import cuda

from .utilities import Engine


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        self.engine.allocate_buffers(
            shape_dict={
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "latent": latent_model_input.shape,
            },
            device=latent_model_input.device,
        )

        noise_pred = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class UNet2DConditionModelSDXLEngine:
    """SDXL UNet engine — adds text_embeds + time_ids inputs to the plain UNet engine."""

    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        self.engine.allocate_buffers(
            shape_dict={
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "text_embeds": text_embeds.shape,
                "time_ids": time_ids.shape,
                "latent": latent_model_input.shape,
            },
            device=latent_model_input.device,
        )

        noise_pred = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class UNet2DConditionModelWithControlEngine:
    """UNet engine variant that accepts ControlNet residuals as runtime inputs.

    Inputs match `UNetWithControlInputs.get_input_names()` —
      sample, timestep, encoder_hidden_states,
      input_control_00..N-1, input_control_middle.
    Output: latent (same as plain UNet).

    Caller is responsible for producing the residuals (e.g. via the
    standalone TRT ControlNet engine) and passing them in.
    """

    def __init__(
        self,
        filepath: str,
        stream: cuda.Stream,
        num_down_residuals: int,
        use_cuda_graph: bool = False,
    ):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.num_down_residuals = num_down_residuals
        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_residuals,  # list[Tensor] of length num_down_residuals
        mid_block_residual: torch.Tensor,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        shape_dict = {
            "sample": latent_model_input.shape,
            "timestep": timestep.shape,
            "encoder_hidden_states": encoder_hidden_states.shape,
            "latent": latent_model_input.shape,
            "input_control_middle": mid_block_residual.shape,
        }
        feed = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "input_control_middle": mid_block_residual,
        }
        for i in range(self.num_down_residuals):
            shape_dict[f"input_control_{i:02d}"] = down_block_residuals[i].shape
            feed[f"input_control_{i:02d}"] = down_block_residuals[i]

        self.engine.allocate_buffers(shape_dict=shape_dict, device=latent_model_input.device)
        noise_pred = self.engine.infer(
            feed, self.stream, use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class ControlNetEngine:
    """Runtime wrapper for a standalone ControlNet TRT engine.

    Mimics diffusers ControlNetModel.__call__ — takes the same kwargs and
    returns ``(down_block_res_samples, mid_block_res_sample)``.
    """

    def __init__(
        self,
        filepath: str,
        stream: cuda.Stream,
        num_down_residuals: int,
        use_cuda_graph: bool = False,
    ):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph
        self.num_down_residuals = num_down_residuals
        self.engine.load()
        self.engine.activate()

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_scale: torch.Tensor,
        **kwargs,
    ) -> Any:
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device)
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        if controlnet_scale.dtype != torch.float32:
            controlnet_scale = controlnet_scale.float()
        if controlnet_cond.dtype != torch.float32:
            controlnet_cond = controlnet_cond.float()

        # Compute output residual shapes from sample shape (latent space).
        B, _, lH, lW = sample.shape
        # (channels, downsample_factor) per residual — must match
        # ControlNet.get_shape_dict in models.py.
        # block_out_channels=(320, 640, 1280, 1280) for SD1.5/2.1 standard.
        chans = (320, 640, 1280, 1280)
        spec = [
            (chans[0], 1), (chans[0], 1), (chans[0], 1),
            (chans[0], 2), (chans[1], 2), (chans[1], 2),
            (chans[1], 4), (chans[2], 4), (chans[2], 4),
            (chans[2], 8), (chans[3], 8), (chans[3], 8),
        ]
        shape_dict: Dict[str, tuple] = {
            "sample": sample.shape,
            "timestep": timestep.shape,
            "encoder_hidden_states": encoder_hidden_states.shape,
            "controlnet_cond": controlnet_cond.shape,
            "controlnet_scale": controlnet_scale.shape,
            "mid_block_res_sample": (B, chans[-1], max(1, lH // 8), max(1, lW // 8)),
        }
        for i, (c, ds) in enumerate(spec):
            shape_dict[f"down_block_res_sample_{i}"] = (
                B, c, max(1, lH // ds), max(1, lW // ds),
            )

        self.engine.allocate_buffers(shape_dict=shape_dict, device=sample.device)
        out = self.engine.infer(
            {
                "sample": sample,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "controlnet_cond": controlnet_cond,
                "controlnet_scale": controlnet_scale,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        down_residuals = [
            out[f"down_block_res_sample_{i}"] for i in range(self.num_down_residuals)
        ]
        mid_residual = out["mid_block_res_sample"]
        return down_residuals, mid_residual

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: cuda.Stream,
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

    def _ensure_activated(self, engine):
        """Re-activate the TRT engine if its execution context was lost.

        The cached adapter survives across plugin reinit (module-scope
        cache in ``_trt_cache``), but rapid acceleration_mode / model
        swaps can leave a previously-activated engine with context=None
        if its activation was torn down by a sibling teardown path.
        Repairing here keeps the streaming loop alive instead of dying
        on `AttributeError: 'NoneType' object has no attribute
        'set_input_shape'`.
        """
        if engine.context is None:
            if engine.engine is None:
                engine.load()
            engine.activate()

    def encode(self, images: torch.Tensor, **kwargs):
        self._ensure_activated(self.encoder)
        self.encoder.allocate_buffers(
            shape_dict={
                "images": images.shape,
                "latent": (
                    images.shape[0],
                    4,
                    images.shape[2] // self.vae_scale_factor,
                    images.shape[3] // self.vae_scale_factor,
                ),
            },
            device=images.device,
        )
        latents = self.encoder.infer(
            {"images": images},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):
        self._ensure_activated(self.decoder)
        self.decoder.allocate_buffers(
            shape_dict={
                "latent": latent.shape,
                "images": (
                    latent.shape[0],
                    3,
                    latent.shape[2] * self.vae_scale_factor,
                    latent.shape[3] * self.vae_scale_factor,
                ),
            },
            device=latent.device,
        )
        images = self.decoder.infer(
            {"latent": latent},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["images"]
        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
