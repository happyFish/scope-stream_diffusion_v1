"""Model loading lifecycle for the StreamDiffusion pipeline.

Owns the diffusers pipeline load (HF and curated presets), the SDXL fp16
VAE replacement, the TAESD swap, the LoRA stubs, and the GPU teardown
that runs before a model swap. The pipeline holds an instance as
``self.model_loader`` and calls ``attach(pipe)`` once at __init__.

Compromise (documented in REFACTOR_PLAN.md): like ``TRTLifecycle`` this
helper writes through to ``pipe.unet`` / ``pipe.vae`` /
``pipe.text_encoder`` / ``pipe.scheduler`` / etc. via the back-reference.
The win is consolidation of the ~500 lines of load / swap mechanics, not
purity. The orchestration order in ``swap()`` is the load-flow contract
relied on by sibling helpers (PromptEncoder, ControlNetHandler,
TRTLifecycle).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from diffusers import (
    DiffusionPipeline,
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor

if TYPE_CHECKING:
    pass


# Curated presets — model_id strings that aren't direct HuggingFace repos but
# describe a (base, distillation) recipe. Extending this dict is how we add
# new 1-step / few-step models to the dropdown without exposing the user to
# the underlying repo plumbing.
#
# Schema currently exposes the `unet_swap` shape. Future shapes:
#   "lora": (lora_repo, lora_filename) — fuse a step-LoRA at scale=1.0 onto
#           the base. Works for Hyper-SD-1step / SDXL-Lightning-1step ONLY
#           after `_set_timesteps` is taught about TCD / Euler schedulers
#           (it currently calls LCM-specific
#           `scheduler.get_scalings_for_boundary_condition_discrete`).
#   "scheduler": SchedulerClass — override the LCMScheduler default in
#           swap(). Same caveat as above re: `_set_timesteps`.
#   "timesteps_override": [int, ...] — pin specific timesteps (Hyper-SD-1step
#           wants [800] with TCD).
MODEL_PRESETS: Dict[str, dict] = {
    "dmd2-sdxl-1step": {
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
        # tianweiy/DMD2 ships several distilled UNet checkpoints; the
        # 1-step fp16 variant is the SDXL-Turbo equivalent.
        "unet_swap": ("tianweiy/DMD2", "dmd2_sdxl_1step_unet_fp16.bin"),
        # DMD2 was distilled at this specific timestep — feeding it
        # LCMScheduler's default 1-step pick (~979) produces noise.
        "timesteps_override": [399],
        # DMD2 has CFG distilled into its weights, so its single-shot
        # output already looks like a guidance-shaped result. Implicit
        # txt2img→img2img loopback re-applies that CFG-shape every frame
        # and the chain blows up within a few iterations. Skip the
        # implicit fallback; explicit image_loopback=True still works.
        "implicit_loopback": False,
    },
}


class ModelLoader:
    """Load / swap / release diffusers pipelines for the StreamDiffusion orchestrator.

    Owns the load primitives (HF + curated presets + UNet swap + safetensors
    cache), the SDXL fp16 VAE replacement, the TAESD swap, and the GPU
    teardown that runs before a model swap. Mutates the parent pipeline's
    attributes through ``self.pipe`` (back-reference set in ``attach()``).
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

        # Live back-reference to the StreamDiffusionPipeline — set via
        # ``attach()``. Until then the helper is inert.
        self.pipe: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, pipe: Any) -> None:
        """Wire the helper to the parent StreamDiffusionPipeline.

        Called once from the pipeline's ``__init__``. Subsequent model
        swaps reuse the same back-reference; only the diffusers pipe
        underneath ``self.pipe.pipe`` changes.
        """
        self.pipe = pipe

    # ------------------------------------------------------------------
    # Load primitives
    # ------------------------------------------------------------------

    def load(self, model_id: str) -> DiffusionPipeline:
        """Load the diffusion model.

        For HuggingFace model IDs, loads via DiffusionPipeline.from_pretrained
        directly. For curated presets in MODEL_PRESETS, follows the preset's
        recipe (base load + UNet swap, etc.).
        """
        try:
            preset = MODEL_PRESETS.get(model_id)
            if preset is not None:
                pipe = self._load_preset(preset)
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                )
            pipe = pipe.to(self.device)

            # Enable xformers memory-efficient attention if available.
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[StreamDiffusion] xformers memory-efficient attention enabled")
            except Exception as e:
                print(f"[StreamDiffusion] xformers not available, skipping: {e}")

            return pipe
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            raise

    def _load_preset(self, preset: dict) -> DiffusionPipeline:
        """Build a DiffusionPipeline from a MODEL_PRESETS recipe.

        Currently supports the ``unet_swap`` shape — load the base pipeline,
        then override its UNet weights from a distilled checkpoint. Other
        recipe shapes (LoRA fuse, scheduler override, timesteps_override)
        will land alongside the `_set_timesteps` refactor needed to support
        non-LCM schedulers.
        """
        base = preset["base"]
        print(f"[StreamDiffusion] Loading preset base: {base}")
        pipe = DiffusionPipeline.from_pretrained(
            base,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
        )

        unet_swap = preset.get("unet_swap")
        if unet_swap is not None:
            unet_repo, unet_file = unet_swap
            # Distilled-UNet repos (DMD2, SDXL-Lightning, etc.) often ship
            # weights only — no config.json — because the architecture is
            # identical to the base UNet. Reuse the base pipeline's UNet
            # module and override its state_dict.
            #
            # Move the UNet to GPU *before* loading the state_dict so the
            # checkpoint can stream straight to device — without this, the
            # state_dict lands on CPU, gets copied into the (CPU) UNet, then
            # the whole pipe gets shipped to GPU later via ``load()``'s
            # ``pipe.to(device)``. Skipping the CPU staging cuts ~3-5 s on a
            # 5 GB UNet (DMD2 fp16). The rest of the pipe still moves to GPU
            # in ``load()`` as before.
            pipe.unet.to(self.device)
            state_dict = self._load_unet_swap_state_dict(unet_repo, unet_file)
            pipe.unet.load_state_dict(state_dict)
            print("[StreamDiffusion] Distilled UNet weights loaded")
            return pipe

        return pipe

    def _load_unet_swap_state_dict(
        self, unet_repo: str, unet_file: str
    ) -> Dict[str, torch.Tensor]:
        """Load a distilled UNet state_dict, preferring a local safetensors cache.

        First load: download the original weights via ``hf_hub_download``,
        load to GPU, then write a safetensors copy to
        ``~/.cache/scope-streamdiffusion/converted/<repo>__<file>.safetensors``.
        Subsequent loads mmap the converted file directly into GPU — DMD2's
        5 GB ``.bin`` pickle goes from ~10 s to ~1-2 s on warm cache.

        The conversion is one-shot per (repo, file) pair and survives plugin
        reinit. Native ``.safetensors`` checkpoints (e.g. DMD2 4-step) skip
        the conversion entirely and load straight from the HF cache.
        """
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        from safetensors.torch import load_file, save_file

        # Native safetensors path — load straight to device, no conversion.
        if unet_file.endswith(".safetensors"):
            try:
                ckpt_path = hf_hub_download(unet_repo, unet_file, local_files_only=True)
                print(f"[StreamDiffusion] Loading cached distilled UNet: {unet_repo}/{unet_file}")
            except LocalEntryNotFoundError:
                print(f"[StreamDiffusion] Downloading distilled UNet: {unet_repo}/{unet_file}")
                ckpt_path = hf_hub_download(unet_repo, unet_file)
            return load_file(ckpt_path, device=str(self.device))

        # .bin path — check the local converted-safetensors cache first.
        cache_dir = Path.home() / ".cache" / "scope-streamdiffusion" / "converted"
        cache_name = f"{unet_repo.replace('/', '__')}__{unet_file}.safetensors"
        cached_path = cache_dir / cache_name
        if cached_path.exists():
            print(
                f"[StreamDiffusion] Loading converted-safetensors UNet: {cached_path.name}"
            )
            return load_file(str(cached_path), device=str(self.device))

        # Cache miss — fetch the .bin and load via torch.load.
        try:
            ckpt_path = hf_hub_download(unet_repo, unet_file, local_files_only=True)
            print(f"[StreamDiffusion] Loading cached distilled UNet: {unet_repo}/{unet_file}")
        except LocalEntryNotFoundError:
            print(f"[StreamDiffusion] Downloading distilled UNet: {unet_repo}/{unet_file}")
            ckpt_path = hf_hub_download(unet_repo, unet_file)
        # weights_only=True skips the unpickler dispatch and refuses arbitrary
        # code execution. Required for trust on third-party .bin checkpoints;
        # also a small (~10-15%) read-time win.
        state_dict = torch.load(
            ckpt_path,
            map_location=self.device,
            weights_only=True,
        )

        # Write the safetensors conversion for next time. Best-effort: a
        # failed write (out of disk, permission, etc.) shouldn't block this
        # load. ``save_file`` requires CPU contiguous tensors, so transfer
        # back through CPU; one-time cost on first load only.
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cpu_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
            tmp_path = cached_path.with_suffix(cached_path.suffix + ".tmp")
            save_file(cpu_state, str(tmp_path))
            tmp_path.rename(cached_path)
            print(
                f"[StreamDiffusion] Cached safetensors conversion: {cached_path}"
            )
        except Exception as e:
            print(f"[StreamDiffusion] safetensors conversion cache failed (non-fatal): {e}")

        return state_dict

    # ------------------------------------------------------------------
    # Post-load helpers
    # ------------------------------------------------------------------

    def install_sdxl_fp16_vae(self) -> None:
        """Swap SDXL's default VAE for madebyollin/sdxl-vae-fp16-fix.

        Stability AI's SDXL VAE overflows on certain inputs in fp16 and decodes
        to NaN — even from a perfectly valid UNet prediction. The community
        fp16-fix VAE is a drop-in replacement with the same architecture and
        quality, retuned to be numerically stable in fp16.
        """
        from diffusers import AutoencoderKL

        try:
            print("[StreamDiffusion] Installing madebyollin/sdxl-vae-fp16-fix")
            new_vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=self.dtype
            ).to(self.device)
            self.pipe.pipe.vae = new_vae
            print("[StreamDiffusion] SDXL fp16-fix VAE installed")
        except Exception as e:
            print(f"[StreamDiffusion] Failed to install fp16-fix VAE: {e}")

    def set_taesd(self, enabled: bool) -> None:
        """Switch between TAESD (fast) and full VAE decoder."""
        if enabled == self.pipe._using_taesd:
            return
        if enabled:
            if self.pipe._taesd_vae is None:
                from diffusers import AutoencoderTiny

                taesd_id = "madebyollin/taesdxl" if self.pipe.sdxl else "madebyollin/taesd"
                print(f"[StreamDiffusion] Loading TAESD from {taesd_id}")
                self.pipe._taesd_vae = AutoencoderTiny.from_pretrained(
                    taesd_id, torch_dtype=self.dtype
                ).to(self.device)
                print("[StreamDiffusion] TAESD loaded")
            self.pipe.vae = self.pipe._taesd_vae
            self.pipe._using_taesd = True
            print("[StreamDiffusion] Switched to TAESD (fast decode)")
        else:
            self.pipe.vae = self.pipe._full_vae
            self.pipe._using_taesd = False
            print("[StreamDiffusion] Switched to full VAE")

    # ------------------------------------------------------------------
    # LoRA (stubs — wired by the LoRA plan)
    # ------------------------------------------------------------------

    def load_lora(
        self,
        pretrained_lora_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[Any] = None,
        **kwargs,
    ) -> None:
        self.pipe.pipe.load_lora_weights(
            pretrained_lora_model_name_or_path_or_dict, adapter_name, **kwargs
        )

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ) -> None:
        self.pipe.pipe.fuse_lora(
            fuse_unet=fuse_unet,
            fuse_text_encoder=fuse_text_encoder,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
        )

    # ------------------------------------------------------------------
    # Release / swap
    # ------------------------------------------------------------------

    def release_pipe_state(self) -> None:
        """Drop every GPU-resident reference owned by the parent pipeline.

        Called from :meth:`swap` before loading the new model. Clears module
        references (``unet`` / ``vae`` / ``text_encoder`` / any TRT adapter
        still pinned in ``pipe.unet``), per-step cached tensors, prompt-
        embedding caches, and the ControlNet handler's sub-models. Caller
        (``swap``) is expected to have already run
        :meth:`TRTLifecycle.reset_caches` so the cache-state-held adapter
        references are gone too. Finishes with a ``gc.collect`` +
        ``torch.cuda.empty_cache`` so the next allocation starts clean.
        """
        import gc

        p = self.pipe

        # Module references — these are the big-ticket allocations.
        # pipe.unet may be a TRT adapter that owns engine memory; nulling
        # it here is what actually releases the engine.
        p.unet = None
        p.vae = None
        p.text_encoder = None
        p._taesd_vae = None
        p._full_vae = None
        p.controlnet = None
        p.controlnet_input = None
        if hasattr(p, "_cn") and p._cn is not None:
            p._cn.release()

        # Cached per-frame state.
        p.prev_image_result = None
        if hasattr(p, "inference") and p.inference is not None:
            p.inference.reset_buffers()
            p.inference.timesteps = None
            p.inference.sub_timesteps = []
            p.inference.sub_timesteps_tensor = None
            p.inference.c_skip = None
            p.inference.c_out = None
            p.inference.alpha_prod_t_sqrt = None
            p.inference.beta_prod_t_sqrt = None

        # Reset prompt-encoder caches (text-encoder-specific; the new
        # model will have a different text encoder).
        if hasattr(p, "prompts"):
            p.prompts.reset_caches()

        # Drop the pipeline last so any of the above that aliased its
        # submodules have already been nulled.
        p.pipe = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def swap(self, new_model_id: str) -> None:
        """Replace the loaded model in place.

        Scope routes ``model_id_or_path`` through both the load-time path
        (which would reinit the pipeline cleanly) and the runtime
        ``setNodeParams`` path (which only updates kwargs and never touches
        ``__init__``). When the runtime kwarg disagrees with what we loaded,
        rebuild the model parts here so picking a model in the UI actually
        swaps it. Stalls the frame loop while loading — same as a fresh load.

        Helper attach order: PromptEncoder, InferenceCore, TRTLifecycle.
        TRT runs last because its engine builds need ``pipe.unet`` and the
        freshly-loaded text-encoder / VAE state to be in place.
        """
        p = self.pipe
        print(f"[StreamDiffusion] Swapping model: {p.model_id} -> {new_model_id}")
        # Reset TRT sticky state for the new model. Without this the
        # ``_trt_unet_built`` / ``_trt_taesd_built`` flags from the previous
        # model cause the next ``trt.setup`` to short-circuit, leaving the
        # new model running eager regardless of acceleration_mode.
        p.trt.reset_caches(new_model_id)
        # Tear down everything the old model holds on the GPU before loading
        # the new one — without this we peak at 2x model weights + engines
        # and OOM on large models (SDXL UNet alone is ~5 GB fp16, plus a
        # 2 GB+ TRT engine, plus VAE / text encoders / cached tensors).
        self.release_pipe_state()

        p.model_id = new_model_id
        preset = MODEL_PRESETS.get(new_model_id, {})
        p._timesteps_override = preset.get("timesteps_override")
        p._implicit_loopback = preset.get("implicit_loopback", True)
        p.pipe = self.load(new_model_id)
        print(f"[StreamDiffusion] Model loaded: {p.pipe.__class__.__name__}")
        p.sdxl = type(p.pipe) is StableDiffusionXLPipeline
        if p.sdxl and self.dtype == torch.float16:
            self.install_sdxl_fp16_vae()
        # Re-route ControlNet handler to the SDXL- or SD1.5-keyed weights for
        # the new host. ``release_pipe_state`` cleared the cache, so the next
        # ``update()`` will load the right repo from scratch.
        p._cn.set_sdxl(p.sdxl)

        p.text_encoder = p.pipe.text_encoder
        p.unet = p.pipe.unet
        p.vae = p.pipe.vae
        p._full_vae = p.vae
        p._using_taesd = False
        p.scheduler = LCMScheduler.from_config(p.pipe.scheduler.config)
        p.image_processor = VaeImageProcessor(p.pipe.vae_scale_factor)
        p.prompts.attach(p.pipe, p.sdxl)
        p.inference.attach(p)
        p.trt.attach(p, p.sdxl)

        # Invalidate runtime caches so the next __call__ rebuilds the
        # timestep schedule and noise buffers against the new model.
        # Prompt-encoder caches are reset by ``prompts.attach()`` above.
        p._schedule_key = None
        p._noise_shape = None
        p.prev_image_result = None
        p.inference.cancel_seed_transition()

        # Build TRT engines for the new model now so the next frame doesn't stall.
        if p.trt.acceleration_mode == "trt":
            p.trt.setup(**p.trt.setup_args_from_config())
