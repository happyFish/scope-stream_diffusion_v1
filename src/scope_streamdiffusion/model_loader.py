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

from typing import TYPE_CHECKING, Any, Dict

import torch
from diffusers import (
    DiffusionPipeline,
    LCMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.image_processor import VaeImageProcessor

if TYPE_CHECKING:
    pass


def _spec_path(spec: Any) -> str | None:
    """LoRA spec path accessor that tolerates both LoraSpec and runtime dicts."""
    if isinstance(spec, dict):
        return spec.get("path")
    return getattr(spec, "path", None)


def _spec_scale(spec: Any) -> float:
    if isinstance(spec, dict):
        return float(spec.get("scale", 1.0))
    return float(getattr(spec, "scale", 1.0))


def lora_signature_from_specs(loras: list) -> tuple:
    """Stable signature of a LoRA stack — sorted ((path, scale), ...).

    Used for change detection on the loader and for TRT cache keying.
    Sort by path so picker reorders don't invalidate the engine cache,
    but order in the actual ``set_adapters`` call follows the user's
    list (so ordering-sensitive blends still work).
    """
    return tuple(sorted(
        (str(_spec_path(spec)), float(_spec_scale(spec)))
        for spec in (loras or [])
        if _spec_path(spec)
    ))


def resolve_lora_path(path: str) -> str:
    """Resolve a LoRA spec path to something diffusers can load.

    Picker UIs and timeline files often emit just the filename
    (``pixel-art-xl.safetensors``); diffusers' ``load_lora_weights``
    needs either an HF repo id or an absolute file path. Anything that
    looks like a bare filename / relative path falls back to Scope's
    LoRA library (``DAYDREAM_SCOPE_LORA_DIR`` or
    ``~/.daydream-scope/models/lora/``). HF repo ids
    (``user/repo[/subdir]``) and absolute paths pass through untouched.
    """
    import os
    from pathlib import Path

    if not path:
        return path
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)

    env_dir = os.environ.get("DAYDREAM_SCOPE_LORA_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser() / path
    else:
        candidate = Path.home() / ".daydream-scope" / "models" / "lora" / path
    if candidate.exists():
        return str(candidate)
    # Fall back to original — diffusers may resolve it as an HF repo id.
    return path


def _adapter_registered(pipe: Any, adapter_name: str) -> bool:
    """Did ``load_lora_weights`` actually register ``adapter_name``?

    When the LoRA file's architecture doesn't match the pipeline (e.g. a
    Flux/SD3 DiT LoRA loaded onto an SDXL UNet), diffusers logs warnings
    and silently registers nothing. We probe ``peft_config`` on the UNet
    and text encoders — if none of them know about the adapter, the load
    was a no-op regardless of what ``load_lora_weights`` returned.
    """
    targets = []
    for attr in ("unet", "text_encoder", "text_encoder_2"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            targets.append(mod)

    for mod in targets:
        cfg = getattr(mod, "peft_config", None)
        if isinstance(cfg, dict) and adapter_name in cfg:
            return True
    return False


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

        # LoRA stack currently attached to ``self.pipe.pipe``. ``None`` means
        # "fresh pipe, never seen apply_loras"; ``()`` means "apply_loras ran
        # with an empty stack". Distinguishing the two avoids a spurious
        # ``unload_lora_weights`` call on the very first attach.
        self._lora_signature: tuple | None = None
        # Path order as supplied by the user; used to detect scale-only
        # changes that can be applied via ``set_adapters`` without an
        # unload/reload roundtrip.
        self._lora_paths: tuple = ()
        self._lora_strategy: str = "runtime_peft"

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

    def load(
        self,
        model_id: str,
        *,
        keep_unet_on_cpu: bool = False,
        keep_full_vae_on_cpu: bool = False,
    ) -> DiffusionPipeline:
        """Load the diffusion model.

        For HuggingFace model IDs, loads via DiffusionPipeline.from_pretrained
        directly. For curated presets in MODEL_PRESETS, follows the preset's
        recipe (base load + UNet swap, etc.).

        ``keep_unet_on_cpu`` (TRT path): the eager UNet will be replaced by a
        TRT adapter and only kept for refit / fallback. Leave it on CPU to
        skip the wasted CPU→GPU staging followed by an immediate GPU→CPU
        offload. Build-time ONNX export still moves it to GPU temporarily.

        ``keep_full_vae_on_cpu`` (TAESD path): when TAESD will replace the
        VAE for inference, the full VAE is just held for a "toggle back"
        capability. Same trade — skip the GPU staging.
        """
        try:
            preset = MODEL_PRESETS.get(model_id)
            if preset is not None:
                pipe = self._load_preset(preset, keep_unet_on_cpu=keep_unet_on_cpu)
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    variant="fp16" if self.dtype == torch.float16 else None,
                )
            self._move_pipe_to_device(
                pipe,
                keep_unet_on_cpu=keep_unet_on_cpu,
                keep_full_vae_on_cpu=keep_full_vae_on_cpu,
            )

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

    def _move_pipe_to_device(
        self,
        pipe: DiffusionPipeline,
        *,
        keep_unet_on_cpu: bool,
        keep_full_vae_on_cpu: bool,
    ) -> None:
        """Move pipeline components to ``self.device`` selectively.

        Replaces the blanket ``pipe.to(device)`` so we can leave the UNet
        and / or full VAE on CPU when the TRT / TAESD paths are going to
        immediately offload them anyway. Skip-list flags are caller's
        responsibility; default (both False) reproduces the old behavior.
        """
        # Text encoders run every frame for prompt encoding — always GPU.
        for attr in ("text_encoder", "text_encoder_2"):
            comp = getattr(pipe, attr, None)
            if comp is not None and hasattr(comp, "to"):
                try:
                    comp.to(self.device)
                except Exception as e:
                    print(f"[StreamDiffusion] could not move {attr} to GPU: {e}")
        # UNet — defer to caller's choice. Build-time ONNX export will move
        # it to GPU temporarily if needed.
        if not keep_unet_on_cpu:
            unet = getattr(pipe, "unet", None)
            if unet is not None and hasattr(unet, "to"):
                try:
                    unet.to(self.device)
                except Exception as e:
                    print(f"[StreamDiffusion] could not move UNet to GPU: {e}")
        # Full VAE — similarly deferred. Decoder runs only when TAESD is off.
        if not keep_full_vae_on_cpu:
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "to"):
                try:
                    vae.to(self.device)
                except Exception as e:
                    print(f"[StreamDiffusion] could not move VAE to GPU: {e}")

    def _load_preset(
        self, preset: dict, *, keep_unet_on_cpu: bool = False,
    ) -> DiffusionPipeline:
        """Build a DiffusionPipeline from a MODEL_PRESETS recipe.

        Currently supports the ``unet_swap`` shape — load the base pipeline
        WITHOUT its UNet weights, build an empty UNet from the base's config,
        then stream the distilled checkpoint straight into it. Skipping the
        base UNet load saves ~5-10s of disk I/O and ~2.6 GB of VRAM
        allocation churn — the base UNet would be overwritten immediately,
        so loading it is pure dead weight.

        ``keep_unet_on_cpu`` lets the caller park the distilled UNet on
        CPU (TRT path with cached engine) instead of allocating GPU
        storage. The build-time ONNX export will move it to GPU on
        demand if needed.

        Other recipe shapes (LoRA fuse, scheduler override, timesteps_override)
        will land alongside the `_set_timesteps` refactor needed to support
        non-LCM schedulers.
        """
        base = preset["base"]
        unet_swap = preset.get("unet_swap")

        if unet_swap is not None:
            return self._load_with_unet_swap(
                base, unet_swap, keep_unet_on_cpu=keep_unet_on_cpu,
            )

        print(f"[StreamDiffusion] Loading preset base: {base}")
        return DiffusionPipeline.from_pretrained(
            base,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
        )

    def _load_with_unet_swap(
        self, base: str, unet_swap: tuple[str, str],
        *, keep_unet_on_cpu: bool = False,
    ) -> DiffusionPipeline:
        """Load ``base`` with no UNet, attach a fresh UNet loaded from the
        distilled checkpoint at ``unet_swap = (repo, file)``.

        Two-step trick:
          1. ``from_pretrained(..., unet=None)`` skips downloading and
             materializing the base UNet's safetensors entirely.
          2. ``init_empty_weights()`` builds the UNet topology on the ``meta``
             device (no storage allocated), ``to_empty()`` then allocates
             real GPU storage but doesn't initialize it, and finally
             ``load_state_dict`` fills it from the distilled checkpoint.

        On DMD2 / SDXL-Lightning / similar this trims ~5-10s off model load
        and avoids the brief VRAM doubling we'd otherwise see between
        "base UNet on GPU" and "distilled state_dict materialized".
        """
        from diffusers import UNet2DConditionModel
        unet_repo, unet_file = unet_swap

        print(
            f"[StreamDiffusion] Loading preset base (no UNet): {base} "
            f"+ distilled UNet from {unet_repo}/{unet_file}"
        )

        # Load the base pipeline with unet=None. Diffusers tolerates this
        # for SDXL / SD1.5 pipelines — the UNet attribute ends up None,
        # which we replace below. Avoids ever materializing the base UNet
        # weights from disk.
        pipe = DiffusionPipeline.from_pretrained(
            base,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
            unet=None,
        )

        # Build an empty UNet from the base's config (small JSON, ~100 KB
        # vs the 2.6 GB safetensors). ``init_empty_weights`` puts every
        # parameter on the ``meta`` device — no allocation.
        try:
            from accelerate import init_empty_weights
        except ImportError as e:
            raise RuntimeError(
                "scope-streamdiffusion preset unet_swap requires `accelerate`. "
                "Install with: uv pip install accelerate"
            ) from e

        unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
        with init_empty_weights():
            unet = UNet2DConditionModel.from_config(unet_config)

        # Allocate real (uninitialized) storage in fp16, then stream the
        # distilled weights into it. Target device is CPU when the engine
        # is already cached (TRT path) — saves the GPU staging cycle.
        target_device = "cpu" if keep_unet_on_cpu else self.device
        unet = unet.to_empty(device=target_device).to(self.dtype)
        state_dict = self._load_unet_swap_state_dict(
            unet_repo, unet_file, device=target_device,
        )
        unet.load_state_dict(state_dict)
        pipe.unet = unet
        print("[StreamDiffusion] Distilled UNet weights loaded (no base-UNet allocation)")
        return pipe

    def _load_unet_swap_state_dict(
        self, unet_repo: str, unet_file: str, *, device: Any = None,
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

        target_device = device if device is not None else self.device

        # Native safetensors path — load straight to device, no conversion.
        if unet_file.endswith(".safetensors"):
            try:
                ckpt_path = hf_hub_download(unet_repo, unet_file, local_files_only=True)
                print(f"[StreamDiffusion] Loading cached distilled UNet: {unet_repo}/{unet_file}")
            except LocalEntryNotFoundError:
                print(f"[StreamDiffusion] Downloading distilled UNet: {unet_repo}/{unet_file}")
                ckpt_path = hf_hub_download(unet_repo, unet_file)
            return load_file(ckpt_path, device=str(target_device))

        # .bin path — check the local converted-safetensors cache first.
        cache_dir = Path.home() / ".cache" / "scope-streamdiffusion" / "converted"
        cache_name = f"{unet_repo.replace('/', '__')}__{unet_file}.safetensors"
        cached_path = cache_dir / cache_name
        if cached_path.exists():
            print(
                f"[StreamDiffusion] Loading converted-safetensors UNet: {cached_path.name}"
            )
            return load_file(str(cached_path), device=str(target_device))

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
            map_location=target_device,
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
        """Switch between TAESD (fast) and full VAE decoder.

        Park the inactive decoder on CPU so we only pay GPU storage for
        the one we're actually running. Full SDXL VAE is ~300 MB fp16;
        not huge, but adds up alongside the UNet / ControlNet offloads
        on tight-VRAM cards. Round-trip on toggle.
        """
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
            self._offload_full_vae_to_cpu()
            print("[StreamDiffusion] Switched to TAESD (fast decode)")
        else:
            self._restore_full_vae_to_gpu()
            self.pipe.vae = self.pipe._full_vae
            self.pipe._using_taesd = False
            print("[StreamDiffusion] Switched to full VAE")

    def _offload_full_vae_to_cpu(self) -> None:
        vae = getattr(self.pipe, "_full_vae", None)
        if vae is None:
            return
        try:
            first_param = next(vae.parameters(), None)
        except StopIteration:
            first_param = None
        if first_param is None or first_param.device.type == "cpu":
            return
        try:
            vae.to("cpu")
            torch.cuda.empty_cache()
            print("[StreamDiffusion] full VAE offloaded to CPU", flush=True)
        except Exception as e:
            print(f"[StreamDiffusion] failed to offload full VAE to CPU: {e}")

    def _restore_full_vae_to_gpu(self) -> None:
        vae = getattr(self.pipe, "_full_vae", None)
        if vae is None:
            return
        try:
            first_param = next(vae.parameters(), None)
        except StopIteration:
            first_param = None
        if first_param is None or first_param.device.type == "cuda":
            return
        try:
            vae.to(self.device)
        except Exception as e:
            print(f"[StreamDiffusion] failed to restore full VAE to GPU: {e}")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def apply_loras(self, loras: list, strategy: str = "runtime_peft") -> None:
        """Attach the requested LoRA stack to the diffusers pipe.

        Idempotent: returns early when the requested signature already matches
        what's attached. Three change shapes:

          * **No change** — same paths, same scales, same strategy: no-op.
          * **Scale-only, runtime_peft** — same paths in same order, only
            scales differ: ``set_adapters`` with new weights. ~free.
          * **Stack change or strategy change** — unload everything, reload
            requested adapters, ``set_adapters``, optional ``fuse_lora``.

        ``strategy='permanent_merge'`` calls ``fuse_lora`` so subsequent
        inference (and TRT engine builds) see the merged weights baked in.
        ``runtime_peft`` keeps adapters live so per-frame scale tweaks are
        cheap. TRT mode bakes weights at compile time, so a runtime_peft
        scale change there forces an engine rebuild — handled at the
        pipeline level, not here.

        The pipe must be loaded; safe to call before any LoRAs were ever
        attached (handles the first-attach path).
        """
        if self.pipe is None or self.pipe.pipe is None:
            return

        new_sig = lora_signature_from_specs(loras)
        cur_sig = self._lora_signature
        if new_sig == (cur_sig or ()) and strategy == self._lora_strategy and cur_sig is not None:
            return

        new_paths_in_order = tuple(_spec_path(s) for s in loras if _spec_path(s))

        # Scale-only fast path: paths (and order) unchanged, both old and
        # new strategies are runtime_peft, and we have something attached.
        scale_only = (
            new_paths_in_order == self._lora_paths
            and bool(self._lora_paths)
            and strategy == "runtime_peft"
            and self._lora_strategy == "runtime_peft"
        )
        if scale_only:
            names = [f"lora_{i}" for i in range(len(new_paths_in_order))]
            scales = [_spec_scale(s) for s in loras if _spec_path(s)]
            try:
                self.pipe.pipe.set_adapters(names, adapter_weights=scales)
                self._lora_signature = new_sig
                print(
                    f"[StreamDiffusion] LoRA scales updated: {dict(zip(names, scales))}"
                )
                return
            except Exception as e:
                print(f"[StreamDiffusion] set_adapters fast-path failed, falling back to reload: {e}")

        # Slow path: unload, then reload everything.
        if cur_sig:
            try:
                self.pipe.pipe.unload_lora_weights()
            except Exception as e:
                print(f"[StreamDiffusion] unload_lora_weights failed (non-fatal): {e}")

        self._lora_paths = ()
        self._lora_signature = ()
        self._lora_strategy = strategy

        if not loras:
            return

        names: list[str] = []
        scales: list[float] = []
        loaded_paths: list[str] = []
        for i, spec in enumerate(loras):
            path = _spec_path(spec)
            if not path:
                continue
            scale = _spec_scale(spec)
            adapter_name = f"lora_{i}"
            resolved = resolve_lora_path(path)
            try:
                self.pipe.pipe.load_lora_weights(resolved, adapter_name=adapter_name)
            except Exception as e:
                print(f"[StreamDiffusion] Failed to load LoRA {path}: {e}")
                continue

            # Diffusers' load_lora_weights silently no-ops when the file's
            # architecture doesn't match this pipeline (e.g. a Flux/SD3 DiT
            # LoRA loaded into an SDXL UNet). It logs a warning and registers
            # zero adapters, then set_adapters later fails with a confusing
            # "Adapter name(s) not in present adapters" error. Detect that
            # here so the user sees a clear architecture-mismatch message.
            if not _adapter_registered(self.pipe.pipe, adapter_name):
                print(
                    f"[StreamDiffusion] LoRA {path} registered no adapters — "
                    f"likely architecture mismatch (file may be DiT/Flux/SD3, "
                    f"not SDXL/SD1.5). Skipping."
                )
                continue

            names.append(adapter_name)
            scales.append(scale)
            loaded_paths.append(path)
            print(
                f"[StreamDiffusion] Loaded LoRA: {path} "
                f"(scale={scale}, adapter={adapter_name})"
            )

        if names:
            try:
                self.pipe.pipe.set_adapters(names, adapter_weights=scales)
            except Exception as e:
                print(f"[StreamDiffusion] set_adapters failed: {e}")
            if strategy == "permanent_merge":
                try:
                    self.pipe.pipe.fuse_lora()
                    print("[StreamDiffusion] LoRA weights fused (permanent_merge)")
                except Exception as e:
                    print(f"[StreamDiffusion] fuse_lora failed: {e}")

        # Recompute the signature against what actually loaded — failures
        # would otherwise leave a stale "intent" signature pinned.
        loaded_specs = [
            spec for spec in loras
            if _spec_path(spec) in set(loaded_paths)
        ]
        self._lora_paths = tuple(loaded_paths)
        self._lora_signature = lora_signature_from_specs(loaded_specs)
        self._lora_strategy = strategy

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

        # The fresh pipe will have no LoRAs attached. Reset to ``None``
        # (not ``()``) so the next ``apply_loras`` skips the unload call.
        self._lora_signature = None
        self._lora_paths = ()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def swap(
        self,
        new_model_id: str,
        *,
        loras: list | None = None,
        lora_strategy: str | None = None,
        setup_kwargs: dict | None = None,
    ) -> None:
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
        # Skip the CPU→GPU staging for components we're going to offload
        # anyway. UNet stays on CPU when going to TRT; full VAE stays on
        # CPU when TAESD is the active decoder. Build-time ONNX export
        # for the UNet will pull it up to GPU temporarily.
        trt_path = p.trt.acceleration_mode == "trt"
        use_taesd = bool((setup_kwargs or {}).get("use_taesd", True))
        p.pipe = self.load(
            new_model_id,
            keep_unet_on_cpu=trt_path,
            keep_full_vae_on_cpu=trt_path and use_taesd,
        )
        print(f"[StreamDiffusion] Model loaded: {p.pipe.__class__.__name__}")
        p.sdxl = type(p.pipe) is StableDiffusionXLPipeline
        if p.sdxl and self.dtype == torch.float16:
            self.install_sdxl_fp16_vae()
        # Re-route ControlNet handler to the SDXL- or SD1.5-keyed weights for
        # the new host. ``release_pipe_state`` cleared the cache, so the next
        # ``update()`` will load the right repo from scratch.
        p._cn.set_sdxl(p.sdxl)

        # Attach LoRAs before TRT setup so engine builds compile against
        # the (optionally fused) post-LoRA weights. Caller-supplied
        # overrides win over ``p.config`` because the latter can be the
        # stale init-time snapshot when the swap was triggered by a
        # runtime kwarg change (e.g. live LoRA picker swap under TRT).
        if loras is None:
            cfg = getattr(p, "config", None)
            loras = list(getattr(cfg, "loras", None) or [])
        if lora_strategy is None:
            cfg = getattr(p, "config", None)
            lora_strategy = (
                getattr(cfg, "lora_merge_strategy", None) or "runtime_peft"
            )
        self.apply_loras(loras, lora_strategy)

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

        # Build TRT engines for the new model now so the next frame doesn't
        # stall. Prefer caller-supplied runtime sig over ``p.config`` (the
        # latter is the load-time snapshot and may be stale if the scene
        # changed CN-mode / resolution since load — would otherwise force
        # a second rebuild on the very next frame).
        if p.trt.acceleration_mode == "trt":
            p.trt.setup(**(setup_kwargs or p.trt.setup_args_from_config()))
