"""TensorRT engine lifecycle for the StreamDiffusion pipeline.

Owns the build / restore / swap / teardown of TRT adapters for the UNet,
ControlNet, and TAESD VAE, plus the runtime acceleration-mode toggle.
The pipeline holds an instance as ``self.trt`` and calls ``attach()``
after each model load. Per-frame ``__call__`` flow asks ``setup()`` to
ensure the right engines are live for the current ``(height, width,
controlnet_mode, use_taesd)`` quadruple.

Lifecycle: ``attach(pipe, sdxl)`` rebinds the helper to the freshly
loaded pipeline. ``reset_caches(new_model_id)`` invalidates sticky
build flags and the per-instance adapter handles before a model swap;
``release()`` is the heavier teardown for full pipeline teardown.

Compromise (documented in REFACTOR_PLAN.md): the engine swap mutates
``pipe.unet`` / ``pipe.controlnet`` / ``pipe.vae`` / ``pipe._taesd_vae``
directly via the back-reference. Untangling that is a separate refactor;
the win here is the ~500 lines of TRT-specific code leaving the
orchestrator.
"""

from __future__ import annotations

from typing import Any

import torch

from . import _trt_cache


class TRTLifecycle:
    """Build, cache, and swap TRT adapters for the StreamDiffusion pipeline.

    The pipeline owns the diffusers modules (``pipe.unet``, ``pipe.vae``,
    ``pipe.controlnet``, ``pipe._taesd_vae``); this helper writes through
    those references via ``self.pipe`` after ``attach()``.

    Acceleration mode: ``"none"`` runs the eager modules, ``"trt"`` swaps
    in built engines. Toggle at runtime via ``set_acceleration_mode()``.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        node_id: str | None = None,
        acceleration_mode: str = "none",
    ) -> None:
        self.device = device
        self.dtype = dtype

        # Live pipe reference — set via ``attach()``. Until then the helper
        # is inert; ``setup()`` short-circuits on missing pipe via the
        # acceleration-mode gate.
        self.pipe: Any = None
        self.sdxl: bool = False

        # The user-supplied graph node id (stable across plugin reinit) is
        # used to key adapters in the cross-instance ``_trt_cache._CACHE``.
        # Falls back to a model-scoped anon key when absent.
        self._node_id: str | None = node_id

        # Acceleration mode (the last-applied value). Pipeline reads this
        # via ``self.trt.acceleration_mode``.
        if acceleration_mode not in ("none", "trt"):
            acceleration_mode = "none"
        self._acceleration_mode: str = acceleration_mode

        # Sticky build flags + per-mode caches. Cleared on ``reset_caches()``.
        self._trt_taesd_built: bool = False
        self._trt_eager_taesd: Any = None

        # TRT engine state — set once on first call when acceleration_mode='trt'.
        # pipe.unet is swapped to a TRT UNet adapter; pipe.controlnet (when
        # controlnet_mode != 'none') is independently swapped to a TRT
        # ControlNet adapter. Two separate engines: each fits well under
        # ONNX's 2 GB limit individually, avoiding the combined-graph
        # cask-convolution bug we hit trying to merge them.
        self._trt_unet_built: bool = False
        self._trt_unet_has_controlnet: bool = False  # legacy flag from combined-engine attempt
        self._trt_cn_built_modes: set[str] = set()
        self._trt_cn_engines: dict[str, Any] = {}  # mode -> TRTControlNetAdapter
        self._trt_eager_controlnets: dict[str, Any] = {}  # mode -> diffusers ControlNetModel (fallback)
        self._trt_cuda_stream: Any = None
        self._trt_eager_unet: Any = None  # original; kept for fallback
        # (height, width, controlnet_mode, use_taesd) of the last setup() call.
        # Pipeline's __call__ compares the current values against this and
        # re-runs setup only on real divergence — otherwise the per-frame
        # TRT block is a no-op.
        self._trt_setup_signature: tuple | None = None

        # Cache key for the cross-instance ``_trt_cache._CACHE``.
        self._trt_cache_key: str = _trt_cache.cache_key(self._node_id, "")

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def acceleration_mode(self) -> str:
        return self._acceleration_mode

    @property
    def setup_signature(self) -> tuple | None:
        return self._trt_setup_signature

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, pipe: Any, sdxl: bool) -> None:
        """Wire to a loaded pipeline.

        Call from ``_ensure_pipe_loaded`` and ``_swap_model`` after the new
        pipe is available. The cache key is updated to match the freshly
        loaded model id (read off ``pipe.model_id``-equivalent via the
        pipeline's ``self.model_id`` attribute).
        """
        self.pipe = pipe
        self.sdxl = sdxl
        # Pipeline's ``model_id`` reflects the just-loaded model. Refresh
        # the TRT cache key so subsequent ``setup`` calls land in the
        # right slot in the cross-instance cache.
        model_id = getattr(pipe, "model_id", "") or ""
        self._trt_cache_key = _trt_cache.cache_key(self._node_id, model_id)

    def reset_caches(self, new_model_id: str) -> None:
        """Invalidate TRT sticky state so the next ``setup()`` rebuilds.

        Called from ``_swap_model`` before loading the new model. Without
        this, the sticky ``_trt_unet_built`` / ``_trt_taesd_built`` flags
        cause subsequent ``ensure_*`` calls to short-circuit and the
        new model runs eager regardless of ``acceleration_mode``.
        """
        # Drop the module-scope cache entry for the previous model. Without
        # this its ``unet_adapter`` / ``cn_adapters`` / ``taesd_adapter``
        # references stay live in ``_trt_cache._CACHE`` and pin engine
        # memory across the swap — direct cause of OOM on a 24 GB card
        # when going SD1.5 → SDXL with TRT on.
        old_key = self._trt_cache_key
        if old_key:
            _trt_cache.clear(old_key)
        self._trt_unet_built = False
        self._trt_unet_has_controlnet = False
        self._trt_taesd_built = False
        self._trt_eager_unet = None
        self._trt_eager_taesd = None
        self._trt_cn_built_modes.clear()
        self._trt_cn_engines.clear()
        self._trt_eager_controlnets.clear()
        self._trt_cache_key = _trt_cache.cache_key(self._node_id, new_model_id)
        self._trt_setup_signature = None

    def release(self) -> None:
        """Drop all TRT adapter handles owned by this helper.

        Adapter objects pin engine memory; nulling them here releases the
        only strong ref once any cached entry in ``_trt_cache._CACHE`` is
        also cleared. The module-scope cache is intentionally not flushed —
        it must survive plugin reinit (that's the whole point of the
        cross-instance cache).
        """
        self._trt_taesd_built = False
        self._trt_eager_taesd = None
        self._trt_unet_built = False
        self._trt_unet_has_controlnet = False
        self._trt_cn_built_modes.clear()
        self._trt_cn_engines.clear()
        self._trt_eager_controlnets.clear()
        self._trt_eager_unet = None
        self._trt_cuda_stream = None
        self._trt_setup_signature = None

    # ------------------------------------------------------------------
    # Acceleration-mode toggle
    # ------------------------------------------------------------------

    def set_acceleration_mode(self, mode: str) -> None:
        """Swap between TRT-accelerated and eager modules at runtime.

        TRT engines themselves are immutable after build, but the choice of
        which UNet / ControlNet / TAESD module ``pipe.*`` points at *can* be
        flipped per frame. Cached adapters (in ``_trt_cache._CACHE`` and on
        the helper) stay alive across the swap so toggling back to 'trt'
        is instant after the first build.
        """
        if mode not in ("none", "trt") or mode == self._acceleration_mode:
            return
        print(
            f"[StreamDiffusion] acceleration_mode swap: "
            f"{self._acceleration_mode} -> {mode}"
        )
        if mode == "none":
            self.deactivate()
            self._acceleration_mode = "none"
        else:
            self._acceleration_mode = "trt"
            self.setup(**self._setup_args_from_config())

    def deactivate(self) -> None:
        """Restore eager UNet / ControlNet / TAESD; keep adapters cached.

        Resets the sticky ``_trt_*_built`` flags so a future ``setup()``
        re-enters the cache-restore path and re-attaches the same adapters
        without rebuilding.
        """
        if self._trt_eager_unet is not None and self.pipe.unet is not self._trt_eager_unet:
            self.pipe.unet = self._trt_eager_unet
        if self._trt_eager_taesd is not None:
            self.pipe._taesd_vae = self._trt_eager_taesd
            if self.pipe._using_taesd:
                self.pipe.vae = self.pipe._taesd_vae
        if self.pipe._cn.model is not None:
            self.pipe.controlnet = self.pipe._cn.model
        self._trt_unet_built = False
        self._trt_unet_has_controlnet = False
        self._trt_taesd_built = False
        self._trt_cn_built_modes.clear()
        self._trt_setup_signature = None

    # ------------------------------------------------------------------
    # Setup orchestration
    # ------------------------------------------------------------------

    def setup(
        self,
        *,
        height: int,
        width: int,
        controlnet_mode: str,
        use_taesd: bool,
    ) -> None:
        """Build or attach TRT engines for the current model.

        Called at load time (``_ensure_pipe_loaded`` and ``_swap_model``)
        so the first frame doesn't stall on a 5-10 minute compile, and
        again from ``__call__`` only when ``(height, width, controlnet_mode,
        use_taesd)`` diverges from the last setup. The inner ``ensure_*``
        methods short-circuit when nothing needs to change.
        """
        if self._acceleration_mode != "trt":
            return
        try:
            self._ensure_unet(
                controlnet_mode,
                image_height=int(height),
                image_width=int(width),
            )
        except Exception as e:
            print(f"[TRT] UNet engine swap failed, falling back to eager: {e}")
            import traceback
            traceback.print_exc()
            if self._trt_eager_unet is not None:
                self.pipe.unet = self._trt_eager_unet
        if controlnet_mode in ("depth", "scribble"):
            try:
                self._ensure_controlnet(controlnet_mode)
            except Exception as e:
                print(
                    f"[TRT] ControlNet engine swap failed for {controlnet_mode}, using eager: {e}"
                )
                import traceback
                traceback.print_exc()
        if use_taesd:
            try:
                self._ensure_taesd()
            except Exception as e:
                print(f"[TRT] TAESD engine swap failed, using eager: {e}")
                import traceback
                traceback.print_exc()
        self._trt_setup_signature = (
            int(height),
            int(width),
            controlnet_mode,
            bool(use_taesd),
        )

    def setup_args_from_config(self) -> dict:
        """Public alias used by the pipeline's load-time hooks."""
        return self._setup_args_from_config()

    def _setup_args_from_config(self) -> dict:
        """Resolve setup() args from ``self.pipe.config`` with schema-default fallbacks."""
        cfg = getattr(self.pipe, "config", None) if self.pipe is not None else None
        return {
            "height": int(getattr(cfg, "height", 512)) if cfg else 512,
            "width": int(getattr(cfg, "width", 512)) if cfg else 512,
            "controlnet_mode": getattr(cfg, "controlnet_mode", "none") if cfg else "none",
            "use_taesd": bool(getattr(cfg, "use_taesd", True)) if cfg else True,
        }

    # ------------------------------------------------------------------
    # Per-component ensure_* (build or restore from cache)
    # ------------------------------------------------------------------

    def _ensure_taesd(self) -> None:
        """Build TRT engines for the TAESD encoder + decoder, swap pipe.vae.

        Called only when use_taesd is on AND acceleration_mode='trt'.
        TAESD is small so engine builds are fast (~30-60s per engine on
        first run). One-shot per process; cached engines load instantly.
        """
        if self._trt_taesd_built:
            return
        if self.pipe._taesd_vae is None:
            return

        signature = (self.pipe.model_id, int(self.pipe.height), int(self.pipe.width))
        cache_state, restored = _trt_cache.get_or_create(self._trt_cache_key, signature)
        if self._trt_cuda_stream is None and cache_state.cuda_stream is not None:
            self._trt_cuda_stream = cache_state.cuda_stream
        if restored and cache_state.taesd_adapter is not None:
            self._trt_eager_taesd = self.pipe._taesd_vae
            self.pipe._taesd_vae = cache_state.taesd_adapter
            if self.pipe._using_taesd:
                self.pipe.vae = cache_state.taesd_adapter
            self._trt_taesd_built = True
            print(
                f"[TRT] TAESD adapter restored from cache (key={self._trt_cache_key})",
                flush=True,
            )
            return

        self._trt_taesd_built = True  # prevent retry on failure

        from .trt_engines import (
            TRTTaesdAdapter,
            build_taesd_engines,
            make_cuda_stream,
        )
        if self._trt_cuda_stream is None:
            self._trt_cuda_stream = make_cuda_stream()
        cache_state.cuda_stream = self._trt_cuda_stream
        print(
            "[TRT] Preparing TAESD engines — first build takes ~1 min, cached after",
            flush=True,
        )
        try:
            taesd_model_id = "madebyollin/taesdxl" if self.sdxl else "madebyollin/taesd"
            enc_path, dec_path = build_taesd_engines(
                self.pipe._taesd_vae,
                model_id=taesd_model_id,
                image_height=int(self.pipe.height),
                image_width=int(self.pipe.width),
                min_batch_size=1,
                max_batch_size=4,
            )
        except Exception as e:
            print(f"[TRT] TAESD engine build failed, using eager: {e}")
            return
        scaling_factor = float(self.pipe._taesd_vae.config.scaling_factor)
        vae_scale_factor = self.pipe.pipe.vae_scale_factor
        adapter = TRTTaesdAdapter(
            enc_path, dec_path, self._trt_cuda_stream,
            scaling_factor=scaling_factor,
            vae_scale_factor=vae_scale_factor,
            dtype=self.dtype,
        )
        self._trt_eager_taesd = self.pipe._taesd_vae
        self.pipe._taesd_vae = adapter
        if self.pipe._using_taesd:
            self.pipe.vae = adapter
        cache_state.taesd_adapter = adapter
        print(f"[TRT] TAESD engines active: enc={enc_path.name}, dec={dec_path.name}", flush=True)

    def _ensure_controlnet(self, mode: str) -> None:
        """Build (or reuse) TRT engine for the active ControlNet, swap pipe.controlnet.

        Per-mode engine cached on disk + once built per process. Each mode
        (depth/scribble) gets its own engine because the underlying
        diffusers ControlNetModel weights differ. On build failure or
        unsupported mode we leave pipe.controlnet as the eager model.
        """
        if mode in ("none", None) or self.pipe._cn.model is None:
            return
        if mode in self._trt_cn_built_modes:
            # Already attempted (success or failure). Restore the engine
            # adapter if we previously built one for this mode.
            adapter = self._trt_cn_engines.get(mode)
            if adapter is not None and self.pipe.controlnet is not adapter:
                self._trt_eager_controlnets[mode] = self.pipe._cn.model
                self.pipe.controlnet = adapter
            return

        signature = (self.pipe.model_id, int(self.pipe.height), int(self.pipe.width))
        cache_state, restored = _trt_cache.get_or_create(self._trt_cache_key, signature)
        if self._trt_cuda_stream is None and cache_state.cuda_stream is not None:
            self._trt_cuda_stream = cache_state.cuda_stream
        cached_cn = cache_state.cn_adapters.get(mode) if restored else None
        if cached_cn is not None:
            self._trt_eager_controlnets[mode] = self.pipe._cn.model
            self._trt_cn_engines[mode] = cached_cn
            self._trt_cn_built_modes.add(mode)
            self.pipe.controlnet = cached_cn
            print(
                f"[TRT] ControlNet adapter restored from cache "
                f"(mode={mode}, key={self._trt_cache_key})",
                flush=True,
            )
            return

        self._trt_cn_built_modes.add(mode)  # mark before build to prevent retry storm

        from .trt_engines import (
            TRTControlNetAdapter,
            build_controlnet_engine,
            make_cuda_stream,
        )

        if self._trt_cuda_stream is None:
            self._trt_cuda_stream = make_cuda_stream()
        cache_state.cuda_stream = self._trt_cuda_stream

        # ControlNet ONNX export needs default attention too (same xformers
        # issue as the UNet path).
        try:
            self.pipe._cn.model.set_default_attn_processor()
        except AttributeError:
            pass

        print(
            f"[TRT] Preparing ControlNet engine ({mode}) — first build takes 3-5 min, cached after",
            flush=True,
        )
        try:
            engine_path = build_controlnet_engine(
                self.pipe._cn.model,
                model_id=self.pipe.model_id,
                controlnet_id=mode,
                image_height=int(self.pipe.height),
                image_width=int(self.pipe.width),
                min_batch_size=1,
                max_batch_size=4,
            )
        except Exception as e:
            print(f"[TRT] ControlNet engine build failed for {mode}, using eager: {e}")
            return
        adapter = TRTControlNetAdapter(engine_path, self._trt_cuda_stream)
        self._trt_eager_controlnets[mode] = self.pipe._cn.model
        self._trt_cn_engines[mode] = adapter
        self.pipe.controlnet = adapter
        cache_state.cn_adapters[mode] = adapter
        print(f"[TRT] ControlNet engine active ({mode}): {engine_path}", flush=True)

    def _ensure_unet(
        self,
        controlnet_mode: str = "none",
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> None:
        """Build TRT engine for the UNet and swap pipe.unet to the adapter.

        Two variants depending on controlnet_mode:
          * 'none'              → plain UNet engine (3 inputs, fastest)
          * 'depth' | 'scribble' → UNet engine WITH ControlNet residual input
            slots so the ControlNet output (from the standalone engine)
            actually reaches the UNet's inner blocks. Without this, the
            residuals get silently dropped and ControlNet conditioning has
            no effect on the output.

        ``image_height`` / ``image_width`` should be the runtime spatial
        dims for this build. Falls back to ``pipe.height`` / ``pipe.width``
        when omitted, but caller should pass them explicitly because
        ``_prepare_runtime_state`` (which sets ``pipe.{height,width}``)
        normally runs *after* this method.

        Engines are cached separately on disk because they have different
        signatures. Switching modes mid-process may trigger a rebuild.
        """
        want_control = controlnet_mode in ("depth", "scribble")

        # If we previously built the wrong variant, rebuild now.
        if self._trt_unet_built and self._trt_unet_has_controlnet == want_control:
            return
        if self._trt_unet_built and self._trt_unet_has_controlnet != want_control:
            print(
                f"[TRT] UNet ctrl-input variant changed "
                f"(had={self._trt_unet_has_controlnet}, want={want_control}); rebuilding"
            )

        eff_h = int(image_height if image_height is not None else self.pipe.height)
        eff_w = int(image_width if image_width is not None else self.pipe.width)
        signature = (self.pipe.model_id, eff_h, eff_w)
        cache_state, restored = _trt_cache.get_or_create(self._trt_cache_key, signature)
        if self._trt_cuda_stream is None and cache_state.cuda_stream is not None:
            self._trt_cuda_stream = cache_state.cuda_stream
        if (
            restored
            and cache_state.unet_adapter is not None
            and cache_state.unet_has_controlnet == want_control
        ):
            self._trt_eager_unet = self.pipe.pipe.unet
            self.pipe.unet = cache_state.unet_adapter
            self._trt_unet_built = True
            self._trt_unet_has_controlnet = want_control
            print(
                f"[TRT] UNet adapter restored from cache "
                f"(want_control={want_control}, key={self._trt_cache_key})",
                flush=True,
            )
            return

        # Set sticky flags before the build so failures don't retry every frame.
        self._trt_unet_built = True
        self._trt_unet_has_controlnet = want_control

        # Restore eager forward if torch.compile wrapped it earlier.
        if self.pipe._unet_compiled:
            self.pipe.unet = self.pipe.pipe.unet
            self.pipe._unet_compiled = False

        # xformers flash-attention ops can't be ONNX-exported.
        try:
            self.pipe.pipe.unet.set_default_attn_processor()
            print("[TRT] swapped UNet to default attention for ONNX export")
        except AttributeError:
            try:
                self.pipe.pipe.disable_xformers_memory_efficient_attention()
                print("[TRT] disabled xformers on pipe for ONNX export")
            except Exception as e:
                print(f"[TRT] could not disable xformers attention: {e}")

        from .trt_engines import (
            TRTUNetAdapter,
            TRTUNetSDXLAdapter,
            TRTUNetWithControlAdapter,
            build_unet_engine,
            build_unet_sdxl_engine,
            build_unet_with_control_engine,
            make_cuda_stream,
        )

        if self._trt_cuda_stream is None:
            self._trt_cuda_stream = make_cuda_stream()
        cache_state.cuda_stream = self._trt_cuda_stream

        if want_control:
            if self.sdxl:
                # SDXL + ControlNet + TRT: not yet wired. The ControlNet
                # path uses UNetWithControlInputs which assumes SD1.5
                # signature (no text_embeds/time_ids). Falling through to
                # eager keeps SDXL+ControlNet working until that variant
                # gets the same SDXL aug-conditioning treatment as
                # build_unet_sdxl_engine.
                raise NotImplementedError(
                    "SDXL + ControlNet + TRT not yet supported. Use "
                    "acceleration_mode='none' with controlnet on SDXL models."
                )
            print(
                "[TRT] Preparing UNet+ctrl engine — first build takes 5-10 min, cached after",
                flush=True,
            )
            engine_path = build_unet_with_control_engine(
                self.pipe.pipe.unet,
                model_id=self.pipe.model_id,
                image_height=eff_h,
                image_width=eff_w,
                min_batch_size=1,
                max_batch_size=4,
                num_down_residuals=12,
            )
            self._trt_eager_unet = self.pipe.pipe.unet
            self.pipe.unet = TRTUNetWithControlAdapter(
                engine_path, self._trt_cuda_stream, num_down_residuals=12,
            )
            cache_state.unet_adapter = self.pipe.unet
            cache_state.unet_has_controlnet = True
            print(f"[TRT] UNet+ctrl engine active: {engine_path}", flush=True)
        elif self.sdxl:
            print(
                "[TRT] Preparing SDXL UNet engine — first build takes 5-10 min, cached after",
                flush=True,
            )
            # TRT's builder TACTIC_DRAM allocator can race our resident
            # pipeline allocations during engine build. Free what we can
            # (VAE + text encoders, ~5 GB combined) without disturbing
            # the UNet — the ONNX tracer needs it on GPU. Restore after
            # the build completes.
            print("[TRT] Moving VAE + text encoders to CPU during build", flush=True)
            cpu_components = []
            for attr in ("vae", "text_encoder", "text_encoder_2"):
                comp = getattr(self.pipe.pipe, attr, None)
                if comp is not None and hasattr(comp, "to"):
                    try:
                        comp.to("cpu")
                        cpu_components.append((attr, comp))
                    except Exception as e:
                        print(f"[TRT] could not move {attr} to CPU: {e}", flush=True)
            torch.cuda.empty_cache()
            # batch=1 + static shape (set in build_unet_sdxl_engine) make
            # TRT's tactic search bounded enough to fit on a 24 GB card.
            # Engine is only valid at the (height, width, batch=1) profile
            # it was built for; resolution changes will trigger a rebuild.
            engine_path = build_unet_sdxl_engine(
                self.pipe.pipe.unet,
                model_id=self.pipe.model_id,
                image_height=eff_h,
                image_width=eff_w,
                min_batch_size=1,
                max_batch_size=1,
            )
            print("[TRT] Restoring VAE + text encoders to GPU", flush=True)
            for attr, comp in cpu_components:
                try:
                    comp.to(self.device)
                except Exception as e:
                    print(f"[TRT] could not restore {attr} to {self.device}: {e}", flush=True)
            self._trt_eager_unet = self.pipe.pipe.unet
            self.pipe.unet = TRTUNetSDXLAdapter(engine_path, self._trt_cuda_stream)
            cache_state.unet_adapter = self.pipe.unet
            cache_state.unet_has_controlnet = False
            print(f"[TRT] SDXL UNet engine active: {engine_path}", flush=True)
        else:
            print(
                "[TRT] Preparing UNet engine — first build takes 5-10 min, cached after",
                flush=True,
            )
            engine_path = build_unet_engine(
                self.pipe.pipe.unet,
                model_id=self.pipe.model_id,
                image_height=eff_h,
                image_width=eff_w,
                min_batch_size=1,
                max_batch_size=4,
            )
            self._trt_eager_unet = self.pipe.pipe.unet
            self.pipe.unet = TRTUNetAdapter(engine_path, self._trt_cuda_stream)
            cache_state.unet_adapter = self.pipe.unet
            cache_state.unet_has_controlnet = False
            print(f"[TRT] UNet engine active: {engine_path}", flush=True)
