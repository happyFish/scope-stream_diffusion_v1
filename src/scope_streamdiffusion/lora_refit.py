"""Hot LoRA scale refit for built TRT UNet engines.

Whenever a LoRA stack is present, the UNet engine is built with
``build_enable_refit=True`` so its weight tables can be replaced
in-place at runtime via TensorRT's ``IRefitter`` API. Scale changes
(paths unchanged, only the floats moving) then skip the 5-15 minute
engine rebuild and instead push fresh weights into the live engine
in a few hundred ms.

Side benefit: the on-disk engine cache key drops scale from its hash
when refit is enabled, so a single engine on disk serves any scale.
Without this the cache filled with one ~5 GB engine per scale tweak.

Cost:
  * Engine size grows ~10-15% (refittable weights aren't kernel-fused).
  * Inference slows ~5-8% (less aggressive layer fusion).
  * First-frame latency is unchanged.

Architecture
------------
``apply_loras`` already drives ``load_lora_weights`` + ``set_adapters`` +
``fuse_lora`` on the eager UNet (``pipe.pipe.unet``). We piggyback on
that: when scales change we let it re-run, then read the updated
parameters off the eager UNet's ``state_dict()`` and push them into
the TRT engine via ``IRefitter.set_named_weights``.

Name mapping is the only fiddly part. ONNX export usually preserves
torch parameter names as initializer names (e.g.
``down_blocks.0.attentions.0.to_q.weight``), and TRT's refitter
exposes those names directly. For weights that don't match, we log
and skip — the engine keeps its baked-in values for those tensors,
which is acceptable for a spike (LoRAs only touch a small subset of
attention weights anyway).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RefitContext:
    """Per-engine state needed to perform a runtime refit.

    Lives on the TRT adapter while the engine is active. Captured at
    build time so the eager UNet's parameters at the moment of build
    are the reference.
    """

    # Reference to the live diffusers eager UNet (``pipe.pipe.unet``).
    # We don't own this — the model_loader does. Held only so we can
    # re-read its state_dict() after a fresh fuse_lora().
    eager_unet: Any
    # The TRT ``Engine`` instance from ``_trt/utilities.py`` — we walk
    # ``Refitter(self.engine.engine)`` and call ``refit_cuda_engine``.
    trt_engine: Any
    # The diffusers pipeline (``pipe.pipe``) so we can call fuse_lora /
    # unfuse_lora around the refit. Needed under ``runtime_peft`` because
    # PEFT keeps base weights pristine and adds the LoRA contribution at
    # forward time — without fusing, state_dict() returns base weights
    # regardless of scale and the refit silently writes base weights into
    # the engine (LoRA effect lost).
    diffusers_pipe: Any = None
    # Callable returning the current ``lora_merge_strategy`` string
    # (``"runtime_peft"`` or ``"permanent_merge"``). Lets the refit
    # decide whether to fuse before reading state_dict. ``None`` →
    # assume ``permanent_merge`` and skip the fuse roundtrip.
    get_strategy: Any = None
    # Names of TRT weights present in the engine — populated lazily on
    # first refit by enumerating the Refitter.
    _trt_weight_names: set[str] = field(default_factory=set)


def _enumerate_refittable_weights(trt_engine: Any) -> set[str]:
    """Return the set of weight names the engine exposes for refit.

    TensorRT's ``Refitter`` enumerates *layer* names + roles via
    ``get_all()``; the modern named-weights API exposes initializer
    names via ``get_all_weights()``. We use the latter — those names
    are typically ONNX initializer names which match torch parameter
    names from the original UNet.
    """
    import tensorrt as trt

    refitter = trt.Refitter(trt_engine.engine, trt.Logger(trt.Logger.WARNING))
    try:
        # TRT 8.6+ — preferred. Returns (names_tuple,) on some versions
        # and a flat tuple of names on others.
        names = refitter.get_all_weights()
    except AttributeError:
        # Fallback: legacy layer/role enumeration.
        layer_names, _roles = refitter.get_all()
        names = layer_names

    if isinstance(names, tuple) and len(names) == 1 and isinstance(names[0], (list, tuple)):
        names = names[0]
    return {str(n) for n in names}


def refit_unet_lora_scales(ctx: RefitContext) -> bool:
    """Push the eager UNet's current params into the live TRT engine.

    Caller is expected to have already updated the eager UNet —
    typically by calling ``apply_loras(new_loras)`` so ``fuse_lora``
    bakes the new scales into ``pipe.pipe.unet`` weights. We then read
    those weights off the state_dict and push them into TRT.

    Returns True on successful refit, False if the refit failed
    (caller should fall back to a full engine rebuild).
    """
    import tensorrt as trt

    eager = ctx.eager_unet
    trt_engine = ctx.trt_engine

    if eager is None or trt_engine is None or trt_engine.engine is None:
        print("[LoRA refit] missing eager UNet or live engine — aborting refit")
        return False

    if not ctx._trt_weight_names:
        ctx._trt_weight_names = _enumerate_refittable_weights(trt_engine)
        if not ctx._trt_weight_names:
            print("[LoRA refit] engine exposes no refittable weights — was it built with refit?")
            return False
        print(f"[LoRA refit] engine exposes {len(ctx._trt_weight_names)} refittable weights")

    # Under ``runtime_peft`` (the default), PEFT keeps base weights pristine
    # and applies the LoRA delta at forward time — ``state_dict()`` therefore
    # returns *base* weights regardless of the active scale. We need merged
    # weights for refit, so fuse-then-read-then-unfuse. ``permanent_merge``
    # already has the deltas baked into base, so we skip the roundtrip.
    strategy = ctx.get_strategy() if ctx.get_strategy is not None else "permanent_merge"
    pipe = ctx.diffusers_pipe
    fused_here = False
    if strategy == "runtime_peft" and pipe is not None and hasattr(pipe, "fuse_lora"):
        try:
            pipe.fuse_lora()
            fused_here = True
        except Exception as e:
            print(f"[LoRA refit] fuse_lora before refit failed: {e}")
            return False

    try:
        state = eager.state_dict()
        refitter = trt.Refitter(trt_engine.engine, trt.Logger(trt.Logger.WARNING))

        # ONNX export goes through a wrapper that stores the UNet as
        # ``self.unet`` (see ``_trt/models.py:UNetSDXLExportWrapper`` and
        # the plain-UNet variant). The wrapper's torch parameter names
        # get an ``unet.`` prefix during export, which carries into the
        # ONNX initializer names that TRT keeps as the refit identifiers.
        # So state_dict key ``down_blocks.0.attentions.0.to_q.weight``
        # matches engine name ``unet.down_blocks.0.attentions.0.to_q.weight``.
        def _candidate_names(key: str) -> list[str]:
            return [key, f"unet.{key}"]

        matched = 0
        skipped = 0
        unmatched_samples: list[str] = []
        for name, tensor in state.items():
            engine_name = next(
                (cand for cand in _candidate_names(name) if cand in ctx._trt_weight_names),
                None,
            )
            if engine_name is None:
                skipped += 1
                if len(unmatched_samples) < 5:
                    unmatched_samples.append(name)
                continue
            # TRT wants contiguous CPU numpy; engine ingests in fp16 so cast.
            arr = tensor.detach().to("cpu", dtype=torch.float16).contiguous().numpy()
            weights = trt.Weights(arr)
            try:
                refitter.set_named_weights(engine_name, weights)
                matched += 1
            except Exception as e:
                print(f"[LoRA refit] set_named_weights({engine_name}) failed: {e}")
                skipped += 1

        if matched == 0:
            print("[LoRA refit] no UNet param names matched engine weights — name mapping bust")
            print(f"[LoRA refit] sample state_dict keys: {unmatched_samples}")
            sample_trt = sorted(ctx._trt_weight_names)[:5]
            print(f"[LoRA refit] sample engine weights:  {sample_trt}")
            return False

        print(f"[LoRA refit] matched {matched} / {matched + skipped} weights — refitting engine")
        if not refitter.refit_cuda_engine():
            print("[LoRA refit] refit_cuda_engine failed")
            return False

        print(
            f"[LoRA refit] refit complete ({matched} weights updated, "
            f"strategy={strategy})"
        )
        return True
    finally:
        # Always restore the eager UNet to its pre-refit state so PEFT's
        # forward path doesn't double-apply the LoRA delta (fused base
        # weights + active adapter).
        if fused_here:
            try:
                pipe.unfuse_lora()
            except Exception as e:
                print(f"[LoRA refit] unfuse_lora after refit failed: {e}")


def is_scale_only_change(prev_paths: tuple, prev_sig: tuple, new_paths: tuple, new_sig: tuple) -> bool:
    """True iff the two LoRA stacks share paths but differ in scales.

    Used to decide whether to attempt a refit (cheap) or fall back to
    a full UNet engine rebuild (expensive). Matches the signature
    layout from ``lora_signature_from_specs``: a tuple of
    ``(path, scale)`` pairs sorted by path.
    """
    if not prev_paths or not new_paths:
        return False
    if tuple(sorted(prev_paths)) != tuple(sorted(new_paths)):
        return False
    if prev_sig == new_sig:
        return False  # no change at all
    return True
