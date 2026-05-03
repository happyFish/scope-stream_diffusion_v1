"""Process-wide cache of built TRT adapters keyed by graph node id.

Scope rebuilds plugin instances on every graph edit (see
`scope/src/scope/server/graph_executor.py`); a fresh `StreamDiffusionPipeline`
loses its in-memory `_trt_*_built` flags and rebuilds engines on first call,
even when the on-disk engine cache hits. Loading and binding a TRT engine
context costs ~hundreds of ms per engine, and ONNX→TRT compile costs minutes
when the disk cache misses. Both are visible stalls during graph edits.

This module holds the built adapters at module scope so a new pipeline
instance for the same logical node can swap them straight back in without
touching the engine builder.

Cache key: the user-supplied graph node id when Scope passes it through
`__init__` kwargs. Until that upstream change lands the plugin falls back to
`_anon_<model_id>`, which is correct for the common single-SD-node setup but
collides if two SD nodes ever coexist with different engine signatures.

Engines are tied to (model_id, image_height, image_width); changing any of
those invalidates the cached state and forces a clean rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CachedTRTState:
    signature: tuple  # (model_id, height, width)
    cuda_stream: Any | None = None
    unet_adapter: Any | None = None
    unet_has_controlnet: bool = False
    cn_adapters: dict[str, Any] = field(default_factory=dict)
    taesd_adapter: Any | None = None


_CACHE: dict[str, CachedTRTState] = {}


def cache_key(node_id: str | None, model_id: str) -> str:
    """Return the cache key for this pipeline instance.

    Prefers the node id (stable across graph edits, unique per logical node);
    falls back to a model-id-scoped anon key for compatibility with Scope
    versions that don't yet pass node_id to plugin __init__.
    """
    if node_id:
        return f"node:{node_id}"
    return f"_anon_{model_id}"


def get_or_create(key: str, signature: tuple) -> tuple[CachedTRTState, bool]:
    """Look up an entry; return (state, restored).

    `restored=True` means the cached signature matched and the caller should
    reuse `state.*_adapter`. `restored=False` means either no entry existed or
    the signature changed (engines built for different dims/model); the entry
    is reset to a fresh state so callers can populate it after building.
    """
    existing = _CACHE.get(key)
    if existing is not None and existing.signature == signature:
        return existing, True
    fresh = CachedTRTState(signature=signature)
    _CACHE[key] = fresh
    return fresh, False


def peek(key: str) -> CachedTRTState | None:
    return _CACHE.get(key)


def clear(key: str | None = None) -> None:
    """Drop one entry, or the whole cache when key is None.

    Adapters hold CUDA memory; clearing here releases the only strong ref
    once the previous pipeline instance is also gone.
    """
    if key is None:
        _CACHE.clear()
    else:
        _CACHE.pop(key, None)
