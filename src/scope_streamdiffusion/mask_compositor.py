"""Mask compositing for the StreamDiffusion pipeline.

Blends SD output with an upstream background source, gated by a binary
mask from a segmenter (``yolo_mask``, ``scope-sam3``). SD output where
mask=1, background where mask=0; flip via the upstream segmenter's
"Invert Mask" toggle.

Stateless — the entire concern is "given a few kwargs and the SD output,
produce the composited frame." No caches, no per-instance state, no
attach lifecycle. Lives as a module of functions rather than a class.

Shape handling is the bulk of the code. ``vace_input_masks`` and
``vace_input_frames`` can arrive in two shapes depending on whether
Scope's graph executor treated the port as a video stream (re-chunked
into per-frame THWC tensors) or passed the producer's raw 5-D tensor
through. Both paths land in ``__call__``; the coercion helpers
normalize before the compositing math runs.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def coerce_mask_to_bchw(masks_in: Any) -> torch.Tensor:
    """Normalize a mask kwarg to ``(1, 1, H, W)`` regardless of producer shape.

    Accepted shapes:
      * ``(1, 1, F, H, W)`` — raw producer emit, takes frame 0.
      * ``(T, H, W, C)`` — Scope's THWC video-port chunk; first frame, first channel.
      * ``(B, C, H, W)`` — already BCHW; collapses to single channel.
      * ``(H, W, C)`` — single un-batched frame.
      * ``(H, W)`` — bare 2-D mask.

    Caller is responsible for moving the result to the desired
    device / dtype.
    """
    m = masks_in[0] if isinstance(masks_in, list) else masks_in
    if m.ndim == 5:
        return m[:, :, 0]
    if m.ndim == 4 and m.shape[-1] in (1, 3):
        m = m[:1].permute(0, 3, 1, 2)
        return m[:, :1] if m.shape[1] > 1 else m
    if m.ndim == 4:
        return m[:, :1] if m.shape[1] > 1 else m
    if m.ndim == 3 and m.shape[-1] in (1, 3):
        m = m.permute(2, 0, 1).unsqueeze(0)
        return m[:, :1] if m.shape[1] > 1 else m
    if m.ndim == 2:
        return m.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"coerce_mask_to_bchw: unsupported shape {tuple(m.shape)}")


def coerce_background_to_bhwc(vace_frames: Any) -> Optional[torch.Tensor]:
    """Normalize a ``vace_input_frames`` kwarg to ``(1, H, W, C)`` in [0, 1].

    Producer emits ``(1, C, F, H, W)`` in [-1, 1] VAE space. After
    Scope's video-chunking it may arrive as ``(T, H, W, C)`` in either
    range. We range-detect against -0.01 so a [-1, 1] frame gets
    rescaled to [0, 1] before the composite math.

    Returns ``None`` when the input is missing or in an unsupported
    shape; caller falls back to the ``video`` frame.
    """
    if vace_frames is None:
        return None
    vf = vace_frames[0] if isinstance(vace_frames, list) else vace_frames
    if vf is None:
        return None
    if vf.ndim == 5:
        out = vf[:, :, 0].permute(0, 2, 3, 1)
        return ((out + 1.0) / 2.0).clamp(0, 1)
    if vf.ndim == 4 and vf.shape[-1] in (1, 3):
        out = vf[:1]
        if out.min() < -0.01:
            out = ((out + 1.0) / 2.0).clamp(0, 1)
        return out
    if vf.ndim == 3 and vf.shape[-1] in (1, 3):
        out = vf.unsqueeze(0)
        if out.min() < -0.01:
            out = ((out + 1.0) / 2.0).clamp(0, 1)
        return out
    return None


def composite(
    *,
    sd_output: torch.Tensor,
    height: int,
    width: int,
    masks_in: Any,
    vace_frames: Any,
    fallback_frame: Optional[torch.Tensor],
    mask_strength: float,
    mask_feather: float,
) -> torch.Tensor:
    """Blend ``sd_output`` with a background according to ``masks_in``.

    Returns ``sd_output`` unchanged when:
      * ``masks_in`` is missing (nothing to blend against), OR
      * no usable background source is available (no ``vace_frames`` and
        no ``fallback_frame``).

    Inputs / outputs are in Scope's BHWC convention (``(1, H, W, C)`` for
    the single-frame case the SD pipeline produces).
    """
    if masks_in is None or mask_strength <= 0:
        return sd_output

    m = coerce_mask_to_bchw(masks_in).to(
        device=sd_output.device, dtype=sd_output.dtype
    )
    if m.shape[-2:] != (height, width):
        m = torch.nn.functional.interpolate(
            m, size=(height, width), mode="bilinear", align_corners=False,
        )
    if mask_feather > 0:
        k = max(1, int(mask_feather) * 2 + 1)
        m = torch.nn.functional.avg_pool2d(m, k, stride=1, padding=k // 2)
    m = (m * mask_strength).clamp(0, 1).permute(0, 2, 3, 1)  # (1, H, W, 1)

    orig = coerce_background_to_bhwc(vace_frames)
    if orig is None and fallback_frame is not None:
        orig = fallback_frame.unsqueeze(0)
    if orig is None:
        return sd_output

    orig = orig.to(device=sd_output.device, dtype=sd_output.dtype)
    if orig.shape[1:3] != (height, width):
        orig = torch.nn.functional.interpolate(
            orig.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)

    return m * sd_output + (1.0 - m) * orig
