"""Prompt encoding, blending, transitions, and negative subtraction.

Owns everything text-encoder-related so the main pipeline doesn't have to.
The pipeline holds an instance as ``self.prompts`` and calls
``encode_for_frame()`` once per ``__call__``, then optionally
``apply_negative_subtraction()``. Inference reads the produced embeds via
``self.prompts.prompt_embeds`` / ``add_text_embeds`` / ``add_time_ids``.

Lifecycle: ``attach(pipe, sdxl)`` after a model load (or model swap) wires
us to the live pipeline and resets all text-encoder-dependent caches.
``reset_caches()`` is the lighter version called during teardown without
re-attaching.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import torch

from scope.core.pipelines.blending import EmbeddingBlender, parse_transition_config


def normalize_prompts(prompts: str | list[str] | list[dict]) -> list[dict]:
    """Coerce a prompts payload into ``list[{"text": str, "weight": float}]``.

    Module-level so the pipeline can call it on raw kwargs before any
    PromptEncoder method that expects normalized input.
    """
    if isinstance(prompts, str):
        return [{"text": prompts, "weight": 1.0}]
    if isinstance(prompts, list):
        if len(prompts) == 0:
            return [{"text": "", "weight": 1.0}]
        if isinstance(prompts[0], str):
            return [{"text": text, "weight": 1.0} for text in prompts]
        return prompts
    return [{"text": str(prompts), "weight": 1.0}]


class PromptEncoder:
    """Per-frame prompt encoding with transitions, caching, and negative
    subtraction.

    Attach to a loaded ``DiffusionPipeline`` via ``attach(pipe, sdxl)``;
    re-attach on every model swap because the text encoder identity (and
    hidden dim, for SDXL) changes between SD 1.5 and SDXL.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

        # Live pipe references — set via ``attach()``. Until then the encoder
        # is inert; calling encode_for_frame would raise.
        self.pipe: Any = None
        self.sdxl: bool = False

        self.embedding_blender = EmbeddingBlender(device=device, dtype=dtype)

        # Current-frame outputs the inference path reads. Inference accesses
        # ``self.prompts.prompt_embeds`` / ``add_text_embeds`` / ``add_time_ids``.
        self.prompt_embeds: Optional[torch.Tensor] = None
        self.add_text_embeds: Optional[torch.Tensor] = None
        self.add_time_ids: Optional[torch.Tensor] = None

        # Per-text-encoder caches. All invalidate on attach() and reset_caches().
        self._cached_base_embed: Optional[torch.Tensor] = None
        self._previous_prompt_embeddings: Optional[torch.Tensor] = None
        self._prompts_key: Optional[tuple] = None

        # Negative-prompt cache.
        self._cached_negative_text: Optional[str] = None
        self._cached_negative_embed: Optional[torch.Tensor] = None
        self._cached_negative_pooled: Optional[torch.Tensor] = None

        # Pooled (SDXL) transition state — main embedding queue lives in
        # ``embedding_blender``; pooled is interpolated linearly in lockstep.
        self._pooled_source: Optional[torch.Tensor] = None
        self._pooled_target: Optional[torch.Tensor] = None
        self._transition_total_steps: int = 0

        # Transition-id guard so repeated identical explicit transition dicts
        # don't restart the transition every frame.
        self._last_transition_id: Optional[str] = None

        # One-shot warning when slerp is requested for temporal interpolation
        # (fp16 NaN bug in upstream blender). We fall back to linear silently
        # after the first warn.
        self._slerp_fallback_warned: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, pipe: Any, sdxl: bool) -> None:
        """Wire to a loaded pipeline and reset all caches.

        Call from ``_ensure_pipe_loaded`` and ``_swap_model`` after the new
        pipe is available; SD 1.5 and SDXL have different text-encoder
        hidden dims, so cached embeds from the prior model would mismatch.
        """
        self.pipe = pipe
        self.sdxl = sdxl
        self.reset_caches()

    def reset_caches(self) -> None:
        """Drop every cached tensor and cancel any in-flight transition."""
        self._cached_base_embed = None
        self._previous_prompt_embeddings = None
        self._prompts_key = None
        self._cached_negative_text = None
        self._cached_negative_embed = None
        self._cached_negative_pooled = None
        self._pooled_source = None
        self._pooled_target = None
        self._transition_total_steps = 0
        self._last_transition_id = None
        try:
            self.embedding_blender.cancel_transition()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-frame encode
    # ------------------------------------------------------------------

    def encode_for_frame(
        self,
        prompts: list[dict],
        interpolation_method: str,
        width: int,
        height: int,
        batch_size: int,
        transition: Optional[dict] = None,
        transition_steps: int = 0,
    ) -> None:
        """Update ``self.prompt_embeds`` (and SDXL extras) for this frame.

        Handles: prompts-changed re-encoding, explicit transition-dict
        starts, auto-transitions on prompt change, blender advance for
        in-flight transitions, and pooled (SDXL) lockstep lerp.
        """
        # When an explicit transition dict is present, its target_prompts is
        # the authoritative destination; keying against the source prompts
        # would make prompts_changed flap during/after the transition and
        # snap steady state back to the source.
        key_prompts = prompts
        if transition is not None:
            target_raw = transition.get("target_prompts")
            if target_raw:
                key_prompts = normalize_prompts(target_raw)
        new_prompts_key = self._make_prompts_key(
            key_prompts, interpolation_method, width, height
        )
        prompts_changed = new_prompts_key != self._prompts_key

        transition_id = self._hash_transition(transition) if transition else None
        new_explicit_transition = (
            transition_id is not None and transition_id != self._last_transition_id
        )

        started_transition = False

        # Cancel any in-flight transition if a new target has arrived so we
        # redirect from the current interpolated position rather than
        # snapping after the old transition drains.
        if self.embedding_blender.is_transitioning() and (
            new_explicit_transition
            or (transition is None and transition_steps > 0 and prompts_changed)
        ):
            self.embedding_blender.cancel_transition()
            self._finish_pooled_transition()

        if new_explicit_transition and not self.embedding_blender.is_transitioning():
            transition_config = parse_transition_config(transition)
            target_prompts_raw = transition.get("target_prompts", [])
            if transition_config.num_steps > 0 and target_prompts_raw:
                target_prompts = normalize_prompts(target_prompts_raw)
                started_transition = self._begin_transition(
                    target_prompts=target_prompts,
                    interpolation_method=interpolation_method,
                    num_steps=transition_config.num_steps,
                    temporal_method=transition_config.temporal_interpolation_method,
                    width=width,
                    height=height,
                )
            self._last_transition_id = transition_id
        elif (
            transition is None
            and transition_steps > 0
            and prompts_changed
            and self._previous_prompt_embeddings is not None
            and not self.embedding_blender.is_transitioning()
        ):
            started_transition = self._begin_transition(
                target_prompts=prompts,
                interpolation_method=interpolation_method,
                num_steps=transition_steps,
                temporal_method=interpolation_method,
                width=width,
                height=height,
            )

        # --- Produce prompt_embeds for this frame ---
        if self.embedding_blender.is_transitioning():
            next_embedding = self.embedding_blender.get_next_embedding()
            if next_embedding is not None:
                self.prompt_embeds = next_embedding.repeat(batch_size, 1, 1)
                self._advance_pooled_transition()
            else:
                self.prompt_embeds = self._cached_base_embed.repeat(batch_size, 1, 1)
                self._finish_pooled_transition()
        else:
            # Steady state — re-encode if prompts changed and we didn't start
            # a transition for it (hard cut path, e.g. transition_steps == 0).
            if prompts_changed and not started_transition:
                raw_embeds, _ = self._encode_prompts_array(
                    key_prompts,
                    interpolation_method,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                )
                self._cached_base_embed = raw_embeds[0:1]
                self._prompts_key = new_prompts_key
            # Drop the transition-id guard once the explicit dict is gone so
            # a later identical dict is treated as a fresh request.
            if transition is None:
                self._last_transition_id = None
            self._finish_pooled_transition()
            self.prompt_embeds = self._cached_base_embed.repeat(batch_size, 1, 1)

        # Cache embedding as source for the next transition.
        self._previous_prompt_embeddings = self.prompt_embeds[0:1].detach()

    # ------------------------------------------------------------------
    # Negative-prompt subtraction (single-pass models)
    # ------------------------------------------------------------------

    def apply_negative_subtraction(
        self, negative_prompt: str, negative_prompt_scale: float
    ) -> None:
        """Norm-preserving negative subtraction in embedding space.

        Single-pass models (Turbo, DMD2) can't use standard CFG without
        doubling UNet cost. Embedding subtraction is the cheap alternative,
        but raw ``pos - scale * neg`` blows up the L2 norm of each token,
        knocking the conditioning out of the training distribution and
        the UNet predicts pure noise.

        We do the subtraction directionally and then renormalize each
        token's embedding back to the original L2 norm. Same treatment
        applied to SDXL's pooled ``add_text_embeds``. ``add_time_ids``
        are positional / size-derived, not text-derived, so they stay put.

        Encoded negative is cached on text; empty text or scale 0 is a
        no-op. Cache invalidates on model swap (text-encoder dim changes).
        """
        if negative_prompt_scale <= 0 or not negative_prompt:
            return
        if self.prompt_embeds is None:
            return
        if (
            self._cached_negative_text != negative_prompt
            or self._cached_negative_embed is None
        ):
            neg_embed, neg_pooled = self._encode_single_prompt(negative_prompt)
            self._cached_negative_text = negative_prompt
            self._cached_negative_embed = neg_embed.detach()
            self._cached_negative_pooled = (
                neg_pooled.detach() if neg_pooled is not None else None
            )

        self.prompt_embeds = _norm_preserving_subtract(
            self.prompt_embeds, self._cached_negative_embed, negative_prompt_scale
        )
        if self.sdxl and self._cached_negative_pooled is not None and self.add_text_embeds is not None:
            self.add_text_embeds = _norm_preserving_subtract(
                self.add_text_embeds,
                self._cached_negative_pooled,
                negative_prompt_scale,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_prompts_key(
        self,
        prompts: list[dict],
        interpolation_method: str,
        width: int,
        height: int,
    ) -> tuple:
        return (
            tuple((p.get("text", ""), p.get("weight", 1.0)) for p in prompts),
            interpolation_method,
            (width, height) if self.sdxl else (),
        )

    @staticmethod
    def _hash_transition(transition: dict) -> str:
        payload = {
            "num_steps": int(transition.get("num_steps", 0) or 0),
            "method": transition.get("temporal_interpolation_method", "linear"),
            "target": [
                {
                    "text": p.get("text", "") if isinstance(p, dict) else str(p),
                    "weight": float(p.get("weight", 1.0)) if isinstance(p, dict) else 1.0,
                }
                for p in (transition.get("target_prompts") or [])
            ],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()

    def _begin_transition(
        self,
        target_prompts: list[dict],
        interpolation_method: str,
        num_steps: int,
        temporal_method: str,
        width: int,
        height: int,
    ) -> bool:
        source_embedding = self._previous_prompt_embeddings
        if source_embedding is None:
            return False

        target_embed, target_pooled = self._encode_prompts_array(
            target_prompts,
            interpolation_method,
            apply_sdxl_conditioning=False,
            width=width,
            height=height,
            batch_size=1,
        )
        target_embed_single = target_embed[0:1]

        # Eagerly move steady-state cache to the target so once the queue
        # drains we land on the target prompts with no bounce-back.
        self._cached_base_embed = target_embed_single
        self._prompts_key = self._make_prompts_key(
            target_prompts, interpolation_method, width, height
        )

        # Slerp NaNs at fp16 in the upstream blender (acos at the [-1, 1]
        # boundary) — fall back to linear with a one-shot warn.
        if temporal_method == "slerp":
            if not self._slerp_fallback_warned:
                print(
                    "[StreamDiffusion] slerp temporal interpolation is not "
                    "supported (fp16 NaN in upstream blender); falling back "
                    "to linear."
                )
                self._slerp_fallback_warned = True
            temporal_method = "linear"

        self.embedding_blender.start_transition(
            source_embedding=source_embedding,
            target_embedding=target_embed_single,
            num_steps=num_steps,
            temporal_interpolation_method=temporal_method,
        )

        if self.sdxl and target_pooled is not None:
            self._pooled_source = (
                self.add_text_embeds.detach().clone()
                if self.add_text_embeds is not None
                else target_pooled.clone()
            )
            self._pooled_target = target_pooled.clone()
            self._transition_total_steps = max(1, num_steps)
        else:
            self._pooled_source = None
            self._pooled_target = None
            self._transition_total_steps = 0

        # start_transition short-circuits when source ≈ target
        # (MIN_EMBEDDING_DIFF_THRESHOLD); report accurately so the caller
        # falls to steady state instead of assuming a transition is live.
        if not self.embedding_blender.is_transitioning():
            self._finish_pooled_transition()
            return False
        return True

    def _advance_pooled_transition(self) -> None:
        """Linearly interpolate ``add_text_embeds`` toward the target pooled."""
        if not self.sdxl or self._pooled_target is None:
            return
        if self._transition_total_steps <= 0:
            return
        remaining = len(self.embedding_blender._transition_queue)
        done_steps = self._transition_total_steps - remaining
        t = min(1.0, max(0.0, done_steps / self._transition_total_steps))
        source = (
            self._pooled_source
            if self._pooled_source is not None
            else self._pooled_target
        )
        self.add_text_embeds = torch.lerp(source, self._pooled_target, t).to(
            dtype=self.dtype, device=self.device
        )

    def _finish_pooled_transition(self) -> None:
        """Snap pooled to the target and clear transition state."""
        if self.sdxl and self._pooled_target is not None:
            self.add_text_embeds = self._pooled_target.to(
                dtype=self.dtype, device=self.device
            )
        self._pooled_source = None
        self._pooled_target = None
        self._transition_total_steps = 0

    def _encode_single_prompt(
        self, prompt_text: str
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt_text,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )
        prompt_embeds = encoder_output[0]
        pooled_embeds = encoder_output[2] if self.sdxl else None
        return prompt_embeds, pooled_embeds

    def _encode_prompts_array(
        self,
        prompt_items: list[dict],
        interpolation_method: str,
        *,
        width: int,
        height: int,
        batch_size: int,
        apply_sdxl_conditioning: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not prompt_items:
            prompt_items = [{"text": "", "weight": 1.0}]

        texts = [item.get("text", "") for item in prompt_items]
        weights = [item.get("weight", 1.0) for item in prompt_items]

        all_prompt_embeds = []
        all_pooled_embeds = [] if self.sdxl else None

        for text in texts:
            prompt_embeds, pooled_embeds = self._encode_single_prompt(text)
            all_prompt_embeds.append(prompt_embeds)
            if self.sdxl and pooled_embeds is not None:
                all_pooled_embeds.append(pooled_embeds)

        blended_prompt_embeds = self.embedding_blender.blend(
            all_prompt_embeds,
            weights,
            interpolation_method,
            cache_result=True,
        )

        blended_pooled_embeds = None
        if self.sdxl and all_pooled_embeds:
            blended_pooled_embeds = self.embedding_blender.blend(
                all_pooled_embeds,
                weights,
                interpolation_method,
                cache_result=False,
            )

        # SDXL aug-conditioning: write add_text_embeds and add_time_ids for
        # the steady-state encode. Skipped for transition-target encodes so
        # the in-flight pooled / time_ids aren't overwritten mid-morph.
        if apply_sdxl_conditioning and self.sdxl and blended_pooled_embeds is not None:
            self.add_text_embeds = blended_pooled_embeds
            self.add_time_ids = self._compute_add_time_ids(
                width=width, height=height, dtype=self.dtype
            )

        return blended_prompt_embeds.repeat(batch_size, 1, 1), blended_pooled_embeds

    def _compute_add_time_ids(
        self, width: int, height: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build SDXL aug-conditioning time_ids from the current dims.

        Reads ``self.pipe.unet.config.addition_time_embed_dim`` and
        ``self.pipe.unet.add_embedding.linear_1.in_features`` to validate
        the vector length matches what the UNet expects. Raises if not.
        """
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)
        text_encoder_projection_dim = int(self.add_text_embeds.shape[-1])

        add_time_ids_list = list(original_size + crops_coords_top_left + target_size)
        unet = self.pipe.unet
        passed_add_embed_dim = (
            unet.config.addition_time_embed_dim * len(add_time_ids_list)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length "
                f"{expected_add_embed_dim}, but a vector of "
                f"{passed_add_embed_dim} was created."
            )
        return torch.tensor([add_time_ids_list], dtype=dtype)


def _norm_preserving_subtract(
    positive: torch.Tensor, negative: torch.Tensor, scale: float
) -> torch.Tensor:
    """Subtract ``scale * negative`` then rescale to match positive's
    original per-row L2 norm. Direction shifts, magnitude is preserved,
    UNet stays inside training distribution.
    """
    neg = negative.to(device=positive.device, dtype=positive.dtype)
    if neg.shape[0] != positive.shape[0]:
        neg = neg[:1].expand_as(positive)
    orig_norm = positive.norm(dim=-1, keepdim=True)
    shifted = positive - scale * neg
    new_norm = shifted.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return shifted * (orig_norm / new_norm)
