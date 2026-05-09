# Plans

Hand-off plans for the next round of work on `sd-multi-model`. Each plan is self-contained and intended to be executed by another agent without needing the originating conversation.

- [REFACTOR_PLAN.md](REFACTOR_PLAN.md) — decompose `pipeline.py` into helper classes (`TRTLifecycle`, `ModelLoader`, `InferenceCore`) following the `PromptEncoder` / `ControlNetHandler` pattern.
- [SDXL_CONTROLNET_PLAN.md](SDXL_CONTROLNET_PLAN.md) — wire SDXL ControlNet through the eager and TRT paths (currently raises `NotImplementedError` on TRT for SDXL).
- [LORA_PLAN.md](LORA_PLAN.md) — schema, loader wiring, change detection, and the TRT refit path for live LoRA swaps.

## Recommended order
1. Refactor (lands first — the LoRA plan assumes the `ModelLoader` and `TRTLifecycle` helpers exist).
2. SDXL ControlNet (independent of LoRA).
3. LoRA (depends on refactor; benefits from but does not require ControlNet work).

## Architectural pattern (read first)
All three plans assume the helper-class composition pattern. The canonical examples in the repo:
- `src/scope_streamdiffusion/prompt_encoder.py`
- `src/scope_streamdiffusion/controlnet.py`

Helpers take `(device, dtype)` at construction, gain a pipe back-reference via `attach(pipe, sdxl)`, and expose runtime state as instance attributes.
