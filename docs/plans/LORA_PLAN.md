# Plan: LoRA Support

## Context
`schema.py` has `supports_lora = True` already. `pipeline.py` has stub `load_lora` and `fuse_lora` methods that aren't called. Scope has a `download_lora` endpoint already — verify in the parent repo `daydreamlive-scope`.

## Schema Changes (`src/scope_streamdiffusion/schema.py`)

Add a `LoraSpec` model and a `loras` list field on `StreamDiffusionConfig`:
```python
class LoraSpec(BaseModel):
    repo_id: str  # HF repo or local path
    weight_name: Optional[str] = None  # for repos with multiple files
    adapter_name: str  # diffusers adapter name; required for stack/swap
    scale: float = 1.0  # 0..2 typical

class StreamDiffusionConfig(BaseModel):
    ...
    loras: list[LoraSpec] = Field(
        default_factory=list,
        json_schema_extra=ui_field_config(order=..., label="LoRAs"),
    )
```

Order field: place after model selection but before ControlNet config. Reuse Scope's existing LoRA picker UI if one exists in the parent repo's other pipelines.

## Loader Wiring (`ModelLoader` post-refactor, or `pipeline.py` if pre-refactor)

LoRAs attach via `pipe.load_lora_weights(repo_id, weight_name=..., adapter_name=...)`. After loading all requested adapters, call `pipe.set_adapters([names...], adapter_weights=[scales...])`.

**Lifecycle order:**
1. `ModelLoader._load_model` loads the diffusers pipe.
2. SDXL fp16 VAE swap.
3. **LoRA attach.** Iterate `config.loras`, call `pipe.load_lora_weights` per spec.
4. `pipe.set_adapters(...)` with names + scales.
5. **Do NOT call `fuse_lora`.** Keep adapters live so scales/swaps work without reload. Only fuse before TRT compilation (next step).
6. PromptEncoder.attach, ControlNetHandler.attach.
7. TRTLifecycle.attach. **If TRT is enabled, fuse_lora here** before compiling — TRT bakes weights at compile time, so fused-then-compiled is the only correct path (unless using the refit path — see TRT Refit below).

## Change Detection
Track a "LoRA signature" (sorted tuple of `(repo_id, weight_name, adapter_name, scale)`) on the model loader. On `_swap_model` / `_ensure_pipe_loaded`:
- Same model + same LoRA signature → no-op.
- Same model + different LoRA signature, **eager mode** → call `pipe.unload_lora_weights()`, then re-attach. Cheap, no reload needed.
- Same model + different LoRA signature, **TRT mode without refit** → full reload required. Treat this as a model swap. Surface the cost in the UI — recompiling SDXL UNet is 10+ minutes.
- Same model + different LoRA signature, **TRT mode with refit-capable engine** → refit (see below). 1–10s instead of 10+ min.
- Scale-only change with same adapters loaded, **eager mode** → `pipe.set_adapters(...)` with new weights. No reload.
- Scale-only change, **TRT non-refit** → full reload. **TRT refit** → refit.

## Cache Coordination with TRT
The TRT cache key (in `_trt_cache.py` / `trt_engines.py`) must include the LoRA signature. Otherwise two different LoRA stacks will collide on the same cache slot and you'll silently load the wrong engine. Hash the sorted signature into the engine filename.

When using refit, the cache key for the *engine* uses only the base model + refit-capable flag (LoRA signature does NOT affect the engine identity). The fused weights are applied at refit time. The LoRA signature is tracked separately as the "currently refit-applied state" and used only for change detection.

## Scope Integration
The user mentioned Scope has a `download_lora` endpoint already. Find it in the parent repo (`daydreamlive-scope`) and confirm:
- Whether it returns a local path or a repo_id.
- Whether the UI already has a LoRA picker in other pipelines that we can match.
- Whether LoRA management is per-pipeline or global.

Match the existing pattern. Don't invent a new one.

## Testing
1. Eager SD-Turbo + a single style LoRA from CivitAI (download via Scope, attach via config).
2. Live scale change 0.0 → 1.0 → 1.5. Should update without reload.
3. Live LoRA swap (different adapter). Should be fast (unload + load), no model reload.
4. Toggle TRT on with LoRAs attached. Confirm fuse-then-compile path runs and engine is cached with LoRA-aware key.
5. Live LoRA change with TRT on (non-refit) — confirm full reload + recompile triggers and completes.
6. Stack 2 LoRAs simultaneously. Verify `set_adapters` with multiple names works and scales are independent.
7. SDXL + LoRA (eager and TRT).

## Out of Scope (defer)
- Multi-LoRA blending UI beyond stack-with-scales.
- LoRA training or merging.

---

# Addendum: TRT Refit Path for LoRAs

The base plan above says "LoRA change with TRT → full reload" — correct but expensive (10+ min for SDXL). TensorRT's **refit** feature lets you update weights in a built engine without rebuilding it. This is the right answer for live LoRA swaps on TRT.

## What Refit Buys You
- Engine structure (layers, shapes, fusions) stays compiled.
- Only the weight tensors get re-uploaded.
- Typical refit time: **1–10 seconds** for SDXL UNet vs. 10+ minutes for full rebuild.
- Works for scale changes AND adapter swaps, as long as the LoRA targets the same layers.

## Build-Time Requirements
The engine must be compiled with refit enabled. Two flags in the TRT builder:
- `BuilderFlag.REFIT` — required.
- `BuilderFlag.STRIP_PLAN` (TRT 10+) — optional but recommended; strips weights from the engine file so you ship a smaller cache and refit at load. Trade-off: load is no longer instant — must refit before first inference.

**Decision:** use `REFIT` only (not `STRIP_PLAN`). Cached engines stay self-sufficient; refit only runs when LoRAs change. The size penalty for `REFIT`-only is small (~5%) and inference perf is unchanged.

## Implementation Sketch

### Builder changes (`src/scope_streamdiffusion/_trt/builder.py` or wherever the network config lives)
Add `network_flags` / `builder_config.flags |= 1 << int(trt.BuilderFlag.REFIT)` to all UNet builders (`build_unet_engine`, `build_unet_sdxl_engine`, `build_unet_with_control_engine`, and the new SDXL+control variant). VAE/TAESD/ControlNet engines don't need it — LoRAs target UNet only (cross-attention layers).

### Refit at runtime (new method on `TRTLifecycle`)
```python
def refit_lora(self, lora_signature):
    # 1. Load base UNet weights into a temporary diffusers UNet (CPU OK).
    # 2. Apply LoRA stack to that UNet (load_lora_weights + set_adapters + fuse_lora).
    # 3. Use trt.Refitter to push the fused weights into the live engine.
    # 4. Discard the temp UNet.
```

The refitter API:
```python
refitter = trt.Refitter(self._trt_unet_engine.engine, TRT_LOGGER)
for name in refitter.get_all_weights():  # or get_missing()
    weights = fused_unet_state_dict[map_trt_name_to_torch(name)]
    refitter.set_named_weights(name, weights)
assert refitter.refit_cuda_engine()
```

### Name mapping (the hard part)
TRT weight names come from the ONNX export and don't match diffusers' `state_dict` keys 1:1. You need a map. Two approaches:
1. **Build the map at compile time.** During ONNX export, record the `(torch_param_name → onnx_initializer_name)` mapping and persist it next to the engine in the cache. At refit time, load the map and translate.
2. **Reconstruct the map at refit time** by re-running ONNX export on a dummy UNet with the same architecture and reading the resulting initializer names. Slower but simpler.

Recommend approach 1. Save the map as `<engine>.refit_map.json` alongside the engine file. The TRT cache key already covers architecture variants, so the map is valid for the engine.

### Cache key change
Refit-capable engines and refit-incapable engines are different artifacts. Add `refit=True` to the cache key path component so old (non-refit) cached engines aren't reused. Old engines stay valid for non-LoRA streams; new ones get used when LoRAs are configured.

## Updated LoRA Lifecycle (replaces "full reload" branch in the base plan)

| Change | Eager | TRT (refit-capable engine) | TRT (legacy non-refit engine) |
|---|---|---|---|
| Scale only | `set_adapters` | refit | rebuild |
| Adapter swap, same layers | unload + load + `set_adapters` | refit | rebuild |
| Adapter swap, different layers | same | refit (zero out unused) | rebuild |
| Add ControlNet, etc. | rebuild pipeline state | rebuild engine | rebuild |

"Different layers" case: if a new LoRA targets layers the previous one didn't, those original-weight slots need to be restored to the base model's weights during refit. The fused-state-dict approach handles this naturally since the temp UNet is built from base weights + new LoRA stack.

## When to Skip Refit
- First time TRT is enabled with LoRAs configured → fuse first, then build (current plan). Refit only helps on subsequent changes.
- Engine compiled before this feature lands → fall back to rebuild. Detect via the cache-key version bump.
- Refitter reports missing weights → log and rebuild. Don't run a partially-refit engine.

## Testing (Refit-specific)
1. Cold start with one LoRA + TRT. Confirm engine builds with `REFIT` flag (check `engine.refittable`).
2. Live scale change 0.0 → 1.5. Should complete in <10s, no recompile log.
3. Live adapter swap (different LoRA, same target layers). Same speed.
4. Live adapter swap to a LoRA that targets *additional* layers. Confirm refit covers all weights and output is correct.
5. Stress test: 20 rapid scale/adapter changes. Memory should stay stable (the temp UNet must actually free).
6. SDXL refit specifically — name-map size is larger; verify no missing weights.

## Risk
- TRT refit name mapping is fiddly. Budget time for debugging the ONNX-name ↔ torch-name mapping.
- Some TRT optimizations bake constants. If a LoRA's effective rank changes the optimal kernel choice, refit produces correct but suboptimal output. Acceptable trade-off.
- `STRIP_PLAN` is tempting but adds first-inference latency. Skip it.

This makes live LoRA swaps on TRT actually viable instead of "technically supported but never used."
