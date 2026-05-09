# Refactor Plan: pipeline.py Decomposition

## Goal
Reduce `pipeline.py` from ~1900 lines to a thin orchestrator (~400 lines) by extracting cohesive responsibilities into helper classes. Follow the pattern established by `PromptEncoder` (commit `b1b5478`) and the existing `ControlNetHandler`.

## Architectural Pattern (non-negotiable — already established)
- Helper class lives in its own module under `src/scope_streamdiffusion/`.
- Constructor takes `(device, dtype)` and any static config.
- `attach(pipe, sdxl: bool)` lifecycle method called from `_ensure_pipe_loaded` and `_swap_model` after the diffusers pipeline is loaded. Helpers re-bind to the new pipe here.
- Helper owns its caches and exposes runtime state as instance attributes the pipeline reads through (e.g., `self.prompts.prompt_embeds`).
- Helper has explicit `reset_caches()` / `release()` methods called on model swap or teardown.
- **No mixins.** Composition only. The user explicitly rejected mixins.

## Reference Files
- `src/scope_streamdiffusion/prompt_encoder.py` — the template. Read this first.
- `src/scope_streamdiffusion/controlnet.py` — second example of the pattern.
- `src/scope_streamdiffusion/pipeline.py` — the source to extract from.

## Extraction Order (do them in this order, commit between each)

### Extraction 1: `TRTLifecycle` → `src/scope_streamdiffusion/trt_lifecycle.py`
**Methods to move:**
- `_ensure_trt_taesd`
- `_ensure_trt_controlnet`
- `_ensure_trt_unet`
- `_setup_trt`
- `_reset_trt_state`
- `_set_acceleration_mode`
- `_deactivate_trt`
- `_trt_setup_args_from_config`

**Compromise to accept:** these methods currently mutate `self.unet`, `self.controlnet`, `self.vae`, `self._taesd_vae` directly. Don't fight it — give the helper a back-reference to the pipeline (`self.pipe = pipe` set in `attach()`) and have it write through. The win is moving 500 lines of TRT-specific lifecycle code out of the orchestrator, not pretending TRT doesn't touch pipeline state.

**Caches the helper owns:** `_trt_taesd_paths`, `_trt_controlnet_paths`, `_trt_unet_paths`, `_trt_unet_engine`, `_trt_controlnet_engine`, the `acceleration_mode` last-applied value, and the `_trt_cache` adapter handles. The module-scope `_trt_cache._CACHE` stays where it is — it must survive plugin reinit.

**Pipeline-side after extraction:**
```python
self.trt = TRTLifecycle(device=self.device, dtype=self.dtype)
# in _ensure_pipe_loaded / _swap_model:
self.trt.attach(self, self.sdxl)
# in __call__'s pre-inference setup:
self.trt.ensure_engines(config, want_control=...)
```

**Testing checkpoint after this extraction:**
1. Cold-load each model with `acceleration_mode="trt"`: SD-Turbo, SDXL-Turbo, DMD2.
2. Live-swap from SD-Turbo → SDXL-Turbo → DMD2 → SD-Turbo. Confirm no `context=None` crashes (the band-aid `_ensure_activated` in `_trt/engine.py` should still cover this; if it triggers, that's a regression in the swap teardown path).
3. Toggle ControlNet on SD1.5 + SD-Turbo while running.
4. Switch `acceleration_mode` between `none` / `xformers` / `trt` mid-stream.

---

### Extraction 2: `ModelLoader` → `src/scope_streamdiffusion/model_loader.py`
**Methods to move:**
- `_load_model`
- `_load_preset`
- `_release_pipe_state`
- `_swap_model`
- `_install_sdxl_fp16_vae`
- `_set_taesd`
- `load_lora` (currently a stub — leave as-is, the LoRA plan wires it up)
- `fuse_lora` (stub — same)

**State the helper owns:** the `MODEL_PRESETS` dict (move it to this module), last-loaded `model_id`, last-loaded preset signature, the SDXL fp16 VAE replacement state, TAESD-installed flag.

**Compromise:** like TRT, this writes through to `self.pipe`, `self.unet`, `self.vae`, `self.text_encoder`, `self.text_encoder_2`, `self.tokenizer`, `self.tokenizer_2`, `self.scheduler`, `self.sdxl`. Use the back-reference; the goal is consolidation, not purity.

**Order matters in `attach`/swap flow:** ModelLoader runs first, then PromptEncoder.attach, then ControlNetHandler.attach, then TRTLifecycle.attach. Document this in a comment at the top of `pipeline._ensure_pipe_loaded`.

**Testing checkpoint:**
1. Cold load each preset.
2. Swap each direction. Verify no double-loaded models in VRAM (`nvidia-smi` while swapping).
3. Verify SDXL fp16 VAE replacement still happens on SDXL-Turbo and DMD2.
4. Verify TAESD eager and TRT both still work.

---

### Extraction 3: `InferenceCore` → `src/scope_streamdiffusion/inference_core.py`
**Methods to move:**
- `_set_timesteps`
- `_initialize_noise`
- `_setup_seed_transition`
- `_slerp_noise`
- `_advance_seed_transition`
- `_cancel_seed_transition`
- `_encode_image`
- `_decode_image`
- `_add_noise`
- `_scheduler_step_batch`
- `_unet_step`
- `_predict_x0_batch`

**State the helper owns:** `alpha_prod_t_sqrt`, `beta_prod_t_sqrt`, `c_skip`, `c_out`, `sub_timesteps_tensor`, `init_noise`, `x_t_latent_buffer`, the seed-transition fields (`_pending_seed`, `_transition_remaining`, etc.).

**Reads (not writes) from pipeline:** `self.pipe.prompts.prompt_embeds`, `self.pipe.unet`, `self.pipe.controlnet`, `self.pipe.controlnet_input`, `self.pipe.vae`, `self.pipe.scheduler`. Pass these through the back-reference.

**`__call__` after this extraction shrinks to roughly:**
```python
def __call__(self, **kwargs):
    config = self._validate_config(kwargs)
    self._prepare_runtime_state(config)
    self.prompts.encode_for_frame(...)
    if self.controlnet_handler:
        self.controlnet_handler.update(...)
    latent = self.inference.run_step(video, config)
    return {"video": self.inference.to_scope_format(latent)}
```

**Testing checkpoint:** full smoke test — every model × (txt2img / img2img / loopback) × (eager / xformers / TRT) × (with/without negative prompt) × seed transitions.

## Cross-cutting Rules
- **Don't change behavior.** This is a pure move. If you find a bug, note it in a comment — fix it in a separate commit after the refactor lands.
- **Commit per extraction.** Three commits. Each must pass the testing checkpoint before moving to the next.
- **Don't extract `__init__`, `prepare`, `_prepare_runtime_state`, `__call__`, `get_config_class`, or the schema-driven setters.** These are the orchestrator's job.
- **Don't add abstract base classes or interfaces** for the helpers. Three concrete classes is fine.
- **Don't introduce a `BaseHelper` parent class.** They share a pattern, not behavior.
