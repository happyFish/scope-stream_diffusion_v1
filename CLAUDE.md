# scope-streamdiffusion Plugin — Claude Code Guide

Real-time Stable Diffusion pipeline for Daydream Scope using StreamDiffusion. Supports SD 1.5, SDXL, Turbo models with LCM scheduling, ControlNet, TensorRT acceleration, and multi-model orchestration.

## Design

This is a Scope `Pipeline` subclass that wraps diffusion inference. The plugin is **entry-point discovered** via `pyproject.toml` and loads into Scope's pipeline selector automatically.

### Core Principles

- **Init/Runtime separation.** `__init__()` loads models once; `__call__(**kwargs)` handles per-frame params (prompt, seed, guidance scale, strength). Parameters can change frame-to-frame without reloading.
- **Tensor format aware.** Scope uses `(T, H, W, C)` in [0, 1]; diffusion expects `(B, C, H, W)`. Conversions happen in `__call__()`.
- **Schema-driven config.** All parameters defined in `schema.py` using Pydantic. UI fields auto-generated via `ui_field_config(order=N, label="...")`.
- **Lazy model loading.** Models load on first init; subsequent calls reuse weights. Model changes trigger full reinitialization.

## Project Structure

```
.
├── CLAUDE.md                         # This file
├── README.md                         # User-facing features and usage
├── ADAPTATION_NOTES.md               # How StreamDiffusion was adapted to Scope
├── INSTALL.md                        # Quick install guide
├── pyproject.toml                    # Package config, entry point, deps
│
├── src/scope_streamdiffusion/
│   ├── __init__.py                   # Plugin registration via hookimpl
│   ├── schema.py                     # StreamDiffusionConfig (Pydantic + UI)
│   ├── pipeline.py                   # StreamDiffusionPipeline (main logic)
│   ├── controlnet.py                 # ControlNet handler for multi-ControlNet support
│   ├── trt_engines.py                # TensorRT engine discovery/caching
│   ├── _trt_cache.py                 # TensorRT compile cache management
│   │
│   └── _trt/                         # TensorRT backend utilities
│       ├── __init__.py
│       ├── models.py                 # TRT model configs
│       ├── engine.py                 # Engine compilation and inference
│       ├── builder.py                # ONNX → TRT conversion
│       └── utilities.py              # Device/precision helpers
│
└── (no tests directory yet)
```

## Key Files

**Schema & Configuration:**
- `schema.py` — `StreamDiffusionConfig`: Pydantic model with 50+ fields defining model, scheduler, sampler, guidance, seed, ControlNet setup, TensorRT flags. Fields use `ui_field_config()` for Scope UI auto-generation.

**Pipeline Implementation:**
- `pipeline.py` — `StreamDiffusionPipeline`: implements `Pipeline` interface. Methods:
  - `get_config_class()`: returns `StreamDiffusionConfig`
  - `prepare(**kwargs) → Requirements`: returns resource hints
  - `__call__(**kwargs) → dict`: main inference loop; returns `{"video": tensor}`

**ControlNet:**
- `controlnet.py` — `ControlNetHandler`: manages multi-ControlNet attachment, caching, and inference integration. Supports Canny, pose, depth, etc.

**TensorRT:**
- `trt_engines.py` — discovers cached engines, auto-selects by device/precision
- `_trt/engine.py` — compiles ONNX models to TensorRT `.engine` format with dynamic shapes
- `_trt_cache.py` — caches compiled engines locally for rapid reuse

**Entry Point:**
- `__init__.py` — `@hookimpl` function that Scope's plugin loader calls at discovery

## Architecture

### Inference Flow

```
Input frame (from Scope)
  ↓
[Tensor format conversion] (T,H,W,C) → (B,C,H,W)
  ↓
[Load/prepare model] (on first call; cached after)
  ↓
[Encode prompt] (Compel for weighting; cache embeddings)
  ↓
[VAE encode] frame → latent
  ↓
[ControlNet encode] (if enabled; pre-compute for all conditions)
  ↓
[Denoising loop] 
  for each step in scheduler:
    - Add noise if img2img
    - Denoise with UNet
    - Apply ControlNet
    - Apply guidance
  ↓
[VAE decode] latent → image
  ↓
[Tensor format conversion] (B,C,H,W) → (T,H,W,C)
  ↓
Output (PIL.Image or tensor dict)
```

### Model Loading Lifecycle

1. **First `__init__`:** Load diffusion model from HuggingFace or local path. Setup VAE, UNet, text encoder, scheduler. Warm up GPU.
2. **Subsequent `__init__` calls:** Reuse loaded weights (unless model_id changed).
3. **Model change:** Trigger full reload (detected via signature comparison).
4. **ControlNet attach:** Load and fuse ControlNet weights; cache encoders.

### Parameter Handling

**Initialization-time (requires model reload):**
- `model_id`: changes which model to load
- `torch_dtype`: precision (float16 vs float32)
- `acceleration`: xformers vs none

**Runtime (can change per-frame):**
- `prompt`: text input (re-encoded each frame or cached if unchanged)
- `seed`: random seed
- `guidance_scale`: classifier-free guidance strength
- `strength`: how much to denoise (img2img)
- `num_inference_steps`: denoising steps
- `scheduler`: LCM, DDPM, etc. (some require reinit)

### TensorRT Compilation

- Disabled by default (requires additional setup).
- When enabled: UNet compiled to device-specific `.engine` file.
- Compilation happens on first inference (slow; ~1-5 min depending on model).
- Cached engines reused on subsequent runs (instant load).
- Cache dir: `~/.cache/scope-streamdiffusion/trt/`

### ControlNet Support

- Multi-ControlNet: attach multiple conditions (e.g., Canny + pose).
- Conditions pre-computed once per prompt.
- Inference: scales applied per denoising step.
- Encoder caching to avoid re-encoding images.

## Development Workflow

### Before Starting

1. **Read `ADAPTATION_NOTES.md`** — explains how the original StreamDiffusion code was adapted to Scope's architecture.
2. **Understand init/runtime separation** — this is the foundation of how parameters flow.
3. **Check `schema.py`** for existing fields — don't add duplicate params.

### Adding a New Parameter

1. **Add to schema** (`schema.py`):
   ```python
   new_param: float = Field(
       default=1.0,
       ge=0.0,
       le=10.0,
       description="What this does",
       json_schema_extra=ui_field_config(order=50, label="New Param"),
   )
   ```

2. **Use in pipeline** (`pipeline.py`):
   - If runtime-safe (doesn't require model reload): read from `kwargs` in `__call__()`
   - If initialization-time (e.g., model architecture): pass to `__init__()` and track via signature

3. **Test in Scope:**
   - Run Scope (`SCOPE_REPL` or `scope serve`)
   - Select StreamDiffusion pipeline
   - Parameter should appear in UI with label and order

### Adding ControlNet Support

1. **Update schema** to expose ControlNet config:
   ```python
   controlnet_id: Optional[str] = Field(
       default=None,
       description="ControlNet model ID",
       json_schema_extra=ui_field_config(order=40, label="ControlNet"),
   )
   controlnet_conditioning: Optional[str] = Field(
       default=None,
       description="Encoded ControlNet condition",
   )
   ```

2. **Update pipeline** to attach ControlNet:
   ```python
   if kwargs.get("controlnet_id"):
       controlnet = ControlNetHandler(kwargs["controlnet_id"], device=self.device)
       # Attach to diffusion pipeline
   ```

3. **Reference `controlnet.py`** for handler patterns.

### TensorRT Integration

Only attempt if you have CUDA 12+ and understand TensorRT compilation:

1. Update `_trt/builder.py` if you need new precision/shape configs.
2. Call `_trt_cache.get_engine()` to auto-compile and cache.
3. Swap UNet for TRT engine in inference loop.
4. See `trt_engines.py` for caching logic.

## Testing

No test suite exists yet. Manual testing approach:

```bash
# 1. Install in dev mode
pip install -e .

# 2. Start Scope
SCOPE_REPL  # or: scope serve

# 3. In Scope, select StreamDiffusion pipeline

# 4. Set parameters and verify:
#    - Prompt changes take effect immediately
#    - Model changes trigger reload (check logs)
#    - Output looks reasonable
#    - Performance is acceptable
```

### Debugging

```bash
# Check plugin is discovered:
python -c "import scope_streamdiffusion; print('OK')"

# Check config loads:
python -c "from scope_streamdiffusion import StreamDiffusionConfig; print(StreamDiffusionConfig.__fields__.keys())"

# Test pipeline init:
from scope_streamdiffusion.pipeline import StreamDiffusionPipeline
p = StreamDiffusionPipeline()
print("Pipeline initialized")
```

## Important Constraints

- **Model reloads are expensive.** Changing `model_id`, `torch_dtype`, or `acceleration` causes full reload (10-30s).
- **VRAM is limited.** Default to float16 and xformers acceleration. SDXL needs 8GB+ VRAM.
- **Scheduler matters.** LCM is fast (1-4 steps); DDPM is slow but more flexible (20-50 steps). Some model + scheduler combos don't work well.
- **TensorRT engines are device-specific.** Moving to a different GPU requires recompilation.
- **Prompt encoding is cached.** If prompt doesn't change, embeddings are reused (fast). If it does, encoding happens every frame (slower).

## Dependencies & xformers

**Core deps** (in `pyproject.toml`):
- `torch` — deep learning framework
- `diffusers` — HuggingFace diffusion models
- `logfire` — Scope logging integration
- `numpy`, `pillow` — image processing

**Optional (xformers acceleration):**
xformers is NOT in dependencies because it ships with strict (often wrong) torch pins that break Scope's GPU stack.

Install manually after setup, choosing the version for your torch:
```bash
torch 2.9.x  →  uv pip install --no-deps xformers==0.0.33.post2
torch 2.10.x →  uv pip install --no-deps xformers==0.0.34
```

Use `--no-deps` to skip xformers' bogus torch pin.

## Scope Integration Points

**Entry point discovery:**
```toml
[project.entry-points."scope"]
scope_streamdiffusion = "scope_streamdiffusion"
```
Scope calls `hookimpl()` function in `__init__.py` to register the pipeline.

**Config schema:**
Schema fields with `ui_field_config()` are discovered by Scope and rendered in the pipeline UI. Changes to schema are reflected on next Scope restart.

**Requirements:**
`prepare()` returns `Requirements` (e.g., minimum VRAM, input resolution). Scope uses this for validation.

**Tensor I/O:**
`__call__()` receives `video` tensor from Scope in `(T, H, W, C)` format; must return same format.

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Plugin doesn't appear in Scope | Entry point not registered | Run `pip install -e .` again; restart Scope |
| Model loading fails | HF auth needed or model not found | Check HF cache; verify internet; login to HuggingFace if needed |
| OOM errors | Model too big for GPU | Use SDXL Turbo instead of base SDXL; reduce batch size; enable xformers |
| Slow inference | No GPU acceleration | Install xformers; check `torch.cuda.is_available()` returns True |
| ControlNet not working | Handler not attached properly | Review `controlnet.py` logic; check config passes condition tensor |
| TensorRT compile fails | CUDA version mismatch | Ensure CUDA 12+; check triton compatibility |

## Code Style & Conventions

- Type hints required on all public functions.
- Docstrings on classes and complex methods.
- Config validation via Pydantic (no manual validation).
- Logging via `logfire` (not print).
- No magic constants — all tunable params go in schema.

## References

- **Scope plugin tutorials:** https://docs.daydream.live/scope/tutorials/build-video-effects-plugin
- **Diffusers docs:** https://huggingface.co/docs/diffusers
- **StreamDiffusion:** https://github.com/cumulo-autumn/StreamDiffusion
- **Scope Pydantic patterns:** Check other Scope pipelines in `daydreamlive-scope` repo
