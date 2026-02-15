# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Daydream Scope plugin** that integrates StreamDiffusion for real-time Stable Diffusion video generation. It's not a standalone application—it's designed to be installed and discovered by the Daydream Scope framework via Python entry points.

## Installation & Development

### Install Plugin (Development Mode)
```bash
pip install -e .
```
Development mode allows code changes to take effect immediately without reinstalling.

### Verify Plugin Registration
```bash
python -c "import scope_streamdiffusion; print('Plugin loaded successfully')"
```

### Testing in Scope
The plugin is automatically discovered by Scope once installed. Start Scope and look for "StreamDiffusion" in the pipeline selector.

## Architecture

### Plugin Structure
```
src/scope_streamdiffusion/
├── __init__.py      # Plugin registration via @hookimpl
├── schema.py        # Configuration schema (UI fields + validation)
└── pipeline.py      # Pipeline implementation (model + inference)
```

### Entry Point System
The plugin is discovered via `pyproject.toml` entry point:
```toml
[project.entry-points."scope"]
scope_streamdiffusion = "scope_streamdiffusion"
```
Scope automatically loads all registered plugins at startup.

## Critical Architectural Patterns

### 1. Initialization vs Runtime Separation

**This is the most important pattern in the codebase.**

- **`__init__()`**: One-time model loading, GPU setup, component initialization
  - Loads diffusion model from HuggingFace/local path
  - Sets up VAE, UNet, text encoder, scheduler
  - Initializes Compel for prompt weighting
  - NO runtime parameters here

- **`__call__(**kwargs)`**: Per-frame processing with runtime parameters
  - Receives all generation params (prompt, seed, strength, etc.) from kwargs
  - Calls `_prepare_runtime_state()` to set up state from kwargs
  - Processes frame and returns `{"video": tensor}`
  - Parameters can change between frames without reloading model

**Why:** Enables efficient real-time streaming where the model stays loaded but parameters can change dynamically.

### 2. Configuration Schema Pattern

All pipeline parameters are defined in `schema.py` using:
```python
class StreamDiffusionConfig(BasePipelineConfig):
    param_name: type = Field(
        default=value,
        description="...",
        json_schema_extra=ui_field_config(order=N, label="Display Name")
    )
```

- Inherits from `BasePipelineConfig` (provided by Scope)
- Each field gets `ui_field_config()` for UI generation
- `order` determines UI layout order
- Validation happens automatically via Pydantic

### 3. Tensor Format Conversions

**Scope's tensor format:** `(T, H, W, C)` normalized to [0, 1]
**Diffusion model format:** `(B, C, H, W)` for processing

Conversions happen in `__call__()`:
```python
# Input: Scope format → Model format
frame = video[0]  # (H, W, C)
input_tensor = frame.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

# Output: Model format → Scope format
output = result.permute(0, 2, 3, 1).clamp(0, 1)  # (T, H, W, C)
```

### 4. Pipeline Interface Methods

Must implement:
- `get_config_class()`: Returns config schema class
- `prepare()`: Returns `Requirements` (e.g., input size)
- `__call__(**kwargs)`: Main processing method

## Key Implementation Details

### StreamDiffusion Specifics
- Uses LCM (Latent Consistency Models) scheduler for fast inference
- Supports batch denoising for better performance
- Single-step denoising with `t_index_list = [0]` for Turbo models
- Delta parameter controls temporal consistency in streams

### Model Support
- SD 1.5, SDXL, SD Turbo, SDXL Turbo
- Auto-detects SDXL vs SD 1.5 for proper prompt encoding
- Supports LoRA loading via `load_lora()` and `fuse_lora()`

### Prompt Encoding
- Uses Compel library for advanced prompt weighting
- SDXL requires pooled embeddings + add_time_ids
- SD 1.5 uses standard CLIP embeddings

### ControlNet
- Basic support implemented but not exposed in UI
- Set via `self.controlnet` and `self.controlnet_pipeline`
- To expose: add fields to `StreamDiffusionConfig`

## Adding New Parameters

1. **Add to schema** (`schema.py`):
   ```python
   new_param: float = Field(
       default=1.0,
       ge=0.0,
       le=2.0,
       description="Parameter description",
       json_schema_extra=ui_field_config(order=99, label="New Parameter"),
   )
   ```

2. **Use in pipeline** (`pipeline.py`):
   ```python
   def __call__(self, **kwargs) -> dict:
       new_param = kwargs.get("new_param", 1.0)
       # Use new_param in processing...
   ```

3. **(Optional) Add to `_prepare_runtime_state()`** if it affects state initialization

## Important Files Referenced

- `ADAPTATION_NOTES.md`: Detailed explanation of how original StreamDiffusion code was adapted to Scope's architecture
- `README.md`: User-facing documentation with features and usage
- `INSTALL.md`: Quick installation guide

## Dependencies

Core dependencies defined in `pyproject.toml`:
- `torch`: PyTorch (requires CUDA for GPU)
- `diffusers`: Stable Diffusion models and pipelines
- `compel`: Advanced prompt weighting
- `logfire`: Logging (Scope requirement)
- `numpy`, `pillow`: Image processing

## Debugging

Common issues:
- **Plugin not appearing in Scope**: Check entry point registration
- **Model loading fails**: Verify model path and GPU availability
- **Import errors**: Ensure Scope framework is installed
- **Performance issues**: Enable xformers acceleration, reduce inference steps, or use Turbo models

## Development Workflow

1. Make code changes in `src/scope_streamdiffusion/`
2. Changes are immediately available (development mode)
3. Restart Scope to reload plugin
4. Test in Scope UI with various parameters
5. Check Scope logs for errors/warnings
