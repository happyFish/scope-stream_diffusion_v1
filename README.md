# StreamDiffusion Plugin for Daydream Scope

This plugin integrates StreamDiffusion into Daydream Scope for real-time Stable Diffusion video generation.

## Features

- Real-time Stable Diffusion inference
- Support for SD 1.5, SDXL, and Turbo models
- LCM LoRA acceleration
- Configurable denoising parameters
- Similar image filtering
- GPU acceleration with xformers/TensorRT

## Installation

### Development Mode

From the plugin directory:

```bash
cd scope-streamdiffusion
pip install -e .
```

### Production Mode

```bash
cd scope-streamdiffusion
pip install .
```

## Usage

Once installed, the plugin will be automatically discovered by Daydream Scope through the entry point system.

### In Scope

1. Start Daydream Scope
2. Select "StreamDiffusion" from the available pipelines
3. Configure parameters in the UI:
   - **Model**: Choose your Stable Diffusion model
   - **Prompt**: Enter generation prompt
   - **Strength**: Control denoising strength (0.0-1.2)
   - **Guidance Scale**: CFG scale
   - **Inference Steps**: Number of denoising steps
   - **Width/Height**: Output resolution

### Parameters

#### Core Generation
- **Prompt**: Text description of desired output
- **Negative Prompt**: What to avoid in generation
- **Seed**: Random seed for reproducibility
- **Guidance Scale**: Classifier-free guidance strength (0 = none)
- **Inference Steps**: More steps = better quality but slower
- **Strength**: How much to transform the input (1.0 = full transformation)

#### StreamDiffusion Specific
- **Delta**: StreamDiffusion delta parameter
- **Batch Denoising**: Use batch processing for better performance
- **Add Noise**: Add noise between denoising steps
- **Use LCM LoRA**: Enable LCM LoRA for faster inference

#### Optimization
- **Similar Image Filter**: Skip processing similar frames
- **Filter Threshold**: Similarity threshold (0.9-1.0)
- **Acceleration**: Choose hardware acceleration (xformers/tensorrt)

## Architecture

The plugin follows Scope's plugin architecture:

```
scope-streamdiffusion/
├── pyproject.toml                    # Plugin metadata & entry point
└── src/scope_streamdiffusion/
    ├── __init__.py                   # Hook registration
    ├── schema.py                     # Configuration schema
    └── pipeline.py                   # Pipeline implementation
```

### Key Components

1. **StreamDiffusionConfig** (`schema.py`):
   - Pydantic model defining all configurable parameters
   - UI field configurations for Scope's web interface
   - Inherits from `BasePipelineConfig`

2. **StreamDiffusionPipeline** (`pipeline.py`):
   - Main pipeline class inheriting from `Pipeline`
   - Implements `__call__()` for frame processing
   - Handles model loading and inference

3. **Hook Registration** (`__init__.py`):
   - Registers the pipeline with Scope via `@hookimpl`

## Design Decisions

### Initialization vs Runtime

Following Scope's architecture:
- **`__init__()`**: Model loading, GPU setup (one-time)
- **`__call__()`**: Frame processing with runtime params (per-frame)

This separation allows:
- Efficient model reuse across frames
- Dynamic parameter changes without reloading
- Better performance in streaming scenarios

### Tensor Format

The plugin converts between formats:
- **Scope format**: `(T, H, W, C)` normalized to [0, 1]
- **Internal format**: `(B, C, H, W)` for diffusion models

### State Management

Runtime state is prepared fresh each call from `kwargs`:
- Prompt embeddings
- Timestep schedules
- Noise tensors
- Latent buffers

This ensures thread-safety and allows parameter changes between frames.

## Requirements

- Python >= 3.12
- PyTorch with CUDA support
- diffusers
- transformers
- compel (for prompt weighting)
- xformers (optional, for acceleration)

## Models

The plugin supports:
- Stable Diffusion 1.5
- Stable Diffusion XL
- SD Turbo / SDXL Turbo
- ByteDance SDXL Lightning
- Custom models from HuggingFace or local paths

## Troubleshooting

### Model Loading Issues
- Ensure models are downloaded to the correct path
- Check CUDA/GPU availability
- Verify model ID is correct

### Performance Issues
- Enable xformers acceleration
- Use Turbo/Lightning models for speed
- Reduce inference steps
- Lower resolution

### Memory Issues
- Reduce batch size
- Use fp16 (default)
- Close other GPU applications

## Development

### Adding New Features

1. **Add parameter to schema** (`schema.py`):
   ```python
   my_param: float = Field(
       default=1.0,
       description="...",
       json_schema_extra=ui_field_config(order=N, label="..."),
   )
   ```

2. **Use in pipeline** (`pipeline.py`):
   ```python
   def __call__(self, **kwargs) -> dict:
       my_param = kwargs.get("my_param", 1.0)
       # Use my_param...
   ```

### Testing

Run the plugin in Scope and verify:
- Pipeline appears in UI
- Parameters are configurable
- Generation works with various settings
- Performance is acceptable

## License

Same as your main project.

## Credits

Based on [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) by Cumulo Autumn.
