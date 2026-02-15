# Quick Installation Guide

## Prerequisites

Ensure you have:
- Python >= 3.12
- CUDA-capable GPU
- Daydream Scope installed

## Install Plugin

### Option 1: Development Mode (Recommended for testing)

```bash
cd scope-streamdiffusion
pip install -e .
```

This allows you to modify the code and see changes immediately.

### Option 2: Production Install

```bash
cd scope-streamdiffusion
pip install .
```

## Verify Installation

```bash
python -c "import scope_streamdiffusion; print('✓ Plugin loaded successfully')"
```

## Start Using

1. Launch Daydream Scope
2. The "StreamDiffusion" pipeline should appear in the pipeline selector
3. Select it and configure parameters
4. Start generating!

## Troubleshooting

### Plugin Not Appearing

Check that the entry point is registered:
```bash
python -c "from importlib.metadata import entry_points; print([ep for ep in entry_points().get('scope', []) if 'streamdiffusion' in ep.name])"
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install torch diffusers transformers compel logfire
```

### Model Loading Issues

1. Check model path in config
2. Verify models are downloaded
3. Ensure sufficient GPU memory

## Next Steps

- See `README.md` for usage details
- See `ADAPTATION_NOTES.md` for architecture details
- Check Scope documentation for advanced configuration
