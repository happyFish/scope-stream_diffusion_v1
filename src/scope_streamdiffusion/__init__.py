"""StreamDiffusion plugin for Daydream Scope."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register StreamDiffusion pipeline with Scope."""
    from .pipeline import StreamDiffusionPipeline

    register(StreamDiffusionPipeline)
