"""Public rendering API; CUDA dependencies load when the API is called."""

from ..splatting.contracts import Camera


def render_gaussians(*args, **kwargs):
    from .renderer import render_gaussians as implementation

    return implementation(*args, **kwargs)


def render_cameras(*args, **kwargs):
    from .cli import render_cameras as implementation

    return implementation(*args, **kwargs)


__all__ = ["Camera", "render_cameras", "render_gaussians"]
