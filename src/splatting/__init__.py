"""Refined VGGT geometry to trainable Gaussian splats, export and viewing."""

from .config import SplattingConfig
from .contracts import Camera, SplattingResult
from .data import RefinedScene, load_scene
from .sfm_config import SfmSplatConfig
from .sfm_scene import SfmScene, load_sfm_scene


def train_splat(*args, **kwargs) -> SplattingResult:
    from .training import train_splat as implementation

    return implementation(*args, **kwargs)


def load_model(*args, **kwargs):
    from .storage import load_model as implementation

    return implementation(*args, **kwargs)


def export_ply(*args, **kwargs):
    from .storage import export_ply as implementation

    return implementation(*args, **kwargs)


def train_sfm_splat(*args, **kwargs) -> dict:
    from .sfm_training import train_sfm_splat as implementation

    return implementation(*args, **kwargs)


def view_splat(*args, **kwargs):
    from .viewer import view_splat as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "Camera",
    "RefinedScene",
    "SfmScene",
    "SfmSplatConfig",
    "SplattingConfig",
    "SplattingResult",
    "export_ply",
    "load_model",
    "load_scene",
    "load_sfm_scene",
    "train_sfm_splat",
    "train_splat",
    "view_splat",
]
