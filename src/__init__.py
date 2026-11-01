"""Submission pipeline APIs; GPU dependencies load only when a stage is requested."""

from importlib import import_module

from .config import PipelineConfig
from .contracts import Camera, look_at
from .ingest import ingest
from .preprocessing import PreprocessConfig, PreprocessResult, load_result, preprocess_capture

_STAGE_EXPORTS = {
    "STAGES": "pipeline",
    "run": "pipeline",
    "SceneFrame": "scene_frame",
    "estimate_frame": "scene_frame",
    "Solution": "sfm",
    "load_solution": "sfm",
    "solve_cameras": "sfm",
    "GaussianModel": "splat",
    "load_ply": "splat",
    "render": "splat",
    "save_ply": "splat",
    "train_splat": "splat",
    "compute_support": "support",
    "support_map": "support",
    "render_tour": "tour",
}


def __getattr__(name):
    if name not in _STAGE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_STAGE_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "Camera",
    "PipelineConfig",
    "PreprocessConfig",
    "PreprocessResult",
    "ingest",
    "load_result",
    "look_at",
    "preprocess_capture",
    *_STAGE_EXPORTS,
]
