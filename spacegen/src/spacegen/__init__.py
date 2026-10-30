"""spacegen: any indoor video (or image folder) → a photoreal, gap-filled 3D splat + tour."""

from .config import PipelineConfig
from .contracts import Camera, look_at
from .fill import fill_gaps
from .ingest import ingest
from .pipeline import STAGES, run
from .scene_frame import SceneFrame, estimate_frame
from .sfm import Solution, load_solution, solve_cameras
from .splat import GaussianModel, load_ply, render, save_ply, train_splat
from .support import compute_support, support_map
from .tour import render_tour

__all__ = [
    "STAGES",
    "Camera",
    "GaussianModel",
    "PipelineConfig",
    "SceneFrame",
    "Solution",
    "compute_support",
    "estimate_frame",
    "fill_gaps",
    "ingest",
    "load_ply",
    "load_solution",
    "look_at",
    "render",
    "render_tour",
    "run",
    "save_ply",
    "solve_cameras",
    "support_map",
    "train_splat",
]
