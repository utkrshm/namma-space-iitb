"""spacegen: any indoor video (or image folder) → a photoreal, gap-filled 3D splat + tour."""

from .audit import audit_model
from .config import PipelineConfig
from .contracts import Camera, look_at
from .depth import align_metric_depth, near_inverse_depth
from .fill import fill_gaps
from .ingest import ingest
from .mvs import compute_mvs, gap_seeds
from .pipeline import STAGES, run
from .planar import planar_priors
from .reflectors import find_reflectors
from .reslice import reslice, run_solution
from .scene_frame import SceneFrame, estimate_frame, run_frame
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
    "align_metric_depth",
    "audit_model",
    "compute_mvs",
    "compute_support",
    "estimate_frame",
    "fill_gaps",
    "find_reflectors",
    "gap_seeds",
    "ingest",
    "load_ply",
    "load_solution",
    "look_at",
    "near_inverse_depth",
    "planar_priors",
    "render",
    "render_tour",
    "reslice",
    "run",
    "run_frame",
    "run_solution",
    "save_ply",
    "solve_cameras",
    "support_map",
    "train_splat",
]
