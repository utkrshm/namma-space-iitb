"""Dense full-resolution structure from motion for video captures (pycolmap)."""

from .contracts import SfmConfig
from .evaluation import (
    camera_poses,
    compare_with_vggt,
    model_report,
    trajectory_continuity,
    umeyama,
)
from .frames import export_frames, select_frames
from .matching import build_pairs, planar_fraction
from .pipeline import run_sfm


def densify(*args, **kwargs) -> dict:
    from .densify import densify as implementation

    return implementation(*args, **kwargs)


def floor_bundle_adjust(*args, **kwargs) -> dict:
    from .floor_ba import floor_bundle_adjust as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "SfmConfig",
    "build_pairs",
    "camera_poses",
    "compare_with_vggt",
    "densify",
    "export_frames",
    "floor_bundle_adjust",
    "model_report",
    "planar_fraction",
    "run_sfm",
    "select_frames",
    "trajectory_continuity",
    "umeyama",
]
