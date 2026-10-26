"""Independent multi-view pose/depth refinement; learned models load on execution."""

from .contracts import ObservationTracks, RefinementConfig, RefinementResult, RefinementScene
from .controls import heldout_protection_report, run_refinement_controls
from .inputs import load_refinement_scene, world_observations
from .matching import build_observation_tracks, load_observation_tracks
from .optimization import evaluate_refinement, optimize_multiview
from .pipeline import refine_multiview
from .validation import audit_refinement

__all__ = [
    "ObservationTracks",
    "RefinementConfig",
    "RefinementResult",
    "RefinementScene",
    "audit_refinement",
    "build_observation_tracks",
    "evaluate_refinement",
    "heldout_protection_report",
    "load_observation_tracks",
    "load_refinement_scene",
    "optimize_multiview",
    "refine_multiview",
    "run_refinement_controls",
    "world_observations",
]
