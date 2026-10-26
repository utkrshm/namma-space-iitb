"""Loop closure for global camera drift: VGGT loop chunks (A) and bundle adjustment (B)."""

from .bundle_adjustment import BundleProblem, build_tracks, bundle_adjust
from .contracts import FrameMatches, FrameModel, LoopConfig, LoopPair
from .detection import FeatureBank, MatchStore, detect_loops, retrieve_candidates, select_loops
from .evaluation import correspondence_disagreement, render_camera_slices, render_comparison
from .loop_chunks import build_loop_edges, chunk_frames, cross_side_agreement, infer_loop_chunk
from .manhattan import ManhattanPrior, camera_wall_normals, world_up
from .pipeline import run_loop_closure, summary_table
from .store import SectionCache, build_frame_models, export_cloud, load_graph_cameras

__all__ = [
    "BundleProblem",
    "FeatureBank",
    "FrameMatches",
    "FrameModel",
    "LoopConfig",
    "LoopPair",
    "ManhattanPrior",
    "MatchStore",
    "SectionCache",
    "build_frame_models",
    "build_loop_edges",
    "build_tracks",
    "bundle_adjust",
    "camera_wall_normals",
    "chunk_frames",
    "correspondence_disagreement",
    "cross_side_agreement",
    "detect_loops",
    "export_cloud",
    "infer_loop_chunk",
    "load_graph_cameras",
    "render_camera_slices",
    "render_comparison",
    "retrieve_candidates",
    "run_loop_closure",
    "select_loops",
    "summary_table",
    "world_up",
]
