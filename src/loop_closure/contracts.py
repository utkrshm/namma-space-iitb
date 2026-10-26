"""Loop-closure stage contracts.

Inputs (read-only):
- A frozen preprocessing directory (`load_result` contract), used only for frame identity,
  timestamps and to re-load source images for VGGT loop chunks.
- A section-alignment directory produced by `src.reconstruction.align_sections`
  (`manifest.json`, `section_XX/predictions.npz|metadata.json`, `graph.npz`,
  `alignment_report.json`). It is never written to.

Outputs (all under one new output directory):
- `loops.json`: retrieved and verified loop pairs, with every rejection reason.
- `matches/`: cached ALIKED/LightGlue matches on the shared 518-pixel model grid.
- `method_a/`: VGGT-Long-style loop chunks (`loop_chunks/loop_XX/`) and a section-graph
  export in the same format as `section_alignment/` plus `cameras.npz`.
- `method_b*/`: bundle-adjusted cameras (`cameras.npz`), `point_cloud.npz|ply`, report.
- `comparison.json` and `inspection/*.png`.

Conventions (unchanged from reconstruction):
- OpenCV cameras: x right, y down, z forward. `camera_to_world` is 4x4 with a proper
  rotation; `X_world = R @ X_cam + C`.
- Pixels are integer-centre coordinates on the padded square model grid (518x518 by
  default), origin top left. All sections and loop chunks share the same image transform
  per frame, so a model pixel identifies the same source pixel everywhere.
- Depth is camera z-depth. Units are arbitrary scene units (the first section's scale),
  not metres. A per-frame `depth_scale` multiplies the owner section's predicted depth.
- Global frame index = position in the section manifest `frame_ids` (source time order).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LoopConfig:
    """Retrieval, verification and solver settings.

    Field names used by `src.preprocessing.revisits.verify_feature_pair` keep their meaning
    there: fundamental-matrix RANSAC inliers, inlier ratio and spatial spread on both images.
    """

    # Retrieval (DINOv2 global descriptors on source frames).
    descriptor_model: str = "facebook/dinov2-base"
    temporal_exclusion_s: float = 20.0
    top_k: int = 8
    min_similarity: float = 0.40
    # Local features on the model grid.
    keypoints: int = 2048
    # Geometric verification for loop pairs.
    min_inliers: int = 40
    min_inlier_ratio: float = 0.3
    min_coverage: float = 0.25
    min_hull_area: float = 0.05
    self_match_radius_px: float = 16.0
    self_match_ratio: float = 0.8
    ransac_px: float = 1.5
    confidence: float = 0.999
    max_iterations: int = 10000
    # Loop pairs rejected only as repetitive may pass with clearly non-planar support.
    planar_ransac_px: float = 3.0
    max_planar_fraction: float = 0.5
    max_two_plane_fraction: float = 0.75
    min_off_plane_inliers: int = 30
    # Relaxed verification for short-baseline sequential pairs used by bundle adjustment.
    sequential_offsets: tuple[int, ...] = (1, 2, 3)
    sequential_min_inliers: int = 20
    sequential_min_coverage: float = 0.125
    sequential_min_hull_area: float = 0.02
    # Loop selection: one chunk per (section pair, time cluster).
    max_loops_per_section_pair: int = 2
    min_loop_separation_s: float = 6.0
    max_loop_chunks: int = 12
    holdout_every: int = 3
    # Method A: loop chunk half-window (frames on each side of each loop end).
    chunk_half_window: int = 6
    loop_gate_normalized_median: float = 0.10
    # Method B: robust bundle adjustment.
    reprojection_sigma_px: float = 1.0
    depth_sigma_log: float = 0.05
    scale_prior_sigma_log: float = 0.25
    rotation_prior_sigma_rad: float = 0.35
    huber_sigma: float = 2.0
    max_track_length: int = 30
    max_ba_iterations: int = 100
    initial_gate_px: float = 30.0
    # Square-walls (Manhattan) yaw prior, used only by the 'm' arm.
    manhattan_sigma_deg: float = 3.0
    manhattan_min_concentration: float = 0.5


@dataclass(frozen=True)
class LoopPair:
    """A verified image pair between frames far apart in time (global frame indices)."""

    frame_a: int
    frame_b: int
    similarity: float
    inliers: int
    coverage: float
    role: str = "candidate"  # "chunk" (method A centre), "train", or "holdout"


@dataclass
class FrameMatches:
    """Verified model-grid correspondences between two frames."""

    frame_a: int
    frame_b: int
    xy_a: np.ndarray  # N,2 float model pixels
    xy_b: np.ndarray  # N,2 float model pixels
    kind: str  # "sequential" or "loop"
    details: dict = field(default_factory=dict)


@dataclass
class FrameModel:
    """Everything needed to place one frame's dense depth in the global frame."""

    frame_id: str
    timestamp_s: float
    owner_section: int
    intrinsics: np.ndarray  # 3,3 model grid
    camera_to_world: np.ndarray  # 4,4 global
    depth: np.ndarray  # H,W owner-section depth before depth_scale
    depth_scale: float
    confidence: np.ndarray  # H,W
    valid: np.ndarray  # H,W bool, includes the q25 confidence filter
    image: np.ndarray  # H,W,3 uint8

    def unproject(self, xy: np.ndarray) -> np.ndarray:
        """Global points for model pixels (nearest pixel depth); NaN where invalid."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        x = np.clip(np.rint(xy[:, 0]).astype(int), 0, self.depth.shape[1] - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(int), 0, self.depth.shape[0] - 1)
        depth = self.depth[y, x].astype(np.float64) * self.depth_scale
        rays = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(self.intrinsics).T
        camera = rays * depth[:, None]
        points = camera @ self.camera_to_world[:3, :3].T + self.camera_to_world[:3, 3]
        points[~self.valid[y, x]] = np.nan
        return points
