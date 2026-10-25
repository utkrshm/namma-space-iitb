"""Multi-view handoff: OpenCV cameras, positive z-depth, immutable source pixels."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RefinementConfig:
    device: str = "cuda:0"
    keypoints: int = 2048
    feature_size: int = 1024
    neighbor_offsets: tuple[int, ...] = (1, 2, 4, 8)
    retrieval_top_k: int = 5
    retrieval_exclusion_s: float = 10.0
    grid_size: int = 6
    pose_steps: int = 800
    field_steps: int = 800
    learning_rate: float = 0.003
    seed: int = 17
    max_points: int = 2_000_000
    support_tolerance: float = 0.04
    min_support: int = 2
    flow_anchors: bool = False
    geometry_prior_weight: float = 0.02
    dense_anchors: bool = False

    def __post_init__(self):
        if min(self.keypoints, self.feature_size, self.max_points) < 1:
            raise ValueError("Feature sizes and point budget must be positive")
        if self.grid_size < 2 or min(self.pose_steps, self.field_steps) < 0:
            raise ValueError("Require grid_size >= 2 and nonnegative iteration counts")
        if not self.neighbor_offsets or any(i < 1 for i in self.neighbor_offsets):
            raise ValueError("Temporal neighbor offsets must be positive")
        if self.learning_rate <= 0 or self.support_tolerance <= 0 or self.min_support < 1:
            raise ValueError("Learning rate, support tolerance and support count must be positive")
        if self.retrieval_top_k < 0 or self.retrieval_exclusion_s < 0:
            raise ValueError("Invalid retrieval settings")
        if not np.isfinite(self.geometry_prior_weight) or self.geometry_prior_weight < 0:
            raise ValueError("Geometry prior weight must be finite and nonnegative")


@dataclass(frozen=True)
class RefinementScene:
    """All arrays are CPU NumPy; rotations are camera-to-world, C are world centers."""

    images: np.ndarray  # F,H,W,3 uint8
    depth: np.ndarray  # F,H,W, camera z in the common input scene scale
    confidence: np.ndarray  # F,H,W, original uncalibrated scores
    valid_mask: np.ndarray  # F,H,W, original content and finite positive depth
    intrinsics: np.ndarray  # F,3,3, model pixels
    rotations: np.ndarray  # F,3,3 c2w
    centers: np.ndarray  # F,3 world
    source_to_model: np.ndarray  # F,3,3 integer-pixel-center transforms
    content_bounds: np.ndarray  # F,4 inclusive x0,y0,x1,y1
    owners: np.ndarray  # F source section; zero for joint input
    frames: tuple[dict, ...]
    fingerprint: str

    @property
    def frame_count(self):
        return len(self.frames)


@dataclass(frozen=True)
class ObservationTracks:
    frame_indices: np.ndarray  # O observations
    pixel_xy: np.ndarray  # O,2 model pixels
    track_indices: np.ndarray  # O compact track IDs
    split: np.ndarray  # O: 0 train / 1 validation / 2 test, constant within track
    pairs: np.ndarray  # E,2 verified frame pairs
    feature_indices: np.ndarray  # O original ALIKED keypoint identities


@dataclass(frozen=True)
class RefinementResult:
    rotations: np.ndarray
    centers: np.ndarray
    log_scales: np.ndarray
    log_depth_fields: np.ndarray  # F,G,G zero-mean fields over valid content
    report: dict
