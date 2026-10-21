"""CPU reconstruction handoff: RGB pixels, OpenCV cameras, arbitrary scene units."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ImageTransform:
    """Upright source -> model pixels; sizes are (width, height), offsets (x, y)."""

    source_size: tuple[int, int]
    resized_size: tuple[int, int]
    model_size: tuple[int, int]
    offset: tuple[int, int]

    @property
    def source_to_model(self) -> np.ndarray:
        sx, sy = np.array(self.resized_size) / np.array(self.source_size)
        dx, dy = self.offset
        return np.array(
            [[sx, 0, dx + (sx - 1) / 2], [0, sy, dy + (sy - 1) / 2], [0, 0, 1]],
            dtype=np.float32,
        )

    def transform_points(self, xy: np.ndarray, *, inverse: bool = False) -> np.ndarray:
        """Map pixel centres; query points use the first image's model coordinates."""
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Pixel coordinates must be finite with final dimension 2")
        matrix = self.source_to_model
        if inverse:
            matrix = np.linalg.inv(matrix)
        return xy @ matrix[:2, :2].T + matrix[:2, 2]


@dataclass(frozen=True)
class FrameProvenance:
    frame_id: str
    image_path: str | None
    clip_id: str | None = None
    pts: int | None = None
    timestamp_s: float | None = None
    time_base: str | None = None
    selected_frame: dict | None = None  # Frozen preprocessing record, including rotation.


@dataclass(frozen=True)
class ImageBatch:
    images: np.ndarray  # S,3,H,W float32 RGB in [0,1]
    valid_mask: np.ndarray  # S,H,W bool; excludes padding
    transforms: tuple[ImageTransform, ...]
    frames: tuple[FrameProvenance, ...]


@dataclass(frozen=True)
class CameraParameters:
    intrinsics: np.ndarray  # S,3,3, model-image pixels
    original_intrinsics: np.ndarray  # S,3,3, upright exported-image pixels
    world_to_camera: np.ndarray  # S,3,4, X_cam = R X_world + t
    camera_to_world: np.ndarray  # S,4,4


@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray  # N,3 float32 world XYZ
    colors: np.ndarray  # N,3 uint8 RGB
    confidence: np.ndarray  # N float32, uncalibrated head confidence
    frame_indices: np.ndarray  # N integer indices into result.frames
    pixel_xy: np.ndarray  # N,2 integer model-image pixel centres


@dataclass(frozen=True)
class ReconstructionResult:
    """One jointly inferred scene. S is retained even for a single image."""

    images: np.ndarray  # S,H,W,3 uint8 RGB on the model grid
    valid_mask: np.ndarray  # S,H,W bool
    transforms: tuple[ImageTransform, ...]
    frames: tuple[FrameProvenance, ...]
    cameras: CameraParameters
    pose_encoding: np.ndarray  # S,9
    pose_encoding_iterations: np.ndarray  # I,S,9
    depth: np.ndarray  # S,H,W, camera z-depth (not ray distance)
    depth_confidence: np.ndarray  # S,H,W
    world_points: np.ndarray  # S,H,W,3, direct point-head prediction
    world_points_confidence: np.ndarray  # S,H,W
    depth_world_points: np.ndarray  # S,H,W,3, unprojected depth
    model_id: str
    precision: str
    tracks: np.ndarray | None = None  # S,N,2, model-image pixels
    track_visibility: np.ndarray | None = None  # S,N
    track_confidence: np.ndarray | None = None  # S,N
    preprocessing_dir: str | None = None

    def to_dict(self) -> dict[str, np.ndarray]:
        """All processed arrays, with no tensors or object arrays."""
        names = (
            "images",
            "valid_mask",
            "pose_encoding",
            "pose_encoding_iterations",
            "depth",
            "depth_confidence",
            "world_points",
            "world_points_confidence",
            "depth_world_points",
            "tracks",
            "track_visibility",
            "track_confidence",
        )
        result = {name: getattr(self, name) for name in names if getattr(self, name) is not None}
        result.update(
            intrinsics=self.cameras.intrinsics,
            original_intrinsics=self.cameras.original_intrinsics,
            world_to_camera=self.cameras.world_to_camera,
            camera_to_world=self.cameras.camera_to_world,
            source_to_model=np.stack([t.source_to_model for t in self.transforms]),
        )
        return result

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "precision": self.precision,
            "preprocessing_dir": self.preprocessing_dir,
            "coordinate_convention": "OpenCV: camera x right, y down, z forward",
            "extrinsic_convention": "world_to_camera: X_cam = R @ X_world + t",
            "units": "arbitrary scene units, not metres",
            "depth_convention": "camera z-depth, not Euclidean ray distance",
            "pixel_convention": "integer pixel centres, origin top left, x right, y down",
            "frames": [asdict(frame) for frame in self.frames],
            "transforms": [asdict(transform) for transform in self.transforms],
            "arrays": {name: list(value.shape) for name, value in self.to_dict().items()},
        }

    def save(self, output_dir: str | Path) -> Path:
        """Save processed arrays and JSON provenance; original images remain unchanged."""
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "predictions.npz", **self.to_dict())
        (root / "metadata.json").write_text(
            json.dumps(self.metadata(), indent=2, allow_nan=False) + "\n"
        )
        return root

    def point_cloud(
        self,
        *,
        source: Literal["depth", "point_head"] = "depth",
        confidence_percentile: float = 25.0,
        min_confidence: float | None = None,
        max_points: int | None = None,
    ) -> PointCloud:
        """Concatenate valid observations, with provenance; this is not geometric fusion."""
        if source not in ("depth", "point_head"):
            raise ValueError("source must be 'depth' or 'point_head'")
        if not np.isfinite(confidence_percentile) or not 0 <= confidence_percentile <= 100:
            raise ValueError("confidence_percentile must be in [0,100]")
        if min_confidence is not None and not np.isfinite(min_confidence):
            raise ValueError("min_confidence must be finite")
        if max_points is not None and (not isinstance(max_points, int) or max_points < 1):
            raise ValueError("max_points must be a positive integer")

        points = self.depth_world_points if source == "depth" else self.world_points
        confidence = self.depth_confidence if source == "depth" else self.world_points_confidence
        valid = self.valid_mask & np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
        if source == "depth":
            valid &= np.isfinite(self.depth) & (self.depth > 0)
        else:
            for i, camera in enumerate(self.cameras.world_to_camera):
                valid[i] &= (points[i] @ camera[2, :3] + camera[2, 3]) > 0

        if valid.any():
            threshold = float(np.percentile(confidence[valid], confidence_percentile))
            if min_confidence is not None:
                threshold = max(threshold, min_confidence)
            valid &= confidence >= threshold

        indices = np.flatnonzero(valid)
        if max_points is not None and len(indices) > max_points:
            indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)]

        _, height, width = confidence.shape
        frame_indices = indices // (height * width)
        pixel_xy = np.column_stack((indices % width, indices // width % height))
        return PointCloud(
            points.reshape(-1, 3)[indices].astype(np.float32),
            self.images.reshape(-1, 3)[indices],
            confidence.reshape(-1)[indices].astype(np.float32),
            frame_indices,
            pixel_xy,
        )
