"""File contracts shared with upstream reconstruction and downstream rendering."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def load_array(source: Any, root: Path) -> np.ndarray:
    """Read inline matrices, .npy, or {path, key?, index?} array references.

    NPZ requires an explicit key. Index selects a frame in a stacked file.
    Pickled/object arrays are never part of this contract.
    """
    if isinstance(source, list):
        return np.asarray(source, dtype=np.float64)
    reference = {"path": source} if isinstance(source, str) else source
    if not isinstance(reference, dict) or "path" not in reference:
        raise ValueError("Expected an inline matrix or an array file reference")
    if set(reference) - {"path", "key", "index"}:
        raise ValueError(f"Unknown array reference fields: {reference}")

    path = (root / reference["path"]).resolve()
    if path.suffix == ".npz":
        if "key" not in reference:
            raise ValueError(f"NPZ reference needs a key: {path}")
        with np.load(path, allow_pickle=False) as archive:
            array = archive[reference["key"]]
    elif path.suffix == ".npy":
        if "key" in reference:
            raise ValueError(f"A .npy reference cannot have a key: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError(f"Expected .npy or .npz: {path}")

    if "index" in reference:
        index = reference["index"]
        if type(index) is not int or index < 0 or index >= len(array):
            raise ValueError(f"Invalid frame index for {path}: {index}")
        array = array[index]
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ValueError(f"Expected a numeric array: {path}")
    return array


@dataclass(frozen=True)
class Camera:
    """Undistorted OpenCV pinhole camera; K refers to this exact H×W image.

    Integer pixel coordinates address pixel centres. The 4×4 transform maps
    homogeneous world coordinates into x-right, y-down, z-forward camera space.
    """

    frame_id: str
    width: int
    height: int
    intrinsics: np.ndarray
    world_to_camera: np.ndarray

    def __post_init__(self):
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("Camera frame_id must be a nonempty string")
        if any(type(v) is not int or v < 1 for v in (self.width, self.height)):
            raise ValueError("Camera width and height must be positive integers")
        if np.iscomplexobj(self.intrinsics) or np.iscomplexobj(self.world_to_camera):
            raise ValueError("Camera matrices must be real")
        k = np.array(self.intrinsics, dtype=np.float64, copy=True)
        transform = np.array(self.world_to_camera, dtype=np.float64, copy=True)
        if transform.shape == (3, 4):
            transform = np.vstack((transform, [0, 0, 0, 1]))
        if k.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError("Expected intrinsics [3,3] and world_to_camera [3,4] or [4,4]")
        if not np.isfinite(k).all() or not np.isfinite(transform).all():
            raise ValueError("Camera matrices must be finite")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("Invalid pinhole intrinsics")
        if not np.allclose([k[0, 1], k[1, 0]], 0):
            raise ValueError("Skewed intrinsics are unsupported by the renderer")
        if not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous camera transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
            raise ValueError("Camera rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise ValueError("Camera rotation must be right-handed")

        k.setflags(write=False)
        transform.setflags(write=False)
        object.__setattr__(self, "intrinsics", k)
        object.__setattr__(self, "world_to_camera", transform)

    @property
    def camera_to_world(self) -> np.ndarray:
        result = np.eye(4)
        rotation = self.world_to_camera[:3, :3]
        result[:3, :3] = rotation.T
        result[:3, 3] = -rotation.T @ self.world_to_camera[:3, 3]
        return result

    def resized(self, width: int, height: int) -> "Camera":
        # OpenCV resize uses half-pixel centres: u' = sx * (u + .5) - .5.
        sx, sy = width / self.width, height / self.height
        transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        return Camera(
            self.frame_id, width, height, transform @ self.intrinsics, self.world_to_camera
        )

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "intrinsics": self.intrinsics.tolist(),
            "world_to_camera": self.world_to_camera.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict, root: Path = Path(".")) -> "Camera":
        return cls(
            frame_id=value["frame_id"],
            width=value["width"],
            height=value["height"],
            intrinsics=load_array(value["intrinsics"], root),
            world_to_camera=load_array(value["world_to_camera"], root),
        )


@dataclass(frozen=True)
class FrameRecord:
    camera: Camera
    image: Path
    depth: Any
    points: Any
    timestamp_s: float
    source_to_image: np.ndarray
    split: str | None = None
    confidence: Any = None
    mask: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SplattingResult:
    output_dir: Path
    checkpoint: Path
    ply: Path
    metrics: Path
