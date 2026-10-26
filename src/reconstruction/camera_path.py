"""Load ordered VGGT camera poses for point-cloud traversal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CAMERA_FILES = ("cameras.npz", "point_cloud.npz", "predictions.npz")


@dataclass(frozen=True)
class CameraPath:
    """Camera-to-world poses in capture order, using the OpenCV camera convention."""

    camera_to_world: np.ndarray  # S,4,4; X_world = R @ X_camera + C
    intrinsics: np.ndarray | None = None  # S,3,3 model-image pixels, when available

    def __post_init__(self):
        poses = np.asarray(self.camera_to_world, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
            raise ValueError("Camera poses must have shape S,4,4 with S > 0")
        if not np.isfinite(poses).all():
            raise ValueError("Camera poses must be finite")
        if not np.allclose(poses[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("Camera poses must be proper homogeneous transforms")

        rotations = poses[:, :3, :3]
        if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4):
            raise ValueError("Camera pose rotations must be orthonormal")
        if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4):
            raise ValueError("Camera pose rotations must have determinant 1")

        intrinsics = self.intrinsics
        if intrinsics is not None:
            intrinsics = np.asarray(intrinsics, dtype=np.float64)
            if intrinsics.shape != (len(poses), 3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError("Camera intrinsics must be finite S,3,3 and match the poses")

        object.__setattr__(self, "camera_to_world", poses)
        object.__setattr__(self, "intrinsics", intrinsics)

    def __len__(self) -> int:
        return len(self.camera_to_world)

    @property
    def centers(self) -> np.ndarray:
        return self.camera_to_world[:, :3, 3]

    @property
    def world_to_camera(self) -> np.ndarray:
        return np.linalg.inv(self.camera_to_world)


def _poses_from_archive(archive) -> tuple[np.ndarray, np.ndarray | None] | None:
    if "camera_to_world" in archive:
        poses = np.asarray(archive["camera_to_world"], dtype=np.float64)
    elif "camera_centers" in archive and "camera_rotations" in archive:
        poses = _poses_from_centers(archive["camera_centers"], archive["camera_rotations"])
    elif "centers" in archive and "rotations" in archive:
        poses = _poses_from_centers(archive["centers"], archive["rotations"])
    elif "world_to_camera" in archive:
        world_to_camera = np.asarray(archive["world_to_camera"], dtype=np.float64)
        if world_to_camera.ndim == 3 and world_to_camera.shape[1:] in ((3, 4), (4, 4)):
            if world_to_camera.shape[1:] == (3, 4):
                homogeneous = np.tile(np.eye(4), (len(world_to_camera), 1, 1))
                homogeneous[:, :3] = world_to_camera
                world_to_camera = homogeneous
            poses = np.linalg.inv(world_to_camera)
        else:
            raise ValueError("world_to_camera must have shape S,3,4 or S,4,4")
    else:
        return None

    intrinsics = None
    if "intrinsics" in archive:
        candidate = np.asarray(archive["intrinsics"], dtype=np.float64)
        if candidate.shape == (len(poses), 3, 3):
            intrinsics = candidate
    return poses, intrinsics


def _poses_from_centers(centers, rotations) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Camera centers must have shape S,3")
    if rotations.shape != (len(centers), 3, 3):
        raise ValueError("Camera rotations must have shape S,3,3 and match the centers")

    poses = np.tile(np.eye(4), (len(centers), 1, 1))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = centers
    return poses


def _load_npz(path: Path) -> CameraPath:
    with np.load(path, allow_pickle=False) as archive:
        loaded = _poses_from_archive(archive)
    if loaded is None:
        raise ValueError(f"No camera poses found in {path}")

    poses, intrinsics = loaded
    return CameraPath(poses, intrinsics)


def load_camera_path(source: str | Path) -> CameraPath:
    """Load poses from a camera NPZ or a directory containing one."""
    source = Path(source)
    if source.is_dir():
        return _load_from_directory(source)
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"Camera source must be an NPZ file or directory: {source}")
        return _load_npz(source)
    raise FileNotFoundError(source)


def _load_from_directory(directory: Path) -> CameraPath:
    failures = []
    for name in _CAMERA_FILES:
        candidate = directory / name
        if not candidate.is_file():
            continue
        try:
            return _load_npz(candidate)
        except ValueError as error:
            failures.append(str(error))
    detail = f" ({'; '.join(failures)})" if failures else ""
    raise FileNotFoundError(f"No camera poses found in {directory}{detail}")


def find_camera_path(cloud_path: str | Path) -> Path:
    """Find the camera artifact associated with a saved point cloud."""
    cloud_path = Path(cloud_path)
    if cloud_path.is_dir():
        candidates = [cloud_path / name for name in _CAMERA_FILES]
    else:
        candidates = [
            cloud_path.with_suffix(".npz"),
            cloud_path.parent / "cameras.npz",
            cloud_path.parent / "point_cloud.npz",
            cloud_path.parent / "predictions.npz",
        ]

    seen = set()
    failures = []
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            _load_npz(candidate)
        except ValueError as error:
            failures.append(str(error))
        else:
            return candidate

    detail = f" ({'; '.join(failures)})" if failures else ""
    raise FileNotFoundError(
        f"No camera poses found beside {cloud_path}; pass --follow-cameras CAMERA_PATH{detail}"
    )
