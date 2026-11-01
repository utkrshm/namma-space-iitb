"""Data contracts shared by every stage.

Coordinate frames and units
---------------------------
- **World:** the SfM frame, in arbitrary units (the SfM gauge). After `scene_frame` the
  world is also *gravity-aligned*: +z points up and the floor lies at about z = 0. Lengths
  are then reported in *camera heights* (the median camera height above the floor), the
  only physical scale a monocular video gives.
- **Camera:** OpenCV axes, x right, y down, z forward. ``world_to_camera`` is a 4×4
  matrix mapping homogeneous world points into camera space.
- **Pixels:** integer coordinates address pixel centres. COLMAP's (0.5, 0.5) centre
  convention is converted when reading COLMAP models (`sfm.solve`), and gsplat's at the
  renderer boundary (`splat.render`).

Stage directories (under one run directory ``<out>/``)
------------------------------------------------------
``frames/``  `ingest`  → ``frames.json`` + ``images/`` (upright RGB JPEGs)
``sfm/``     `sfm`     → ``sparse/0`` (raw), ``undistorted/{images,sparse}`` (PINHOLE)
``depth/``   `depth`   → ``<image stem>.npy`` monocular inverse depth
``splat/``   `splat`   → ``scene.ply``, ``cameras.json``, ``metrics.json``
``tour/``    `tour`    → ``tour.mp4``, ``tour.json``
``report.json``         stage timings, quality checks and the paths above
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Undistorted pinhole camera; ``intrinsics`` refer to this exact width × height."""

    name: str
    width: int
    height: int
    intrinsics: np.ndarray
    world_to_camera: np.ndarray

    def __post_init__(self):
        k = np.array(self.intrinsics, dtype=np.float64)
        transform = np.array(self.world_to_camera, dtype=np.float64)
        if transform.shape == (3, 4):
            transform = np.vstack((transform, [0, 0, 0, 1]))
        if k.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError("Expected intrinsics [3,3] and world_to_camera [4,4]")
        if not (np.isfinite(k).all() and np.isfinite(transform).all()):
            raise ValueError("Camera matrices must be finite")
        if k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("Focal lengths must be positive")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
            raise ValueError("Camera rotation must be orthonormal")
        k.setflags(write=False)
        transform.setflags(write=False)
        object.__setattr__(self, "intrinsics", k)
        object.__setattr__(self, "world_to_camera", transform)

    @property
    def camera_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.world_to_camera)

    @property
    def centre(self) -> np.ndarray:
        return self.camera_to_world[:3, 3]

    def resized(self, width: int, height: int) -> Camera:
        # OpenCV resizing keeps pixel centres aligned: u' = s·(u + ½) − ½.
        sx, sy = width / self.width, height / self.height
        scale = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        return Camera(self.name, width, height, scale @ self.intrinsics, self.world_to_camera)

    def moved(self, world_to_camera: np.ndarray) -> Camera:
        return Camera(self.name, self.width, self.height, self.intrinsics, world_to_camera)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "intrinsics": self.intrinsics.tolist(),
            "world_to_camera": self.world_to_camera.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict) -> Camera:
        return cls(
            value["name"],
            int(value["width"]),
            int(value["height"]),
            np.asarray(value["intrinsics"]),
            np.asarray(value["world_to_camera"]),
        )


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera matrix (OpenCV axes) for a camera at ``eye`` looking at ``target``."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down
        right = np.cross(forward, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.column_stack((right, down, forward))
    camera_to_world[:3, 3] = eye
    return np.linalg.inv(camera_to_world)
