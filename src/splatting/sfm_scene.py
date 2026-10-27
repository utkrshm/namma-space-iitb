"""Undistorted COLMAP model (from `src.sfm`) as posed training images and seed points.

COLMAP puts the centre of the top-left pixel at (0.5, 0.5). `Camera` uses integer
centres, so principal points are shifted by -0.5 here; the renderer adds +0.5 back at
the gsplat boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .contracts import Camera


@dataclass
class SfmScene:
    cameras: list[Camera]
    image_paths: list[Path]
    timestamps: np.ndarray
    geometry: np.ndarray  # True for preprocessing geometry frames
    points: np.ndarray  # [N,3] SfM world coordinates
    colors: np.ndarray  # [N,3] in 0..1
    train_indices: list[int]
    val_indices: list[int]

    @property
    def frame_ids(self) -> list[str]:
        return [camera.frame_id for camera in self.cameras]


def load_sfm_scene(
    sfm_dir: str | Path, holdout_geometry_every: int = 8, holdout_exclusion_s: float = 0.0
) -> SfmScene:
    """Read ``<sfm_dir>/undistorted`` and ``frames.json``; images are ordered by time."""
    import pycolmap

    root = Path(sfm_dir).resolve()
    frames = {f["name"]: f for f in json.loads((root / "frames.json").read_text())}
    model = pycolmap.Reconstruction(root / "undistorted" / "sparse")
    records = []
    for image in model.images.values():
        if not image.has_pose:
            continue
        camera = model.cameras[image.camera_id]
        if camera.model.name != "PINHOLE":
            raise ValueError("Expected an undistorted PINHOLE model")
        fx, fy, cx, cy = camera.params
        intrinsics = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.0]])
        world_to_camera = np.eye(4)
        world_to_camera[:3] = image.cam_from_world().matrix()
        frame = frames[image.name]
        records.append(
            (
                frame["timestamp_s"],
                Camera(frame["frame_id"], camera.width, camera.height, intrinsics, world_to_camera),
                root / "undistorted" / "images" / image.name,
                bool(frame["geometry"]),
            )
        )
    records.sort(key=lambda r: r[0])
    timestamps = np.array([r[0] for r in records])
    geometry = np.array([r[3] for r in records])

    geometry_positions = np.flatnonzero(geometry)
    val = [
        int(index)
        for k, index in enumerate(geometry_positions)
        if holdout_geometry_every and (k + 1) % holdout_geometry_every == 0
    ]
    near_val = np.zeros(len(records), bool)
    for index in val:
        near_val |= np.abs(timestamps - timestamps[index]) <= holdout_exclusion_s
    train = [i for i in range(len(records)) if not near_val[i]]

    xyz = np.array([p.xyz for p in model.points3D.values()], dtype=np.float32)
    rgb = np.array([p.color for p in model.points3D.values()], dtype=np.float32) / 255.0
    return SfmScene(
        [r[1] for r in records],
        [r[2] for r in records],
        timestamps,
        geometry,
        xyz,
        rgb,
        train,
        val,
    )


def read_rgb(path: Path, downscale: int) -> np.ndarray:
    """uint8 RGB; area downscaling keeps the OpenCV half-pixel convention of `Camera.resized`."""
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    if downscale > 1:
        size = (image.shape[1] // downscale, image.shape[0] // downscale)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image
