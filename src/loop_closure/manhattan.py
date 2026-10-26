"""Square-walls (Manhattan) yaw prior for bundle adjustment.

Indoor walls are mostly at right angles. Each frame's own depth gives wall normals in
camera coordinates; after rotating them into the world with the frame's current camera
rotation and projecting onto the horizontal plane, their angles modulo 90° should agree
with one house-wide direction. Comparing angles modulo 90° uses the circular mean of
exp(4i·angle). The prior constrains yaw where feature tracks are missing, such as long
plain walls. Frames whose normals do not show a clear right-angle structure get little
or no weight.
"""

from __future__ import annotations

import numpy as np

from .contracts import FrameModel


def camera_wall_normals(model: FrameModel, stride: int = 4) -> np.ndarray:
    """Unit surface normals (camera coordinates) from a frame's depth, on a pixel grid."""
    height, width = model.depth.shape
    y, x = np.mgrid[0:height:stride, 0:width:stride].astype(np.float64)
    depth = model.depth[::stride, ::stride].astype(np.float64) * model.depth_scale
    valid = model.valid[::stride, ::stride]
    rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(model.intrinsics).T
    points = rays * depth[..., None]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    length = np.linalg.norm(normals, axis=-1)
    ok = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
        & (length > 0)
    )
    return normals[ok] / length[ok, None]


def world_up(models: list[FrameModel]) -> np.ndarray:
    """Phones are held upright: the mean of image-up directions approximates world up."""
    up = -np.mean([m.camera_to_world[:3, :3] @ np.array([0.0, 1.0, 0.0]) for m in models], axis=0)
    return up / np.linalg.norm(up)


class ManhattanPrior:
    """Per-frame residual: wrap(4·(θ_f − θ_ref)) / 4 / σ, weighted by concentration."""

    def __init__(
        self,
        models: list[FrameModel],
        sigma_deg: float,
        min_concentration: float,
        max_normals: int = 400,
        horizontal_tolerance: float = 0.2,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        self.up = world_up(models)
        axis_a = np.cross(self.up, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis_a) < 1e-3:
            axis_a = np.cross(self.up, [0.0, 0.0, 1.0])
        self.axis_a = axis_a / np.linalg.norm(axis_a)
        self.axis_b = np.cross(self.up, self.axis_a)
        self.sigma = np.radians(sigma_deg)
        self.normals: list[np.ndarray] = []
        concentration = np.zeros(len(models))
        for index, model in enumerate(models):
            normals = camera_wall_normals(model)
            world = normals @ model.camera_to_world[:3, :3].T
            normals = normals[np.abs(world @ self.up) < horizontal_tolerance]
            if len(normals) > max_normals:
                normals = normals[rng.choice(len(normals), max_normals, replace=False)]
            self.normals.append(normals)
            if len(normals) >= 50:
                concentration[index] = abs(self._phasor(normals, model.camera_to_world[:3, :3]))
        self.concentration = concentration
        self.weight = np.where(concentration >= min_concentration, concentration, 0.0)
        phasors = np.asarray(
            [
                self._phasor(n, m.camera_to_world[:3, :3]) if len(n) >= 50 else 0.0
                for n, m in zip(self.normals, models)
            ]
        )
        # The house direction is the dominant mode of the initial frame angles, not their
        # mean: drifted frames form a second cluster that a mean would average in.
        angles = np.angle(phasors) / 4
        grid = np.radians(np.arange(-45.0, 45.0, 0.25))
        density = [np.sum(self.weight * np.exp(20 * np.cos(4 * (angles - g)))) for g in grid]
        peak = grid[int(np.argmax(density))]
        near = np.abs(np.angle(np.exp(4j * (angles - peak))) / 4) < np.radians(10)
        self.reference = float(np.angle(np.sum((self.weight * near) * phasors)) / 4)

    def _phasor(self, normals: np.ndarray, rotation: np.ndarray) -> complex:
        world = normals @ rotation.T
        angle = np.arctan2(world @ self.axis_b, world @ self.axis_a)
        return complex(np.mean(np.exp(4j * angle)))

    def frame_angles(self, rotations: np.ndarray) -> np.ndarray:
        """Dominant wall angle modulo 90° (radians) for every frame; NaN without normals."""
        angles = np.full(len(rotations), np.nan)
        for index, (normals, rotation) in enumerate(zip(self.normals, rotations)):
            if len(normals) >= 50:
                angles[index] = np.angle(self._phasor(normals, rotation)) / 4
        return angles

    def residuals(self, rotations: np.ndarray) -> np.ndarray:
        """One residual per frame except frame 0 (gauge); zero where unweighted."""
        angles = np.nan_to_num(self.frame_angles(rotations[1:]))
        difference = np.angle(np.exp(4j * (angles - self.reference))) / 4
        return np.sqrt(self.weight[1:]) * difference / self.sigma

    def summary(self, rotations: np.ndarray) -> dict:
        angles = self.frame_angles(rotations)
        used = self.weight > 0
        deviation = np.degrees(np.abs(np.angle(np.exp(4j * (angles - self.reference))) / 4))
        return {
            "reference_deg": float(np.degrees(self.reference)),
            "weighted_frames": int(used.sum()),
            "median_abs_deviation_deg": float(np.nanmedian(deviation[used])),
            "p90_abs_deviation_deg": float(np.nanpercentile(deviation[used], 90)),
            "frames_above_10_deg": int(np.sum(deviation[used] > 10)),
        }
