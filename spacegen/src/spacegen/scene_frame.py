"""Gravity, floor, scale and areas from the solved cameras and points.

SfM leaves the world in an arbitrary rotation and scale. Later stages need "up", the
floor and a physical length unit, so this module estimates:

- **Up:** the normal of the dominant plane among the SfM points below the cameras
  (RANSAC, normals within 30° of the cameras' mean up direction). When no floor plane is
  found (a table-top capture), the cameras' mean up direction (−y in OpenCV axes) is
  used. People hold phones upright on average, and tilts toward different view
  directions cancel.
- **Floor height** (that plane) and **camera height:** the median camera height above the
  floor. Lengths downstream are in camera heights. For a handheld phone, one camera
  height ≈ 1.4–1.6 m.
- **Areas:** cameras grouped by plan position with Ward clustering (cut at
  ``area_size`` camera heights), numbered in capture order. These replace hand-typed room
  time ranges.

Output: ``<run>/scene_frame.json`` with ``rotation`` (world → aligned, 3×3), ``origin``
(world point on the floor below the mean camera), ``camera_height`` (world units),
``floor_found`` and ``areas`` (per area: camera indices, plan centre and radius in
camera heights). In the aligned frame, z is up and the floor is z = 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import Camera


@dataclass
class SceneFrame:
    rotation: np.ndarray  # world → aligned
    origin: np.ndarray  # world, on the floor
    camera_height: float  # world units
    floor_found: bool
    areas: list[dict]

    def to_aligned(self, points: np.ndarray) -> np.ndarray:
        """World points → aligned coordinates in camera heights (z up, floor at 0)."""
        return (points - self.origin) @ self.rotation.T / self.camera_height

    def to_world(self, aligned: np.ndarray) -> np.ndarray:
        return aligned * self.camera_height @ self.rotation + self.origin

    @property
    def up(self) -> np.ndarray:
        return self.rotation[2]

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "origin": self.origin.tolist(),
            "camera_height": self.camera_height,
            "floor_found": self.floor_found,
            "areas": self.areas,
        }

    @classmethod
    def load(cls, path) -> SceneFrame:
        value = json.loads(Path(path).read_text())
        return cls(
            np.array(value["rotation"]),
            np.array(value["origin"]),
            float(value["camera_height"]),
            bool(value["floor_found"]),
            value["areas"],
        )


def _ransac_plane(points: np.ndarray, prior_up: np.ndarray, threshold: float, rng, iterations=2000):
    """Best plane (normal, offset, inliers) with the normal within 30° of ``prior_up``."""
    best = (None, None, np.zeros(len(points), bool))
    if len(points) < 50:
        return best
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        if normal @ prior_up < 0:
            normal = -normal
        if normal @ prior_up < np.cos(np.radians(30)):
            continue
        inliers = np.abs((points - sample[0]) @ normal) < threshold
        if inliers.sum() > best[2].sum():
            best = (normal, float(sample[0] @ normal), inliers)
    normal, _, inliers = best
    if normal is None:
        return best
    # Least-squares refit on the inliers.
    chosen = points[inliers]
    centre = chosen.mean(0)
    _, _, axes = np.linalg.svd(chosen - centre, full_matrices=False)
    normal = axes[2] * np.sign(axes[2] @ prior_up)
    return normal, float(centre @ normal), inliers


def estimate_frame(cameras: list[Camera], points: np.ndarray, area_size: float = 2.5) -> SceneFrame:
    rng = np.random.default_rng(0)
    camera_to_world = np.stack([c.camera_to_world for c in cameras])
    centres = camera_to_world[:, :3, 3]
    prior_up = -camera_to_world[:, :3, 1].mean(0)
    prior_up /= np.linalg.norm(prior_up)

    # Floor candidates: points clearly below every camera's height band.
    spread = np.linalg.norm(centres - centres.mean(0), axis=1)
    scale = float(np.median(spread)) + 1e-9
    heights = (points - centres.mean(0)) @ prior_up
    camera_heights = (centres - centres.mean(0)) @ prior_up
    below = points[heights < np.percentile(camera_heights, 5) - 0.3 * scale]
    if len(below) > 20000:
        below = below[rng.choice(len(below), 20000, replace=False)]
    normal, offset, inliers = _ransac_plane(below, prior_up, 0.02 * scale, rng)
    floor_found = normal is not None and inliers.sum() >= max(200, 0.05 * len(below))
    if floor_found:
        up = normal
        floor = offset
    else:
        up = prior_up
        floor = (
            float(np.percentile(points @ up, 2))
            if len(points)
            else float(centres @ up).min() - scale
        )
    camera_height = float(np.median(centres @ up - floor))
    if camera_height <= 0:
        camera_height = scale

    # Aligned axes: x along the main horizontal direction of the camera path.
    horizontal = centres - centres.mean(0)
    horizontal -= np.outer(horizontal @ up, up)
    _, _, axes = np.linalg.svd(horizontal, full_matrices=False)
    across = axes[0] - (axes[0] @ up) * up
    across /= np.linalg.norm(across)
    rotation = np.stack((across, np.cross(up, across), up))
    mean_centre = centres.mean(0)
    origin = mean_centre - (mean_centre @ up - floor) * up

    frame = SceneFrame(rotation, origin, camera_height, bool(floor_found), [])
    frame.areas = find_areas(frame.to_aligned(centres), area_size)
    return frame


def find_areas(aligned_centres: np.ndarray, area_size: float) -> list[dict]:
    """Groups of cameras no wider than ``2 × area_size`` (complete linkage on plan positions),
    numbered by first appearance in capture order."""
    from scipy.cluster.hierarchy import fcluster, linkage

    plan = aligned_centres[:, :2]
    if len(plan) < 2:
        labels = np.ones(len(plan), int)
    else:
        labels = fcluster(linkage(plan, "complete"), t=2 * area_size, criterion="distance")
    order = list(dict.fromkeys(labels.tolist()))
    areas = []
    for number, label in enumerate(order):
        members = np.flatnonzero(labels == label)
        centre = plan[members].mean(0)
        areas.append(
            {
                "name": f"Area {number + 1}",
                "cameras": members.tolist(),
                "centre": centre.tolist(),
                "radius": float(np.linalg.norm(plan[members] - centre, axis=1).max()),
            }
        )
    return areas
