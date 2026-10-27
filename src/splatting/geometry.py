"""Training-only point fusion with geometric support and bounded candidate counts."""

from dataclasses import dataclass

import numpy as np

from .config import InitializationConfig
from .data import RefinedScene


@dataclass
class InitialPoints:
    means: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    scene_scale: float
    diagnostics: dict


def multiview_support(scene: RefinedScene, source_index: int, points: np.ndarray) -> np.ndarray:
    """Count depth agreement in nearby OTHER training cameras, never holdout maps.

    Behind-surface observations are occluded and receive no supporting vote.
    Out-of-frame/masked/invalid observations also receive no vote.
    """
    config = scene.config
    if config.min_consistent_views == 0:
        return np.ones(len(points), dtype=bool)
    center = scene.frames[source_index].camera.camera_to_world[:3, 3]
    neighbors = [i for i in scene.train_indices if i != source_index]
    neighbors.sort(
        key=lambda i: np.linalg.norm(scene.frames[i].camera.camera_to_world[:3, 3] - center)
    )
    support = np.zeros(len(points), dtype=np.int32)
    for index in neighbors[: config.consistency_views]:
        other = scene.frame(index)
        camera = other.camera
        local = points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
        z = local[:, 2]
        projected = local @ camera.intrinsics.T
        uv = projected[:, :2] / np.maximum(z[:, None], 1e-12)
        inside = (
            (z > config.depth_min)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < camera.width - 0.5)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < camera.height - 0.5)
        )
        selected = np.flatnonzero(inside)
        pixel = np.floor(uv[selected] + 0.5).astype(np.int64)
        x, y = pixel[:, 0], pixel[:, 1]
        target = other.depth[y, x]
        agreement = other.geometry_mask[y, x] & (
            np.abs(z[selected] - target) <= config.depth_relative_tolerance * target
        )
        support[selected[agreement]] += 1
    return support >= config.min_consistent_views


def fuse_voxels(points: np.ndarray, colors: np.ndarray, voxel_size: float):
    """Average points/colors in occupied voxels, retaining world coordinates."""
    if len(points) == 0 or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Voxel fusion needs points and a positive finite voxel size")
    origin = points.min(axis=0)
    cells = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    fused = np.zeros((len(counts), 3), dtype=np.float64)
    rgb = np.zeros_like(fused)
    np.add.at(fused, inverse, points)
    np.add.at(rgb, inverse, colors)
    return (fused / counts[:, None]).astype(np.float32), (rgb / counts[:, None]).astype(np.float32)


def initialize_points(
    scene: RefinedScene, config: InitializationConfig, seed: int
) -> InitialPoints:
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    budget = config.max_points * config.candidate_multiplier
    quota = min(config.points_per_frame, budget // len(scene.train_indices))
    if quota < 1:
        raise ValueError("Candidate budget is smaller than the number of training frames")

    all_points, all_colors, per_frame = [], [], {}
    for index in scene.train_indices:
        frame = scene.frame(index)
        candidates = np.flatnonzero(frame.geometry_mask)
        if len(candidates) > quota:
            candidates = rng.choice(candidates, quota, replace=False)
        points = frame.points.reshape(-1, 3)[candidates]
        colors = frame.rgb.reshape(-1, 3)[candidates]
        supported = multiview_support(scene, index, points)
        all_points.append(points[supported])
        all_colors.append(colors[supported])
        per_frame[frame.camera.frame_id] = {
            "sampled": len(candidates),
            "multiview_supported": int(supported.sum()),
        }

    points, colors = np.concatenate(all_points), np.concatenate(all_colors)
    if len(points) < 4:
        raise ValueError("Too few geometrically supported points; inspect input alignment/coverage")
    center = np.median(points, axis=0)
    radius = float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95))
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("Degenerate reconstruction: scene radius is zero or nonfinite")

    points, colors = fuse_voxels(points, colors, config.voxel_fraction * radius)
    if len(points) < 4:
        raise ValueError("Voxel fusion left fewer than four points; reduce voxel_fraction")
    distances, _ = cKDTree(points).query(points, k=4, workers=1)
    neighbor_distances = np.sqrt(np.mean(distances[:, 1:] ** 2, axis=1))
    typical = float(np.median(neighbor_distances))
    keep = neighbor_distances <= config.neighbor_distance_factor * max(typical, 1e-12)
    points, colors = points[keep], colors[keep]
    if len(points) < 4:
        raise ValueError("Neighbor filtering left fewer than four points")
    if len(points) > config.max_points:
        indices = rng.choice(len(points), config.max_points, replace=False)
        points, colors = points[indices], colors[indices]

    # Recompute after filtering/subsampling so scales describe the actual seed cloud.
    distances, _ = cKDTree(points).query(points, k=4, workers=1)
    scale = np.sqrt(np.mean(distances[:, 1:] ** 2, axis=1)) * config.scale_multiplier
    scale = np.clip(scale, config.min_scale_fraction * radius, config.max_scale_fraction * radius)
    scales = np.repeat(scale[:, None], 3, axis=1).astype(np.float32)
    diagnostics = {
        "per_frame": per_frame,
        "candidate_budget": budget,
        "points_per_frame": quota,
        "initial_gaussians": len(points),
        "scene_scale": radius,
        "voxel_size": radius * config.voxel_fraction,
    }
    return InitialPoints(points, colors, scales, radius, diagnostics)
