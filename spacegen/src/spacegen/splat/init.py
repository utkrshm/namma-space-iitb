"""Dense initialization from monocular depth fitted to the SfM points.

SfM points exist only where there is texture, so blank walls, ceilings and floors start
with no Gaussians. MCMC must then grow them from far away, and the first renders have
black holes (playroom: 36k points, alpha 0.5 at step 0). Here, for a spread of training
images:

1. Project the SfM points that image observes; they give sparse metric depths.
2. Fit inverse depth = a · mono + b on those pixels (robust: refit after trimming the worst
   20%). A fit with fewer than 30 points, or with a ≤ 0, is skipped.
3. Back-project a pixel grid, dropping pixels at depth edges (where monocular depth
   smears foreground into background) and outside the depth range the SfM points span.

The new points join the SfM points as seeds. Their colours come from the image.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..depth import read_rgb
from ..sfm import Solution


def depth_seeds(
    solution: Solution, depth_dir: Path, train: list[int], count: int, max_long_side: int
):
    """Extra ``(points, colours, views)`` from monocular depth; empty when no depth exists."""
    if count <= 0 or depth_dir is None:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), []
    per_camera: list[list[int]] = [[] for _ in solution.cameras]
    for point, views in enumerate(solution.point_views):
        for view in views:
            per_camera[view].append(point)

    images = train[:: max(1, len(train) // 80)]
    per_image = max(1, count // max(1, len(images)))
    points, colours, views = [], [], []
    for index in images:
        path = Path(depth_dir) / (solution.image_paths[index].stem + ".npy")
        if not path.exists() or len(per_camera[index]) < 30:
            continue
        mono = np.load(path).astype(np.float32)
        height, width = mono.shape
        camera = solution.cameras[index].resized(width, height)
        local = (
            solution.points[per_camera[index]] @ camera.world_to_camera[:3, :3].T
            + camera.world_to_camera[:3, 3]
        )
        z = local[:, 2]
        uv = (
            local[:, :2] / np.maximum(z[:, None], 1e-9) * np.diag(camera.intrinsics)[:2]
            + camera.intrinsics[:2, 2]
        )
        pixel = np.round(uv).astype(int)
        inside = (
            (z > 0)
            & (pixel[:, 0] >= 0)
            & (pixel[:, 0] < width)
            & (pixel[:, 1] >= 0)
            & (pixel[:, 1] < height)
        )
        if inside.sum() < 30:
            continue
        x = mono[pixel[inside, 1], pixel[inside, 0]]
        y = 1 / z[inside]
        a, b = np.polyfit(x, y, 1)
        for _ in range(2):
            residual = np.abs(a * x + b - y)
            keep = residual <= np.percentile(residual, 80)
            a, b = np.polyfit(x[keep], y[keep], 1)
        if a <= 0:
            continue
        inverse = a * mono + b
        near, far = np.percentile(z[inside], [2, 98])
        gy, gx = np.gradient(inverse)
        edge = np.hypot(gx, gy) / np.maximum(inverse, 1e-9) > 0.05
        stride = max(1, int(np.sqrt(height * width / per_image)))
        ys, xs = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
        depth = 1 / np.maximum(inverse[ys, xs], 1e-9)
        keep = (inverse[ys, xs] > 0) & ~edge[ys, xs] & (depth > 0.8 * near) & (depth < 1.25 * far)
        ys, xs, depth = ys[keep], xs[keep], depth[keep]
        k = camera.intrinsics
        local = np.stack(
            ((xs - k[0, 2]) / k[0, 0] * depth, (ys - k[1, 2]) / k[1, 1] * depth, depth), 1
        )
        camera_to_world = camera.camera_to_world
        points.append(local @ camera_to_world[:3, :3].T + camera_to_world[:3, 3])
        rgb = read_rgb(solution.image_paths[index], max_long_side)
        scale = rgb.shape[1] / width
        rows = np.clip((ys * scale).astype(int), 0, rgb.shape[0] - 1)
        columns = np.clip((xs * scale).astype(int), 0, rgb.shape[1] - 1)
        colours.append(rgb[rows, columns].astype(np.float32) / 255)
        views += [np.array([index])] * len(depth)
    if not points:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), []
    return np.concatenate(points).astype(np.float32), np.concatenate(colours), views


def with_depth_seeds(
    solution: Solution, depth_dir, train, count: int, max_long_side: int, log=print
):
    points, colours, views = depth_seeds(solution, depth_dir, train, count, max_long_side)
    log(f"depth seeds: {len(points)} points added to {len(solution.points)} SfM points")
    if not len(points):
        return solution
    return replace(
        solution,
        points=np.concatenate((solution.points, points)),
        colors=np.concatenate((solution.colors, colours)),
        point_views=solution.point_views + views,
    )
