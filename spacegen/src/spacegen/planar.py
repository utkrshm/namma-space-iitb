"""Planar depth priors for walls, floors and ceilings (``splat.planar_prior``).

Blank walls give the photometric loss nothing to hold on to, and monocular depth bends on
them (a wall bulges or curls at its ends), so Gaussians settle a little in front of or
behind the true surface and smear as soon as the viewer moves (PlanarGS, AtlasGS and
3D Gaussian Flats report the same for indoor scenes). Here each training frame is
segmented (Mask2Former on ADE20K, as in `spacegen.reflectors`); each large ``wall``,
``floor`` or ``ceiling`` region gets up to two planes fitted (RANSAC, in camera space)
to that frame's depth target (MVS where measured, else the fitted monocular depth). Where
a plane explains the depth, or where there was no depth, the plane's depth replaces it.

Output: ``<depth_dir>/planar/<stem>.npz`` with ``inverse`` (float16 inverse z-depth in
SfM units on the depth grid, 0 off planes) and ``planar`` (bool), plus ``summary.json``.
`spacegen.splat.train.Priors` uses them as trusted depth (full weight, like MVS) with a
normal target on every planar pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .contracts import write_atomic, write_text_atomic
from .depth import read_rgb
from .reflectors import SEGMENTER, _frame_depth
from .sfm import Solution

PLANAR = ("wall", "floor", "ceiling")


def fit_planes(points: np.ndarray, threshold: float, rng, planes: int = 2, iterations: int = 300):
    """Up to ``planes`` planes ``(normal, offset, inliers)`` found one after another
    (RANSAC, then a least-squares refit on the inliers). A point is an inlier when its
    distance to the plane is below ``threshold`` × its distance to the camera (the
    origin), since depth errors grow with distance."""
    found = []
    free = np.ones(len(points), bool)
    distance = np.linalg.norm(points, axis=1)
    for _ in range(planes):
        candidates = np.flatnonzero(free)
        if len(candidates) < 50:
            break
        best, best_count = None, 0
        for _ in range(iterations):
            sample = points[rng.choice(candidates, 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-12:
                continue
            normal /= length
            inliers = free & (np.abs(points @ normal - sample[0] @ normal) < threshold * distance)
            count = int(inliers.sum())
            if count > best_count:
                best, best_count = inliers, count
        if best is None or best_count < 50:
            break
        centre = points[best].mean(0)
        _, _, axes = np.linalg.svd(points[best] - centre, full_matrices=False)
        normal = axes[2]
        offset = -float(centre @ normal)
        inliers = free & (np.abs(points @ normal + offset) < threshold * distance)
        found.append((normal, offset, inliers))
        free &= ~inliers
    return found


def plane_inverse_depth(rays: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Inverse z-depth along camera rays ``(x, y, 1)`` [...,3] of the plane n·p + d = 0
    (camera space); 0 where the ray misses it (parallel, or behind the camera)."""
    along = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        z = -offset / along
    return np.where(np.isfinite(z) & (z > 0), 1 / np.where(z > 0, z, 1), 0.0)


def planar_priors(
    solution: Solution,
    train: list[int],
    depth_dir,
    device: str = "cuda:0",
    log=print,
    min_share: float = 0.01,
    threshold: float = 0.015,
    min_inlier_share: float = 0.6,
) -> dict:
    """Write ``planar/`` (see the module docstring). Frames that already have their file
    are kept, so re-sliced extra frames added later get theirs."""
    depth_dir = Path(depth_dir)
    output = depth_dir / "planar"
    summary_path = output / "summary.json"
    fits_path = depth_dir / "alignment.json"
    if not fits_path.exists():
        log("planar: no depth alignment; skipped")
        return {"frames": 0}
    todo = [
        i for i in sorted(train) if not (output / f"{solution.image_paths[i].stem}.npz").exists()
    ]
    if not todo and summary_path.exists():
        return json.loads(summary_path.read_text())

    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(SEGMENTER)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEGMENTER)
    # Half precision on the GPU; the CPU has no fast half kernels for this model.
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    model = model.to(device, dtype).eval()
    names = {i: label.split(",")[0].strip() for i, label in model.config.id2label.items()}
    wanted = [i for i, name in names.items() if name in PLANAR]
    fits = json.loads(fits_path.read_text())
    rng = np.random.default_rng(0)
    output.mkdir(parents=True, exist_ok=True)
    shares = []

    for index in todo:
        stem = solution.image_paths[index].stem
        depth = _frame_depth(depth_dir, stem, fits)
        if depth is None:
            continue
        rows, columns = depth.shape
        rgb = read_rgb(solution.image_paths[index], 640)
        with torch.no_grad():
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            result = model(pixel_values=inputs["pixel_values"].to(dtype))
            labels = (
                processor.post_process_semantic_segmentation(
                    result, target_sizes=[(rows, columns)]
                )[0]
                .cpu()
                .numpy()
            )
        k = solution.cameras[index].resized(columns, rows).intrinsics
        ys, xs = np.mgrid[0:rows, 0:columns]
        rays = np.stack(
            ((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones((rows, columns))), -1
        )
        inverse = np.zeros((rows, columns), np.float32)
        planar = np.zeros((rows, columns), bool)
        for label in wanted:
            count, parts = cv2.connectedComponents((labels == label).astype(np.uint8))
            for part in range(1, count):
                # Away from the region's border, where segmentation and depth disagree.
                region = cv2.erode((parts == part).astype(np.uint8), np.ones((5, 5), np.uint8))
                region = region.astype(bool)
                if region.mean() < min_share:
                    continue
                known = region & (depth > 0)
                if known.sum() < 0.3 * region.sum():
                    continue
                points = rays[known] * depth[known][:, None]
                # A plane must hold a quarter of the region's depth, and together the
                # planes most of it; otherwise the region is clutter the segmenter called
                # a wall, and keeps its depth.
                found = [
                    fit
                    for fit in fit_planes(points, threshold, rng)
                    if fit[2].sum() >= 0.25 * len(points)
                ]
                if sum(int(fit[2].sum()) for fit in found) < min_inlier_share * len(points):
                    continue
                for normal, offset, inliers in found:
                    plane = plane_inverse_depth(rays, normal, offset)
                    agrees = np.zeros_like(region)
                    agrees[known] = inliers
                    # Pixels of the region without depth take the plane only next to
                    # its inliers (the same wall, not a wall seen past a corner).
                    near = cv2.dilate(agrees.astype(np.uint8), np.ones((9, 9), np.uint8))
                    take = (agrees | (region & (depth <= 0) & near.astype(bool))) & (plane > 0)
                    take &= ~planar
                    inverse[take] = plane[take]
                    planar |= take
        write_atomic(
            output / f"{stem}.npz",
            lambda stream, inverse=inverse, planar=planar: np.savez_compressed(
                stream, inverse=inverse.astype(np.float16), planar=planar
            ),
        )
        shares.append(float(planar.mean()))

    for path in output.glob("*.npz"):
        if path.stem not in {solution.image_paths[i].stem for i in todo}:
            with np.load(path) as planes:
                shares.append(float(planes["planar"].mean()))
    summary = {
        "frames": len(shares),
        "planar_share_mean": float(np.mean(shares)) if shares else 0.0,
        "threshold": threshold,
    }
    write_text_atomic(summary_path, json.dumps(summary, indent=1) + "\n")
    log(f"planar: {summary}")
    return summary
