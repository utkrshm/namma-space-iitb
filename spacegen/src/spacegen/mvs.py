"""Multi-view stereo: metric depth from parallax, for the textured part of every image.

The monocular depth (`spacegen.depth`) is a learned guess, fitted to the SfM points with
two numbers per image; it bends across a room. Here depth is *measured* instead: a plane
sweep on the GPU, written in PyTorch because the installed pycolmap has no CUDA (COLMAP's
patch-match stereo needs it).

For each image (the *reference*):

1. Source views: up to ``sources`` images sharing the most SfM points with it, preferring
   a triangulation angle between ``min_angle_deg`` and ``max_angle_deg``; nearly
   co-located frames give no parallax, and very different ones see other surfaces.
2. Sweep ``planes`` fronto-parallel planes, evenly spaced in inverse depth over the depth
   range of the SfM points the image sees. For each plane, warp every source image onto
   the reference and score each pixel by 1 − NCC over a ``window`` × ``window`` patch.
   The cost of a plane is the mean over the best half of the sources, so a source that
   cannot see the pixel (occlusion) does not vote.
3. The cheapest plane wins, refined by a parabola through its neighbours. Pixels keep a
   depth only if the patch has texture (grey standard deviation ≥ ``min_texture``), the
   NCC is at least 1 − ``max_cost``, and the winner is not at the end of the range.
4. Geometric check: a depth is kept only when at least ``consistent_views`` source depth
   maps agree with it. Following the pixel into the source, back by the source's depth,
   it must land within ``reprojection_px`` and ``relative_depth`` of where it started.

Outputs (``<run>/depth/mvs/``)
------------------------------
- ``<image stem>.npy``: float16 z-depth in SfM units on the monocular depth grid; 0 where
  nothing was measured;
- ``mvs.json``: ``{"images": {stem: {"measured", "sources", "mono_error"}}, "summary"}``.
  ``measured`` is the share of pixels with a depth; ``mono_error`` is the median relative
  difference to the fitted monocular depth on those pixels.

Coordinates follow `spacegen.contracts` (OpenCV camera axes, integer pixel centres).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import MvsConfig
from .contracts import Camera, write_text_atomic
from .depth import near_inverse_depth, read_rgb


def choose_sources(cameras: list[Camera], point_views: list[np.ndarray], points, config):
    """Per image, ``(sources, near, far)``: source indices, best first, and its depth range."""
    count = len(cameras)
    per_camera: list[list[int]] = [[] for _ in range(count)]
    for point, views in enumerate(point_views):
        for view in views:
            per_camera[view].append(point)
    centres = np.stack([c.centre for c in cameras])

    chosen = []
    for index in range(count):
        seen = np.array(per_camera[index], dtype=int)
        if len(seen) < 10:
            chosen.append(([], 0.0, 0.0))
            continue
        local = points[seen] @ cameras[index].world_to_camera[:3, :3].T
        z = local[:, 2] + cameras[index].world_to_camera[2, 3]
        z = z[z > 0]
        if len(z) < 10:
            chosen.append(([], 0.0, 0.0))
            continue
        near, far = np.percentile(z, [1, 99])
        median = float(np.median(z))
        shared = {}
        for point in seen:
            for view in point_views[point]:
                if view != index:
                    shared[view] = shared.get(view, 0) + 1
        scored = []
        for view, number in shared.items():
            if number < 10:
                continue
            baseline = float(np.linalg.norm(centres[view] - centres[index]))
            angle = math.degrees(2 * math.atan(baseline / (2 * median)))
            useful = config.min_angle_deg <= angle <= config.max_angle_deg
            scored.append((number * (1.0 if useful else 0.05), view))
        scored.sort(reverse=True)
        chosen.append(([v for _, v in scored[: config.sources]], 0.7 * near, 1.5 * far))
    return chosen


def _rays(camera: Camera, device) -> torch.Tensor:
    """[3, H·W] camera-frame directions with z = 1."""
    k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
    ys, xs = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=torch.float32),
        torch.arange(camera.width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack(
        ((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], torch.ones_like(xs))
    ).reshape(3, -1)


def _relative(reference: Camera, source: Camera, device):
    """``(rotation, translation)`` from reference camera space to source camera space."""
    transform = source.world_to_camera @ reference.camera_to_world
    return (
        torch.tensor(transform[:3, :3], dtype=torch.float32, device=device),
        torch.tensor(transform[:3, 3], dtype=torch.float32, device=device),
    )


def _project(points: torch.Tensor, camera: Camera):
    """Pixel x, y and z of camera-space ``points`` [3, N]."""
    k = camera.intrinsics
    z = points[2]
    safe = z.clamp_min(1e-6)
    return points[0] / safe * k[0, 0] + k[0, 2], points[1] / safe * k[1, 1] + k[1, 2], z


def _sample(image: torch.Tensor, x: torch.Tensor, y: torch.Tensor, mode="bilinear"):
    """Values of ``image`` [H,W] at pixel coordinates (any shape); NaN outside."""
    height, width = image.shape
    grid = torch.stack((x / (width - 1) * 2 - 1, y / (height - 1) * 2 - 1), -1)
    values = F.grid_sample(
        image[None, None], grid.reshape(1, -1, 1, 2), mode=mode, align_corners=True
    ).reshape(x.shape)
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    return torch.where(inside, values, torch.full_like(values, float("nan")))


def _box(value: torch.Tensor, window: int) -> torch.Tensor:
    return F.avg_pool2d(value, window, stride=1, padding=window // 2, count_include_pad=False)


def plane_sweep(
    reference: Camera,
    grey: torch.Tensor,
    sources: list[tuple[Camera, torch.Tensor]],
    near: float,
    far: float,
    config: MvsConfig,
) -> torch.Tensor:
    """Raw z-depth [H,W] of ``reference`` (0 = rejected), before the geometric check."""
    device = grey.device
    height, width = grey.shape
    window = config.window
    if not sources or not 0 < near < far:
        return torch.zeros_like(grey)

    inverse = torch.linspace(1 / near, 1 / far, config.planes, device=device)
    rays = _rays(reference, device)
    mean_r = _box(grey[None, None], window)
    variance_r = (_box(grey[None, None] ** 2, window) - mean_r**2).clamp_min(0)

    costs = []
    for camera, image in sources:
        rotation, translation = _relative(reference, camera, device)
        turned = rotation @ rays  # [3, HW]; a point at depth d is d · turned + translation
        plane_costs = []
        for chunk in inverse.split(32):
            depth = (1 / chunk)[:, None, None]  # [P,1,1]
            points = depth * turned[None] + translation[None, :, None]  # [P,3,HW]
            x, y, z = _project(points.permute(1, 0, 2).reshape(3, -1), camera)
            warped = _sample(image, x, y).reshape(len(chunk), 1, height, width)
            behind = (z <= 0).reshape(len(chunk), 1, height, width)
            missing = torch.isnan(warped) | behind
            warped = torch.nan_to_num(warped)
            mean_w = _box(warped, window)
            variance_w = (_box(warped**2, window) - mean_w**2).clamp_min(0)
            covariance = _box(warped * grey[None, None], window) - mean_w * mean_r
            ncc = covariance / torch.sqrt(variance_w * variance_r + 1e-8)
            cost = (1 - ncc).clamp(0, 2)
            plane_costs.append(torch.where(_box(missing.float(), window) > 0, 2.0, cost)[:, 0])
        costs.append(torch.cat(plane_costs))
    costs = torch.stack(costs)  # [S,P,H,W]
    best_half = max(1, math.ceil(len(sources) / 2))
    cost = costs.sort(dim=0).values[:best_half].mean(0)  # [P,H,W]
    del costs

    best = cost.argmin(0)
    lowest = cost.gather(0, best[None])[0]
    inner = best.clamp(1, config.planes - 2)
    before = cost.gather(0, (inner - 1)[None])[0]
    after = cost.gather(0, (inner + 1)[None])[0]
    middle = cost.gather(0, inner[None])[0]
    curvature = before - 2 * middle + after
    offset = torch.where(
        curvature > 1e-6, (0.5 * (before - after) / curvature).clamp(-0.5, 0.5), 0.0
    )
    step = inverse[1] - inverse[0]
    refined = inverse[inner] + offset * step

    texture = variance_r[0, 0].sqrt()
    keep = (
        (lowest <= config.max_cost)
        & (texture >= config.min_texture)
        & (best > 0)
        & (best < config.planes - 1)
        & (refined > 0)
    )
    return torch.where(keep, 1 / refined.clamp_min(1e-9), 0.0)


def consistent(
    reference: Camera, depth: torch.Tensor, sources: list[tuple[Camera, torch.Tensor]], config
) -> torch.Tensor:
    """``depth`` with pixels kept only where ``consistent_views`` source depths agree."""
    device = depth.device
    height, width = depth.shape
    rays = _rays(reference, device)
    flat = depth.reshape(-1)
    agree = torch.zeros_like(flat, dtype=torch.int32)
    pixels = torch.stack(
        torch.meshgrid(
            torch.arange(width, device=device, dtype=torch.float32),
            torch.arange(height, device=device, dtype=torch.float32),
            indexing="xy",
        )
    ).reshape(2, -1)
    for camera, source_depth in sources:
        rotation, translation = _relative(reference, camera, device)
        points = rotation @ (rays * flat) + translation[:, None]
        x, y, _ = _project(points, camera)
        found = _sample(source_depth, x.round(), y.round(), mode="nearest")
        found = torch.nan_to_num(found)
        k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
        back = (
            torch.stack(
                (
                    (x.round() - k[0, 2]) / k[0, 0],
                    (y.round() - k[1, 2]) / k[1, 1],
                    torch.ones_like(x),
                )
            )
            * found
        )
        returned = rotation.T @ (back - translation[:, None])
        rx, ry, rz = _project(returned, reference)
        moved = torch.hypot(rx - pixels[0], ry - pixels[1])
        relative = (rz - flat).abs() / flat.clamp_min(1e-9)
        agree += (
            (flat > 0)
            & (found > 0)
            & (moved <= config.reprojection_px)
            & (relative <= config.relative_depth)
        ).int()
    return torch.where(agree >= config.consistent_views, flat, 0.0).reshape(height, width)


def _grey(path: Path, size: tuple[int, int]) -> np.ndarray:
    import cv2

    rgb = read_rgb(path)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).mean(-1).astype(np.float32) / 255


def compute_mvs(solution, depth_dir, config: MvsConfig, device="cuda:0", log=print) -> dict:
    """Write ``<depth_dir>/mvs/``; a finished run (``mvs.json``) is reused."""
    target = Path(depth_dir) / "mvs"
    summary_path = target / "mvs.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())["summary"]
    target.mkdir(parents=True, exist_ok=True)

    stems = [p.stem for p in solution.image_paths]
    height, width = np.load(Path(depth_dir) / f"{stems[0]}.npy").shape
    cameras = [c.resized(width, height) for c in solution.cameras]
    plan = choose_sources(solution.cameras, solution.point_views, solution.points, config)
    greys: dict[int, torch.Tensor] = {}

    def grey(index):
        if index not in greys:
            image = _grey(solution.image_paths[index], (width, height))
            greys[index] = torch.from_numpy(image).to(device)
        return greys[index]

    raw = []
    with torch.no_grad():
        for index, (sources, near, far) in enumerate(plan):
            depth = plane_sweep(
                cameras[index],
                grey(index),
                [(cameras[s], grey(s)) for s in sources],
                near,
                far,
                config,
            )
            raw.append(depth.half().cpu())
            # Images are needed again only by later references that use them as sources.
            for old in [k for k in greys if k < index - 2 * config.sources]:
                del greys[old]
            if index % 100 == 0:
                log(f"mvs: plane sweep {index}/{len(plan)}")
        greys.clear()

        fits_path = Path(depth_dir) / "alignment.json"
        fits = json.loads(fits_path.read_text()) if fits_path.exists() else {}
        images = {}
        for index, (sources, _, _) in enumerate(plan):
            depth = (
                consistent(
                    cameras[index],
                    raw[index].to(device).float(),
                    [(cameras[s], raw[s].to(device).float()) for s in sources],
                    config,
                )
                .cpu()
                .numpy()
            )
            np.save(target / f"{stems[index]}.npy", depth.astype(np.float16))
            measured = depth > 0
            record = {"measured": float(measured.mean()), "sources": [int(s) for s in sources]}
            fit = fits.get(stems[index])
            if fit is not None and measured.any():
                mono = np.load(Path(depth_dir) / f"{stems[index]}.npy").astype(np.float32)
                fitted = 1 / np.maximum(fit["a"] * mono + fit["b"], 1e-9)
                record["mono_error"] = float(
                    np.median(np.abs(fitted[measured] - depth[measured]) / depth[measured])
                )
            images[stems[index]] = record

    shares = [r["measured"] for r in images.values()]
    errors = [r["mono_error"] for r in images.values() if "mono_error" in r]
    summary = {
        "images": len(images),
        "measured_median": float(np.median(shares)),
        "mono_error_median": float(np.median(errors)) if errors else None,
    }
    write_text_atomic(
        summary_path, json.dumps({"summary": summary, "images": images}, indent=1) + "\n"
    )
    log(f"mvs: {json.dumps(summary)}")
    return summary


def mvs_seeds(solution, depth_dir, train: list[int], count: int, max_long_side: int):
    """Up to ``count`` ``(points, colours, views)`` back-projected from the MVS depths."""
    folder = Path(depth_dir) / "mvs" if depth_dir is not None else None
    empty = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), []
    if count <= 0 or folder is None or not (folder / "mvs.json").exists():
        return empty
    images = train[:: max(1, len(train) // 100)]
    per_image = max(1, count // max(1, len(images)))
    rng = np.random.default_rng(0)
    points, colours, views = [], [], []
    for index in images:
        path = folder / f"{solution.image_paths[index].stem}.npy"
        if not path.exists():
            continue
        depth = np.load(path).astype(np.float32)
        ys, xs = np.nonzero(depth > 0)
        if not len(ys):
            continue
        pick = rng.choice(len(ys), min(per_image, len(ys)), replace=False)
        ys, xs = ys[pick], xs[pick]
        z = depth[ys, xs]
        height, width = depth.shape
        camera = solution.cameras[index].resized(width, height)
        k = camera.intrinsics
        local = np.stack(((xs - k[0, 2]) / k[0, 0] * z, (ys - k[1, 2]) / k[1, 1] * z, z), 1)
        points.append(local @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3])
        rgb = read_rgb(solution.image_paths[index], max_long_side)
        scale = rgb.shape[1] / width
        rows = np.clip((ys * scale).astype(int), 0, rgb.shape[0] - 1)
        columns = np.clip((xs * scale).astype(int), 0, rgb.shape[1] - 1)
        colours.append(rgb[rows, columns].astype(np.float32) / 255)
        views += [np.array([index])] * len(z)
    if not points:
        return empty
    return np.concatenate(points).astype(np.float32), np.concatenate(colours), views


def gap_seeds(
    solution,
    depth_dir,
    train: list[int],
    count: int,
    max_long_side: int,
    device: str = "cuda:0",
    neighbours: int = 2,
    reprojection_px: float = 2.0,
    relative_depth: float = 0.03,
):
    """Up to ``count`` ``(points, colours, views)`` from fitted monocular depth, only in gaps.

    A near door leaf, filmed for a second while walking through it, has no texture and
    lies closer than the SfM points, so neither SfM, MVS nor the monocular seeds (which
    keep to the SfM depth range) put a Gaussian there. MCMC only adds Gaussians next to
    existing ones, so the door never appeared, not even in its own training frames
    (vyas-room 00038–00041). Here a gap pixel (no MVS depth, away from depth edges) is
    seeded from the fitted monocular depth (near surfaces from metric depth,
    `spacegen.depth.near_inverse_depth`) when the fitted depths of at least 2 of the
    ``neighbours`` frames on each side agree with it, as in `consistent`.
    """
    folder = Path(depth_dir) if depth_dir is not None else None
    empty = np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), []
    fits_path = folder / "alignment.json" if folder is not None else None
    if count <= 0 or fits_path is None or not fits_path.exists():
        return empty
    fits = json.loads(fits_path.read_text())
    scales_path = folder / "metric" / "scale.json"
    scales = json.loads(scales_path.read_text()) if scales_path.exists() else {}
    stems = [p.stem for p in solution.image_paths]
    usable = [
        i for i in sorted(train) if stems[i] in fits and (folder / f"{stems[i]}.npy").exists()
    ]
    if not usable:
        return empty
    height, width = np.load(folder / f"{stems[usable[0]]}.npy").shape
    check = MvsConfig(
        consistent_views=2, reprojection_px=reprojection_px, relative_depth=relative_depth
    )
    per_image = max(1, count // len(usable))
    rng = np.random.default_rng(0)
    cache: dict[int, torch.Tensor] = {}

    def fitted(index):
        if index not in cache:
            mono = torch.from_numpy(np.load(folder / f"{stems[index]}.npy").astype(np.float32))
            fit = fits[stems[index]]
            inverse = fit["a"] * mono.to(device) + fit["b"]
            metric_path = folder / "metric" / f"{stems[index]}.npy"
            if metric_path.exists():
                metres = torch.from_numpy(np.load(metric_path).astype(np.float32)).to(device)
                inverse = near_inverse_depth(inverse, metres, scales.get(stems[index]))
            cache[index] = torch.where(inverse > 0, 1 / inverse.clamp_min(1e-9), 0.0)
        return cache[index]

    points, colours, views = [], [], []
    with torch.no_grad():
        for position, index in enumerate(usable):
            depth = fitted(index)
            camera = solution.cameras[index].resized(width, height)
            nearby = usable[max(0, position - neighbours) : position + neighbours + 1]
            sources = [
                (solution.cameras[i].resized(width, height), fitted(i))
                for i in nearby
                if i != index
            ]
            kept = consistent(camera, depth, sources, check)

            mvs_path = folder / "mvs" / f"{stems[index]}.npy"
            if mvs_path.exists():
                measured = torch.from_numpy(np.load(mvs_path).astype(np.float32)).to(device) > 0
                kept = torch.where(measured, 0.0, kept)
            inverse = torch.where(depth > 0, 1 / depth.clamp_min(1e-9), 0.0)
            gy, gx = torch.gradient(inverse)
            edge = torch.hypot(gx, gy) / inverse.clamp_min(1e-9) > 0.05
            kept = torch.where(edge, 0.0, kept).cpu().numpy()
            for old in [i for i in cache if i < usable[max(0, position - neighbours)]]:
                del cache[old]

            ys, xs = np.nonzero(kept > 0)
            if not len(ys):
                continue
            pick = rng.choice(len(ys), min(per_image, len(ys)), replace=False)
            ys, xs = ys[pick], xs[pick]
            z = kept[ys, xs]
            k = camera.intrinsics
            local = np.stack(((xs - k[0, 2]) / k[0, 0] * z, (ys - k[1, 2]) / k[1, 1] * z, z), 1)
            points.append(local @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3])
            rgb = read_rgb(solution.image_paths[index], max_long_side)
            scale = rgb.shape[1] / width
            rows = np.clip((ys * scale).astype(int), 0, rgb.shape[0] - 1)
            columns = np.clip((xs * scale).astype(int), 0, rgb.shape[1] - 1)
            colours.append(rgb[rows, columns].astype(np.float32) / 255)
            views += [np.array([index])] * len(z)
    if not points:
        return empty
    return np.concatenate(points).astype(np.float32), np.concatenate(colours), views
