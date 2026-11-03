"""Reflector planes for the splat's reflection layer (`spacegen.splat.reflect`).

1. **Floor.** Always a candidate: polished stone, tiles and wooden floors mirror the room,
   and a floor that reflects too little learns a near-zero strength. The plane comes from
   `spacegen.scene_frame` (RANSAC on the SfM points below the cameras). It is *opaque*:
   nothing real lies under it. When no floor plane is found, only step 2 runs.
2. **Windows, glass doors and mirrors.** A semantic segmenter (Mask2Former trained on
   ADE20K) marks ``windowpane``, ``screen door``, ``glass`` and ``mirror`` pixels in up to
   ``max_frames`` training frames. Depth on glass is unreliable (stereo sees the reflection
   or the view through it), so each region's plane is fitted (RANSAC, vertical normals
   only) to the depth just *around* it: the wall and frame it sits in. The region's own
   pixels, cast onto that plane, give its extent. Planes found in at least ``min_frames``
   frames, within 12° and 0.1 camera heights of each other, are merged. Mirrors are
   opaque; windows and glass doors are not (what is seen through them is real).

Depth on each frame is multi-view stereo where measured, else the fitted monocular depth.

Output (also cached as ``<depth_dir>/reflectors.json``, with ``reflectors.jpg`` showing
the regions used): ``{"planes" [[nx, ny, nz, d]], "tau", "kinds", "axes", "bounds",
"opaque", "frames"}`` in world (SfM) units (the cache adds ``key``: the settings and
solution size it was made for), with ``n · x + d = 0`` and ``n`` pointing to
the cameras' side; ``tau`` (0.03 camera heights) is how far from a plane a Gaussian can
still carry its reflection. The fields are those of `spacegen.splat.reflect.make_reflectors`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .depth import near_inverse_depth, read_rgb
from .scene_frame import estimate_frame
from .sfm import Solution

SEGMENTER = "facebook/mask2former-swin-large-ade-semantic"
# ADE20K labels that reflect, and whether they are opaque (nothing real behind them).
GLASS = {"windowpane": False, "screen door": False, "glass": False, "mirror": True}
NO_BOUNDS = [-1e30, 1e30, -1e30, 1e30]


def find_reflectors(
    solution: Solution,
    train: list[int],
    depth_dir,
    device: str = "cuda:0",
    log=print,
    glass: bool = True,
    max_frames: int = 200,
    min_frames: int = 3,
    floor: bool = True,
) -> dict | None:
    """The planes (see the module docstring); ``floor=False`` drops the floor plane, which
    is still estimated for the up direction and scale. None when no plane is left."""
    # The cache is reused only for the same solution and settings: the re-slice preview
    # runs before the extra frames are added, and glass can be switched off.
    key = {"glass": glass, "cameras": len(solution.cameras), "points": len(solution.points)}
    cache = Path(depth_dir) / "reflectors.json" if depth_dir is not None else None
    if cache is not None and cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("key") == key:
            return _without_floor(cached, floor)

    cameras = [solution.cameras[i] for i in train]
    frame = estimate_frame(cameras, solution.points)
    height_unit = frame.camera_height
    up = np.asarray(frame.up, dtype=np.float64)
    offset = -float(up @ frame.origin)
    centres = np.stack([c.centre for c in cameras])
    if np.median(centres @ up + offset) < 0:
        up, offset = -up, -offset
    result = {"key": key, "planes": [], "tau": 0.03 * height_unit, "kinds": [], "axes": []}
    result.update({"bounds": [], "opaque": [], "frames": []})
    if frame.floor_found:
        result["planes"].append([*up.tolist(), offset])
        result["kinds"].append("floor")
        result["axes"].append([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result["bounds"].append(NO_BOUNDS)
        result["opaque"].append(True)
        result["frames"].append(len(train))
    else:
        # Mirrors and windows need only the up direction and the scale.
        log("reflectors: no floor plane found; no floor reflector")

    if glass and depth_dir is not None and (Path(depth_dir) / "alignment.json").exists():
        found = _glass_planes(
            solution, train, Path(depth_dir), up, height_unit, device, log, max_frames
        )
        for plane in found:
            if plane["frames"] < min_frames:
                continue
            result["planes"].append(plane["plane"])
            result["kinds"].append(plane["kind"])
            result["axes"].append(plane["axes"])
            result["bounds"].append(plane["bounds"])
            result["opaque"].append(GLASS[plane["kind"]])
            result["frames"].append(plane["frames"])
            result.setdefault("stems", {})[len(result["planes"]) - 1] = plane["stems"]
    log(
        "reflectors: "
        + (
            ", ".join(
                f"{k} ({n} frames)" for k, n in zip(result["kinds"], result["frames"], strict=True)
            )
            or "none"
        )
    )
    if cache is not None:
        cache.write_text(json.dumps(result, indent=1) + "\n")
    if not result["planes"]:
        return None
    return _without_floor(result, floor)


def _without_floor(result: dict, floor: bool) -> dict | None:
    if floor:
        return result
    keep = [i for i, kind in enumerate(result["kinds"]) if kind != "floor"]
    if not keep:
        return None
    trimmed = {
        key: [value[i] for i in keep]
        for key, value in result.items()
        if isinstance(value, list) and len(value) == len(result["kinds"])
    }
    stems = {str(k): v for k, v in result.get("stems", {}).items()}
    trimmed["stems"] = {
        str(new): stems[str(old)] for new, old in enumerate(keep) if str(old) in stems
    }
    return {**result, **trimmed}


def _frame_depth(depth_dir: Path, stem: str, fits: dict) -> np.ndarray | None:
    """z-depth on the depth grid: MVS where measured, else the fitted monocular depth."""
    path = depth_dir / f"{stem}.npy"
    if not path.exists() or stem not in fits:
        return None
    inverse = fits[stem]["a"] * np.load(path).astype(np.float32) + fits[stem]["b"]
    metric_path = depth_dir / "metric" / f"{stem}.npy"
    scales_path = depth_dir / "metric" / "scale.json"
    if metric_path.exists() and scales_path.exists():
        entry = json.loads(scales_path.read_text()).get(stem)
        inverse = near_inverse_depth(inverse, np.load(metric_path).astype(np.float32), entry)
    depth = np.where(inverse > 0, 1 / np.maximum(inverse, 1e-9), 0.0)
    mvs_path = depth_dir / "mvs" / f"{stem}.npy"
    if mvs_path.exists():
        measured = np.load(mvs_path).astype(np.float32)
        depth = np.where(measured > 0, measured, depth)
    return depth


def _vertical_plane(points: np.ndarray, up: np.ndarray, threshold: float, rng):
    """RANSAC plane with a normal within ~15° of horizontal: ``(normal, offset, inliers)``."""
    best = (None, 0.0, np.zeros(len(points), bool))
    for _ in range(300):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        if abs(normal @ up) > 0.25:
            continue
        inliers = np.abs((points - sample[0]) @ normal) < threshold
        if inliers.sum() > best[2].sum():
            best = (normal, -float(normal @ sample[0]), inliers)
    normal, _, inliers = best
    if normal is None or inliers.sum() < 30:
        return best
    chosen = points[inliers]
    centre = chosen.mean(0)
    _, _, axes = np.linalg.svd(chosen - centre, full_matrices=False)
    normal = axes[2] - (axes[2] @ up) * up  # keep it exactly vertical
    normal /= np.linalg.norm(normal)
    return normal, -float(normal @ centre), inliers


def _glass_planes(solution, train, depth_dir, up, height_unit, device, log, max_frames):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(SEGMENTER)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEGMENTER)
    model = model.to(device).eval().half()
    names = {i: label.split(",")[0].strip() for i, label in model.config.id2label.items()}
    wanted = {i: n for i, n in names.items() if n in GLASS}
    fits = json.loads((depth_dir / "alignment.json").read_text())
    rng = np.random.default_rng(0)
    frames = sorted(train)[:: max(1, math.ceil(len(train) / max_frames))]
    candidates, previews = [], []

    for index in frames:
        stem = solution.image_paths[index].stem
        depth = _frame_depth(depth_dir, stem, fits)
        if depth is None:
            continue
        rows, columns = depth.shape
        rgb = read_rgb(solution.image_paths[index], 640)
        with torch.no_grad():
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            output = model(pixel_values=inputs["pixel_values"].half())
            labels = (
                processor.post_process_semantic_segmentation(
                    output, target_sizes=[(rows, columns)]
                )[0]
                .cpu()
                .numpy()
            )
        any_glass = np.isin(labels, list(wanted))
        if any_glass.mean() < 0.01:
            continue
        camera = solution.cameras[index].resized(columns, rows)
        k = camera.intrinsics
        ys, xs = np.mgrid[0:rows, 0:columns]
        rays = np.stack(
            ((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones_like(xs, float)), -1
        )
        world_rays = rays @ camera.camera_to_world[:3, :3].T
        centre = camera.centre
        used = np.zeros_like(any_glass)
        for label, kind in wanted.items():
            count, parts = cv2.connectedComponents((labels == label).astype(np.uint8))
            for part in range(1, count):
                region = parts == part
                if region.mean() < 0.005:
                    continue
                grown = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=3)
                ring = grown.astype(bool) & ~any_glass & (depth > 0)
                if ring.sum() < 40:
                    continue
                points = centre + world_rays[ring] * depth[ring][:, None]
                normal, offset, inliers = _vertical_plane(points, up, 0.02 * height_unit, rng)
                if normal is None or inliers.mean() < 0.4:
                    continue
                if normal @ centre + offset < 0:
                    normal, offset = -normal, -offset
                # The region's pixels, cast onto the plane, give its extent.
                along = world_rays[region] @ normal
                hit = -(centre @ normal + offset) / np.where(np.abs(along) > 1e-6, along, np.nan)
                inside = np.isfinite(hit) & (hit > 0)
                if inside.sum() < 20:
                    continue
                surface = centre + world_rays[region][inside] * hit[inside][:, None]
                pick = rng.choice(len(surface), min(400, len(surface)), replace=False)
                candidates.append(
                    {
                        "kind": kind,
                        "normal": normal,
                        "offset": offset,
                        "points": surface[pick],
                        "stem": stem,
                    }
                )
                used |= region
        if used.any() and len(previews) < 8:
            preview = cv2.resize(rgb, (columns, rows), interpolation=cv2.INTER_AREA)
            preview[used] = (0.4 * preview[used] + [0, 150, 255]).clip(0, 255).astype(np.uint8)
            previews.append(preview)

    planes = _merge(candidates, up, height_unit)
    if previews:
        height = min(p.shape[0] for p in previews)
        sheet = np.concatenate([p[:height] for p in previews], 1)
        cv2.imwrite(str(depth_dir / "reflectors.jpg"), sheet[..., ::-1])
    log(f"reflectors: {len(candidates)} glass/mirror regions in {len(frames)} frames")
    del model
    torch.cuda.empty_cache()
    return planes


def _merge(candidates, up, height_unit, limit: int = 4):
    """Group per-frame planes (within 12° and 0.1 camera heights, overlapping extents)."""
    clusters: list[list[dict]] = []
    for candidate in candidates:
        for cluster in clusters:
            normal = np.mean([c["normal"] for c in cluster], 0)
            normal /= np.linalg.norm(normal)
            if candidate["normal"] @ normal < math.cos(math.radians(12)):
                continue
            points = np.concatenate([c["points"] for c in cluster])
            offset = -float(np.median(points @ normal))
            if abs(-float(np.median(candidate["points"] @ normal)) - offset) > 0.1 * height_unit:
                continue
            gap = np.linalg.norm(candidate["points"].mean(0) - points.mean(0))
            spread = np.linalg.norm(points.max(0) - points.min(0))
            if gap > spread / 2 + 0.5 * height_unit:
                continue
            cluster.append(candidate)
            break
        else:
            clusters.append([candidate])

    planes = []
    for cluster in clusters:
        normal = np.mean([c["normal"] for c in cluster], 0)
        normal -= (normal @ up) * up
        normal /= np.linalg.norm(normal)
        points = np.concatenate([c["points"] for c in cluster])
        offset = -float(np.median(points @ normal))
        u = np.cross(up, normal)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        margin = 0.02 * height_unit
        low_u, high_u = np.percentile(points @ u, [2, 98])
        low_v, high_v = np.percentile(points @ v, [2, 98])
        kinds = [c["kind"] for c in cluster]
        planes.append(
            {
                "plane": [*normal.tolist(), offset],
                "kind": max(set(kinds), key=kinds.count),
                "axes": [u.tolist(), v.tolist()],
                "bounds": [low_u - margin, high_u + margin, low_v - margin, high_v + margin],
                # Distinct frames: one frame can give several regions (a window split
                # by its mullions), which must not pass the min_frames gate alone.
                "frames": len({c["stem"] for c in cluster}),
                "stems": sorted({c["stem"] for c in cluster}),
            }
        )
    planes.sort(key=lambda p: -p["frames"])
    return planes[:limit]
