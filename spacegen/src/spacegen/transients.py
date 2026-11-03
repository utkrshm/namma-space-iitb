"""Pixels that most other views see straight through: moving content, or a wrong depth.

A splat is one static scene. A door that swings shut, a person walking past or a chair
moved between two passes is in some frames and not in others. Training then fights
over it. On vyas-room the open door leaf of 00038–00044 (the door closer shut it, and
00422–00430 film it closed) was voted out by the 45 frames that look through where it
had been. Hard-view sampling kept returning to the door frames, and what was left was a
dark haze over the room.

The check uses only the depth targets that training already trusts (`splat.train.Priors`:
fitted monocular depth, metric depth for near surfaces, MVS, planes). For every
training image, points on a ``stride``-pixel grid of its target, away from depth edges,
are projected into every other training image. A view *sees through* a point when its own
target there lies beyond ``ratio`` × the point's depth. A pixel is flagged when at least
``min_views`` views see through its point, and they are at least ``share`` of the views
that see it at all. The flags are then opened with a 3 × 3 kernel on the grid: thin
structures, which monocular depth misses in some views, are not flagged. Held-out
images are neither flagged nor used as evidence. This is the visibility test of
depth-map fusion (Merrell et al. 2007) applied to the training targets.

A flag means "do not trust this pixel". Usually it is moving content. Sometimes it is
a monocular depth that other views contradict, such as a wall strip at a frame's edge
in vyas-room 00427, which 12–16 views saw 1.9× beyond. Training leaves flagged pixels out of
the photometric and depth losses, and seed points from them are dropped.

Output: ``<depth>/transient/<stem>.npy`` (uint8 mask at the depth-map size) and
``transient.json`` (parameters, inputs and the flagged share per image).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .contracts import file_stamp, write_atomic, write_text_atomic


def free_space_flags(
    inverse: torch.Tensor,
    cameras,
    min_views: int,
    share: float,
    ratio: float,
    stride: int = 4,
) -> list[torch.Tensor | None]:
    """Per camera, a bool mask [H,W] of flagged pixels (None without a depth target).

    ``inverse`` [V,H,W] holds each camera's inverse-depth target (0 = none); ``cameras``
    are the matching `Camera` objects (any size: they are resized to the maps)."""
    count, height, width = inverse.shape
    device = inverse.device
    resized = [camera.resized(width, height) for camera in cameras]
    rotation = torch.tensor(
        np.stack([c.world_to_camera[:3, :3] for c in resized]), dtype=torch.float32, device=device
    )
    translation = torch.tensor(
        np.stack([c.world_to_camera[:3, 3] for c in resized]), dtype=torch.float32, device=device
    )
    k = torch.tensor(np.stack([c.intrinsics for c in resized]), dtype=torch.float32, device=device)
    ys, xs = torch.meshgrid(
        torch.arange(stride // 2, height, stride, device=device),
        torch.arange(stride // 2, width, stride, device=device),
        indexing="ij",
    )
    views = torch.arange(count, device=device)[:, None]

    flags: list[torch.Tensor | None] = []
    for view in range(count):
        target = inverse[view]
        if not (target > 0).any():
            flags.append(None)
            continue
        gy, gx = torch.gradient(target)
        smooth = torch.hypot(gx, gy) / target.clamp_min(1e-6) < 0.05
        grid = target[ys, xs]
        valid = (grid > 0) & smooth[ys, xs]
        depth = 1 / grid[valid].clamp_min(1e-9)
        x, y = xs[valid].float(), ys[valid].float()
        own = k[view]
        local = torch.stack(
            ((x - own[0, 2]) / own[0, 0] * depth, (y - own[1, 2]) / own[1, 1] * depth, depth), -1
        )
        world = (local - translation[view]) @ rotation[view]

        seen = world @ rotation.transpose(1, 2) + translation[:, None]  # [V,P,3]
        distance = seen[..., 2]
        u = seen[..., 0] / distance.clamp_min(1e-6) * k[:, None, 0, 0] + k[:, None, 0, 2]
        v = seen[..., 1] / distance.clamp_min(1e-6) * k[:, None, 1, 1] + k[:, None, 1, 2]
        inside = (distance > 1e-3) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
        found = inverse[
            views, v.round().long().clamp(0, height - 1), u.round().long().clamp(0, width - 1)
        ]
        inside &= found > 0
        inside[view] = False
        through = inside & (found * ratio * distance < 1)
        votes, visible = through.sum(0), inside.sum(0)

        flagged = torch.zeros_like(grid, dtype=torch.bool)
        flagged[valid] = (votes >= min_views) & (votes >= share * visible)
        # Opening (erode, then dilate) on the grid: only regions at least 3 cells wide.
        cells = flagged[None, None].float()
        cells = 1 - F.max_pool2d(1 - cells, 3, stride=1, padding=1)
        cells = F.max_pool2d(cells, 3, stride=1, padding=1)
        flags.append(F.interpolate(cells, size=(height, width), mode="nearest")[0, 0] > 0)
    return flags


def find_transients(solution, depth_dir, train: list[int], config, device: str, log) -> dict:
    """Write the masks of the ``train`` images (`SplatConfig` ``transient_*`` settings);
    reused while the parameters, the images and the depth fits are unchanged."""
    from .splat.train import Priors

    folder = Path(depth_dir) / "transient"
    summary_path = folder / "transient.json"
    stems = [solution.image_paths[i].stem for i in train]
    inputs = {
        name: file_stamp(Path(depth_dir) / name)
        for name in ("alignment.json", "metric/scale.json")
        if (Path(depth_dir) / name).exists()
    }
    parameters = {
        "min_views": config.transient_views,
        "share": config.transient_share,
        "ratio": config.transient_ratio,
        "planar_prior": config.planar_prior,
    }
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        same = summary.get("parameters") == parameters and summary.get("inputs") == inputs
        if same and sorted(summary.get("images", {})) == sorted(stems):
            log(f"transients: reusing {summary_path}")
            return summary
    folder.mkdir(parents=True, exist_ok=True)

    priors = Priors(depth_dir, solution.image_paths, config.planar_prior)
    maps = []
    for index in train:
        prior = priors.get(index, device)
        maps.append(None if prior is None or prior[1] is None else prior[1])
    shape = next((m.shape for m in maps if m is not None), None)
    if shape is None:
        log("transients: no depth targets; nothing flagged")
        return {"parameters": parameters, "inputs": inputs, "images": {}}
    inverse = torch.stack([m if m is not None else torch.zeros(shape, device=device) for m in maps])
    flags = free_space_flags(
        inverse,
        [solution.cameras[i] for i in train],
        config.transient_views,
        config.transient_share,
        config.transient_ratio,
    )

    shares = {}
    for stem, flag in zip(stems, flags, strict=True):
        mask = np.zeros(shape, np.uint8) if flag is None else flag.cpu().numpy().astype(np.uint8)
        write_atomic(folder / f"{stem}.npy", lambda stream, mask=mask: np.save(stream, mask))
        shares[stem] = float(mask.mean())
    summary = {"parameters": parameters, "inputs": inputs, "images": shares}
    write_text_atomic(summary_path, json.dumps(summary, indent=1) + "\n")
    flagged = {s: v for s, v in shares.items() if v > 0}
    worst = sorted(flagged, key=flagged.get, reverse=True)[:5]
    log(
        f"transients: {len(flagged)} of {len(shares)} images have flagged pixels"
        + (f"; most in {', '.join(f'{s} {flagged[s]:.0%}' for s in worst)}" if worst else "")
    )
    return summary


class TransientMasks:
    """`find_transients` masks per image, loaded lazily (None where an image has none)."""

    def __init__(self, depth_dir, image_paths):
        folder = Path(depth_dir) / "transient"
        self.paths = [folder / f"{Path(p).stem}.npy" for p in image_paths]
        self.cache: dict[int, torch.Tensor | None] = {}

    def get(self, index: int, device) -> torch.Tensor | None:
        if index not in self.cache:
            path = self.paths[index]
            mask = np.load(path).astype(bool) if path.exists() else None
            self.cache[index] = None if mask is None or not mask.any() else torch.from_numpy(mask)
        mask = self.cache[index]
        return None if mask is None else mask.to(device)


def drop_flagged_seeds(solution, first: int, depth_dir, log):
    """``solution`` without the seed points (index ≥ ``first``, one source view each) that
    lie on flagged pixels of their source image."""
    from dataclasses import replace

    masks = TransientMasks(depth_dir, solution.image_paths)
    keep = np.ones(len(solution.points), bool)
    sources = np.array([views[0] if len(views) else -1 for views in solution.point_views[first:]])
    for source in np.unique(sources):
        mask = masks.get(int(source), "cpu") if source >= 0 else None
        if mask is None:
            continue
        chosen = first + np.flatnonzero(sources == source)
        height, width = mask.shape
        camera = solution.cameras[int(source)].resized(width, height)
        local = solution.points[chosen] @ camera.world_to_camera[:3, :3].T
        local = local + camera.world_to_camera[:3, 3]
        pixel = local[:, :2] / np.maximum(local[:, 2:3], 1e-9) * np.diag(camera.intrinsics)[:2]
        pixel = np.round(pixel + camera.intrinsics[:2, 2]).astype(int)
        x, y = pixel[:, 0].clip(0, width - 1), pixel[:, 1].clip(0, height - 1)
        keep[chosen] = ~mask.numpy()[y, x]
    if keep.all():
        return solution
    log(f"transients: {int((~keep).sum())} seed points on flagged pixels dropped")
    return replace(
        solution,
        points=solution.points[keep],
        colors=solution.colors[keep],
        point_views=[v for v, k in zip(solution.point_views, keep, strict=True) if k],
    )
