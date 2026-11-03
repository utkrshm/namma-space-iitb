"""Splat stage: one model for small captures, spatial blocks + merge for large ones.

Output: ``<run>/splat/scene.ply``, ``cameras.json`` and ``metrics.json`` (see
`spacegen.splat.train`). Large captures also leave ``<run>/blocks/<k>/`` behind.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .sfm import Solution
from .splat import train_splat


def build_splat(solution: Solution, output: Path, config: PipelineConfig, depth_dir, device, log):
    # Hold out by position in the *source* order, so unregistered frames do not shift the
    # split away from the standard every-8th-image one. Re-sliced extra frames
    # (`spacegen.reslice`) only ever train.
    every = config.splat.holdout_every
    val = [
        i
        for i, f in enumerate(solution.frames)
        if every and not f.get("extra") and f.get("position", f["index"]) % every == 0
    ]
    train = [i for i in range(len(solution.cameras)) if i not in set(val)]
    finished = (output / "splat" / "metrics.json").exists()
    if config.depth.metric_model and depth_dir is not None and not finished:
        # Metric depth for surfaces nearer than the SfM points (`spacegen.depth`). Here,
        # not in the depth stage, so that re-sliced extra frames get it too.
        from .depth import align_metric_depth, compute_depth

        compute_depth(
            solution.image_paths,
            Path(depth_dir) / "metric",
            config.depth,
            device,
            model_name=config.depth.metric_model,
        )
        align_metric_depth(solution, depth_dir)
    if (
        (
            config.splat.depth_seed_points
            or config.splat.mvs_seed_points
            or config.splat.gap_seed_points
        )
        and depth_dir is not None
        and not finished
    ):
        from .splat.init import with_depth_seeds

        solution = with_depth_seeds(
            solution,
            depth_dir,
            train,
            config.splat.depth_seed_points,
            config.splat.max_long_side,
            log,
            mvs_count=config.splat.mvs_seed_points,
            gap_count=config.splat.gap_seed_points,
            device=device,
        )
    reflectors = None
    if config.splat.reflections and not finished:
        from .reflectors import find_reflectors

        reflectors = find_reflectors(
            solution,
            train,
            depth_dir,
            device,
            log,
            glass=config.splat.glass_reflectors,
            floor=config.splat.floor_reflector,
        )
    if reflectors is not None:
        # Points seen through a floor or mirror are reflections triangulated as a room
        # behind the mirror (or under the floor); the reflection layer renders those
        # instead. A point is dropped when most cameras that matched it saw it through the
        # plane, so a real room behind the mirror's wall keeps its points.
        import torch

        from .splat.reflect import reflectors_from_dict, seen_through_opaque

        lengths = np.array([len(views) for views in solution.point_views])
        owner = np.repeat(np.arange(len(lengths)), lengths)
        views = np.concatenate(solution.point_views) if len(owner) else np.zeros(0, int)
        centres = np.stack([camera.centre for camera in solution.cameras])
        through = seen_through_opaque(
            torch.from_numpy(solution.points[owner]).float(),
            torch.from_numpy(centres[views]).float(),
            reflectors_from_dict(reflectors, "cpu"),
        ).numpy()
        share = np.bincount(owner, through, len(lengths)) / np.maximum(lengths, 1)
        keep = share < 0.5
        log(f"reflectors: {int((~keep).sum())} of {len(keep)} seed points lie behind a plane")
        solution = replace(
            solution,
            points=solution.points[keep],
            colors=solution.colors[keep],
            point_views=[v for v, k in zip(solution.point_views, keep, strict=True) if k],
        )
    if len(train) > config.partition.max_images_per_block:
        from .partition import train_in_blocks

        return train_in_blocks(
            solution, output, config, depth_dir, train, val, device, log, reflectors
        )
    return train_splat(
        solution.cameras,
        solution.image_paths,
        solution.points,
        solution.colors,
        output / "splat",
        config.splat,
        depth_dir=depth_dir,
        train_indices=train,
        val_indices=val,
        reflectors=reflectors,
        device=device,
        log=log,
    )
