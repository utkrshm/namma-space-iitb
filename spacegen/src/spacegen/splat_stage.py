"""Splat stage: one model for small captures, spatial blocks + merge for large ones.

Output: ``<run>/splat/scene.ply``, ``cameras.json`` and ``metrics.json`` (see
`spacegen.splat.train`). Large captures also leave ``<run>/blocks/<k>/`` behind.
"""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .sfm import Solution
from .splat import train_splat


def build_splat(solution: Solution, output: Path, config: PipelineConfig, depth_dir, device, log):
    # Hold out by position in the *source* order, so unregistered frames do not shift the
    # split away from the standard every-8th-image one.
    every = config.splat.holdout_every
    val = [
        i
        for i, f in enumerate(solution.frames)
        if every and f.get("position", f["index"]) % every == 0
    ]
    train = [i for i in range(len(solution.cameras)) if i not in set(val)]
    if (
        config.splat.depth_seed_points
        and depth_dir is not None
        and not (output / "splat" / "metrics.json").exists()
    ):
        from .splat.init import with_depth_seeds

        solution = with_depth_seeds(
            solution,
            depth_dir,
            train,
            config.splat.depth_seed_points,
            config.splat.max_long_side,
            log,
        )
    if len(train) > config.partition.max_images_per_block:
        from .partition import train_in_blocks

        return train_in_blocks(solution, output, config, depth_dir, train, val, device, log)
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
        device=device,
        log=log,
    )
