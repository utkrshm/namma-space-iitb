"""Merge splats trained on separate time blocks of one SfM model into one scene.

One model for a whole house has to share its Gaussian budget across every room; more
Gaussians made Coorg's full-house runs unstable, while a single block (74.5–180 s) reached
27.8 dB colour-corrected held-out PSNR. Blocks are trained with `time_range_s` on the
*same* SfM model, so they share one world frame. Like VastGaussian/CityGS, each block then
keeps only the Gaussians in its own region: a Gaussian is kept by the block that owns the
camera nearest to it (a partition of space by the camera trajectory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from .model import GaussianModel
from .sfm_scene import load_sfm_scene
from .storage import export_ply, write_json


def block_owner(centres: np.ndarray, labels: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Block label of the camera nearest to each point."""
    _, nearest = cKDTree(centres).query(points)
    return labels[nearest]


def merge_blocks(sfm_dir: str | Path, block_dirs: list[str | Path], output_dir: str | Path):
    scene = load_sfm_scene(sfm_dir)
    centres = np.stack([camera.camera_to_world[:3, 3] for camera in scene.cameras])
    labels = np.full(len(centres), -1)
    ranges = []
    for label, block in enumerate(block_dirs):
        run = json.loads((Path(block) / "run.json").read_text())
        start, end = run["sfm_splat_config"]["data"]["time_range_s"]
        ranges.append((start, end))
        labels[(scene.timestamps >= start) & (scene.timestamps <= end)] = label
    if (labels < 0).any():
        raise ValueError("Every camera must belong to one block's time range")

    merged, counts = {}, []
    for label, block in enumerate(block_dirs):
        state = torch.load(Path(block) / "checkpoint.pt", map_location="cpu", weights_only=True)
        params = state["params"]
        keep = block_owner(centres, labels, params["means"].numpy()) == label
        counts.append({"block": str(block), "kept": int(keep.sum()), "total": len(keep)})
        for key, value in params.items():
            merged.setdefault(key, []).append(value[torch.from_numpy(keep)])
        degree = state["sh_degree"]
    model = GaussianModel({key: torch.cat(value) for key, value in merged.items()}, degree)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "trainer": "src.splatting.merge_blocks",
        "sfm_dir": str(Path(sfm_dir).resolve()),
        "blocks": counts,
        "time_ranges_s": ranges,
        "active_sh_degree": degree,
        "initial_camera": scene.cameras[scene.train_indices[0]].to_dict(),
    }
    export_ply(output / "scene.ply", model, metadata)
    write_json(output / "merge.json", metadata)
    return model, scene, metadata


def main(argv=None) -> int:
    from .web_export import held_out_psnr

    parser = argparse.ArgumentParser(description="Merge time-block splats into one scene")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("blocks", type=Path, nargs="+", help="Block run directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    model, scene, metadata = merge_blocks(args.sfm_dir, args.blocks, args.output_dir)
    model = model.to(args.device)
    degree = metadata["active_sh_degree"]
    metadata["val_psnr_cc_db"] = held_out_psnr(
        model, scene, scene.val_indices, args.downscale, degree
    )
    write_json(args.output_dir / "merge.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("blocks", "val_psnr_cc_db")}, indent=1))
    return 0
