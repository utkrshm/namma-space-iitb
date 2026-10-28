"""Merge splats trained on separate time blocks of one SfM model into one scene.

One model for a whole house has to share its Gaussian budget across every room; more
Gaussians made Coorg's full-house runs unstable, while a single block (74.5–180 s) reached
27.8 dB colour-corrected held-out PSNR. Blocks are trained with `time_range_s` on the
*same* SfM model, so they share one world frame. Like VastGaussian/CityGS, each block then
keeps only the Gaussians in its own region. The default ownership is visibility-weighted
(the block whose cameras saw a place best); nearest-camera ownership failed on Coorg
because a walk passes through rooms in several time windows.
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


@torch.no_grad()
def visibility_owner(scene, labels: np.ndarray, points: np.ndarray, device: str) -> np.ndarray:
    """Block whose cameras observe each point best: argmax over blocks of Σ 1/depth².

    1/depth² is proportional to the pixel area a surface covers, so this prefers the
    block that saw a place from closest and most often; a room walked past briefly in
    another block's time window does not take it over.
    """
    count = int(labels.max()) + 1
    world_to_camera = torch.tensor(
        np.stack([c.world_to_camera for c in scene.cameras]), dtype=torch.float32, device=device
    )
    intrinsics = torch.tensor(
        np.stack([c.intrinsics for c in scene.cameras]), dtype=torch.float32, device=device
    )
    sizes = torch.tensor(
        [[c.width, c.height] for c in scene.cameras], dtype=torch.float32, device=device
    )
    label_tensor = torch.tensor(labels, device=device)
    owners = []
    for start in range(0, len(points), 100_000):
        chunk = torch.tensor(points[start : start + 100_000], dtype=torch.float32, device=device)
        camera = torch.einsum("cij,nj->cni", world_to_camera[:, :3, :3], chunk)
        camera = camera + world_to_camera[:, None, :3, 3]
        depth = camera[..., 2]
        pixels = torch.einsum("cij,cnj->cni", intrinsics, camera)
        pixels = pixels[..., :2] / depth.clamp_min(1e-6)[..., None]
        visible = (depth > 0.05) & (pixels >= 0).all(-1) & (pixels < sizes[:, None]).all(-1)
        weight = torch.where(visible, 1 / depth.clamp_min(0.3) ** 2, 0.0)
        per_block = torch.zeros(count, len(chunk), device=device)
        per_block.index_add_(0, label_tensor, weight)
        owners.append(per_block.argmax(0).cpu().numpy())
    return np.concatenate(owners)


def merge_blocks(
    sfm_dir: str | Path,
    block_dirs: list[str | Path],
    output_dir: str | Path,
    ownership: str = "visibility",
    device: str = "cuda:0",
):
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
        means = params["means"].numpy()
        if ownership == "visibility":
            keep = visibility_owner(scene, labels, means, device) == label
        else:
            keep = block_owner(centres, labels, means) == label
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
        "ownership": ownership,
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
    parser.add_argument(
        "--ownership", choices=("visibility", "nearest-camera"), default="visibility"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    model, scene, metadata = merge_blocks(
        args.sfm_dir, args.blocks, args.output_dir, args.ownership, args.device
    )
    model = model.to(args.device)
    degree = metadata["active_sh_degree"]
    metadata["val_psnr_cc_db"] = held_out_psnr(
        model, scene, scene.val_indices, args.downscale, degree
    )
    write_json(args.output_dir / "merge.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("blocks", "val_psnr_cc_db")}, indent=1))
    return 0
