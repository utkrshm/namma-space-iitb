"""Large captures: train spatial blocks, merge them by visibility, fine-tune jointly.

One Gaussian budget shared by a whole house leaves every room short. On Coorg, budgets
above 1.6M were unstable, and the best single model scored 21.6 dB against 27.6 dB for
blocks. Coorg's blocks were typed-in time ranges; here they are found automatically:

1. **Regions:** balanced median cuts of the training camera centres (`balanced_split`), with
   ``k = ceil(train images / max_images_per_block)``.
2. **Block cameras:** a block trains on its own cameras, plus any other camera that
   observes at least ``overlap`` of the SfM points its own cameras see. Its borders are
   then trained from both sides, as in VastGaussian.
3. **Merge:** each Gaussian goes to the block whose own cameras observe its position
   best (argmax of Σ 1/depth² over visible cameras). Nearest-camera ownership failed on
   Coorg, because a walk passes through the same room at several times.
4. **Joint fine-tune** of the merged model on every training image, at full resolution.

Output: ``<run>/blocks/<k>/`` (block runs), ``<run>/blocks/merged.ply``, and the final
``<run>/splat/`` as for a single model.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import PipelineConfig
from .sfm import Solution
from .splat import load_ply, save_ply, train_splat
from .splat.model import GaussianModel


def balanced_split(points: np.ndarray, count: int) -> np.ndarray:
    """Labels for ``count`` spatial blocks of near-equal size.

    Cameras are cut along their widest axis in proportion to the blocks on each side
    (k-d style), recursively. k-means was tried first: on a 991-frame apartment it made
    blocks of 854 and 13 cameras (the balcony), which defeats the point of blocks.
    """
    labels = np.zeros(len(points), int)

    def split(members: np.ndarray, first: int, blocks: int) -> None:
        if blocks == 1:
            labels[members] = first
            return
        left = blocks // 2
        axis = np.ptp(points[members], axis=0).argmax()
        order = members[np.argsort(points[members, axis], kind="stable")]
        cut = round(len(order) * left / blocks)
        split(order[:cut], first, left)
        split(order[cut:], first + left, blocks - left)

    split(np.arange(len(points)), 0, count)
    return labels


def block_cameras(solution: Solution, labels: np.ndarray, train: list[int], overlap: float):
    """Training camera indices per block: own cameras plus strongly overlapping ones."""
    views_of_point = solution.point_views
    count = int(labels.max()) + 1
    per_camera_points: list[set[int]] = [set() for _ in solution.cameras]
    for point, views in enumerate(views_of_point):
        for view in views:
            per_camera_points[view].add(point)
    train_set = set(train)
    blocks = []
    for block in range(count):
        own = [i for i in train if labels[i] == block]
        points = set().union(*(per_camera_points[i] for i in own)) if own else set()
        extra = [
            i
            for i in train_set - set(own)
            if per_camera_points[i]
            and len(per_camera_points[i] & points) >= overlap * len(per_camera_points[i])
        ]
        blocks.append({"own": own, "cameras": sorted(own + extra), "points": sorted(points)})
    return blocks


@torch.no_grad()
def visibility_owner(cameras, labels: np.ndarray, points: np.ndarray, device: str) -> np.ndarray:
    """Block whose own cameras observe each point best: argmax over blocks of Σ 1/depth²."""
    count = int(labels.max()) + 1
    world_to_camera = torch.tensor(
        np.stack([c.world_to_camera for c in cameras]), dtype=torch.float32, device=device
    )
    intrinsics = torch.tensor(
        np.stack([c.intrinsics for c in cameras]), dtype=torch.float32, device=device
    )
    sizes = torch.tensor([[c.width, c.height] for c in cameras], dtype=torch.float32, device=device)
    label_tensor = torch.tensor(labels, device=device)
    owners = []
    for start in range(0, len(points), 50_000):
        chunk = torch.tensor(points[start : start + 50_000], dtype=torch.float32, device=device)
        local = (
            torch.einsum("cij,nj->cni", world_to_camera[:, :3, :3], chunk)
            + world_to_camera[:, None, :3, 3]
        )
        depth = local[..., 2]
        pixels = (
            torch.einsum("cij,cnj->cni", intrinsics, local)[..., :2]
            / depth.clamp_min(1e-6)[..., None]
        )
        visible = (depth > 0) & (pixels >= 0).all(-1) & (pixels < sizes[:, None]).all(-1)
        weight = torch.where(visible, 1 / depth.clamp_min(1e-3) ** 2, 0.0)
        per_block = torch.zeros(count, len(chunk), device=device)
        per_block.index_add_(0, label_tensor, weight)
        owners.append(per_block.argmax(0).cpu().numpy())
    return np.concatenate(owners) if owners else np.zeros(0, int)


def train_in_blocks(
    solution: Solution,
    output: Path,
    config: PipelineConfig,
    depth_dir,
    train,
    val,
    device,
    log,
    reflectors=None,
):
    final = output / "splat"
    if (final / "metrics.json").exists():
        return json.loads((final / "metrics.json").read_text())
    centres = np.stack([c.centre for c in solution.cameras])
    count = math.ceil(len(train) / config.partition.max_images_per_block)
    labels = np.full(len(solution.cameras), -1)
    labels[train] = balanced_split(centres[train], count)
    # Held-out cameras join the block of their nearest training camera.
    for index in val:
        nearest = train[int(np.linalg.norm(centres[train] - centres[index], axis=1).argmin())]
        labels[index] = labels[nearest]
    blocks = block_cameras(solution, labels, train, config.partition.overlap)
    folder = output / "blocks"
    folder.mkdir(parents=True, exist_ok=True)
    summary = []
    for number, block in enumerate(blocks):
        log(
            f"block {number}: {len(block['own'])} own + "
            f"{len(block['cameras']) - len(block['own'])} shared cameras"
        )
        points = block["points"] or range(len(solution.points))
        block_val = [i for i in val if labels[i] == number]
        metrics = train_splat(
            solution.cameras,
            solution.image_paths,
            solution.points[list(points)],
            solution.colors[list(points)],
            folder / str(number),
            config.splat,
            depth_dir=depth_dir,
            train_indices=block["cameras"],
            val_indices=block_val,
            reflectors=reflectors,
            device=device,
            log=log,
        )
        summary.append(
            {"own": len(block["own"]), "cameras": len(block["cameras"]), "val": metrics.get("val")}
        )
        torch.cuda.empty_cache()

    merged_path = folder / "merged.ply"
    if not merged_path.exists():
        parts = []
        for number in range(len(blocks)):
            model = load_ply(folder / str(number) / "scene.ply", device)
            owner = visibility_owner(
                solution.cameras, labels, model.params["means"].detach().cpu().numpy(), device
            )
            parts.append(model.subset(torch.from_numpy(owner == number).to(device)))
            log(f"block {number}: kept {int((owner == number).sum())} of {model.count}")
        save_ply(merged_path, GaussianModel.concatenate(parts))
        del parts
        torch.cuda.empty_cache()

    joint = replace(
        config.splat,
        steps=config.partition.finetune_steps,
        resolution_schedule=((0, 1),),
        sh_interval=1,
        means_lr=config.splat.means_lr * 0.25,
        refine_start=200,
        refine_stop=int(config.partition.finetune_steps * 0.8),
        max_gaussians=0,  # keep the merged count
        pose_start_step=0,
        # Keep the depth prior at its final weight: without it the joint fine-tune of a
        # large capture lets blank walls bend again (new home trained this way).
        depth_prior_weight=config.splat.depth_prior_weight_final,
        evaluate_every=10**9,
    )
    metrics = train_splat(
        solution.cameras,
        solution.image_paths,
        solution.points,
        solution.colors,
        final,
        joint,
        depth_dir=depth_dir,
        train_indices=train,
        val_indices=val,
        initial=merged_path,
        reflectors=reflectors,
        device=device,
        log=log,
    )
    metrics["blocks"] = summary
    (final / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics
