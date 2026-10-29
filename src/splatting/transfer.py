"""Move a trained splat onto a re-solved camera set (same images, corrected geometry).

Retraining from scratch after the cameras are re-solved takes hours. Both solves share
the same images and keypoints, so the change of geometry is known at every camera and
every 3D point. The splat is moved with it:

1. **Correspondences:** 3D points that share a keypoint in the same image in both solves.
2. **Local similarities:** around every corresponding point, a similarity
   ``x' = s R x + t`` is fitted to its 32 nearest correspondences (weighted Umeyama).
   Neighbours that fit badly (tracks merged across a false loop in one solve) are
   dropped and the fit is repeated. This also covers the change of gauge (the new
   solve's arbitrary scale and frame) and follows bends of the old solve.
3. **Blending:** each Gaussian blends the similarities of its 4 nearest points by inverse
   squared distance. Its covariance is mapped by the blended linear part, ``Σ' = A Σ Aᵀ``.

View-dependent colour is not rotated; local rotations are a few degrees. A short
fine-tune on the new cameras (`train_sfm_splat.py --initial-ply`) corrects what is left.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .model import GaussianModel
from .storage import export_ply, load_model


def _poses(reconstruction):
    """Image name → (world-to-camera rotation, camera centre, image)."""
    poses = {}
    for image in reconstruction.images.values():
        if image.has_pose:
            world_to_camera = image.cam_from_world().matrix()
            rotation = world_to_camera[:, :3]
            poses[image.name] = (rotation, -rotation.T @ world_to_camera[:, 3], image)
    return poses


def correspondences(old, new):
    """Shared cameras (old/new similarity parts) and shared 3D points of two solves.

    Both solves must come from the same feature database, so the keypoint indices agree.
    """
    old_poses, new_poses = _poses(old), _poses(new)
    names = sorted(set(old_poses) & set(new_poses))
    cameras = {"old_centre": [], "new_centre": [], "linear": []}
    pairs = {}
    for name in names:
        old_rotation, old_centre, old_image = old_poses[name]
        new_rotation, new_centre, new_image = new_poses[name]
        old_depths, new_depths = [], []
        for index, (a, b) in enumerate(zip(old_image.points2D, new_image.points2D)):
            if a.has_point3D() and b.has_point3D():
                old_xyz = old.points3D[a.point3D_id].xyz
                new_xyz = new.points3D[b.point3D_id].xyz
                old_depths.append(old_rotation[2] @ (old_xyz - old_centre))
                new_depths.append(new_rotation[2] @ (new_xyz - new_centre))
                pairs.setdefault((a.point3D_id, b.point3D_id), (old_xyz, new_xyz))
        if len(old_depths) < 20:
            continue
        ratio = np.array(new_depths) / np.maximum(np.array(old_depths), 1e-9)
        scale = float(np.median(ratio[np.isfinite(ratio) & (ratio > 0)]))
        cameras["old_centre"].append(old_centre)
        cameras["new_centre"].append(new_centre)
        cameras["linear"].append(scale * new_rotation.T @ old_rotation)
    cameras = {key: np.array(value) for key, value in cameras.items()}
    points = np.array(list(pairs.values()))
    return cameras, points[:, 0], points[:, 1]


def local_similarities(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    """Batched weighted Umeyama: ``target ≈ s R source + t`` per row of (n, k, 3) sets."""
    total = weights.sum(1, keepdims=True)
    mean_source = (weights[..., None] * source).sum(1) / total
    mean_target = (weights[..., None] * target).sum(1) / total
    a, b = source - mean_source[:, None], target - mean_target[:, None]
    covariance = np.einsum("nk,nki,nkj->nij", weights, b, a) / total[..., None]
    u, d, vt = np.linalg.svd(covariance)
    sign = np.ones_like(d)
    sign[:, 2] = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    rotation = u @ (sign[..., None] * vt)
    variance = np.einsum("nk,nki,nki->n", weights, a, a) / total[:, 0]
    scale = (d * sign).sum(1) / np.maximum(variance, 1e-12)
    linear = scale[:, None, None] * rotation
    return linear, mean_target - np.einsum("nij,nj->ni", linear, mean_source)


class Warp:
    """Old-world → new-world map from robust local similarities at corresponding points."""

    def __init__(self, old_points, new_points, neighbours=32, blend_neighbours=4):
        self.blend_neighbours = blend_neighbours
        tree = cKDTree(old_points)
        _, near = tree.query(old_points, k=neighbours)
        source, target = old_points[near], new_points[near]
        weights = np.ones(near.shape)

        # Two passes: fit, then drop neighbours that the fit explains badly (tracks merged
        # across a false loop in one solve, or mismatched keypoints) and refit.
        for _ in range(2):
            linear, offset = local_similarities(source, target, weights)
            predicted = np.einsum("nij,nkj->nki", linear, source) + offset[:, None]
            error = np.linalg.norm(predicted - target, axis=2)
            weights = (error <= 3 * np.median(error, axis=1, keepdims=True) + 1e-12).astype(float)
        self.inlier_fraction = float(weights.mean())
        self.linear, self.offset = linear, offset
        self.tree = tree

    def blend(self, points: np.ndarray):
        distance, index = self.tree.query(points, k=self.blend_neighbours)
        weight = 1 / np.maximum(distance, 1e-9) ** 2
        weight /= weight.sum(1, keepdims=True)
        linear = self.linear[index]  # (n, k, 3, 3)
        moved = np.einsum("nkij,nj->nki", linear, points) + self.offset[index]
        return np.einsum("nk,nki->ni", weight, moved), np.einsum("nk,nkij->nij", weight, linear)

    def error(self, old_points, new_points) -> np.ndarray:
        return np.linalg.norm(self.blend(old_points)[0] - new_points, axis=1)


@torch.no_grad()
def transfer_model(model: GaussianModel, warp: Warp, chunk: int = 200_000) -> GaussianModel:
    params = {key: value.detach().cpu() for key, value in model.params.items()}
    means = params["means"].numpy().astype(np.float64)
    quats = params["quats"].numpy().astype(np.float64)
    log_scales = params["scales"].numpy().astype(np.float64)
    new_means = np.empty_like(means)
    new_quats = np.empty_like(quats)
    new_scales = np.empty_like(log_scales)
    for start in range(0, len(means), chunk):
        part = slice(start, start + chunk)
        positions, linear = warp.blend(means[part])
        new_means[part] = positions

        # Σ' = A R S² Rᵀ Aᵀ; the SVD of A R S gives the new rotation and scales.
        rotation = Rotation.from_quat(quats[part][:, [1, 2, 3, 0]]).as_matrix()
        stretched = linear @ rotation * np.exp(log_scales[part])[:, None, :]
        u, singular, _ = np.linalg.svd(stretched)
        u[np.linalg.det(u) < 0, :, 2] *= -1
        new_scales[part] = np.log(np.maximum(singular, 1e-12))
        new_quats[part] = Rotation.from_matrix(u).as_quat()[:, [3, 0, 1, 2]]

    tensors = dict(params)
    tensors["means"] = torch.from_numpy(new_means.astype(np.float32))
    tensors["quats"] = torch.from_numpy(new_quats.astype(np.float32))
    tensors["scales"] = torch.from_numpy(new_scales.astype(np.float32))
    return GaussianModel(tensors, model.sh_degree)


def transfer_splat(
    model_path: str | Path, old_model_dir: str | Path, new_model_dir: str | Path, output: str | Path
) -> dict:
    import pycolmap

    old = pycolmap.Reconstruction(old_model_dir)
    new = pycolmap.Reconstruction(new_model_dir)
    cameras, old_points, new_points = correspondences(old, new)
    warp = Warp(old_points, new_points)

    # Held-out check: points left out of the fit.
    rng = np.random.default_rng(0)
    held = rng.random(len(old_points)) < 0.1
    check = Warp(old_points[~held], new_points[~held])
    scale = np.median(np.linalg.norm(new_points - new_points.mean(0), axis=1))

    model, metadata = load_model(model_path, "cpu")
    moved = transfer_model(model, warp)
    report = {
        "model": str(model_path),
        "old_model": str(old_model_dir),
        "new_model": str(new_model_dir),
        "shared_cameras": len(cameras["old_centre"]),
        "shared_points": len(old_points),
        "inlier_fraction": warp.inlier_fraction,
        "camera_scale_ratio_percentiles": np.percentile(
            np.linalg.det(cameras["linear"]) ** (1 / 3), [5, 50, 95]
        ).tolist(),
        "heldout_error_over_scene_radius_percentiles": (
            np.percentile(check.error(old_points[held], new_points[held]), [50, 90, 99]) / scale
        ).tolist(),
        "gaussians": moved.count,
    }
    export_ply(output, moved, {**metadata, "transfer": report})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Move a splat onto a re-solved camera set")
    parser.add_argument("model", type=Path, help="scene.ply or checkpoint.pt")
    parser.add_argument("old_model", type=Path, help="COLMAP model the splat was trained on")
    parser.add_argument("new_model", type=Path, help="Re-solved COLMAP model (same images)")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY")
    args = parser.parse_args(argv)
    report = transfer_splat(args.model, args.old_model, args.new_model, args.output)
    print(json.dumps(report, indent=1))
    return 0
