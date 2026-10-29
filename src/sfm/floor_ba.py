"""Floor-constrained bundle adjustment: remove a sag of the solved model along a weak stretch.

On Coorg, the global mapper leaves the bay-window foyer and the mat corridor (24–72 s)
about 0.3 camera heights too deep. The cameras there sit 1.2–1.36× as high above the
floor as elsewhere, although the house has one flat floor (door heights agree to 5%).
Reprojection barely constrains such a smooth, low-frequency bend, and plain bundle
adjustment keeps it.

This step solves for:

- a smooth deformation of the whole model, as displacements on a trilinear grid;
- a small rotation and translation correction per camera.

Its energy has four terms:

- reprojection error of every observation (Huber, pixels);
- floor points on one plane, perpendicular to the camera-plane normal ("up");
- smoothness of the deformation (second differences);
- a weak pull towards zero, which fixes the gauge.

Cameras move with the deformation: their centres by its displacement, their rotation by
its local rotation. Their own corrections absorb the rest.

Floor points are sparse points in the lowest dense height layer of each plan cell, a
layer that follows a sagging floor. The result is a new COLMAP model with the same images
and tracks. Its deformation field (``deformation.npz``) can move a splat trained on the
old model (`src.splatting.transfer`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .mapping import undistort


def scene_up(camera_to_world: np.ndarray) -> np.ndarray:
    """Normal of the plane through the camera centres, signed like the cameras' up (−y)."""
    centres = camera_to_world[:, :3, 3]
    _, _, axes = np.linalg.svd(centres - centres.mean(0), full_matrices=False)
    return axes[2] * np.sign(axes[2] @ -camera_to_world[:, :3, 1].mean(0))


def floor_points(points, track_lengths, centres, up, cell_fraction=0.3, band_fraction=0.03):
    """Indices of points in the lowest dense layer under the cameras, and the camera height."""
    heights = (points - centres.mean(0)) @ up
    camera_heights = (centres - centres.mean(0)) @ up
    plan_axes = np.linalg.svd(np.eye(3) - np.outer(up, up))[0][:, :2]
    plan = points @ plan_axes

    # Camera height above the floor, first from the global densest layer below the cameras.
    below = heights[heights < camera_heights.min() - 0.2 * np.ptp(camera_heights)]
    counts, edges = np.histogram(below, bins=100, range=tuple(np.percentile(below, [1, 99])))
    camera_height = float(np.median(camera_heights) - edges[np.argmax(counts)])

    cell = cell_fraction * camera_height
    band = band_fraction * camera_height
    candidates = np.flatnonzero(
        (heights < np.median(camera_heights) - 0.5 * camera_height) & (track_lengths >= 3)
    )
    keys = np.floor(plan[candidates] / cell).astype(int)
    cells, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    floors = {}
    for index, key in enumerate(map(tuple, cells)):
        values = heights[candidates[inverse == index]]
        if len(values) < 15:
            continue
        if np.ptp(values) < 2 * band:
            floors[key] = float(np.median(values))
            continue
        counts, edges = np.histogram(
            values, bins=np.arange(values.min(), values.max() + band, band)
        )
        lowest = np.argmax(counts >= 0.3 * counts.max())  # lowest strong layer, not furniture
        layer = values[np.abs(values - edges[lowest] - band / 2) < 1.5 * band]
        floors[key] = float(np.median(layer))
    smoothed = {}
    for (i, j), value in floors.items():
        near = [floors.get((i + a, j + b)) for a in (-1, 0, 1) for b in (-1, 0, 1)]
        near = [v for v in near if v is not None]
        smoothed[(i, j)] = float(np.median(near)) if len(near) >= 3 else value
    local = np.array([smoothed.get(tuple(k), np.nan) for k in keys])
    selected = candidates[np.abs(heights[candidates] - local) < band]
    return selected, camera_height


class Grid:
    """Trilinear displacement field on a regular grid (world units), clamped at the border."""

    def __init__(self, origin, spacing, shape, device):
        self.origin = torch.as_tensor(origin, dtype=torch.float64, device=device)
        self.spacing = float(spacing)
        self.shape = tuple(int(s) for s in shape)

    def __call__(self, delta: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        position = (points - self.origin) / self.spacing
        upper = torch.tensor(self.shape, device=points.device, dtype=points.dtype) - 1 - 1e-6
        position = torch.minimum(position.clamp_min(0), upper)
        base = position.floor().long()
        fraction = position - base
        result = torch.zeros_like(points)
        for corner in range(8):
            offset = torch.tensor([(corner >> k) & 1 for k in range(3)], device=points.device)
            index = base + offset
            weight = torch.prod(torch.where(offset.bool(), fraction, 1 - fraction), dim=1)
            flat = (index[:, 0] * self.shape[1] + index[:, 1]) * self.shape[2] + index[:, 2]
            result = result + weight[:, None] * delta[flat]
        return result


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vector[:, 0])
    x, y, z = vector.unbind(1)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=1
    ).reshape(-1, 3, 3)  # fmt: skip


def _exp(vector: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula with a series fallback near zero."""
    angle = vector.norm(dim=1, keepdim=True).clamp_min(1e-12)[..., None]
    skew = _skew(vector)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(skew)
    small = angle < 1e-6
    a = torch.where(small, 1 - angle**2 / 6, torch.sin(angle) / angle)
    b = torch.where(small, 0.5 - angle**2 / 24, (1 - torch.cos(angle)) / angle**2)
    return eye + a * skew + b * skew @ skew


def local_rotation(grid: Grid, delta, points, step):
    """Small-angle rotation of the deformation at points: the antisymmetric Jacobian part."""
    columns = []
    for axis in range(3):
        shift = torch.zeros_like(points)
        shift[:, axis] = step
        columns.append((grid(delta, points + shift) - grid(delta, points - shift)) / (2 * step))
    jacobian = torch.stack(columns, dim=2)  # ∂D_i/∂x_j
    antisymmetric = (jacobian - jacobian.transpose(1, 2)) / 2
    return torch.stack(
        (antisymmetric[:, 2, 1], antisymmetric[:, 0, 2], antisymmetric[:, 1, 0]), dim=1
    )


def _observations(reconstruction, names):
    image_index, point_index, pixels = [], [], []
    point_ids = {pid: k for k, pid in enumerate(reconstruction.points3D)}
    for k, name in enumerate(names):
        image = reconstruction.find_image_with_name(name)
        for point in image.points2D:
            if point.has_point3D():
                image_index.append(k)
                point_index.append(point_ids[point.point3D_id])
                pixels.append(point.xy)
    return np.array(image_index), np.array(point_index), np.array(pixels), point_ids


def _undistorted_copy(source: Path, output: Path, reconstruction, images) -> None:
    import pycolmap

    undistorted = pycolmap.Reconstruction(source / "undistorted" / "sparse")
    for pid, point in reconstruction.points3D.items():
        undistorted.points3D[pid].xyz = point.xyz
    for image in images:
        target = undistorted.images[image.image_id]
        assert target.name == image.name
        undistorted.frames[target.frame_id].rig_from_world = reconstruction.frames[
            image.frame_id
        ].rig_from_world
    (output / "undistorted" / "sparse").mkdir(parents=True, exist_ok=True)
    undistorted.write(output / "undistorted" / "sparse")
    link = output / "undistorted" / "images"
    if not link.exists():
        link.symlink_to((source / "undistorted" / "images").resolve())


def floor_bundle_adjust(
    sfm_dir: str | Path,
    output_dir: str | Path,
    *,
    floor_weight: float = 1.0,
    smooth_weight: float = 10.0,
    grid_fraction: float = 0.5,
    iterations: int = 300,
    refine_intrinsics: bool = True,
    dry_run: bool = False,
    device: str = "cuda:0",
) -> dict:
    """Write ``<output_dir>/{sparse/0, undistorted/sparse, frames.json, deformation.npz, report.json}``.

    ``images`` and ``undistorted/images`` link to the source (intrinsics are unchanged).
    """
    import pycolmap

    source, output = Path(sfm_dir).resolve(), Path(output_dir).resolve()
    reconstruction = pycolmap.Reconstruction(source / "sparse" / "0")
    camera = next(iter(reconstruction.cameras.values()))
    focal, cx, cy, k = camera.params
    images = [image for image in reconstruction.images.values() if image.has_pose]
    names = [image.name for image in images]
    world_to_camera = np.array([image.cam_from_world().matrix() for image in images])
    rotations = world_to_camera[:, :, :3]
    centres = -np.einsum("nji,nj->ni", rotations, world_to_camera[:, :, 3])
    camera_to_world = np.tile(np.eye(4), (len(images), 1, 1))
    camera_to_world[:, :3, :3] = rotations.transpose(0, 2, 1)
    camera_to_world[:, :3, 3] = centres
    image_index, point_index, pixels, point_ids = _observations(reconstruction, names)
    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    track_lengths = np.array([p.track.length() for p in reconstruction.points3D.values()])

    up = scene_up(camera_to_world)
    floor, camera_height = floor_points(points, track_lengths, centres, up)
    origin = centres.mean(0)

    # Grid over the scene (1st–99th percentile of points, padded by one cell).
    spacing = grid_fraction * camera_height
    low = np.percentile(points, 1, axis=0) - spacing
    high = np.percentile(points, 99, axis=0) + spacing
    shape = np.ceil((high - low) / spacing).astype(int) + 1
    grid = Grid(low, spacing, shape, device)

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64, device=device)

    points_t, centres_t, rotations_t = tensor(points), tensor(centres), tensor(rotations)
    pixels_t, up_t = tensor(pixels), tensor(up)
    image_t = torch.as_tensor(image_index, device=device)
    point_t = torch.as_tensor(point_index, device=device)
    floor_t = torch.as_tensor(floor, device=device)
    delta = torch.zeros((int(np.prod(shape)), 3), dtype=torch.float64, device=device)
    delta.requires_grad_(True)
    camera_rotation = torch.zeros((len(images), 3), dtype=torch.float64, device=device)
    camera_rotation.requires_grad_(True)
    camera_shift = torch.zeros_like(camera_rotation, requires_grad=True)
    # The floor plane stays at the dominant floor level (the densest height), so the part of
    # the floor that is right is kept and only the sagging part moves.
    floor_heights = (points[floor] - origin) @ up
    counts, edges = np.histogram(floor_heights, bins=np.arange(
        floor_heights.min(), floor_heights.max() + 0.01 * camera_height, 0.01 * camera_height
    ))  # fmt: skip
    level = float(edges[np.argmax(counts)] + 0.005 * camera_height)
    sigma = 0.01 * camera_height
    # Relative focal change and radial coefficient: a mis-estimated lens also bends geometry.
    intrinsics = torch.tensor([0.0, k], dtype=torch.float64, device=device, requires_grad=True)

    def solve(delta, camera_rotation, camera_shift):
        moved = points_t + grid(delta, points_t)
        world = local_rotation(grid, delta, centres_t, 0.5 * spacing)
        new_rotations = _exp(camera_rotation) @ rotations_t @ _exp(world).transpose(1, 2)
        new_centres = centres_t + grid(delta, centres_t) + camera_shift
        return moved, new_rotations, new_centres

    def reprojection(moved, new_rotations, new_centres):
        local = torch.einsum(
            "nij,nj->ni", new_rotations[image_t], moved[point_t] - new_centres[image_t]
        )
        normalized = local[:, :2] / local[:, 2:].clamp_min(1e-6)
        radial = 1 + intrinsics[1] * (normalized**2).sum(1, keepdim=True)
        projected = focal * (1 + intrinsics[0]) * normalized * radial + tensor([cx, cy])
        return (projected - pixels_t).norm(dim=1)

    def second_differences(delta):
        field = delta.reshape(*shape, 3)
        total = 0
        for axis in range(3):
            d = torch.diff(field, n=2, dim=axis)
            total = total + (d**2).sum()
        return total / spacing**2

    def energy():
        moved, new_rotations, new_centres = solve(delta, camera_rotation, camera_shift)
        error = reprojection(moved, new_rotations, new_centres)
        huber = torch.where(error < 2, 0.5 * error**2, 2 * (error - 1))
        floor_residual = (((moved[floor_t] - tensor(origin)) @ up_t - level) / sigma).abs()
        floor_huber = torch.where(
            floor_residual < 2, 0.5 * floor_residual**2, 2 * (floor_residual - 1)
        )
        return (
            huber.mean()
            + floor_weight * floor_huber.mean()
            + smooth_weight * second_differences(delta) / len(delta)
            + 1e-4 * (delta**2).sum() / (len(delta) * spacing**2)
            + 1e-2 * (camera_shift**2).sum() / (len(images) * spacing**2)
        )

    def summary():
        with torch.no_grad():
            moved, new_rotations, new_centres = solve(delta, camera_rotation, camera_shift)
            error = reprojection(moved, new_rotations, new_centres)
            heights = (moved[floor_t] - tensor(origin)) @ up_t
        return {
            "reprojection_px_mean": float(error.mean()),
            "reprojection_px_median": float(error.median()),
            "floor_height_std_over_camera_height": float(heights.std() / camera_height),
            "floor_height_p5_p95_over_camera_height": (
                torch.quantile(
                    heights - heights.median(),
                    torch.tensor([0.05, 0.95], device=device, dtype=heights.dtype),
                )
                / camera_height
            ).tolist(),
        }

    before = summary()
    optimizer = torch.optim.LBFGS(
        [delta, camera_rotation, camera_shift] + ([intrinsics] if refine_intrinsics else []),
        max_iter=iterations,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def closure():
        optimizer.zero_grad()
        value = energy()
        value.backward()
        return value

    optimizer.step(closure)
    after = summary()
    if dry_run:
        return {
            "floor_points": len(floor),
            "before": before,
            "after": after,
            "intrinsics": [float(focal * (1 + intrinsics[0].item())), float(intrinsics[1].item())],
        }

    # Write the corrected model: same images, cameras and tracks; moved points and poses.
    with torch.no_grad():
        moved, new_rotations, new_centres = solve(delta, camera_rotation, camera_shift)
    moved, new_rotations, new_centres = (
        x.cpu().numpy() for x in (moved, new_rotations, new_centres)
    )
    for pid, index in point_ids.items():
        reconstruction.points3D[pid].xyz = moved[index]
    for image, rotation, centre in zip(images, new_rotations, new_centres):
        reconstruction.frames[image.frame_id].rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(rotation), -rotation @ centre
        )
    new_focal = float(focal * (1 + intrinsics[0].item()))
    camera.params = [new_focal, cx, cy, float(intrinsics[1].item())]
    reconstruction.cameras[camera.camera_id] = camera
    sparse = output / "sparse" / "0"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    reconstruction.write(sparse)
    for name in ("frames.json", "images"):
        link = output / name
        if not link.exists():
            link.symlink_to((source / name).resolve())

    # With unchanged intrinsics the undistorted images are those of the source; only the
    # undistorted model's points and poses change (same image and point ids).
    if refine_intrinsics:
        undistort(output, sparse)
    else:
        _undistorted_copy(source, output, reconstruction, images)
    np.savez(
        output / "deformation.npz",
        origin=low,
        spacing=spacing,
        shape=shape,
        displacement=delta.detach().cpu().numpy(),
    )
    report = {
        "source": str(source),
        "up": up.tolist(),
        "camera_height": camera_height,
        "floor_points": len(floor),
        "observations": len(pixels),
        "grid_shape": shape.tolist(),
        "grid_spacing": spacing,
        "weights": {"floor": floor_weight, "smooth": smooth_weight},
        "intrinsics_before": [float(focal), float(k)],
        "intrinsics_after": [
            float(focal * (1 + intrinsics[0].item())),
            float(intrinsics[1].item()),
        ],
        "before": before,
        "after": after,
        "camera_shift_max_over_camera_height": float(
            camera_shift.detach().norm(dim=1).max() / camera_height
        ),
        "displacement_max_over_camera_height": float(
            delta.detach().norm(dim=1).max() / camera_height
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Flatten a sagging floor by constrained BA")
    parser.add_argument("sfm_dir", type=Path, help="src.sfm directory with sparse/0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--floor-weight", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument(
        "--grid-fraction", type=float, default=0.5, help="Grid step / camera height"
    )
    parser.add_argument("--fixed-intrinsics", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    report = floor_bundle_adjust(
        args.sfm_dir, args.output_dir, floor_weight=args.floor_weight,
        smooth_weight=args.smooth_weight, iterations=args.iterations,
        grid_fraction=args.grid_fraction,
        refine_intrinsics=not args.fixed_intrinsics, dry_run=args.dry_run, device=args.device,
    )  # fmt: skip
    print(json.dumps({k: report[k] for k in ("floor_points", "before", "after")}, indent=1))
    return 0
