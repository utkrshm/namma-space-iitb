"""Adapters for joint VGGT and aligned overlapping-section predictions."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .contracts import RefinementScene


def load_refinement_scene(input_dir: str | Path) -> RefinementScene:
    root = Path(input_dir).resolve()
    sectioned = (root / "graph.json").is_file()
    keys = ("images", "depth", "depth_confidence", "valid_mask", "intrinsics", "source_to_model")

    if sectioned:
        manifest = json.loads((root / "manifest.json").read_text())
        frames = manifest["frame_provenance"]
        owners = np.asarray(
            json.loads((root / "alignment_report.json").read_text())["owner_section_by_frame"],
            dtype=np.int32,
        )
        transforms = json.loads((root / "graph.json").read_text())["optimized_transforms"]
        with np.load(root / "point_cloud.npz", allow_pickle=False) as data:
            centers = data["camera_centers"]
            rotations = data["camera_rotations"]
            if list(data["frame_ids"]) != [f["frame_id"] for f in frames]:
                raise ValueError("Section cloud and manifest frame identities disagree")
        blocks = {}
        for sid in sorted(set(owners)):
            directory = root / f"section_{sid:02d}"
            meta = json.loads((directory / "metadata.json").read_text())
            with np.load(directory / "predictions.npz", allow_pickle=False) as data:
                arrays = {k: data[k] for k in keys}
            lookup = {f["frame_id"]: i for i, f in enumerate(meta["frames"])}
            scale = transforms[sid]["scale"]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Section scale must be finite and positive")
            for index in np.flatnonzero(owners == sid):
                local = lookup[frames[index]["frame_id"]]
                blocks[index] = {k: arrays[k][local] for k in keys}
                blocks[index]["depth"] = blocks[index]["depth"] * scale
        arrays = {k: np.stack([blocks[i][k] for i in range(len(frames))]) for k in keys}
    else:
        meta = json.loads((root / "metadata.json").read_text())
        frames = meta["frames"]
        with np.load(root / "predictions.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in keys}
            poses = data["camera_to_world"]
        rotations, centers = poses[:, :3, :3], poses[:, :3, 3]
        owners = np.zeros(len(frames), np.int32)

    valid = (
        arrays["valid_mask"]
        & np.isfinite(arrays["depth"])
        & (arrays["depth"] > 0)
        & np.isfinite(arrays["depth_confidence"])
    )
    bounds = []
    for mask in arrays["valid_mask"]:
        y, x = np.nonzero(mask)
        if not len(x):
            raise ValueError("Input frame contains no image content")
        bounds.append([x.min(), y.min(), x.max(), y.max()])

    for value in (rotations, centers, arrays["intrinsics"], arrays["source_to_model"]):
        if not np.isfinite(value).all():
            raise ValueError("Camera parameters and image transforms must be finite")
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-5):
        raise ValueError("Input cameras are not proper orthonormal rotations")
    if not np.allclose(np.linalg.det(rotations), 1, atol=2e-5):
        raise ValueError("Input camera rotations must have determinant +1")

    # Full array content, not just file timestamps, identifies all cached evidence.
    digest = hashlib.sha256()
    for key, value in (*sorted(arrays.items()), ("rotations", rotations), ("centers", centers)):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(json.dumps(frames, sort_keys=True).encode())
    return RefinementScene(
        arrays["images"],
        arrays["depth"],
        arrays["depth_confidence"],
        valid,
        arrays["intrinsics"],
        rotations,
        centers,
        arrays["source_to_model"],
        np.asarray(bounds),
        owners,
        tuple(frames),
        digest.hexdigest(),
    )


def sample_image(image, xy):
    """Bilinear sampling at integer-centered image coordinates; exterior is NaN."""
    xy = np.asarray(xy, np.float32)
    if not len(xy):
        return np.empty((0, *image.shape[2:]), np.float32)
    result = cv2.remap(
        image.astype(np.float32),
        xy[:, 0, None],
        xy[:, 1, None],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )[:, 0]
    inside = (xy >= 0).all(1) & (xy[:, 0] <= image.shape[1] - 1) & (xy[:, 1] <= image.shape[0] - 1)
    result[~inside] = np.nan
    return result


def field_values(scene, frame, xy, fields):
    """Bilinear field in content coordinates, with constant extension at content edges."""
    x0, y0, x1, y1 = scene.content_bounds[frame]
    size = fields.shape[-1]
    grid_xy = (np.asarray(xy) - [x0, y0]) / np.maximum([x1 - x0, y1 - y0], 1)
    grid_xy = np.clip(grid_xy, 0, 1) * (size - 1)
    lo = np.minimum(grid_xy.astype(int), size - 2)
    x, y = (grid_xy - lo).T
    gx, gy = lo.T
    grid = fields[frame]
    return (
        (1 - x) * (1 - y) * grid[gy, gx]
        + x * (1 - y) * grid[gy, gx + 1]
        + (1 - x) * y * grid[gy + 1, gx]
        + x * y * grid[gy + 1, gx + 1]
    )


def world_observations(scene, frames, xy, rotations=None, centers=None, scales=None, fields=None):
    """Unproject arbitrary source-frame pixels using one consistent camera/depth variant."""
    rotations = scene.rotations if rotations is None else rotations
    centers = scene.centers if centers is None else centers
    points = np.empty((len(frames), 3), np.float64)
    for frame in np.unique(frames):
        selected = frames == frame
        uv = xy[selected]
        depth = sample_image(scene.depth[frame], uv).astype(np.float64)
        if scales is not None:
            depth *= np.exp(scales[frame] + field_values(scene, frame, uv, fields))
        rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(scene.intrinsics[frame]).T
        points[selected] = (rays * depth[:, None]) @ rotations[frame].T + centers[frame]
    return points
