"""Unfused, source-pixel-preserving exports and explicit multi-view support diagnostics."""

from pathlib import Path

import numpy as np

from ..reconstruction.contracts import PointCloud
from ..reconstruction.visualization import save_point_cloud
from .contracts import RefinementResult
from .inputs import field_values, sample_image, world_observations


def refined_depth(scene, result, frame):
    height, width = scene.depth.shape[1:]
    y, x = np.indices((height, width))
    xy = np.column_stack((x.ravel(), y.ravel()))
    correction = result.log_scales[frame] + field_values(scene, frame, xy, result.log_depth_fields)
    return (scene.depth[frame] * np.exp(correction.reshape(height, width))).astype(np.float32)


def _extrinsics(rotations, centers):
    r = rotations.transpose(0, 2, 1)
    return np.concatenate((r, -np.einsum("nij,nj->ni", r, centers)[:, :, None]), axis=2)


def _write(path, scene, result, arrays):
    points = np.asarray(arrays["points"], np.float32)
    if not len(points):
        raise ValueError(f"No supported points for {path}; inspect raw corrected geometry instead")
    arrays = {
        **arrays,
        "points": points,
        "camera_centers": result.centers,
        "camera_rotations": result.rotations,
        "camera_translations": _extrinsics(result.rotations, result.centers),
        "frame_ids": np.asarray([f["frame_id"] for f in scene.frames]),
    }
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    save_point_cloud(
        PointCloud(
            *(arrays[k] for k in ("points", "colors", "confidence", "frame_indices", "pixel_xy"))
        ),
        path,
    )


def _arrays(scene, result, frames, pixels, points=None, **extra):
    frames = np.asarray(frames, np.int32)
    pixels = np.asarray(pixels, np.int32)
    x, y = pixels.T
    if points is None:
        points = world_observations(
            scene,
            frames,
            pixels,
            result.rotations,
            result.centers,
            result.log_scales,
            result.log_depth_fields,
        )
    depth = np.einsum(
        "nij,nj->ni", result.rotations[frames].transpose(0, 2, 1), points - result.centers[frames]
    )[:, 2]
    return {
        "points": points,
        "colors": scene.images[frames, y, x],
        "confidence": scene.confidence[frames, y, x],
        "frame_indices": frames,
        "pixel_xy": pixels,
        "section_indices": scene.owners[frames],
        "depths": depth.astype(np.float32),
        **extra,
    }


def multiview_support(scene, result, depth_maps, frame, xy, neighbors, tolerance):
    """Count other views agreeing on z; occlusion is not free-space contradiction."""
    source_frames = np.full(len(xy), frame, np.int32)
    points = world_observations(
        scene,
        source_frames,
        xy,
        result.rotations,
        result.centers,
        result.log_scales,
        result.log_depth_fields,
    )
    support = np.zeros(len(points), np.int16)
    contradictions = np.zeros(len(points), np.int16)
    for other in neighbors:
        camera = (points - result.centers[other]) @ result.rotations[other]
        projected = camera @ scene.intrinsics[other].T
        positive = camera[:, 2] > 1e-6
        uv = projected[:, :2] / np.maximum(camera[:, 2, None], 1e-6)
        # Keep exterior coordinates bounded before OpenCV's fixed-point remapping.
        uv[~positive] = -10
        uv = np.clip(uv, -10, max(scene.depth.shape[1:]) + 10)
        reference = sample_image(depth_maps[other], uv)
        valid = positive & (sample_image(scene.valid_mask[other], uv) > 0.999)
        valid &= np.isfinite(reference) & (reference > 0)
        residual = (camera[:, 2] - reference) / np.maximum(reference, 1e-6)
        support += (valid & (np.abs(residual) <= tolerance)).astype(np.int16)
        contradictions += (valid & (residual < -2 * tolerance)).astype(np.int16)
    return points, support, contradictions


def export_refinement(scene, result, tracks, source, output, config):
    output = Path(output)
    identity = RefinementResult(
        scene.rotations,
        scene.centers,
        np.zeros(scene.frame_count),
        np.zeros_like(result.log_depth_fields),
        {},
    )
    corrected_depth = np.stack([refined_depth(scene, result, i) for i in range(scene.frame_count)])
    extrinsics = _extrinsics(result.rotations, result.centers)
    np.savez_compressed(
        output / "depths.npz",
        depth=corrected_depth,
        valid_mask=scene.valid_mask,
        frame_ids=[f["frame_id"] for f in scene.frames],
    )
    np.savez_compressed(
        output / "cameras.npz",
        intrinsics=scene.intrinsics,
        rotations=result.rotations,
        centers=result.centers,
        world_to_camera=extrinsics,
        source_to_model=scene.source_to_model,
        original_intrinsics=np.linalg.solve(scene.source_to_model, scene.intrinsics),
    )

    # This comparison changes geometry only: exactly the original exported observations.
    with np.load(Path(source) / "point_cloud.npz", allow_pickle=False) as data:
        frame_ids, pixels = data["frame_indices"], data["pixel_xy"]
        expected = data["points"]
    before = _arrays(scene, identity, frame_ids, pixels)
    if not np.isfinite(before["points"]).all():
        raise ValueError("Source unprojection produced nonfinite observations")
    error = float(np.max(np.abs(before["points"] - expected)))
    if error > 2e-5:
        raise ValueError(f"Source cloud is not explained by input cameras and depth: {error}")
    _write(output / "geometry_before.ply", scene, identity, before)
    _write(output / "geometry_after.ply", scene, result, _arrays(scene, result, frame_ids, pixels))

    neighbors = [set() for _ in range(scene.frame_count)]
    for a, b in tracks.pairs:
        neighbors[a].add(int(b))
        neighbors[b].add(int(a))
    for frame, items in enumerate(neighbors):
        ordered = sorted(items, key=lambda other: abs(other - frame))
        neighbors[frame] = sorted(set(ordered[:8] + ordered[-4:]))

    rng = np.random.default_rng(config.seed)
    budget = max(1, config.max_points // scene.frame_count)
    chunks = {
        name: []
        for name in ("unfiltered_before", "unfiltered_after", "supported_before", "point_cloud")
    }
    records = []
    for frame in range(scene.frame_count):
        eligible = np.flatnonzero(scene.valid_mask[frame])
        chosen = rng.choice(eligible, min(len(eligible), 2 * budget), replace=False)
        width = scene.depth.shape[2]
        xy = np.column_stack((chosen % width, chosen // width))
        initial_points, initial_support, initial_conflict = multiview_support(
            scene, identity, scene.depth, frame, xy, neighbors[frame], config.support_tolerance
        )
        points, support, conflict = multiview_support(
            scene, result, corrected_depth, frame, xy, neighbors[frame], config.support_tolerance
        )
        raw_indices = np.arange(min(len(chosen), budget))
        stable_before = np.flatnonzero(
            (initial_support >= config.min_support) & (initial_conflict <= initial_support)
        )[:budget]
        stable_after = np.flatnonzero((support >= config.min_support) & (conflict <= support))[
            :budget
        ]
        for name, selected, p, s, c, variant in (
            (
                "unfiltered_before",
                raw_indices,
                initial_points,
                initial_support,
                initial_conflict,
                identity,
            ),
            ("unfiltered_after", raw_indices, points, support, conflict, result),
            (
                "supported_before",
                stable_before,
                initial_points,
                initial_support,
                initial_conflict,
                identity,
            ),
            ("point_cloud", stable_after, points, support, conflict, result),
        ):
            chunks[name].append(
                _arrays(
                    scene,
                    variant,
                    np.full(len(selected), frame),
                    xy[selected],
                    p[selected],
                    support_count=s[selected],
                    contradiction_count=c[selected],
                )
            )
        cutoff = np.percentile(scene.confidence[frame][scene.valid_mask[frame]], 25)
        below = scene.confidence[frame, xy[:, 1], xy[:, 0]] < cutoff
        records.append(
            {
                "frame": frame,
                "neighbors": neighbors[frame],
                "candidates": len(chosen),
                "supported_before": len(stable_before),
                "supported_after": len(stable_after),
                "below_quartile_retained": int(below[stable_after].sum()),
                "median_support": float(np.median(support)),
            }
        )
    counts = {}
    for name, parts in chunks.items():
        arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        variant = identity if name.endswith("before") else result
        _write(output / f"{name}.ply", scene, variant, arrays)
        counts[name] = len(arrays["points"])

    return {
        "source_unprojection_max_error": error,
        "same_observation_count": len(frame_ids),
        "counts": counts,
        "sampling": "seeded uniform source-content pixels without replacement per frame",
        "support": "other-view relative z agreement; behind-surface occlusion is not contradiction",
        "min_support": config.min_support,
        "relative_z_tolerance": config.support_tolerance,
        "unsupported_export_frames": [r["frame"] for r in records if r["supported_after"] == 0],
        "below_original_confidence_quartile_retained": sum(
            r["below_quartile_retained"] for r in records
        ),
        "frames": records,
    }
