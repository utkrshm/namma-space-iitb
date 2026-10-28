"""Keep only retrieval (loop) pairs whose relative pose agrees with a reference trajectory.

Repeated structures (window grilles, floor inlays) produce retrieval pairs with many
epipolar inliers between *different* places; a global mapper then folds the trajectory.
A sequential-only model has drift but no such folds, so over one loop its relative
rotation is wrong by a few degrees at most, while a false loop implies an arbitrary
rotation. Each pair's own relative rotation (essential matrix from its matches) is
compared with the reference; disagreeing pairs are dropped.
"""

from __future__ import annotations

import cv2
import numpy as np


def relative_rotation(xy_a, xy_b, intrinsics, ransac_px):
    """Rotation a→b (camera frames) and inlier count from an essential matrix."""
    if len(xy_a) < 16:
        return None, 0
    essential, mask = cv2.findEssentialMat(
        xy_a, xy_b, intrinsics, cv2.USAC_MAGSAC, 0.999, ransac_px
    )
    if essential is None or essential.shape != (3, 3):
        return None, 0
    count, rotation, _, _ = cv2.recoverPose(essential, xy_a, xy_b, intrinsics, mask=mask)
    return rotation, int(count)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def sampson_px(xy_a, xy_b, pose_a, pose_b, intrinsics) -> float:
    """Median Sampson epipolar distance (pixels) of matches under two camera-to-world poses."""
    world_to_a, world_to_b = np.linalg.inv(pose_a), np.linalg.inv(pose_b)
    rotation = world_to_b[:3, :3] @ world_to_a[:3, :3].T
    translation = world_to_b[:3, 3] - rotation @ world_to_a[:3, 3]
    skew = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    inverse = np.linalg.inv(intrinsics)
    fundamental = inverse.T @ skew @ rotation @ inverse
    a = np.column_stack((xy_a, np.ones(len(xy_a))))
    b = np.column_stack((xy_b, np.ones(len(xy_b))))
    fa, ftb = a @ fundamental.T, b @ fundamental
    numerator = np.sum(b * fa, axis=1) ** 2
    denominator = fa[:, 0] ** 2 + fa[:, 1] ** 2 + ftb[:, 0] ** 2 + ftb[:, 1] ** 2
    return float(np.median(np.sqrt(numerator / np.maximum(denominator, 1e-12))))


def consistent_loops(
    matches: dict[tuple[int, int], np.ndarray],
    listing: list[dict],
    features: list[dict],
    frames: list[dict],
    reference_poses: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    max_rotation_deg: float,
    ransac_px: float,
    max_epipolar_px: float = 0.0,
) -> tuple[dict[tuple[int, int], np.ndarray], list[dict]]:
    """Drop retrieval pairs that disagree with ``reference_poses`` (camera-to-world).

    With ``max_epipolar_px`` > 0, a pair must also fit the reference's epipolar geometry
    (median Sampson distance). Use that with a loop-closed reference, whose cameras should
    explain every true loop to about a pixel.
    """
    kept, report = {}, []
    kinds = {(p["i"], p["j"]): p["kind"] for p in listing}
    for (i, j), pair_matches in matches.items():
        if kinds[(i, j)] != "retrieval":
            kept[(i, j)] = pair_matches
            continue
        a, b = frames[i]["frame_id"], frames[j]["frame_id"]
        xy_a = features[i]["keypoints"][pair_matches[:, 0]].astype(np.float64)
        xy_b = features[j]["keypoints"][pair_matches[:, 1]].astype(np.float64)
        measured, inliers = relative_rotation(xy_a, xy_b, intrinsics, ransac_px)
        record = {"i": i, "j": j, "inliers": inliers}
        if measured is not None and a in reference_poses and b in reference_poses:
            # x_b = R x_a for camera coordinates; reference R = R_b^T R_a (world-to-camera).
            expected = reference_poses[b][:3, :3].T @ reference_poses[a][:3, :3]
            record["rotation_error_deg"] = rotation_angle_deg(measured @ expected.T)
            record["accepted"] = record["rotation_error_deg"] <= max_rotation_deg
            if max_epipolar_px > 0:
                record["epipolar_px"] = sampson_px(
                    xy_a, xy_b, reference_poses[a], reference_poses[b], intrinsics
                )
                record["accepted"] &= record["epipolar_px"] <= max_epipolar_px
        else:
            record["accepted"] = False
        if record["accepted"]:
            kept[(i, j)] = pair_matches
        report.append(record)
    return kept, report


def loop_consistency(model, frames, matches, listing, features, segment_s: float = 20.0) -> dict:
    """Epipolar error of every retrieval pair under a solved model, overall and by time.

    True loops should fit a well-closed model to about a pixel; a stretch with large
    errors is a stretch whose revisits disagree, which shows up as blur or doubled
    surfaces in a splat.
    """
    from .evaluation import camera_poses

    poses = camera_poses(model, frames)
    focal, cx, cy = next(iter(model.cameras.values())).params[:3]
    intrinsics = np.array([[focal, 0, cx - 0.5], [0, focal, cy - 0.5], [0, 0, 1.0]])
    kinds = {(p["i"], p["j"]): p["kind"] for p in listing}
    rows = []
    for (i, j), pair_matches in matches.items():
        a, b = frames[i]["frame_id"], frames[j]["frame_id"]
        if kinds[(i, j)] != "retrieval" or a not in poses or b not in poses:
            continue
        xy_a = features[i]["keypoints"][pair_matches[:, 0]].astype(np.float64)
        xy_b = features[j]["keypoints"][pair_matches[:, 1]].astype(np.float64)
        error = sampson_px(xy_a, xy_b, poses[a], poses[b], intrinsics)
        rows.append((frames[i]["timestamp_s"], error))
    if not rows:
        return {"pairs": 0}
    times, errors = np.array(rows).T
    segments = []
    for start in np.arange(0, times.max() + segment_s, segment_s):
        inside = (times >= start) & (times < start + segment_s)
        if inside.any():
            segments.append(
                {
                    "start_s": float(start),
                    "pairs": int(inside.sum()),
                    "median_px": float(np.median(errors[inside])),
                    "over_3px": float(np.mean(errors[inside] > 3)),
                }
            )
    return {
        "pairs": len(errors),
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "over_3px": float(np.mean(errors > 3)),
        "segments": segments,
    }
