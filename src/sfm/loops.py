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


def consistent_loops(
    matches: dict[tuple[int, int], np.ndarray],
    listing: list[dict],
    features: list[dict],
    frames: list[dict],
    reference_poses: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    max_rotation_deg: float,
    ransac_px: float,
) -> tuple[dict[tuple[int, int], np.ndarray], list[dict]]:
    """Drop retrieval pairs that disagree with ``reference_poses`` (camera-to-world)."""
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
        else:
            record["accepted"] = False
        if record["accepted"]:
            kept[(i, j)] = pair_matches
        report.append(record)
    return kept, report
