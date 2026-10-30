"""Keep only loop (retrieval) pairs whose relative rotation agrees with a sequential solve.

Repeated structures (window grilles, floor tiles, identical doors) give retrieval pairs
with many epipolar inliers between *different* places. A global mapper then folds the
trajectory (Coorg: 10–20% of retrieval pairs, 100–180° rotation error). A solve from
sequential pairs only drifts but never folds, so over one loop its relative rotation is
wrong by a few degrees at most, while a false loop implies an arbitrary rotation.

The sequential solve may break into several models (a blank wall, a fast turn). A loop
between frames of the same model is checked against it. A loop between different models
is exactly what can join them, so it is kept if it has twice the usual inliers.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalized_points(model, image, xy: np.ndarray) -> np.ndarray:
    """Pixel keypoints (integer centres) → undistorted normalized camera coordinates."""
    camera = model.cameras[image.camera_id]
    params = camera.params
    if camera.model.name == "OPENCV":
        k = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
        distortion = np.array(params[4:8])
    elif camera.model.name == "SIMPLE_RADIAL":
        k = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]])
        distortion = np.array([params[3], 0, 0, 0])
    else:
        k = camera.calibration_matrix()
        distortion = np.zeros(4)
    shifted = (xy + 0.5).reshape(-1, 1, 2).astype(np.float64)  # COLMAP pixel centres
    if camera.model.name == "OPENCV_FISHEYE":
        k = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
        return cv2.fisheye.undistortPoints(shifted, k, np.array(params[4:8])).reshape(-1, 2)
    return cv2.undistortPoints(shifted, k, distortion).reshape(-1, 2)


def relative_rotation(a: np.ndarray, b: np.ndarray, threshold: float):
    """Rotation a→b (camera frames) and inlier count from normalized correspondences."""
    if len(a) < 16:
        return None, 0
    essential, mask = cv2.findEssentialMat(a, b, np.eye(3), cv2.USAC_MAGSAC, 0.999, threshold)
    if essential is None or essential.shape != (3, 3):
        return None, 0
    count, rotation, _, _ = cv2.recoverPose(essential, a, b, np.eye(3), mask=mask)
    return rotation, int(count)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def check_loops(
    matches: dict[tuple[int, int], np.ndarray],
    listing: list[dict],
    features: list[dict],
    names: list[str],
    reference_models: list,
    max_rotation_deg: float,
    ransac_px: float,
    min_inliers: int,
) -> tuple[dict[tuple[int, int], np.ndarray], dict]:
    """Drop retrieval pairs that disagree with the sequential reference models."""
    kinds = {(p["i"], p["j"]): p["kind"] for p in listing}
    located = {}
    for number, model in enumerate(reference_models):
        for image in model.images.values():
            if image.has_pose:
                located[image.name] = (number, model, image)

    kept, records = {}, []
    for (i, j), pair_matches in matches.items():
        if kinds[(i, j)] != "retrieval":
            kept[(i, j)] = pair_matches
            continue
        record = {"i": i, "j": j}
        a, b = located.get(names[i]), located.get(names[j])
        if a is None or b is None:
            record["accepted"], record["reason"] = False, "unregistered in reference"
        elif a[0] != b[0]:
            # Joins two sequential models: no rotation to compare with.
            strong = len(pair_matches) >= 2 * min_inliers
            record["accepted"], record["reason"] = strong, "joins models"
        else:
            model = a[1]
            xy_a = normalized_points(model, a[2], features[i]["keypoints"][pair_matches[:, 0]])
            xy_b = normalized_points(model, b[2], features[j]["keypoints"][pair_matches[:, 1]])
            focal = model.cameras[a[2].camera_id].mean_focal_length()
            measured, inliers = relative_rotation(xy_a, xy_b, ransac_px / focal)
            record["inliers"] = inliers
            if measured is None:
                record["accepted"], record["reason"] = False, "no essential matrix"
            else:
                # x_b = R x_a in camera coordinates; reference R = R_b R_aᵀ (world-to-camera).
                rotation_a = a[2].cam_from_world().rotation.matrix()
                rotation_b = b[2].cam_from_world().rotation.matrix()
                expected = rotation_b @ rotation_a.T
                error = rotation_angle_deg(measured @ expected.T)
                record["rotation_error_deg"] = error
                record["accepted"] = error <= max_rotation_deg
                record["reason"] = "rotation"
        if record["accepted"]:
            kept[(i, j)] = pair_matches
        records.append(record)

    errors = [r["rotation_error_deg"] for r in records if "rotation_error_deg" in r]
    summary = {
        "checked": len(records),
        "accepted": int(sum(r["accepted"] for r in records)),
        "joins_models": int(sum(r.get("reason") == "joins models" for r in records)),
        "rotation_error_deg_percentiles": {
            str(q): float(np.percentile(errors, q)) for q in (10, 50, 90, 99)
        }
        if errors
        else {},
    }
    return kept, summary
