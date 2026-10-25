"""Additional real texture tracks, spatially disjoint from learned feature observations."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .contracts import ObservationTracks


def _patches(image, xy):
    patches = np.stack(
        [cv2.getRectSubPix(image, (9, 9), tuple(map(float, p))).ravel() for p in xy]
    ).astype(np.float32)
    patches -= patches.mean(1, keepdims=True)
    patches /= np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), 1e-6)
    return patches


def _unique(a, b, xy_a, xy_b):
    cross = np.maximum(2 - 2 * np.sum(a * b, axis=1), 0)
    good = cross < 0.4  # Normalized correlation > 0.8.
    for descriptors, pixels in ((a, xy_a), (b, xy_b)):
        own = np.maximum(2 - 2 * descriptors @ descriptors.T, 0)
        spatial = np.linalg.norm(pixels[:, None] - pixels[None], axis=2)
        own[spatial <= 24] = np.inf
        good &= cross < 0.8**2 * own.min(1)
    return good


def augment_texture_tracks(scene, tracks, output_dir):
    """Add measured LK tracks only to training, protecting all learned-feature pixels."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for value in asdict(tracks).values():
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(b"texture-v2-protect-validation-test-radius10")
    file = destination / f"tracks_texture_{digest.hexdigest()[:16]}.npz"
    if file.is_file():
        from .matching import load_observation_tracks

        return load_observation_tracks(file, scene)

    gray = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in scene.images]
    protected = []
    for frame in range(scene.frame_count):
        mask = cv2.erode(scene.valid_mask[frame].astype(np.uint8), np.ones((19, 19), np.uint8))
        for xy in tracks.pixel_xy[(tracks.frame_indices == frame) & (tracks.split != 0)]:
            cv2.circle(mask, tuple(np.rint(xy).astype(int)), 10, 0, -1)
        protected.append(mask)

    groups, records = [], []
    active_xy, active_ids = np.empty((0, 2), np.float32), np.empty(0, int)
    all_groups = []
    pair_list = []
    for frame in range(scene.frame_count - 1):
        first, second = scene.frames[frame], scene.frames[frame + 1]
        if first.get("clip_id") != second.get("clip_id"):
            active_xy, active_ids = np.empty((0, 2), np.float32), np.empty(0, int)
            continue
        allowed = protected[frame].copy()
        for point in active_xy:
            cv2.circle(allowed, tuple(np.rint(point).astype(int)), 8, 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray[frame],
            maxCorners=max(1, 600 - len(active_xy)),
            qualityLevel=0.005,
            minDistance=8,
            mask=allowed,
            blockSize=5,
        )
        if corners is not None and len(active_xy) < 600:
            corners = corners[:, 0]
            ids = np.arange(len(all_groups), len(all_groups) + len(corners))
            all_groups.extend([[(frame, point.copy())] for point in corners])
            active_xy = np.concatenate((active_xy, corners))
            active_ids = np.concatenate((active_ids, ids))
        if not len(active_xy):
            continue

        target, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[frame],
            gray[frame + 1],
            active_xy[:, None],
            None,
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
        )
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray[frame + 1], gray[frame], target, None, winSize=(31, 31), maxLevel=4
        )
        target, backward = target[:, 0], backward[:, 0]
        valid = status.ravel().astype(bool) & back_status.ravel().astype(bool)
        valid &= np.isfinite(target).all(1) & (np.linalg.norm(backward - active_xy, axis=1) < 0.6)
        x, y = np.rint(np.nan_to_num(target)).astype(int).T
        inside = (x >= 0) & (x < gray[frame].shape[1]) & (y >= 0) & (y < gray[frame].shape[0])
        valid &= inside
        selected = np.flatnonzero(valid)
        valid[selected] &= protected[frame + 1][y[selected], x[selected]] > 0
        selected = np.flatnonzero(valid)
        if len(selected) >= 20:
            a, b = active_xy[selected], target[selected]
            unique = _unique(_patches(gray[frame], a), _patches(gray[frame + 1], b), a, b)
            selected = selected[unique]
        if len(selected) >= 20:
            matrix, inliers = cv2.findFundamentalMat(
                active_xy[selected], target[selected], cv2.USAC_MAGSAC, 1.0, 0.999, 10000
            )
            if matrix is not None and inliers is not None:
                selected = selected[inliers.ravel().astype(bool)]
            else:
                selected = np.empty(0, int)
        else:
            selected = np.empty(0, int)
        if len(selected) >= 20:
            from ..preprocessing.revisits import spatial_spread

            if (
                min(
                    spatial_spread(active_xy[selected], gray[frame].shape)[0],
                    spatial_spread(target[selected], gray[frame + 1].shape)[0],
                )
                < 0.125
            ):
                selected = np.empty(0, int)
        records.append({"first": frame, "second": frame + 1, "accepted": len(selected)})
        if len(selected):
            pair_list.append((frame, frame + 1))
        for index in selected:
            all_groups[active_ids[index]].append((frame + 1, target[index].copy()))
        active_xy, active_ids = target[selected], active_ids[selected]
        if frame % 30 == 0:
            print(
                f"Texture tracking {frame + 1}/{scene.frame_count - 1}: {len(selected)} measured anchors",
                flush=True,
            )

    groups = [g for g in all_groups if len(g) >= 2]
    frames, pixels, ids = [], [], []
    offset = int(tracks.track_indices.max()) + 1
    for index, group in enumerate(groups):
        for frame, pixel in group:
            frames.append(frame)
            pixels.append(pixel)
            ids.append(offset + index)
    if not groups:
        return tracks
    count = len(frames)
    result = ObservationTracks(
        np.r_[tracks.frame_indices, np.asarray(frames, np.int32)],
        np.concatenate((tracks.pixel_xy, np.asarray(pixels, np.float32))),
        np.r_[tracks.track_indices, np.asarray(ids, np.int32)],
        np.r_[tracks.split, np.zeros(count, np.int8)],
        np.unique(np.concatenate((tracks.pairs, np.asarray(pair_list, np.int32))), axis=0),
        np.r_[tracks.feature_indices, np.full(count, -1, np.int32)],
    )
    np.savez_compressed(file, **asdict(result), scene_fingerprint=scene.fingerprint)
    (destination / "texture_tracks.json").write_text(
        json.dumps(
            {
                "tracks": len(groups),
                "observations": count,
                "protected_radius_model_pixels": 10,
                "protected_splits": ["validation", "test"],
                "evaluation": "original ALIKED validation/test only; texture tracks train only",
                "pairs": records,
            },
            indent=2,
        )
        + "\n"
    )
    return result
