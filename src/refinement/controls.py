"""Executable real-model controls for dense matching and held-out flow protection."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .contracts import ObservationTracks, RefinementConfig
from .dense_matching import augment_dense_tracks
from .flow import augment_texture_tracks
from .inputs import RefinementScene


def _singleton_tracks(frame_count: int) -> ObservationTracks:
    frames = np.arange(frame_count, dtype=np.int32)
    return ObservationTracks(
        frames,
        np.zeros((frame_count, 2), np.float32),
        frames.copy(),
        np.zeros(frame_count, np.int8),
        np.empty((0, 2), np.int32),
        np.full(frame_count, -1, np.int32),
    )


def _control_scene(images: np.ndarray, fingerprint: str) -> RefinementScene:
    count, height, width = images.shape[:3]
    intrinsics = np.tile(
        [
            [max(height, width), 0, (width - 1) / 2],
            [0, max(height, width), (height - 1) / 2],
            [0, 0, 1],
        ],
        (count, 1, 1),
    ).astype(np.float64)
    return RefinementScene(
        images,
        np.ones((count, height, width), np.float32),
        np.ones((count, height, width), np.float32),
        np.ones((count, height, width), bool),
        intrinsics,
        np.tile(np.eye(3), (count, 1, 1)),
        np.zeros((count, 3)),
        np.tile(np.eye(3), (count, 1, 1)),
        np.tile([0, 0, width - 1, height - 1], (count, 1)),
        np.zeros(count, np.int32),
        tuple({"frame_id": f"control-{i}"} for i in range(count)),
        fingerprint,
    )


def _protected_mask(scene: RefinementScene, tracks: ObservationTracks, frame: int) -> np.ndarray:
    mask = cv2.erode(scene.valid_mask[frame].astype(np.uint8), np.ones((19, 19), np.uint8))
    for point in tracks.pixel_xy[(tracks.frame_indices == frame) & (tracks.split != 0)]:
        cv2.circle(mask, tuple(np.rint(point).astype(int)), 10, 0, -1)
    return mask


def heldout_protection_report(scene, original, augmented) -> dict:
    """Check unchanged original evidence, train-only additions and exact protected masks."""
    original_count = len(original.frame_indices)
    observation_fields = (
        "frame_indices",
        "pixel_xy",
        "track_indices",
        "split",
        "feature_indices",
    )
    for name in observation_fields:
        np.testing.assert_array_equal(
            getattr(augmented, name)[:original_count], getattr(original, name)
        )
    original_pairs = {tuple(pair) for pair in original.pairs.tolist()}
    augmented_pairs = {tuple(pair) for pair in augmented.pairs.tolist()}
    if not original_pairs.issubset(augmented_pairs):
        raise AssertionError("Augmentation removed an existing verified frame pair")
    extra = slice(original_count, len(augmented.frame_indices))
    added_frames = augmented.frame_indices[extra]
    added_splits = augmented.split[extra]
    protected_hits = 0
    nearest = []
    for frame, point in zip(added_frames, augmented.pixel_xy[extra], strict=True):
        heldout = original.pixel_xy[(original.frame_indices == frame) & (original.split != 0)]
        if len(heldout):
            nearest.append(float(cKDTree(heldout).query(point)[0]))
        x, y = np.rint(point).astype(int)
        mask = _protected_mask(scene, original, int(frame))
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] == 0:
            protected_hits += 1
    return {
        "original_observations": original_count,
        "added_observations": len(added_frames),
        "original_arrays_unchanged": True,
        "original_pairs_preserved": True,
        "added_pairs": len(augmented_pairs - original_pairs),
        "added_training_only": bool(np.all(added_splits == 0)),
        "added_nontraining_observations": int(np.count_nonzero(added_splits)),
        "heldout_mask_hits": protected_hits,
        "nearest_heldout_distance_pixels": {
            "minimum": min(nearest) if nearest else None,
            "median": float(np.median(nearest)) if nearest else None,
        },
    }


def run_refinement_controls(
    scene: RefinementScene,
    tracks: ObservationTracks,
    output_dir: str | Path,
    config: RefinementConfig,
    negative_scene: RefinementScene | None = None,
) -> dict:
    """Run positive/blank LoFTR controls and the real scene's held-out flow audit."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source = scene.images[0]
    # LoFTR's coarse grid is 8 pixels; use a small but resolvable multi-cell shift.
    dx, dy = 24, 16
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(source, matrix, (source.shape[1], source.shape[0]))
    positive_scene = _control_scene(
        np.stack((source, warped)), scene.fingerprint + f"-loftr-positive-{dx}-{dy}"
    )
    positive_tracks = augment_dense_tracks(
        positive_scene, _singleton_tracks(2), output / "loftr_positive", config
    )
    positive_report = heldout_protection_report(
        positive_scene, _singleton_tracks(2), positive_tracks
    )
    positive_points = positive_tracks.pixel_xy[2:].reshape(-1, 2, 2)
    residual = np.linalg.norm(positive_points[:, 1] - positive_points[:, 0] - [dx, dy], axis=1)
    cv2.imwrite(str(output / "loftr_positive_source.png"), cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "loftr_positive_warp.png"), cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))

    blank = np.zeros_like(source)
    negative_pair = np.stack((blank, blank))
    negative_scene_obj = _control_scene(negative_pair, scene.fingerprint + "-loftr-blank-negative")
    negative_tracks = augment_dense_tracks(
        negative_scene_obj, _singleton_tracks(2), output / "loftr_blank_negative", config
    )
    cv2.imwrite(str(output / "loftr_negative_blank.png"), blank)

    flow_tracks = augment_texture_tracks(scene, tracks, output / "flow")
    flow_report = heldout_protection_report(scene, tracks, flow_tracks)
    report = {
        "model": "Kornia pretrained LoFTR indoor; real checkpoint via augment_dense_tracks",
        "positive_translation_model_pixels": [dx, dy],
        "positive_accepted_correspondences": len(positive_points),
        "positive_warp_residual_pixels": {
            "median": float(np.median(residual)) if len(residual) else None,
            "p90": float(np.percentile(residual, 90)) if len(residual) else None,
            "maximum": float(np.max(residual)) if len(residual) else None,
        },
        "positive_match_sha256": hashlib.sha256(
            np.ascontiguousarray(positive_points).tobytes()
        ).hexdigest(),
        "positive_training_and_protection": positive_report,
        "negative_blank_accepted_correspondences": len(negative_tracks.frame_indices) - 2,
        "flow_heldout_protection": flow_report,
        "optional_real_scene_negative": negative_scene is not None,
        "pass": bool(
            len(positive_points) >= 30
            and len(residual) > 0
            and np.percentile(residual, 90) <= 2.5
            and len(negative_tracks.frame_indices) == 2
            and flow_report["original_arrays_unchanged"]
            and flow_report["added_training_only"]
            and flow_report["heldout_mask_hits"] == 0
            and positive_report["added_training_only"]
            and positive_report["heldout_mask_hits"] == 0
        ),
    }
    if negative_scene is not None:
        real_pair = _control_scene(
            np.stack((scene.images[0], negative_scene.images[0])),
            scene.fingerprint + "-" + negative_scene.fingerprint,
        )
        real_negative = augment_dense_tracks(
            real_pair, _singleton_tracks(2), output / "loftr_real_negative", config
        )
        cv2.imwrite(
            str(output / "loftr_real_negative_first.png"),
            cv2.cvtColor(scene.images[0], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(output / "loftr_real_negative_second.png"),
            cv2.cvtColor(negative_scene.images[0], cv2.COLOR_RGB2BGR),
        )
        report["optional_real_scene_negative_accepted"] = len(real_negative.frame_indices) - 2
    (output / "matching_controls.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
