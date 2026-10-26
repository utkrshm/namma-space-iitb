"""Focused checks for control-scene construction and held-out flow audits."""

import numpy as np

from src.refinement.contracts import ObservationTracks, RefinementScene
from src.refinement.controls import heldout_protection_report


def scene():
    height, width = 48, 64
    image = np.zeros((1, height, width, 3), np.uint8)
    depth = np.ones((1, height, width), np.float32)
    return RefinementScene(
        image,
        depth,
        depth.copy(),
        np.ones_like(depth, bool),
        np.array([[[50, 0, 31.5], [0, 50, 23.5], [0, 0, 1]]]),
        np.eye(3)[None],
        np.zeros((1, 3)),
        np.eye(3)[None],
        np.array([[0, 0, width - 1, height - 1]]),
        np.zeros(1, np.int32),
        ({"frame_id": "frame-0"},),
        "test-scene",
    )


def tracks(points, splits, pairs=None):
    count = len(points)
    return ObservationTracks(
        np.zeros(count, np.int32),
        np.asarray(points, np.float32),
        np.arange(count, dtype=np.int32),
        np.asarray(splits, np.int8),
        np.empty((0, 2), np.int32) if pairs is None else np.asarray(pairs, np.int32),
        np.arange(count, dtype=np.int32),
    )


def test_heldout_audit_passes_for_train_addition_outside_protection():
    original = tracks([[25, 24]], [1])
    added = tracks([[25, 24], [48, 24]], [1, 0])
    report = heldout_protection_report(scene(), original, added)
    assert report["original_arrays_unchanged"]
    assert report["added_training_only"]
    assert report["heldout_mask_hits"] == 0
    assert report["nearest_heldout_distance_pixels"]["minimum"] > 10


def test_heldout_audit_finds_nontraining_leak_and_preserves_empty_additions():
    original = tracks([[25, 24]], [2])
    leaked = tracks([[25, 24], [48, 24]], [2, 1])
    report = heldout_protection_report(scene(), original, leaked)
    assert not report["added_training_only"]
    assert report["added_nontraining_observations"] == 1
    empty = heldout_protection_report(scene(), original, original)
    assert empty["added_observations"] == 0
    assert empty["nearest_heldout_distance_pixels"]["minimum"] is None


def test_heldout_audit_preserves_existing_pairs_and_allows_added_pairs():
    original = tracks([[25, 24]], [0], pairs=[[0, 2]])
    augmented = tracks([[25, 24], [48, 24]], [0, 0], pairs=[[0, 1], [0, 2]])
    report = heldout_protection_report(scene(), original, augmented)
    assert report["original_pairs_preserved"]
    assert report["added_pairs"] == 1

    lost_pair = tracks([[25, 24], [48, 24]], [0, 0], pairs=[[0, 1]])
    try:
        heldout_protection_report(scene(), original, lost_pair)
    except AssertionError as error:
        assert "removed an existing verified frame pair" in str(error)
    else:
        raise AssertionError("The audit accepted augmentation that dropped a baseline pair")
