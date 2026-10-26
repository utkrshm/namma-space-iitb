"""Validation of cached feature tracks against real source-image files."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from PIL import Image

from src.refinement import ObservationTracks, RefinementConfig, RefinementScene
from src.refinement.matching import (
    _matcher_signature,
    split_tracks,
    validate_track_cache,
)
from src.refinement.pipeline import _export_tracks


def cache_fixture(tmp_path, seed=17):
    source = tmp_path / "input"
    source.mkdir()
    image_paths = []
    for index in range(4):
        image_path = source / f"frame-{index}.png"
        Image.fromarray(np.full((16, 16, 3), index, dtype=np.uint8)).save(image_path)
        image_paths.append(image_path)

    scene = RefinementScene(
        images=np.zeros((4, 16, 16, 3), np.uint8),
        depth=np.ones((4, 16, 16), np.float32),
        confidence=np.ones((4, 16, 16), np.float32),
        valid_mask=np.ones((4, 16, 16), bool),
        intrinsics=np.tile(np.eye(3), (4, 1, 1)),
        rotations=np.tile(np.eye(3), (4, 1, 1)),
        centers=np.zeros((4, 3)),
        source_to_model=np.tile(np.eye(3), (4, 1, 1)),
        content_bounds=np.tile([0, 0, 15, 15], (4, 1)),
        owners=np.zeros(4, int),
        frames=tuple(
            {"frame_id": str(index), "image_path": str(path)}
            for index, path in enumerate(image_paths)
        ),
        fingerprint="scene-fingerprint",
    )
    config = replace(RefinementConfig(), device="cpu", seed=seed)
    track_ids = np.repeat(np.arange(40), 2)
    tracks = ObservationTracks(
        frame_indices=np.tile([0, 1], 40),
        pixel_xy=np.tile([[2.0, 3.0], [4.0, 5.0]], (40, 1)),
        track_indices=track_ids,
        split=split_tracks(track_ids, seed),
        pairs=np.array([[0, 1]], np.int32),
        feature_indices=np.arange(80, dtype=np.int32),
    )
    cache_path = tmp_path / "tracks.npz"
    np.savez_compressed(cache_path, **asdict(tracks), scene_fingerprint=scene.fingerprint)
    summary = {
        "matcher_signature": _matcher_signature(scene, image_paths, config),
        "split_seed": seed,
    }
    cache_path.with_suffix(".json").write_text(json.dumps(summary))
    return source, scene, config, cache_path, tracks


def test_matching_cache_reuses_identical_settings_and_sources(tmp_path):
    source, scene, config, cache_path, expected = cache_fixture(tmp_path)

    actual = validate_track_cache(cache_path, scene, source, config)

    np.testing.assert_array_equal(actual.track_indices, expected.track_indices)
    np.testing.assert_array_equal(actual.split, expected.split)


@pytest.mark.parametrize(
    "changes",
    [
        {"feature_size": 512},
        {"keypoints": 1024},
        {"neighbor_offsets": (1, 3)},
        {"retrieval_top_k": 2},
        {"retrieval_exclusion_s": 5.0},
    ],
)
def test_changed_matcher_settings_reject_cache(tmp_path, changes):
    source, scene, config, cache_path, _ = cache_fixture(tmp_path)

    with pytest.raises(ValueError, match="matcher settings/images differ"):
        validate_track_cache(cache_path, scene, source, replace(config, **changes))


def test_changed_source_image_contents_reject_cache(tmp_path):
    source, scene, config, cache_path, _ = cache_fixture(tmp_path)
    Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(source / "frame-0.png")

    with pytest.raises(ValueError, match="matcher settings/images differ"):
        validate_track_cache(cache_path, scene, source, config)


def test_changed_split_seed_rejects_cache(tmp_path):
    source, scene, config, cache_path, _ = cache_fixture(tmp_path)

    with pytest.raises(ValueError, match="split seed differs"):
        validate_track_cache(cache_path, scene, source, replace(config, seed=23))


def test_scene_fingerprint_mismatch_rejects_cache(tmp_path):
    source, scene, config, cache_path, _ = cache_fixture(tmp_path)

    with pytest.raises(ValueError, match="matcher settings/images differ"):
        validate_track_cache(cache_path, replace(scene, fingerprint="other-scene"), source, config)


@pytest.mark.parametrize("corruption", ["split", "frame-conflict"])
def test_invalid_whole_track_cache_is_rejected(tmp_path, corruption):
    source, scene, config, cache_path, tracks = cache_fixture(tmp_path)
    values = asdict(tracks)
    if corruption == "split":
        values["split"] = tracks.split.copy()
        values["split"][1] = (values["split"][0] + 1) % 3
    else:
        values["frame_indices"] = tracks.frame_indices.copy()
        values["frame_indices"][1] = values["frame_indices"][0]
    np.savez_compressed(cache_path, **values, scene_fingerprint=scene.fingerprint)

    with pytest.raises(ValueError, match="leaks across evaluation splits|conflicting observations"):
        validate_track_cache(cache_path, scene, source, config)


def test_export_tracks_prefers_effective_tracks_and_validates_integrity(tmp_path):
    _, scene, _, cache_path, tracks = cache_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    altered = replace(
        tracks,
        pixel_xy=tracks.pixel_xy + 1,
    )
    np.savez_compressed(
        output / "effective_tracks.npz",
        **asdict(altered),
        scene_fingerprint=scene.fingerprint,
    )
    previous = {"config": {}, "tracks_source": str(cache_path)}

    actual = _export_tracks(output, previous, scene)

    np.testing.assert_array_equal(actual.pixel_xy, altered.pixel_xy)


def test_export_tracks_allows_only_explicit_path_matching_saved_effective(tmp_path):
    source, scene, _, cache_path, tracks = cache_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    np.savez_compressed(
        output / "effective_tracks.npz",
        **asdict(tracks),
        scene_fingerprint=scene.fingerprint,
    )
    previous = {"config": {}}

    actual = _export_tracks(output, previous, scene, cache_path)

    np.testing.assert_array_equal(actual.track_indices, tracks.track_indices)
    other = source / "different.npz"
    np.savez_compressed(
        other,
        **asdict(replace(tracks, pixel_xy=tracks.pixel_xy + 1)),
        scene_fingerprint=scene.fingerprint,
    )
    with pytest.raises(ValueError, match="must match the saved effective tracks"):
        _export_tracks(output, previous, scene, other)


def test_legacy_export_tracks_rejects_unrecoverable_augmentation(tmp_path):
    _, scene, _, cache_path, _ = cache_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ValueError, match="cannot reconstruct the saved flow tracks"):
        _export_tracks(
            output,
            {"config": {"flow_anchors": True}, "tracks_source": str(cache_path)},
            scene,
        )


def test_legacy_dense_export_uses_single_final_cache(tmp_path):
    _, scene, _, _, tracks = cache_fixture(tmp_path)
    output = tmp_path / "output"
    dense_cache = output / "dense_matches"
    dense_cache.mkdir(parents=True)
    final_path = dense_cache / "tracks_final.npz"
    np.savez_compressed(final_path, **asdict(tracks), scene_fingerprint=scene.fingerprint)

    actual = _export_tracks(output, {"config": {"dense_anchors": True}}, scene)

    np.testing.assert_array_equal(actual.track_indices, tracks.track_indices)
