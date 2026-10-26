"""Inspection input selection and component grouping contracts."""

import numpy as np
import pytest

from src.refinement.contracts import ObservationTracks
from src.refinement.inspection import (
    _component_point_mask,
    _components,
    _residual_stats,
    _tracks_path,
)


def test_tracks_source_priority_and_relocated_manifest(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    effective = root / "effective_tracks.npz"
    effective.touch()
    assert _tracks_path(root, {"tracks_source": "missing.npz"}) == effective
    effective.unlink()
    source = tmp_path / "shared" / "tracks.npz"
    source.parent.mkdir()
    source.touch()
    assert _tracks_path(root, {"tracks_source": str(source)}) == source


def test_tracks_fallback_requires_unambiguous_dense_cache(tmp_path):
    root = tmp_path / "run"
    cache = root / "dense_matches"
    cache.mkdir(parents=True)
    one = cache / "tracks_one.npz"
    one.touch()
    assert _tracks_path(root, {}) == one
    (cache / "tracks_two.npz").touch()
    assert _tracks_path(root, {}) is None


def test_component_groups_include_unconnected_frames():
    tracks = ObservationTracks(
        np.array([0, 1, 3, 4]),
        np.zeros((4, 2)),
        np.array([0, 0, 1, 1]),
        np.zeros(4, dtype=int),
        np.array([[0, 1], [3, 4]]),
        np.zeros(4, dtype=int),
    )
    groups = _components(tracks, 6)
    assert sorted(groups.values()) == [[0, 1], [2], [3, 4], [5]]


def test_component_mask_selects_exact_source_frame_partition():
    frame_indices = np.array([2, 0, 2, 1, 0])
    np.testing.assert_array_equal(
        _component_point_mask(frame_indices, [0, 2]), [True, True, True, False, True]
    )


def test_residual_statistics_are_computed_from_pooled_observations():
    stats = _residual_stats(np.array([0.0, 0.0, 0.0, 10.0]))
    assert stats["count"] == 4
    assert stats["median"] == 0.0
    assert stats["p90"] == pytest.approx(7.0)
    assert _residual_stats(np.empty(0)) == {"count": 0, "median": None, "p90": None}
