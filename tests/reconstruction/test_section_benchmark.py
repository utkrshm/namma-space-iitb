import numpy as np
import pytest

from src.reconstruction.section_benchmark import (
    _bilinear,
    _require_spatial_support,
    _source_pixels,
    map_pixel_centres,
    normalized_disagreement,
)


def test_pixel_mapping_preserves_center_convention():
    matrix = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1]])
    assert np.allclose(
        map_pixel_centres(np.array([[0, 0], [2, 3]]), matrix), [[0.5, 0.5], [4.5, 6.5]]
    )


def test_disagreement_reports_each_arbitrary_scene_scale():
    first = np.array([[1.0, 0, 0], [3, 0, 0], [5, 0, 0]])
    second = first + np.array([0.2, 0, 0])
    centers = np.zeros_like(first)
    result = normalized_disagreement(first, second, centers)
    assert result["count"] == 3
    assert result["median"] == pytest.approx(0.2)
    assert result["first_normalized_median"] == pytest.approx(0.2 / 3.0)


def test_disagreement_rejects_empty_or_mismatched_points():
    with pytest.raises(ValueError):
        normalized_disagreement(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
    with pytest.raises(ValueError):
        map_pixel_centres(np.ones((2, 2)), np.eye(2))


def test_bilinear_samples_require_all_four_valid_neighbors():
    yy, xx = np.indices((6, 6))
    image = np.stack([xx, yy, xx + yy], axis=-1).astype(np.float32)
    valid = np.ones((6, 6), bool)
    xy = np.array([[1.5, 2.5], [2.5, 2.5]])
    valid[2, 1] = False
    result = _bilinear(xy, image, valid)
    assert np.isnan(result[0]).all()
    np.testing.assert_allclose(result[1], [2.5, 2.5, 5.0])


def test_source_feature_mapping_uses_half_pixel_centers():
    xy = _source_pixels(np.array([[0.0, 0.0], [575.0, 1023.0]]), (1024, 576), (1080, 1920))
    np.testing.assert_allclose(xy, [[0.4375, 0.4375], [1078.5625, 1918.5625]])


def test_depth_valid_revisit_patch_cannot_anchor_scale():
    rng = np.random.default_rng(7)
    # Image matches can be broad while reconstructed support has collapsed to a tiny patch.
    xy = rng.uniform(0, 1000, (50, 2))
    points = rng.normal(0, 0.001, (50, 3)) + [0, 0, 10]
    with pytest.raises(ValueError, match="geometrically tiny"):
        _require_spatial_support(xy, (1000, 1000), points, 10.0)
    with pytest.raises(ValueError, match="spatially concentrated"):
        _require_spatial_support(xy * 0.01, (1000, 1000), rng.normal(size=(50, 3)), 1.0)
