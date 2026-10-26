import numpy as np
import pytest

from src.refinement.validation import _check, _check_provenance, _check_support, _read_ply


def test_read_ascii_ply_preserves_points_and_rgb(tmp_path):
    path = tmp_path / "cloud.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        "1 2 3 10 20 30\n4 5 6 40 50 60\n"
    )

    points, colors = _read_ply(path)

    np.testing.assert_array_equal(points, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(colors, [[10, 20, 30], [40, 50, 60]])


def test_read_ply_rejects_truncated_vertices(tmp_path):
    path = tmp_path / "broken.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n")

    with pytest.raises(ValueError, match="Invalid PLY"):
        _read_ply(path)


def test_check_reports_contract_failure():
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _check(False, "fingerprint mismatch")


@pytest.mark.parametrize(
    ("frames", "pixels", "expected_frames", "expected_pixels", "message"),
    [
        ([0, 1], [[2, 3], [4, 5]], [0], [[2, 3]], "provenance"),
        ([0, 0], [[2, 3], [2, 3]], [0], [[2, 3]], "provenance"),
    ],
)
def test_provenance_rejects_deleted_or_duplicated_observations(
    frames, pixels, expected_frames, expected_pixels, message
):
    with pytest.raises(ValueError, match=message):
        _check_provenance(
            np.asarray(frames),
            np.asarray(pixels),
            np.asarray(expected_frames),
            np.asarray(expected_pixels),
            "test",
        )


@pytest.mark.parametrize(
    ("support", "contradictions", "message"),
    [([1], [0], "Minimum support"), ([2], [3], "Contradictions exceed")],
)
def test_supported_export_rejects_invalid_counts(support, contradictions, message):
    with pytest.raises(ValueError, match=message):
        _check_support(np.asarray(support), np.asarray(contradictions), 2, "point_cloud")
