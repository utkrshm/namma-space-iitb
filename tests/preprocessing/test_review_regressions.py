from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from src.preprocessing.decode import decode_frames
from src.preprocessing.probe import probe_clip
from src.preprocessing.revisits import match_image_pair, revisits_available
from src.preprocessing.scoring import blur_flags, score_clip, texture_and_reblur
from src.preprocessing.selection import select_clip
from src.preprocessing.tracking import detect_corners

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_ceiling_global_corner_floor_and_temporal_continuity(config):
    clip = probe_clip(FIXTURES / "coorg_ceiling.mp4")
    with av.open(clip.path) as container:
        assert not any("location" in key.lower() for key in container.metadata)
        assert not any("location" in key.lower() for key in container.streams.video[0].metadata)
    assert clip.rotation_deg == -90
    assert Path(clip.path).stat().st_size < 5_000_000
    frames = list(decode_frames(clip, config.decode))
    assert len(frames) == 480
    frame = min(frames, key=lambda f: abs(f.t_s - 3.0))
    points = detect_corners(frame.image, config.tracking)
    response = cv2.cornerMinEigenVal(frame.image, blockSize=3, ksize=3)
    xy = np.rint(points).astype(int)
    floor = max(
        config.tracking.absolute_eigen_floor,
        config.tracking.global_quality_floor * float(response.max()),
    )
    assert len(points) >= config.tracking.min_tracks
    assert np.all(response[xy[:, 1], xy[:, 0]] >= floor)
    scores = score_clip(clip, config)
    selected, _, _, links = select_clip(clip, scores, config)
    assert all(link["supported"] for link in links), links
    # Previously almost every minimum interval triggered a new reference (~25 per 8s).
    assert 4 <= sum(f.pool == "geometry" for f in selected) <= 18


@pytest.mark.parametrize(
    "name,expected",
    [
        ("5", "sharp"),
        ("12_5", "low_texture"),
        ("25", "sharp"),
        ("103_5", "low_texture"),
        ("140", "sharp"),
        ("172", "low_texture"),
        ("204", "low_texture"),
    ],
)
def test_reviewed_real_frame_labels(name, expected, config):
    gray = cv2.imread(str(FIXTURES / "coorg_labels" / f"coorg_{name}s.png"), 0)
    assert gray is not None and gray.shape == (640, 360)
    evidence = texture_and_reblur(gray, config.scoring)
    flags = blur_flags(*evidence, config.scoring)
    assert "suspected_blur" not in flags
    # A sharp frame can still have insufficient texture for a confident blur decision.
    if expected == "low_texture":
        assert flags == ("low_texture",)
        # Even a neighboring textured scene must not turn this into a quality failure.
        assert blur_flags(*evidence, config.scoring, relative_sharpness=0.01) == flags
    elif name != "5":
        # The 5 s frame is sharp but plain enough to receive informational low_texture.
        assert flags == ()


@pytest.mark.xfail(
    strict=True, reason="640px re-blur evidence overlaps sharp doors; requires context"
)
def test_motion_blurred_door_from_per_frame_evidence_alone(config):
    gray = cv2.imread(str(FIXTURES / "coorg_labels/coorg_49_5s.png"), 0)
    assert "suspected_blur" in blur_flags(*texture_and_reblur(gray, config.scoring), config.scoring)


def test_motion_blurred_door_with_context_and_without_flow(config):
    gray = cv2.imread(str(FIXTURES / "coorg_labels/coorg_49_5s.png"), 0)
    evidence = texture_and_reblur(gray, config.scoring)
    assert "suspected_blur" in blur_flags(*evidence, config.scoring, relative_sharpness=0.2)


@pytest.mark.skipif(not revisits_available(), reason="optional revisits extra not installed")
def test_real_different_buildings_rejected(frames, config, tmp_path):
    room = tmp_path / "room.png"
    assert cv2.imwrite(str(room), frames[0].image)
    assert match_image_pair(room, FIXTURES / "coorg_labels/coorg_5s.png", config.revisits) is None
