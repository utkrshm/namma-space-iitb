"""Dense SfM helpers on a real fixture video and on analytic geometry."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.preprocessing.config import DecodeConfig
from src.preprocessing.contracts import FrameScores
from src.preprocessing.decode import decode_frames
from src.preprocessing.probe import probe_clip
from src.sfm import (
    SfmConfig,
    build_pairs,
    planar_fraction,
    select_frames,
    trajectory_continuity,
    umeyama,
)
from src.sfm.frames import decode_selected

FIXTURE = Path(__file__).parents[1] / "fixtures" / "room_short.mp4"


def test_selective_decoder_matches_preprocessing_frames():
    """Frame IDs, orientation and pixels are identical to the preprocessing decoder."""
    clip = probe_clip(FIXTURE)
    reference = list(decode_frames(clip, DecodeConfig(), full_resolution=True, rgb=True))
    wanted = {reference[i].frame_id for i in (0, 7, len(reference) - 1)}

    decoded = dict(decode_selected(clip, wanted, threads=4))

    assert set(decoded) == wanted
    for frame in reference:
        if frame.frame_id in wanted:
            assert np.array_equal(decoded[frame.frame_id], frame.image)


def _score(index, t, sharpness, flags=()):
    return FrameScores(
        f"clip:{index}", "clip", index, index, t, sharpness, 1.0, 1.0, 0.0, 1.0, True,
        0.0, 0.0, 100.0, 10.0, 1 / 60, tuple(flags),
    )  # fmt: skip


def test_frame_selection_takes_sharpest_clean_frame_per_window_and_all_geometry():
    scores = [
        _score(0, 0.00, 5.0),
        _score(1, 0.10, 9.0, ("suspected_blur",)),  # sharper but flagged
        _score(2, 0.20, 7.0),
        _score(3, 0.30, 1.0, ("suspected_blur",)),  # only frame of its window
        _score(4, 0.60, 2.0),
        _score(5, 0.70, 3.0),
    ]
    config = SfmConfig(frames_per_second=4.0)

    frames = select_frames(scores, {"clip:4"}, config)

    assert [f["frame_id"] for f in frames] == ["clip:2", "clip:3", "clip:4", "clip:5"]
    assert [f["name"] for f in frames] == ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"]
    assert [f["geometry"] for f in frames] == [False, False, True, False]


def test_pairs_cover_sequential_window_and_far_retrieval_only():
    timestamps = np.arange(20) * 0.25
    descriptors = np.eye(20)
    descriptors[15] = descriptors[2]  # frame 15 revisits frame 2, 3.25 s later
    config = SfmConfig(sequential_window=2, retrieval_exclusion_s=3.0, retrieval_top_k=1)

    pairs = build_pairs(
        timestamps, descriptors / np.linalg.norm(descriptors, axis=1)[:, None], config
    )

    sequential = {(i, j) for i, j, kind in pairs if kind == "sequential"}
    retrieval = {(i, j) for i, j, kind in pairs if kind == "retrieval"}
    assert sequential == {(i, j) for i in range(20) for j in (i + 1, i + 2) if j < 20}
    assert retrieval == {(2, 15)}


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(30, 3))
    rotation = Rotation.from_rotvec([0.3, -0.5, 1.1]).as_matrix()
    target = 2.5 * source @ rotation.T + [1.0, -2.0, 0.5]

    scale, estimated, translation = umeyama(source, target)

    assert scale == pytest.approx(2.5)
    assert np.allclose(estimated, rotation)
    assert np.allclose(translation, [1.0, -2.0, 0.5])


def _project(points, rotation, centre, focal=800.0):
    camera = (points - centre) @ rotation.T
    return (focal * camera[:, :2] / camera[:, 2:] + 500.0).astype(np.float32)


def test_planar_fraction_separates_planar_patterns_from_3d_structure():
    """A repeated pattern on one wall is fully explained by a homography; a room is not."""
    rng = np.random.default_rng(1)
    rotation = Rotation.from_rotvec([0.0, 0.25, 0.0]).as_matrix()
    centre = np.array([0.8, 0.0, 0.0])
    wall = np.column_stack((rng.uniform(-2, 2, (300, 2)), np.full(300, 5.0)))
    room = np.column_stack((rng.uniform(-2, 2, (300, 2)), rng.uniform(2.0, 9.0, 300)))

    _, wall_fraction = planar_fraction(
        _project(wall, np.eye(3), np.zeros(3)), _project(wall, rotation, centre), 1.0
    )
    inliers, room_fraction = planar_fraction(
        _project(room, np.eye(3), np.zeros(3)), _project(room, rotation, centre), 1.0
    )

    assert wall_fraction > 0.95
    assert inliers > 250
    assert room_fraction < SfmConfig().retrieval_max_planar_fraction


def test_continuity_flags_misplaced_segments_but_not_smooth_walks():
    frames = [{"frame_id": f"f{i}", "timestamp_s": 0.25 * i} for i in range(40)]
    poses = {}
    for i in range(40):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([0, np.radians(2 * i), 0]).as_matrix()
        pose[:3, 3] = [0.05 * i, 0, 0]
        if i >= 25:  # a segment placed 3 m away and turned around
            pose[:3, :3] = Rotation.from_rotvec([0, np.pi, 0]).as_matrix() @ pose[:3, :3]
            pose[:3, 3] += [0, 0, 3.0]
        poses[f"f{i}"] = pose

    report = trajectory_continuity(poses, frames)

    assert report["median_step"] == pytest.approx(0.05)
    assert [(b["from_s"], b["to_s"]) for b in report["breaks"]] == [(6.0, 6.25)]
    assert report["breaks"][0]["turn_deg"] > 170
