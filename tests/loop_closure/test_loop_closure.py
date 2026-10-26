"""Loop-closure geometry on analytic scenes with exactly known cameras and depth."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.loop_closure import (
    FrameMatches,
    FrameModel,
    LoopConfig,
    LoopPair,
    build_tracks,
    bundle_adjust,
    chunk_frames,
    correspondence_disagreement,
    retrieve_candidates,
)
from src.reconstruction import Sim3

SIZE = 96
K = np.array([[80.0, 0.0, 47.5], [0.0, 80.0, 47.5], [0.0, 0.0, 1.0]])
WALL_Z = 5.0


def _pose(rotvec_deg, center):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(np.radians(rotvec_deg)).as_matrix()
    pose[:3, 3] = center
    return pose


def _wall_depth(pose):
    """Camera z-depth of the plane Z = WALL_Z for every pixel (ray-plane intersection)."""
    y, x = np.indices((SIZE, SIZE), dtype=np.float64)
    rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(K).T
    world_rays = rays @ pose[:3, :3].T
    distance = (WALL_Z - pose[2, 3]) / world_rays[..., 2]
    return (rays[..., 2] * distance).astype(np.float32)


def _model(index, pose, depth_pose=None):
    depth = _wall_depth(pose if depth_pose is None else depth_pose)
    return FrameModel(
        frame_id=f"f{index}",
        timestamp_s=float(index),
        owner_section=0,
        intrinsics=K.copy(),
        camera_to_world=pose.copy(),
        depth=depth,
        depth_scale=1.0,
        confidence=np.ones_like(depth),
        valid=np.ones(depth.shape, dtype=bool),
        image=np.zeros((SIZE, SIZE, 3), np.uint8),
    )


def _project(pose, points):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:]


def _scene():
    rng = np.random.default_rng(0)
    truth = [
        _pose([0, 0, 0], [0.0, 0.0, 0.0]),
        _pose([1, 3, 0], [0.3, 0.0, 0.1]),
        _pose([-2, 6, 1], [0.6, 0.1, 0.0]),
        _pose([2, 9, -1], [0.9, -0.1, 0.2]),
        _pose([0, 12, 0], [1.2, 0.0, 0.1]),
    ]
    points = np.column_stack(
        (rng.uniform(-1.5, 3.0, 600), rng.uniform(-2.0, 2.0, 600), np.full(600, WALL_Z))
    )
    matches = []
    for a in range(len(truth)):
        for b in range(a + 1, len(truth)):
            pa, pb = _project(truth[a], points), _project(truth[b], points)
            inside = np.all((pa > 1) & (pa < SIZE - 2) & (pb > 1) & (pb < SIZE - 2), axis=1)
            matches.append(FrameMatches(a, b, pa[inside], pb[inside], "sequential"))
    return truth, matches


def test_bundle_adjustment_recovers_perturbed_cameras():
    truth, matches = _scene()
    rng = np.random.default_rng(1)
    initial = [truth[0]]
    for pose in truth[1:]:
        noisy = pose.copy()
        noisy[:3, :3] = (
            Rotation.from_rotvec(np.radians(rng.normal(0, 1.5, 3))).as_matrix() @ pose[:3, :3]
        )
        noisy[:3, 3] = pose[:3, 3] + rng.normal(0, 0.05, 3)
        initial.append(noisy)
    # Depth maps are the true per-camera depths; only the global poses are wrong.
    models = [_model(i, initial[i], truth[i]) for i in range(len(truth))]
    before = correspondence_disagreement(models, matches)["median"]

    cameras, report = bundle_adjust(models, matches, LoopConfig(max_ba_iterations=200))

    rotation_errors = [
        np.degrees(Rotation.from_matrix(c[:3, :3].T @ t[:3, :3]).magnitude())
        for c, t in zip(cameras["camera_to_world"], truth)
    ]
    center_errors = [
        np.linalg.norm(c[:3, 3] - t[:3, 3]) for c, t in zip(cameras["camera_to_world"], truth)
    ]
    assert max(rotation_errors) < 0.1
    assert max(center_errors) < 0.01
    assert np.allclose(cameras["depth_scale"], 1.0, atol=1e-3)
    assert report["reprojection_after_px"]["median"] < 0.05

    refined = [_model(i, cameras["camera_to_world"][i], truth[i]) for i in range(len(truth))]
    after = correspondence_disagreement(refined, matches)["median"]
    assert after < before * 0.05


def test_correspondence_disagreement_is_zero_for_true_cameras():
    truth, matches = _scene()
    models = [_model(i, pose) for i, pose in enumerate(truth)]
    result = correspondence_disagreement(models, matches)
    assert result["pairs"] == len(matches)
    assert result["median"] < 1e-3


def test_tracks_merge_shared_keypoints_and_drop_conflicts():
    a = np.array([[10.0, 10.0], [20.0, 20.0]])
    b = np.array([[11.0, 10.0], [21.0, 20.0]])
    c = np.array([[12.0, 10.0], [40.0, 40.0]])
    matches = [
        FrameMatches(0, 1, a, b, "sequential"),
        FrameMatches(1, 2, b, c, "sequential"),
        # Conflicting: keypoint (20,20) in frame 0 would link to a second point in frame 2.
        FrameMatches(0, 2, a[1:], np.array([[50.0, 50.0]]), "loop"),
    ]
    tracks = build_tracks(matches, max_length=10)
    lengths = sorted(len(track) for track in tracks)
    assert lengths == [3]
    assert [frame for frame, _ in tracks[0]] == [0, 1, 2]


def test_loop_edge_composition_maps_section_b_into_section_a():
    rng = np.random.default_rng(2)
    chunk_to_a = Sim3(
        1.7, Rotation.from_rotvec([0.1, 0.5, -0.2]).as_matrix(), np.array([1.0, -2.0, 0.3])
    )
    chunk_to_b = Sim3(
        0.6, Rotation.from_rotvec([-0.3, 0.2, 0.9]).as_matrix(), np.array([0.2, 0.4, 5.0])
    )
    points_chunk = rng.normal(size=(50, 3))
    points_a, points_b = chunk_to_a.apply(points_chunk), chunk_to_b.apply(points_chunk)
    relative = chunk_to_a.compose(chunk_to_b.inverse())
    assert np.allclose(relative.apply(points_b), points_a, atol=1e-9)


def test_retrieval_respects_temporal_exclusion_and_threshold():
    descriptors = np.eye(4)[[0, 1, 0, 2]].astype(float)
    descriptors[2] = descriptors[2] * 0.9 + 0.1 * np.eye(4)[1]
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    timestamps = np.array([0.0, 5.0, 30.0, 31.0])
    pairs = retrieve_candidates(
        descriptors, timestamps, LoopConfig(temporal_exclusion_s=20.0, min_similarity=0.9)
    )
    assert [(a, b) for a, b, _ in pairs] == [(0, 2)]


@pytest.mark.parametrize("frame_a, frame_b", [(10, 100), (2, 159)])
def test_chunk_windows_are_disjoint_and_clipped(frame_a, frame_b):
    side_a, side_b = chunk_frames(LoopPair(frame_a, frame_b, 0.9, 100, 0.5), 161, 6)
    assert not set(side_a) & set(side_b)
    assert frame_a in side_a and frame_b in side_b
    assert min(side_a + side_b) >= 0 and max(side_a + side_b) <= 160


def test_square_walls_prior_measures_a_yaw_error():
    from src.loop_closure import ManhattanPrior

    # Cameras look along +Z at the wall Z = WALL_Z; image "up" is world -Y.
    straight = [_model(i, _pose([0, 0, 0], [0.3 * i, 0.0, 0.0])) for i in range(4)]
    yawed_pose = _pose([0, 20, 0], [1.2, 0.0, 0.0])
    models = [*straight, _model(4, yawed_pose)]
    prior = ManhattanPrior(models, sigma_deg=3.0, min_concentration=0.5)

    angles = np.degrees(prior.frame_angles(np.stack([m.camera_to_world[:3, :3] for m in models])))
    deviation = np.abs((angles - np.degrees(prior.reference) + 45) % 90 - 45)
    assert np.allclose(deviation[:4], 0.0, atol=0.5)
    # The fifth camera sees the same wall: its world wall direction is unchanged, so a
    # correct camera rotation gives no error; rotating the camera by 20° without the
    # wall turning (drift) does.
    assert deviation[4] < 0.5
    drifted = np.stack([m.camera_to_world[:3, :3] for m in models])
    drifted[4] = Rotation.from_rotvec(np.radians([0, 20, 0])).as_matrix() @ drifted[4]
    drift_angles = np.degrees(prior.frame_angles(drifted))
    drift_deviation = abs((drift_angles[4] - np.degrees(prior.reference) + 45) % 90 - 45)
    assert abs(drift_deviation - 20.0) < 0.5
    assert abs(prior.residuals(drifted)[-1]) > 5
