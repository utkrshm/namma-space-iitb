"""CPU tests of the camera, configuration, scene-frame, partition and depth-fit code."""

import math

import numpy as np
import pytest

from spacegen.config import PipelineConfig
from spacegen.contracts import Camera, look_at
from spacegen.fill import align_depth, probe_camera
from spacegen.partition import kmeans
from spacegen.scene_frame import SceneFrame, estimate_frame, find_areas
from spacegen.support import fibonacci_directions


def test_look_at_points_the_camera_at_the_target():
    eye, target = np.array([1.0, 2.0, 1.5]), np.array([4.0, 2.0, 1.5])
    camera = Camera(
        "c", 64, 48, np.array([[50, 0, 31.5], [0, 50, 23.5], [0, 0, 1]]), look_at(eye, target)
    )
    local = camera.world_to_camera @ np.r_[target, 1]
    assert np.allclose(local[:2], 0, atol=1e-9) and local[2] > 0
    # World up (+z) appears as image up (−y).
    above = camera.world_to_camera @ np.r_[target + np.array([0, 0, 1]), 1]
    assert above[1] < 0
    assert np.allclose(camera.centre, eye)


def test_resized_keeps_pixel_centres():
    k = np.array([[100.0, 0, 63.5], [0, 100.0, 47.5], [0, 0, 1]])
    camera = Camera("c", 128, 96, k, np.eye(4)).resized(64, 48)
    assert np.isclose(camera.intrinsics[0, 2], 31.5) and np.isclose(camera.intrinsics[0, 0], 50)


def test_config_overrides_and_rejects_typos():
    config = PipelineConfig().replace(splat={"steps": 123}, fill={"pitches_deg": [0, 10]})
    assert config.splat.steps == 123 and config.fill.pitches_deg == (0, 10)
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"splat": {"stepz": 1}})
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"nonsense": {}})


def _room(seed=0):
    """Cameras at eye height 1.5 walking two 'rooms', looking around; floor and walls."""
    rng = np.random.default_rng(seed)
    cameras = []
    for number in range(80):
        room = number // 40
        centre = np.array([room * 8.0, 0.0])
        position = np.r_[centre + rng.normal(0, 0.8, 2), 1.5 + rng.normal(0, 0.03)]
        angle = rng.uniform(0, 2 * math.pi)
        target = position + np.array([math.cos(angle), math.sin(angle), -0.5])
        k = np.array([[300.0, 0, 159.5], [0, 300.0, 119.5], [0, 0, 1]])
        cameras.append(Camera(f"{number}", 320, 240, k, look_at(position, target)))
    floor = np.c_[rng.uniform(-4, 12, (4000, 2)), rng.normal(0, 0.005, 4000)]
    walls = np.c_[rng.uniform(-4, 12, 2000), np.full(2000, 4.0), rng.uniform(0, 2.8, 2000)]
    return cameras, np.vstack((floor, walls))


def _rotate(cameras, points, rotation, scale, shift):
    """The same scene in an arbitrary SfM gauge."""
    moved = []
    for camera in cameras:
        camera_to_world = camera.camera_to_world
        new = np.eye(4)
        new[:3, :3] = rotation @ camera_to_world[:3, :3]
        new[:3, 3] = scale * rotation @ camera_to_world[:3, 3] + shift
        moved.append(camera.moved(np.linalg.inv(new)))
    return moved, scale * points @ rotation.T + shift


def test_scene_frame_recovers_up_floor_and_scale_in_any_gauge():
    cameras, points = _room()
    angle = 0.7
    rotation = np.array(
        [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]]
    )
    cameras, points = _rotate(cameras, points, rotation, 0.3, np.array([5.0, -2.0, 1.0]))
    frame = estimate_frame(cameras, points)
    assert frame.floor_found
    assert np.degrees(np.arccos(frame.up @ (rotation @ [0, 0, 1]))) < 1.0
    assert abs(frame.camera_height - 0.3 * 1.5) < 0.01
    aligned = frame.to_aligned(np.stack([c.centre for c in cameras]))
    assert abs(np.median(aligned[:, 2]) - 1.0) < 0.02
    assert np.allclose(frame.to_world(frame.to_aligned(points[:5])), points[:5])
    assert len(frame.areas) == 2 and frame.areas[0]["cameras"][0] == 0


def test_scene_frame_round_trips_through_json(tmp_path):
    cameras, points = _room(1)
    frame = estimate_frame(cameras, points)
    (tmp_path / "frame.json").write_text(__import__("json").dumps(frame.to_dict()))
    loaded = SceneFrame.load(tmp_path / "frame.json")
    assert np.allclose(loaded.rotation, frame.rotation) and loaded.areas == frame.areas


def test_areas_split_far_apart_groups_only():
    plan = np.r_[
        np.random.default_rng(0).normal(0, 0.5, (30, 2)),
        np.random.default_rng(1).normal(0, 0.5, (30, 2)) + 10,
    ]
    areas = find_areas(np.c_[plan, np.ones(60)], 2.5)
    assert [len(a["cameras"]) for a in areas] == [30, 30]
    assert len(find_areas(np.c_[plan[:30], np.ones(30)], 2.5)) == 1


def test_probe_camera_looks_along_aligned_yaw():
    cameras, points = _room(2)
    frame = estimate_frame(cameras, points)
    probe = probe_camera(frame, [0.0, 0.0, 1.0], 90.0, 0.0, 64, 48, 75.0, "p")
    forward = frame.rotation @ probe.camera_to_world[:3, 2]
    assert np.allclose(forward, [0, 1, 0], atol=1e-6)


def test_align_depth_recovers_scale_and_shift():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1, 5, (40, 50))
    mono = 0.5 / depth - 0.02  # inverse depth up to an affine map
    known = rng.random(depth.shape) < 0.6
    recovered = align_depth(mono, np.where(known, depth, 0), known, 2.0)
    assert np.allclose(recovered[~known], depth[~known], rtol=1e-3)


def test_kmeans_separates_clusters():
    rng = np.random.default_rng(0)
    points = np.r_[rng.normal(0, 0.1, (50, 3)), rng.normal(5, 0.1, (50, 3))]
    labels = kmeans(points, 2)
    assert len(set(labels[:50])) == 1 and len(set(labels[50:])) == 1 and labels[0] != labels[-1]


def test_fibonacci_directions_cover_the_sphere():
    directions = fibonacci_directions()
    assert np.allclose(np.linalg.norm(directions, axis=1), 1)
    probes = np.random.default_rng(0).normal(size=(2000, 3))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    worst = np.degrees(np.arccos((probes @ directions.T).max(1))).max()
    assert worst < 20  # every direction is within 20° of a bin
