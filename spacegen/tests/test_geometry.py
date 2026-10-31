"""CPU tests of the camera, configuration, scene-frame, partition, depth-fit, loss and tour code."""

import math

import numpy as np
import pytest
import torch

from spacegen.config import PipelineConfig, TourConfig
from spacegen.contracts import Camera, look_at
from spacegen.depth import depth_normals, fit_inverse_depth
from spacegen.fill import align_depth, probe_camera
from spacegen.partition import balanced_split
from spacegen.scene_frame import SceneFrame, estimate_frame, find_areas
from spacegen.splat.train import photometric_l1
from spacegen.support import fibonacci_directions
from spacegen.tour import filmed_arc


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


def test_balanced_split_gives_equal_blocks_even_with_an_outlying_group():
    rng = np.random.default_rng(0)
    # 300 cameras in a long apartment, 15 on a far balcony.
    points = np.r_[
        rng.uniform([0, 0, 1], [12, 5, 1.1], (300, 3)), rng.normal([30, 2, 1], 0.2, (15, 3))
    ]
    labels = balanced_split(points, 3)
    sizes = np.bincount(labels)
    assert len(sizes) == 3 and sizes.max() - sizes.min() <= 1
    # Blocks are spatially compact along the long axis.
    ranges = [np.ptp(points[labels == k, 0]) for k in range(3)]
    assert max(ranges[:2]) < 8


def test_estimate_focal_recovers_a_wide_phone_lens_from_fundamental_matrices():
    import cv2

    from spacegen.sfm.mapping import estimate_focal

    rng = np.random.default_rng(0)
    width, height, focal = 1080, 1920, 1050.0
    k = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    points = rng.uniform([-3, -3, 2], [3, 3, 8], (400, 3))
    fundamentals = []
    for _ in range(30):
        rotation, _ = cv2.Rodrigues(rng.normal(0, 0.15, 3))
        translation = rng.normal(0, 0.4, 3)
        first = points @ k.T
        second = (points @ rotation.T + translation) @ k.T
        a, b = first[:, :2] / first[:, 2:], second[:, :2] / second[:, 2:]
        a, b = a + rng.normal(0, 0.5, a.shape), b + rng.normal(0, 0.5, b.shape)
        f_matrix, _ = cv2.findFundamentalMat(a, b, cv2.FM_8POINT)
        fundamentals.append(f_matrix)

    estimate = estimate_focal(fundamentals, width, height)
    assert abs(estimate / focal - 1) < 0.03
    assert estimate_focal(fundamentals[:5], width, height) is None


def test_fibonacci_directions_cover_the_sphere():
    directions = fibonacci_directions()
    assert np.allclose(np.linalg.norm(directions, axis=1), 1)
    probes = np.random.default_rng(0).normal(size=(2000, 3))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    worst = np.degrees(np.arccos((probes @ directions.T).max(1))).max()
    assert worst < 20  # every direction is within 20° of a bin


def test_split_holes_fills_slivers_and_flags_large_regions():
    from spacegen.fill import split_holes

    rgb = np.zeros((100, 120, 3), np.uint8)
    rgb[:, :60] = (200, 40, 40)
    rgb[:, 60:] = (40, 40, 200)
    hole = np.zeros((100, 120), bool)
    hole[10:90, 30] = True  # a thin sliver inside the red half
    hole[40:80, 80:115] = True  # a large block in the blue half
    base, large = split_holes(rgb, hole, 0.01)
    assert not large[:, 30].any() and large[40:80, 80:115].all()
    assert np.abs(base[10:90, 30].astype(int) - (200, 40, 40)).max() < 10


def test_fit_inverse_depth_ignores_outliers():
    rng = np.random.default_rng(0)
    mono = rng.uniform(0.1, 1.0, 500)
    inverse = 0.8 * mono + 0.05
    wrong = rng.random(500) < 0.15  # e.g. points across an object edge
    inverse[wrong] *= rng.uniform(1.5, 3.0, wrong.sum())
    a, b, error = fit_inverse_depth(mono, inverse)
    assert abs(a - 0.8) < 0.01 and abs(b - 0.05) < 0.005
    assert error < 0.01


def test_depth_normals_of_a_tilted_plane_face_the_camera():
    k = torch.tensor([[60.0, 0, 39.5], [0, 60.0, 29.5], [0, 0, 1]])
    normal = torch.nn.functional.normalize(torch.tensor([0.3, -0.4, -1.0]), dim=0)
    ys, xs = torch.meshgrid(torch.arange(60.0), torch.arange(80.0), indexing="ij")
    rays = torch.stack(((xs - 39.5) / 60, (ys - 29.5) / 60, torch.ones_like(xs)), -1)
    # Plane n · p = d with the camera on the side the normal points to (d < 0).
    depth = -2.0 / (rays @ normal)
    normals = depth_normals(depth, k)
    assert torch.allclose(normals, normal.expand_as(normals), atol=1e-4)


def test_clipped_pixels_only_penalize_a_darker_render():
    target = torch.tensor([[[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]])
    brighter = torch.tensor([[[1.4, 1.2, 1.1], [0.5, 0.5, 0.5]]])
    assert float(photometric_l1(brighter, target, 0.98)) == 0
    darker = torch.tensor([[[0.8, 0.8, 0.8], [0.5, 0.5, 0.5]]])
    assert float(photometric_l1(darker, target, 0.98)) == pytest.approx(0.1)
    # Above 1 the level disables it: a plain L1.
    assert float(photometric_l1(brighter, target, 1.5)) == pytest.approx(0.7 / 6)


def test_filmed_arc_follows_the_cameras_near_the_standing_point():
    frame = SceneFrame(np.eye(3), np.zeros(3), 1.5, True, [])
    k = np.array([[300.0, 0, 159.5], [0, 300.0, 119.5], [0, 0, 1]])
    rng = np.random.default_rng(0)
    cameras = []
    for number in range(40):
        near = number < 30
        position = np.r_[rng.normal(0, 0.1, 2) + (0 if near else 3), 1.5]
        yaw = math.radians(rng.uniform(0, 120) if near else rng.uniform(180, 300))
        pitch = math.radians(-25)
        direction = np.r_[
            math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)
        ]
        cameras.append(Camera(f"{number}", 320, 240, k, look_at(position, position + direction)))
    arc = filmed_arc(cameras, list(range(40)), frame, np.array([0.0, 0.0, 1.0]), TourConfig())
    low, high = (math.degrees(a) for a in arc["yaws"])
    assert 5 <= low <= 25 and 95 <= high <= 115
    assert arc["pitch_deg"] == pytest.approx(-25, abs=0.5)
    assert arc["cameras"] == 30
