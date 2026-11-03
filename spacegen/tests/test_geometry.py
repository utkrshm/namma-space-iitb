"""CPU tests of the camera, configuration, scene-frame, partition, depth-fit, loss, tour, audit
and re-slicing code."""

import json
import math

import numpy as np
import pytest
import torch

from spacegen.audit import between, turn_deg
from spacegen.config import PipelineConfig, TourConfig
from spacegen.contracts import Camera, look_at
from spacegen.depth import depth_normals, fit_inverse_depth, near_inverse_depth
from spacegen.fill import align_depth, probe_camera, standing_points
from spacegen.partition import balanced_split
from spacegen.reslice import choose_extra, with_extra_frames
from spacegen.scene_frame import SceneFrame, estimate_frame, find_areas, run_frame
from spacegen.sfm import Solution
from spacegen.splat.train import needle_penalty, photometric_l1
from spacegen.support import fibonacci_directions
from spacegen.tour import WalkPath, filmed_arc


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


def test_scene_frame_finds_the_floor_when_half_the_video_looks_straight_down():
    """A capture that ends filming the floor and bed from above (room_m_block). A camera
    looking down has its image-up axis horizontal, so the cameras' mean image-up leans far
    from vertical; their x axes stay horizontal and still give the right up."""
    cameras, points = _room(3)
    for number in range(30, 80):
        position = cameras[number].centre
        target = position + np.array([0.15, 0.0, -1.0])
        k = cameras[number].intrinsics
        cameras[number] = Camera(f"{number}", 320, 240, k, look_at(position, target))
    angle = 0.4
    rotation = np.array(
        [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]
    )
    cameras, points = _rotate(cameras, points, rotation, 2.0, np.array([1.0, 3.0, -2.0]))
    frame = estimate_frame(cameras, points)
    assert frame.floor_found
    assert np.degrees(np.arccos(frame.up @ (rotation @ [0, 0, 1]))) < 1.0


def test_scene_frame_round_trips_through_json(tmp_path):
    cameras, points = _room(1)
    frame = estimate_frame(cameras, points)
    (tmp_path / "frame.json").write_text(__import__("json").dumps(frame.to_dict()))
    loaded = SceneFrame.load(tmp_path / "frame.json")
    assert np.allclose(loaded.rotation, frame.rotation) and loaded.areas == frame.areas


def test_run_frame_numbers_area_cameras_in_the_full_camera_list(tmp_path):
    """Areas must index all cameras (training and held-out), whichever stage writes the
    file first; a file counted over training cameras only is remade."""
    cameras, points = _room(4)
    train = [i for i in range(len(cameras)) if i % 8]
    stale = estimate_frame([cameras[i] for i in train], points)
    (tmp_path / "scene_frame.json").write_text(json.dumps(stale.to_dict()))

    frame = run_frame(tmp_path, cameras, points)
    assert frame.camera_count == len(cameras)
    assert SceneFrame.load(tmp_path / "scene_frame.json").camera_count == len(cameras)
    # Cameras 0-39 film the first room and 40-79 the second.
    assert sorted(frame.areas[0]["cameras"]) == list(range(40))
    stands = standing_points(frame, cameras)
    assert stands[0]["camera"] < 40 <= stands[1]["camera"]


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


def test_walk_path_turns_slowly_on_the_spot_and_does_not_stare_at_the_floor():
    frame = SceneFrame(np.eye(3), np.zeros(3), 1.0, True, [])
    k = np.array([[300.0, 0, 159.5], [0, 300.0, 119.5], [0, 0, 1]])

    def camera(x, yaw_deg, pitch_deg):
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        position = np.array([x, 0.0, 1.0])
        direction = np.r_[
            math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)
        ]
        return Camera("c", 320, 240, k, look_at(position, position + direction))

    # Walk 2 camera heights along +x, turn 180° on the spot, walk back looking straight down.
    cameras = [camera(x, 0, -20) for x in np.linspace(0, 2, 40)]
    cameras += [camera(2, yaw, -20) for yaw in np.linspace(0, 180, 20)]
    cameras += [camera(x, 180, -90) for x in np.linspace(2, 0, 40)]
    walk = WalkPath(cameras, frame, 1.8, 45.0)

    times = np.linspace(0, walk.duration, int(walk.duration * 30))
    views = np.stack([np.linalg.inv(walk.pose(t))[:3, 2] for t in times])
    rate = np.degrees(np.arccos(np.clip((views[1:] * views[:-1]).sum(1), -1, 1))) * 30
    pitch = np.degrees(np.arcsin(views[:, 2]))
    assert rate.max() <= 46
    assert walk.duration > 2 * 2 * 1.8 * 0.9 + 180 / 45 * 0.9
    assert pitch.min() >= -35.1  # interpolating between headings dips a little
    assert pitch[0] == pytest.approx(math.degrees(math.atan(0.4 * math.tan(math.radians(-20)))))
    # Looking straight down, the view keeps the last real heading (back along −x).
    assert views[-1, 0] / np.linalg.norm(views[-1, :2]) < -0.95
    assert np.allclose(np.linalg.inv(walk.pose(walk.duration))[:3, 3], [0, 0, 1], atol=0.05)


def test_in_between_view_is_halfway_in_position_and_turn():
    k = np.array([[300.0, 0, 159.5], [0, 300.0, 119.5], [0, 0, 1]])
    a = Camera("a", 320, 240, k, look_at(np.zeros(3), np.array([1.0, 0, 0])))
    b = Camera("b", 320, 240, k, look_at(np.array([0.0, 1, 0]), np.array([0.0, 2, 0])))
    middle = between(a, b)
    assert np.allclose(middle.centre, [0, 0.5, 0])
    assert turn_deg(a, b) == pytest.approx(90, abs=1e-6)
    assert turn_deg(a, middle) == pytest.approx(45, abs=1e-6)
    assert turn_deg(middle, b) == pytest.approx(45, abs=1e-6)


def test_extra_frames_merge_in_capture_order_and_keep_point_tracks(tmp_path):
    k = np.array([[300.0, 0, 159.5], [0, 300.0, 119.5], [0, 0, 1]])

    def camera(name, x):
        return Camera(name, 320, 240, k, look_at(np.array([x, 0, 0]), np.array([x, 0, 5.0])))

    solution = Solution(
        [camera("a", 0), camera("b", 1), camera("c", 2)],
        [tmp_path / n for n in "abc"],
        np.zeros((2, 3), np.float32),
        np.zeros((2, 3), np.float32),
        [np.array([0, 2]), np.array([1])],
        [{"name": n, "index": i * 10, "position": i} for i, n in enumerate("abc")],
    )
    extra = [
        {
            "name": "x000015.jpg",
            "index": 15,
            "registered": True,
            "camera": camera("x", 1.5).to_dict(),
        },
        {"name": "x000005.jpg", "index": 5, "registered": False},
    ]
    (tmp_path / "reslice").mkdir()
    (tmp_path / "reslice" / "extra.json").write_text(json.dumps({"frames": extra}))

    merged = with_extra_frames(solution, tmp_path)
    assert [c.name for c in merged.cameras] == ["a", "b", "x", "c"]
    assert merged.frames[2]["name"] == "x000015.jpg" and "camera" not in merged.frames[2]
    assert [list(v) for v in merged.point_views] == [[0, 3], [1]]


def test_extra_frames_come_from_weak_segments_and_avoid_held_out_frames():
    count = 300
    measures = {
        "times": np.arange(count) / 30.0,
        "sharpness": np.ones(count),
        "motion": np.full(count, 0.002),
    }
    config = PipelineConfig()
    step = 0.03  # 15 video frames of motion
    segments = [{"start_s": 1.0, "end_s": 3.0}, {"start_s": 6.0, "end_s": 7.0}]
    held_out = [60]  # at 2 s, inside the first segment
    picks = choose_extra(measures, segments, {45}, held_out, step, config)
    frames = [index for index, _ in picks]

    assert frames and all(30 < f < 90 or 180 < f < 210 for f in frames)
    assert 45 not in frames
    # Half a step is 7.5 frames of motion.
    assert all(abs(f - 60) >= 7.5 for f in frames)
    assert {number for _, number in picks} == {0, 1}


def test_needle_penalty_spares_flat_discs_and_round_blobs():
    disc = torch.log(torch.tensor([[1.0, 1.0, 0.01]]))
    blob = torch.log(torch.tensor([[1.0, 0.8, 0.9]]))
    needle = torch.log(torch.tensor([[1.0, 0.05, 0.05]]))
    assert needle_penalty(disc, 4.0) == 0
    assert needle_penalty(blob, 4.0) == 0
    assert needle_penalty(needle, 4.0) == pytest.approx(math.log(20 / 4))


def test_near_surfaces_take_the_metric_depth():
    entry = {"scale": 2.0, "near": 0.5}  # SfM points end at 2 units; 2 units per metre
    fitted = np.array([0.25, 0.4, 0.2, 0.3], np.float32)  # inverse depth, SfM units
    metres = np.array([2.0, 0.5, 0.25, 0.0], np.float32)  # 4, 1, 0.5 units; none

    blended = near_inverse_depth(fitted, metres, entry)

    # Far: the fit. Near by the metric depth (1 and 0.5 units): the metric depth. No
    # metric depth: the fit.
    assert np.allclose(blended, [0.25, 1.0, 2.0, 0.3])
    tensor = near_inverse_depth(torch.from_numpy(fitted), torch.from_numpy(metres), entry)
    assert np.allclose(tensor.numpy(), blended)
    assert near_inverse_depth(fitted, None, entry) is fitted


def test_only_points_seen_through_a_mirror_count_as_its_reflection():
    """A 1 × 1 mirror in the plane x = 0 facing +x. A point 1 m behind it is a reflection
    when a camera in the room (x > 0) looks at it through the glass; the same point filmed
    from the next room (x < 0), or seen past the mirror's edge, is a real surface."""
    from spacegen.splat.reflect import make_reflectors, seen_through_opaque

    reflectors = make_reflectors(
        [[1.0, 0.0, 0.0, 0.0]],
        0.02,
        "cpu",
        axes=[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        bounds=[[-0.5, 0.5, 1.0, 2.0]],
        opaque=[True],
    )
    points = torch.tensor([[-1.0, 0.0, 1.5], [-1.0, 0.0, 1.5], [-1.0, 3.0, 1.5], [0.5, 0.0, 1.5]])
    eyes = torch.tensor([[2.0, 0.0, 1.5], [-3.0, 0.0, 1.5], [2.0, 0.0, 1.5], [2.0, 0.0, 1.5]])
    assert seen_through_opaque(points, eyes, reflectors).tolist() == [True, False, False, False]
    # One eye for all points (a depth map's camera).
    assert seen_through_opaque(points, eyes[0], reflectors).tolist() == [True, True, False, False]


def test_reflector_frame_count_counts_distinct_frames():
    """Three regions of one window in one frame (split by mullions) are one frame, so
    they cannot pass the min_frames gate alone."""
    from spacegen.reflectors import _merge

    rng = np.random.default_rng(0)

    def region(stem, y_low):
        points = np.c_[np.zeros(200), rng.uniform(y_low, y_low + 0.3, 200), rng.uniform(1, 2, 200)]
        return {
            "normal": np.array([1.0, 0, 0]),
            "points": points,
            "kind": "windowpane",
            "stem": stem,
        }

    one_frame = [region("00010", y) for y in (0.0, 0.35, 0.7)]
    merged = _merge(one_frame, np.array([0, 0, 1.0]), 1.0)
    assert len(merged) == 1 and merged[0]["frames"] == 1
    three_frames = [region(stem, 0.0) for stem in ("00010", "00011", "00012")]
    assert _merge(three_frames, np.array([0, 0, 1.0]), 1.0)[0]["frames"] == 3


def test_planar_fit_separates_the_two_walls_of_a_corner_and_gives_their_depth():
    """A camera facing a room corner: two walls meeting at 90°, depth with 1% noise. Two
    planes must come out, each holding its own wall, and the plane depth along each ray
    must match the true depth."""
    from spacegen.planar import fit_planes, plane_inverse_depth

    rng = np.random.default_rng(0)
    # Camera at the origin looking along +z; walls x + z = 4 (left) and −x + z = 4 (right).
    xs, ys = np.meshgrid(np.linspace(-0.8, 0.8, 80), np.linspace(-0.6, 0.6, 60))
    rays = np.stack((xs, ys, np.ones_like(xs)), -1)
    left = xs < 0
    z = np.where(left, 4 / (1 - xs), 4 / (1 + xs))
    noisy = z * (1 + rng.normal(0, 0.01, z.shape) * 0.3)
    points = rays.reshape(-1, 3) * noisy.reshape(-1, 1)
    planes = fit_planes(points, 0.015, rng)
    assert len(planes) == 2
    owners = [left.ravel()[inliers].mean() for _, _, inliers in planes]
    assert sorted(round(o) for o in owners) == [0, 1]
    assert sum(int(p[2].sum()) for p in planes) > 0.9 * len(points)
    for normal, offset, inliers in planes:
        inverse = plane_inverse_depth(rays.reshape(-1, 3)[inliers], normal, offset)
        # Inliers lie within 1.5% of their distance from the plane; along a ray at 45°
        # to the wall that is up to about 2% of depth (points next to the corner).
        assert np.allclose(1 / inverse, z.ravel()[inliers], rtol=0.025)
        assert np.median(np.abs(1 / inverse / z.ravel()[inliers] - 1)) < 0.005


def test_priors_use_planes_only_when_switched_on(tmp_path):
    from spacegen.splat.train import Priors

    mono = np.full((6, 8), 0.5, np.float16)
    np.save(tmp_path / "a.npy", mono)
    (tmp_path / "alignment.json").write_text(json.dumps({"a": {"a": 1.0, "b": 0.0}}))
    (tmp_path / "planar").mkdir()
    planar = np.zeros((6, 8), bool)
    planar[:, :4] = True
    np.savez(
        tmp_path / "planar" / "a.npz", inverse=np.full((6, 8), 0.25, np.float16), planar=planar
    )
    image = tmp_path / "a.jpg"

    _, metric, measured, mask = Priors(tmp_path, [image]).get(0, "cpu")
    assert mask is None and measured is None and torch.allclose(metric, torch.tensor(0.5))
    _, metric, _, mask = Priors(tmp_path, [image], planar=True).get(0, "cpu")
    assert mask[:, :4].all() and not mask[:, 4:].any()
    assert torch.allclose(metric[:, :4], torch.tensor(0.25)) and torch.allclose(
        metric[:, 4:], torch.tensor(0.5)
    )


def test_block_ownership_ignores_cameras_behind_a_wall():
    """A 0.1 m wall at x = 0 between two rooms. Camera A (block 0) films its left face
    from 1.95 m; camera B (block 1) films the right face from 0.75 m. Without occlusion
    the left face goes to B, the closer camera, which never saw it; with B's depth map
    it stays with A."""
    from spacegen.partition import visibility_owner

    k = np.array([[100.0, 0, 63.5], [0, 100.0, 47.5], [0, 0, 1]])
    a = Camera("a", 128, 96, k, look_at(np.array([-2.0, 0, 1]), np.array([0.0, 0, 1])))
    b = Camera("b", 128, 96, k, look_at(np.array([0.8, 0, 1]), np.array([0.0, 0, 1])))
    rng = np.random.default_rng(0)
    left_face = np.c_[np.full(200, -0.05), rng.uniform(-0.3, 0.3, 200), rng.uniform(0.7, 1.3, 200)]
    labels = np.array([0, 1])
    assert (visibility_owner([a, b], labels, left_face, "cpu") == 1).all()

    # Depth maps: A sees the left face at 1.95 m, B the right face at 0.75 m (inverse depth).
    surfaces = torch.stack((torch.full((24, 32), 1 / 1.95), torch.full((24, 32), 1 / 0.75)))
    assert (visibility_owner([a, b], labels, left_face, "cpu", surfaces) == 0).all()


def test_bilateral_grid_starts_at_identity_and_maps_by_brightness():
    from spacegen.splat.train import Corrections, bilateral_slice

    rng = np.random.default_rng(0)
    rgb = torch.tensor(rng.uniform(0, 1, (30, 40, 3)), dtype=torch.float32)
    residual = torch.zeros(12, 8, 4, 4)
    assert torch.allclose(bilateral_slice(residual, rgb), rgb)

    # Brighten only the brightest grey levels by 20% (the top grid layer), as a phone's
    # local tone mapping might: dark pixels must not change.
    residual[0, -1] = residual[4, -1] = residual[8, -1] = 0.2
    out = bilateral_slice(residual, rgb)
    grey = rgb @ torch.tensor([0.299, 0.587, 0.114])
    dark = grey < 6 / 7 * 0.5  # below the layer next to the top one
    assert torch.allclose(out[dark], rgb[dark])
    assert (out[grey > 0.95] > rgb[grey > 0.95]).all()

    corrections = Corrections(5, (8, 4, 4))
    with torch.no_grad():
        corrections.grid += torch.randn_like(corrections.grid)
    corrections.centre(torch.tensor([0, 1, 2]))
    assert corrections.grid[:3].mean(0).abs().max() < 1e-6
    loaded = Corrections.for_state(corrections.state_dict())
    loaded.load_state_dict(corrections.state_dict())
    assert torch.equal(loaded.grid, corrections.grid)
    assert Corrections.for_state(Corrections(5).state_dict()).grid is None
