"""Tour geometry: scene up/floor estimation, look-at cameras, cut-away filter and reveal times."""

import numpy as np
import torch

from src.splatting.tour import cutaway_mask, first_seen, local_floor, look_at, scene_frame


def _room(rotation):
    """Cameras walking along x at 1.5 above a floor at z = 0, in a rotated world."""
    rng = np.random.default_rng(0)
    camera_to_world = np.tile(np.eye(4), (40, 1, 1))
    for index in range(40):
        # Camera looks along +x; OpenCV axes: x right (−y world), y down (−z world), z forward.
        camera_to_world[index, :3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        camera_to_world[index, :3, 3] = [0.2 * index, 0.3 * np.sin(index / 5), 1.5]
    floor = np.column_stack((rng.uniform(-1, 9, 20000), rng.uniform(-2, 2, 20000), np.zeros(20000)))
    ceiling = floor + [0, 0, 2.8]
    means = np.concatenate((floor, ceiling[:500]))
    camera_to_world[:, :3, :] = rotation @ camera_to_world[:, :3, :]
    return camera_to_world, means @ rotation.T


def _rotation():
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ np.array(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    )


def test_scene_frame_recovers_up_and_floor():
    rotation = _rotation()
    camera_to_world, means = _room(rotation)
    frame = scene_frame(camera_to_world, means, np.ones(len(means)))

    assert np.allclose(frame.up, rotation[:, 2], atol=0.02)
    assert abs(frame.floor + 1.5) < 0.05
    assert abs(frame.across @ frame.up) < 1e-9 and abs(frame.side @ frame.up) < 1e-9


def test_look_at_is_a_right_handed_opencv_camera():
    eye, target, up = np.array([1.0, 2, 3]), np.array([4.0, -1, 0.5]), np.array([0.0, 0, 1])
    world_to_camera = look_at(eye, target, up)
    rotation = world_to_camera[:3, :3]
    camera_target = rotation @ target + world_to_camera[:3, 3]

    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1)
    assert np.allclose(camera_target[:2], 0) and camera_target[2] > 0
    assert (rotation @ up)[1] < 0  # world up points up in the image (−y)


def test_cutaway_drops_ceiling_and_needles():
    camera_to_world, means = _room(np.eye(3))
    frame = scene_frame(camera_to_world, means, np.ones(len(means)))
    scales = np.full((len(means), 3), 0.01)
    scales[0] = [0.01, 0.01, 0.5]  # a needle on the floor
    means[1] = [4.0, 0.0, 1.0]  # an isolated floater in mid-air
    keep = cutaway_mask(frame, means, scales, np.ones(len(means)), camera_to_world[:, :3, 3])

    assert not keep[0] and not keep[1]
    assert not keep[20000:].any()  # ceiling at 2.8 is above floor + 1.25 × 1.5
    near_path = np.abs(means[2:20000, 1]) < 1.5
    assert keep[2:20000][near_path].mean() > 0.9


def test_first_seen_follows_the_camera_walk():
    camera_to_world, _ = _room(np.eye(3))
    world_to_camera = np.linalg.inv(camera_to_world)
    k = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
    ahead = torch.tensor([[3.0, 0, 1.5], [7.0, 0, 1.5], [-5.0, 0, 1.5]])
    times = np.arange(40, dtype=float)
    seen = first_seen(ahead, world_to_camera, [k] * 40, [(320, 240)] * 40, times, max_depth=2.0)

    # Each point is first seen once the camera is within 2 units behind it.
    assert seen[0] == 6 and seen[1] == 26
    assert torch.isinf(seen[2])  # behind every camera


def test_local_floor_keeps_a_bent_floor():
    camera_to_world, means = _room(np.eye(3))
    # The floor sags by 0.45 (0.3 camera heights) under part of the path, as in a drifted solve.
    sag = means[:, 2] < 1.0
    sag &= (means[:, 0] > 2) & (means[:, 0] < 5)
    means[sag, 2] -= 0.45
    frame = scene_frame(camera_to_world, means, np.ones(len(means)))
    scales = np.full((len(means), 3), 0.01)
    centres = camera_to_world[:, :3, 3]
    floor_at = local_floor(frame, means, np.ones(len(means)))
    near_path = (np.abs(means[:, 1]) < 1.5) & (means[:, 2] < 1.0)

    global_keep = cutaway_mask(frame, means, scales, np.ones(len(means)), centres)
    local_keep = cutaway_mask(frame, means, scales, np.ones(len(means)), centres, floor_at=floor_at)
    assert global_keep[sag & near_path].mean() < 0.05  # the single floor height drops the sag
    assert local_keep[near_path].mean() > 0.9
    assert not local_keep[20000:].any()
