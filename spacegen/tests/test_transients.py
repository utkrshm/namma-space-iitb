"""CPU tests of the free-space check (`spacegen.transients`) and how training uses its masks."""

import json
import math

import cv2
import numpy as np
import torch

from spacegen.config import SplatConfig
from spacegen.contracts import Camera, look_at
from spacegen.sfm import Solution
from spacegen.splat.train import Priors, hide_flagged, photometric_l1
from spacegen.transients import TransientMasks, drop_flagged_seeds, free_space_flags

LOW, HIGH = np.array([-2.0, -2.0, 0.0]), np.array([2.0, 2.0, 2.5])
SIZE = (64, 48)


def ring(count: int = 48) -> list[Camera]:
    """Cameras on a circle of radius 1.2 at eye height, looking at the room's middle."""
    width, height = SIZE
    k = np.array([[0.6 * width, 0, (width - 1) / 2], [0, 0.6 * width, (height - 1) / 2], [0, 0, 1]])
    cameras = []
    for number in range(count):
        angle = 2 * math.pi * number / count
        eye = np.array([1.2 * math.cos(angle), 1.2 * math.sin(angle), 1.2])
        cameras.append(Camera(f"{number:05d}", width, height, k, look_at(eye, [0, 0, 1.2])))
    return cameras


def depth_target(camera: Camera, door: Camera | None = None) -> np.ndarray:
    """Inverse depth of the box room's inside; with ``door``, also a 0.6 × 0.6 plate 0.5 in
    front of that camera (an open door leaf next to the lens)."""
    width, height = SIZE
    k = camera.intrinsics
    ys, xs = np.mgrid[0:height, 0:width]
    rays = np.stack(((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones(xs.shape)), -1)
    to_world = camera.camera_to_world
    direction, origin = rays @ to_world[:3, :3].T, to_world[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = np.where(direction > 0, HIGH, LOW)
        steps = (bound - origin) / direction
    depth = np.where(np.isfinite(steps) & (steps > 0), steps, np.inf).min(-1)
    if door is not None:
        axes = door.camera_to_world[:3, :3]
        centre = door.centre + 0.5 * axes[:, 2]
        along = direction @ axes[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            step = ((centre - origin) @ axes[:, 2]) / along
        hit = origin + step[..., None] * direction
        offset = hit - centre
        on = (step > 0) & (np.abs(offset @ axes[:, 0]) < 0.3) & (np.abs(offset @ axes[:, 1]) < 0.3)
        depth = np.where(on & (step < depth), step, depth)
    return np.where(np.isfinite(depth), 1 / depth, 0.0)


def flags_for(cameras, door_cameras=(), door_at=None):
    maps = [depth_target(c, door_at if i in door_cameras else None) for i, c in enumerate(cameras)]
    inverse = torch.tensor(np.stack(maps), dtype=torch.float32)
    config = SplatConfig()
    flags = free_space_flags(inverse, cameras, 8, config.transient_share, config.transient_ratio)
    return maps, [f.numpy() for f in flags]


def test_a_door_in_three_frames_is_flagged_and_the_static_room_is_not():
    cameras = ring()
    _, static = flags_for(cameras)
    assert max(f.mean() for f in static) == 0

    door = (47, 0, 1)
    maps, flags = flags_for(cameras, door, cameras[0])
    plain = [depth_target(c) for c in cameras]
    for index in door:
        on_door = maps[index] > plain[index] * 1.01
        assert on_door.mean() > 0.2
        assert flags[index][on_door].mean() > 0.8, (index, flags[index][on_door].mean())
        # Flags are made on a 4-pixel grid, so they may spill a cell past the door's edge.
        near_door = cv2.dilate(on_door.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        assert not flags[index][~near_door].any()
    assert max(f.mean() for i, f in enumerate(flags) if i not in door) == 0


def test_flagged_pixels_have_no_depth_target_and_no_photometric_gradient(tmp_path):
    images = [tmp_path / "00000.jpg"]
    (tmp_path / "transient").mkdir()
    np.save(tmp_path / "00000.npy", np.full((8, 10), 0.5, np.float32))
    (tmp_path / "alignment.json").write_text(json.dumps({"00000": {"a": 1.0, "b": 0.0}}))
    mask = np.zeros((8, 10), np.uint8)
    mask[:, 6:] = 1
    np.save(tmp_path / "transient" / "00000.npy", mask)
    masks = TransientMasks(tmp_path, images)

    _, metric, _, _ = Priors(tmp_path, images, masks=masks).get(0, "cpu")
    assert torch.all(metric[:, 6:] == 0) and torch.all(metric[:, :6] == 0.5)

    rgb = torch.rand((16, 20, 3), requires_grad=True)
    target = torch.rand((16, 20, 3))
    photometric_l1(hide_flagged(rgb, target, masks.get(0, "cpu")), target, 0.98).backward()
    assert torch.all(rgb.grad[:, 12:] == 0)
    assert torch.all(rgb.grad[:, :12] != 0)


def test_seeds_on_flagged_pixels_are_dropped_but_sfm_points_are_kept(tmp_path):
    camera = ring(4)[0].resized(10, 8)
    (tmp_path / "transient").mkdir()
    mask = np.zeros((8, 10), np.uint8)
    mask[:, 5:] = 1  # the right half of the image
    np.save(tmp_path / "transient" / "00000.npy", mask)
    to_world = camera.camera_to_world

    def point(column: float) -> np.ndarray:
        k = camera.intrinsics
        local = np.array([(column - k[0, 2]) / k[0, 0], (3.5 - k[1, 2]) / k[1, 1], 1.0]) * 2
        return local @ to_world[:3, :3].T + to_world[:3, 3]

    points = np.stack([point(8), point(2), point(8), point(2)])  # SfM ×2, then seeds ×2
    solution = Solution(
        [camera],
        [tmp_path / "00000.jpg"],
        points,
        np.zeros((4, 3)),
        [np.array([0])] * 4,
        [{"index": 0}],
    )
    kept = drop_flagged_seeds(solution, 2, tmp_path, print)
    assert np.allclose(kept.points, points[[0, 1, 3]])
