"""GPU tests of multi-view stereo and PnP registration on ray-traced textured planes."""

import math

import cv2
import numpy as np
import pytest
import torch

from spacegen.config import MvsConfig, PipelineConfig
from spacegen.contracts import Camera, look_at
from spacegen.mvs import consistent, plane_sweep
from spacegen.reslice import Matcher, register

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

WIDTH, HEIGHT, FOCAL = 320, 240, 300.0


def noise(seed: int, size: int = 512) -> np.ndarray:
    """Smooth random texture in 0..1 (blurred white noise, contrast stretched)."""
    image = np.random.default_rng(seed).random((size, size)).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), 2.0)
    return (image - image.min()) / (image.max() - image.min())


def trace(camera: Camera, textures) -> tuple[np.ndarray, np.ndarray]:
    """Grey image and z-depth: a wall at world z = 4, a 1 × 1 square at z = 2.5 in front of
    its middle and a 0.8 × 0.8 one at z = 1.8 to the lower left."""
    k = camera.intrinsics
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    rays = np.stack(((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones_like(xs)), -1)
    world_rays = rays @ camera.camera_to_world[:3, :3].T
    centre = camera.centre
    grey = np.zeros((HEIGHT, WIDTH), np.float32)
    distance = np.full((HEIGHT, WIDTH), np.inf)
    planes = (
        (4.0, textures[0], 0.0, 0.0, 3.0),
        (2.5, textures[1], 0.0, 0.0, 0.5),
        (1.8, textures[1][::-1].copy(), -0.6, 0.45, 0.4),
    )
    for plane_z, texture, x0, y0, half in planes:
        t = (plane_z - centre[2]) / world_rays[..., 2]
        hit = centre + t[..., None] * world_rays
        x, y = hit[..., 0] - x0, hit[..., 1] - y0
        inside = (t > 0) & (np.abs(x) <= half) & (np.abs(y) <= half) & (t < distance)
        size = texture.shape[0] - 1
        u = ((x / half + 1) / 2 * size).astype(np.float32)
        v = ((y / half + 1) / 2 * size).astype(np.float32)
        value = cv2.remap(texture, u, v, cv2.INTER_LINEAR)
        grey = np.where(inside, value, grey)
        distance = np.where(inside, t, distance)
    return grey, distance  # rays have z = 1 in the camera, so t is the z-depth


def cameras(offsets) -> list[Camera]:
    k = np.array([[FOCAL, 0, (WIDTH - 1) / 2], [0, FOCAL, (HEIGHT - 1) / 2], [0, 0, 1]])
    return [
        Camera(
            f"{n}",
            WIDTH,
            HEIGHT,
            k,
            look_at(np.array([x, y, 0.0]), np.array([0, 0, 4.0]), (0, -1, 0)),
        )
        for n, (x, y) in enumerate(offsets)
    ]


def test_plane_sweep_measures_depth_within_one_percent():
    textures = (noise(1), noise(2))
    views = cameras([(0, 0), (-0.3, 0), (0.3, 0), (0, 0.25), (0, -0.25)])
    traced = [trace(c, textures) for c in views]
    greys = [torch.from_numpy(g).cuda() for g, _ in traced]
    config = MvsConfig()

    raw = []
    for index, camera in enumerate(views):
        others = [(views[s], greys[s]) for s in range(len(views)) if s != index]
        raw.append(plane_sweep(camera, greys[index], others, 1.5, 6.0, config))
    kept = (
        consistent(views[0], raw[0], [(views[s], raw[s]) for s in range(1, len(views))], config)
        .cpu()
        .numpy()
    )

    truth = traced[0][1]
    measured = kept > 0
    error = np.abs(kept[measured] - truth[measured]) / truth[measured]
    assert measured.mean() > 0.6, measured.mean()
    assert np.median(error) < 0.01, np.median(error)
    assert np.percentile(error, 95) < 0.03, np.percentile(error, 95)
    # Both surfaces are found, including the near square's occluding edges.
    assert (np.abs(kept[measured] - 2.5) < 0.05).sum() > 1000
    assert (np.abs(kept[measured] - 4.0) < 0.08).sum() > 10000


def test_pnp_registers_a_frame_between_two_known_cameras():
    textures = (noise(3), noise(4))
    a, query, b = cameras([(-0.3, 0.0), (0.02, 0.05), (0.3, 0.0)])
    turn = np.eye(4)
    turn[:3, :3] = cv2.Rodrigues(np.array([0.0, math.radians(3), 0.0]))[0]
    query = query.moved(turn @ query.world_to_camera)

    def rgb(camera):
        grey, _ = trace(camera, textures)
        return np.repeat((grey * 255).astype(np.uint8)[..., None], 3, -1)

    config = PipelineConfig()
    neighbours = [(c.name, c, rgb(c), trace(c, textures)[1].astype(np.float32)) for c in (a, b)]
    found, info = register(
        rgb(query), query.moved(np.eye(4)), neighbours, Matcher(config, "cuda:0"), config
    )

    assert found is not None, info
    assert np.linalg.norm(found.centre - query.centre) < 0.01, (found.centre, query.centre)
    relative = found.world_to_camera[:3, :3] @ query.camera_to_world[:3, :3]
    assert math.degrees(math.acos(np.clip((np.trace(relative) - 1) / 2, -1, 1))) < 0.2
