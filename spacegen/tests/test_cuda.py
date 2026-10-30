"""GPU tests with real gsplat rendering and training (no mocks)."""

import math

import cv2
import numpy as np
import pytest
import torch

from spacegen.config import SplatConfig
from spacegen.contracts import Camera, look_at
from spacegen.splat.model import SH_C0, GaussianModel, load_ply, render_camera, save_ply
from spacegen.splat.train import train_splat
from spacegen.support import compute_support, support_map

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]


def textured_wall(device="cuda:0", count=60) -> GaussianModel:
    """A 2 × 2 wall in the x-z plane at y = 0, made of flat discs with a checker pattern."""
    u, v = np.meshgrid(np.linspace(-1, 1, count), np.linspace(-1, 1, count))
    means = np.stack((u.ravel(), np.zeros(u.size), v.ravel() + 1), axis=1)
    checker = ((np.floor(u * 4) + np.floor(v * 4)) % 2).ravel()
    colours = np.stack((0.2 + 0.6 * checker, 0.3 + 0.2 * (1 - checker), 0.5 * np.ones(u.size)), 1)
    spacing = 2 / (count - 1)
    scales = np.log(np.tile([spacing * 0.7, spacing * 0.05, spacing * 0.7], (u.size, 1)))
    quats = np.tile([1.0, 0, 0, 0], (u.size, 1))
    tensors = {
        "means": torch.tensor(means, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "quats": torch.tensor(quats, dtype=torch.float32),
        "opacities": torch.full((u.size,), 4.0),
        "sh0": torch.tensor((colours - 0.5) / SH_C0, dtype=torch.float32)[:, None],
        "shn": torch.zeros((u.size, 15, 3)),
    }
    return GaussianModel(tensors, 3).to(device)


def camera_at(eye, target=(0, 0, 1), size=(160, 120), name="c") -> Camera:
    width, height = size
    focal = 0.9 * width
    k = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    return Camera(
        name, width, height, k, look_at(np.asarray(eye, float), np.asarray(target, float))
    )


def front_cameras(count=24):
    cameras = []
    for number in range(count):
        angle = math.radians(-35 + 70 * number / (count - 1))
        eye = (2.5 * math.sin(angle), -2.5 * math.cos(angle), 1.0 + 0.3 * math.sin(3 * angle))
        cameras.append(camera_at(eye, name=f"{number:03d}"))
    return cameras


def test_support_marks_front_views_measured_and_back_views_not():
    wall = textured_wall()
    cameras = front_cameras()
    wall.extras["support"] = compute_support(wall, cameras, long_side=160)
    assert (wall.extras["support"] != 0).float().mean() > 0.95

    front, alpha, _ = support_map(wall, camera_at((0.3, -2.0, 1.1)), 35.0)
    covered = alpha > 0.5
    assert covered.float().mean() > 0.3
    assert front[covered].mean() > 0.95

    back, alpha, _ = support_map(wall, camera_at((0.0, 2.5, 1.0)), 35.0)
    assert back[alpha > 0.5].mean() < 0.05


def test_ply_round_trip_keeps_parameters_and_flags(tmp_path):
    wall = textured_wall()
    wall.extras["support"] = torch.randint(-(2**62), 2**62, (wall.count,), device="cuda:0")
    wall.extras["filled"] = (torch.rand(wall.count, device="cuda:0") > 0.5).long()
    save_ply(tmp_path / "wall.ply", wall)
    loaded = load_ply(tmp_path / "wall.ply")
    for key, value in wall.params.items():
        assert torch.allclose(loaded.params[key], value)
    assert torch.equal(loaded.extras["support"], wall.extras["support"])
    assert torch.equal(loaded.extras["filled"], wall.extras["filled"])


def test_training_recovers_a_rendered_wall(tmp_path):
    """Images rendered from a known wall; a splat trained from sparse points must match them."""
    truth = textured_wall()
    cameras = front_cameras(30)
    paths = []
    for camera in cameras:
        with torch.no_grad():
            rgb, _, _, _ = render_camera(truth, camera)
        path = tmp_path / f"{camera.name}.png"
        cv2.imwrite(str(path), (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()[..., ::-1])
        paths.append(path)
    rng = np.random.default_rng(0)
    points = np.c_[rng.uniform(-1, 1, 400), rng.normal(0, 0.02, 400), rng.uniform(0, 2, 400)]
    config = SplatConfig(
        steps=1500,
        max_gaussians=20_000,
        resolution_schedule=((0, 1),),
        holdout_every=0,
        pose_refinement=False,
        appearance=False,
        depth_prior_weight=0.0,
        refine_stop=1200,
        evaluate_every=10**9,
        checkpoint_every=10**9,
        log_every=500,
    )
    train = [i for i in range(30) if i % 5]
    val = [i for i in range(30) if i % 5 == 0]
    metrics = train_splat(
        cameras,
        paths,
        points.astype(np.float32),
        np.full((400, 3), 0.5, np.float32),
        tmp_path / "run",
        config,
        train_indices=train,
        val_indices=val,
        log=lambda _: None,
    )
    assert metrics["val"]["psnr"] > 24, metrics["val"]
    assert (tmp_path / "run" / "scene.ply").exists()
