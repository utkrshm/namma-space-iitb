"""CPU tests of rendering with the capture's own exposure (`spacegen.exposure`)."""

import numpy as np
import torch

from spacegen.contracts import look_at
from spacegen.exposure import CaptureExposure, ExposureFilter, apply_residual
from spacegen.splat.train import Corrections


def two_cameras():
    """A darkening camera looking along +x and a neutral one 2 units away looking along +y."""
    poses = [look_at(np.zeros(3), [1.0, 0, 0]), look_at(np.array([2.0, 0, 0]), [2.0, 1, 0])]
    centres = np.stack([np.linalg.inv(p)[:3, 3] for p in poses])
    forwards = np.stack([np.linalg.inv(p)[:3, 2] for p in poses])
    maps = torch.zeros((2, 12))
    maps[0, [0, 4, 8]] = -0.5  # half the brightness
    maps[0, 9:] = -0.05
    return poses, CaptureExposure(centres, forwards, maps, camera_height=1.0)


def test_a_view_takes_the_exposure_of_the_capture_cameras_near_it():
    poses, exposure = two_cameras()
    rgb = torch.full((4, 5, 3), 0.8)

    at_first = apply_residual(exposure.residual(poses[0]), rgb)
    expected = 0.8 * 0.5 - 0.05
    assert torch.allclose(at_first, torch.tensor(expected), atol=0.03)

    turned = look_at(np.zeros(3), [0, 0, 1.0])  # same place, 90° away: nobody filmed that
    assert torch.allclose(apply_residual(exposure.residual(turned), rgb), rgb, atol=1e-3)
    far = look_at(np.array([0, 5.0, 0]), [1.0, 5, 0])
    assert torch.allclose(apply_residual(exposure.residual(far), rgb), rgb, atol=1e-3)


def test_the_blend_matches_the_training_colour_map():
    corrections = Corrections(3)
    with torch.no_grad():
        corrections.colour[1] = torch.linspace(-0.2, 0.2, 12)
    rgb = torch.rand((6, 7, 3))
    assert torch.allclose(
        apply_residual(corrections.colour[1].detach(), rgb), corrections.apply_colour(1, rgb)
    )


def test_the_exposure_settles_over_frames_and_restarts_with_a_new_shot():
    poses, exposure = two_cameras()
    smooth = ExposureFilter(exposure, fps=30, seconds=0.25)
    rgb = torch.full((2, 2, 3), 0.8)
    neutral = look_at(np.zeros(3), [0, 0, 1.0])

    assert torch.allclose(smooth(neutral, rgb), rgb, atol=1e-3)
    first = smooth(poses[0], rgb)
    assert float(first.mean()) > 0.7  # one frame in, still close to the old exposure
    for _ in range(60):
        settled = smooth(poses[0], rgb)
    assert float(settled.mean()) < 0.4
    smooth.reset()
    assert torch.allclose(smooth(neutral, rgb), rgb, atol=1e-3)


def test_held_out_psnr_with_the_neighbours_exposure(tmp_path):
    import math

    import cv2

    from spacegen.config import SplatConfig
    from spacegen.contracts import Camera
    from spacegen.splat.model import SH_C0, GaussianModel, render_camera
    from spacegen.splat.train import Images, evaluate, neighbour_exposure

    xs, zs = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(0, 1.5, 24))
    means = np.stack((xs.ravel(), np.zeros(xs.size), zs.ravel()), 1)
    colours = np.stack((0.5 + 0.4 * np.sin(5 * xs), 0.5 + 0.4 * np.cos(4 * zs), 0.6 + 0 * xs), -1)
    model = GaussianModel(
        {
            "means": torch.tensor(means, dtype=torch.float32),
            "scales": torch.full((xs.size, 3), math.log(0.05)),
            "quats": torch.tensor([[1.0, 0, 0, 0]] * xs.size),
            "opacities": torch.full((xs.size,), 3.0),
            "sh0": torch.tensor((colours.reshape(-1, 3) - 0.5) / SH_C0, dtype=torch.float32)[
                :, None
            ],
            "shn": torch.zeros((xs.size, 0, 3)),
        },
        0,
    )
    k = np.array([[80.0, 0, 47.5], [0, 80.0, 31.5], [0, 0, 1]])
    eyes = [(-0.1, -2.5, 0.75), (0.0, -2.5, 0.75), (0.1, -2.5, 0.75), (0.02, -2.5, 0.76)]
    cameras = [
        Camera(f"{n:05d}", 96, 64, k, look_at(np.array(e), np.array([e[0], 0, 0.75])))
        for n, e in enumerate(eyes)
    ]
    # The phone exposed every frame at 60% of the splat's (average) brightness.
    paths = []
    for camera in cameras:
        with torch.no_grad():
            rgb = (render_camera(model, camera)[0] * 0.6).clamp(0, 1)
        paths.append(tmp_path / f"{camera.name}.png")
        cv2.imwrite(str(paths[-1]), (rgb.numpy() * 255).round().astype(np.uint8)[..., ::-1])
    corrections = Corrections(4)
    with torch.no_grad():
        corrections.colour[:3, [0, 4, 8]] = -0.4  # what training learns for those frames

    images = Images(paths, 96, {1})
    exposure = neighbour_exposure(cameras, corrections, [0, 1, 2], camera_height=1.0)
    report = evaluate(
        model, cameras, images, [3], corrections, SplatConfig(), tmp_path, "val", False, exposure
    )
    row = report["frames"][0]
    assert row["psnr_nb"] > 35 and row["psnr_nb"] > row["psnr"] + 8, row
