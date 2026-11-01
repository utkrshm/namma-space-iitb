"""GPU tests of the planar reflection layer: a mirror floor shows the mirrored scene."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from spacegen.contracts import Camera, look_at
from spacegen.splat.model import SH_C0, GaussianModel, load_ply, render_camera, save_ply
from spacegen.splat.reflect import make_reflectors, mirror

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]


def gaussians(means, colours, size, flat=False) -> dict:
    count = len(means)
    scales = torch.full((count, 3), math.log(size))
    if flat:
        scales[:, 2] = math.log(size / 50)
    quats = torch.zeros((count, 4))
    quats[:, 0] = 1
    return {
        "means": torch.as_tensor(means, dtype=torch.float32),
        "scales": scales,
        "quats": quats,
        "opacities": torch.full((count,), 6.0),
        "sh0": ((torch.as_tensor(colours, dtype=torch.float32) - 0.5) / SH_C0)[:, None],
        "shn": torch.zeros((count, 0, 3)),
    }


def join(a: dict, b: dict) -> dict:
    return {k: torch.cat((a[k], b[k])) for k in a}


def scene():
    """A black floor (z = 0, flat discs) and two coloured, tilted ellipsoids above it."""
    xs, ys = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61))
    floor_means = np.stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)), 1)
    floor = gaussians(floor_means, np.zeros_like(floor_means), 0.08, flat=True)
    objects = gaussians([[0.3, 0.0, 0.6], [-0.5, 0.4, 0.3]], [[1, 0.2, 0.1], [0.1, 0.8, 0.2]], 0.15)
    objects["scales"] = torch.log(torch.tensor([[0.3, 0.05, 0.08], [0.06, 0.25, 0.04]]))
    objects["quats"] = F.normalize(
        torch.tensor([[0.9, 0.3, -0.2, 0.25], [0.7, -0.4, 0.5, 0.3]]), dim=-1
    )
    return floor, objects


def mirrored_reference(objects, plane):
    """The objects plus their mirror images, rasterized with explicit covariances M Σ M."""
    from gsplat import rasterization

    w, x, y, z = objects["quats"].unbind(-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)
    covariance = rotation @ torch.diag_embed(objects["scales"].exp() ** 2) @ rotation.mT
    reflection = torch.eye(3) - 2 * torch.outer(plane[:3], plane[:3])
    view = camera()
    k = torch.tensor(view.intrinsics, dtype=torch.float32)
    k[:2, 2] += 0.5
    image, _, _ = rasterization(
        means=torch.cat((objects["means"], mirror(objects["means"], plane))).cuda(),
        quats=None,
        scales=None,
        covars=torch.cat((covariance, reflection @ covariance @ reflection)).cuda(),
        opacities=torch.sigmoid(objects["opacities"]).repeat(2).cuda(),
        colors=(objects["sh0"][:, 0] * SH_C0 + 0.5).repeat(2, 1).cuda(),
        viewmats=torch.tensor(view.world_to_camera, dtype=torch.float32)[None].cuda(),
        Ks=k[None].cuda(),
        width=view.width,
        height=view.height,
        rasterize_mode="antialiased",
    )
    return image[0]


def camera() -> Camera:
    k = np.array([[300.0, 0, 159.5], [0, 300.0, 119.5], [0, 0, 1]])
    return Camera("c", 320, 240, k, look_at(np.array([0, -2.5, 1.5]), np.zeros(3), (0, 0, 1)))


def test_mirror_floor_matches_the_explicitly_mirrored_scene():
    floor, objects = scene()
    plane = torch.tensor([0.0, 0.0, 1.0, 0.0])

    model = GaussianModel(join(floor, objects), 0).cuda()
    model.reflectors = make_reflectors(plane, 0.03, "cuda")
    model.reflectors["f0"] = torch.full((1,), 12.0, device="cuda")  # a perfect mirror
    on_floor = torch.cat((torch.full((len(floor["means"]),), 12.0), torch.full((2,), -12.0)))
    model.params["reflect"] = torch.nn.Parameter(on_floor[:, None].cuda())
    with torch.no_grad():
        composite = render_camera(model, camera())[0]

    # Reference: the objects plus their mirror images as ordinary Gaussians, no floor.
    reference = mirrored_reference(objects, plane)

    def soft(image):  # the reflection pass renders at half resolution
        return F.avg_pool2d(image.permute(2, 0, 1)[None], 4)[0]

    error = (soft(composite) - soft(reference)).abs()
    # Without the reflection the mean error is 0.012 (the mirror images are missing).
    assert float(error.mean()) < 0.002, float(error.mean())
    assert float(error.max()) < 0.15, float(error.max())


def test_reflection_strength_round_trips_through_the_ply(tmp_path):
    floor, objects = scene()
    model = GaussianModel(join(floor, objects), 0).cuda()
    model.reflectors = make_reflectors([[0.0, 0.0, 1.0, 0.0]], 0.03, "cuda")
    model.params["reflect"] = torch.nn.Parameter(torch.randn(model.count, 1, device="cuda"))
    save_ply(tmp_path / "scene.ply", model)
    loaded = load_ply(tmp_path / "scene.ply")
    assert torch.allclose(loaded.params["reflect"], model.params["reflect"])
    assert torch.allclose(loaded.reflectors["planes"], model.reflectors["planes"])
    with torch.no_grad():
        a = render_camera(model, camera())[0]
        b = render_camera(loaded, camera())[0]
    assert float((a - b).abs().max()) < 1e-4
