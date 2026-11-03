"""The CPU rasterizer (`spacegen.splat.raster`) against closed-form blending, and against
gsplat on the GPU when there is one."""

import math

import numpy as np
import pytest
import torch

from spacegen.contracts import Camera, look_at
from spacegen.splat.model import SH_C0, GaussianModel, render_camera
from spacegen.splat.raster import cpu_rasterization
from spacegen.support import blending_weight

K = torch.tensor([[[100.0, 0, 32], [0, 100, 24], [0, 0, 1]]])
VIEW = torch.eye(4)[None]


def blobs(means, sizes, opacities, colours) -> dict:
    count = len(means)
    return {
        "means": torch.tensor(means, dtype=torch.float32),
        "quats": torch.tensor([[1.0, 0, 0, 0]] * count),
        "scales": torch.tensor(sizes, dtype=torch.float32)[:, None].repeat(1, 3),
        "opacities": torch.tensor(opacities, dtype=torch.float32),
        "colors": torch.tensor(colours, dtype=torch.float32),
    }


def draw(scene: dict, **options):
    options = {"render_mode": "RGB+ED", "rasterize_mode": "classic", **options}
    return cpu_rasterization(**scene, viewmats=VIEW, Ks=K, width=64, height=48, **options)


def test_one_gaussian_matches_its_closed_form_footprint():
    x, y, z = 0.1, -0.05, 2.0
    image, alpha, info = draw(blobs([[x, y, z]], [0.05], [0.9], [[1.0, 0.5, 0.25]]))

    # EWA splatting: Σ₂ = J Σ Jᵀ with the projection's Jacobian, plus gsplat's 0.3 px² filter.
    jacobian = np.array([[100 / z, 0, -100 * x / z**2], [0, 100 / z, -100 * y / z**2]])
    covariance = 0.05**2 * jacobian @ jacobian.T + 0.3 * np.eye(2)
    radii = [math.ceil(3.33 * math.sqrt(covariance[i, i])) for i in (0, 1)]
    assert info["radii"][0, 0].tolist() == radii
    ys, xs = np.mgrid[0:48, 0:64] + 0.5  # pixel centres
    delta = np.stack((xs - (32 + 100 * x / z), ys - (24 + 100 * y / z)), -1)
    sigma = 0.5 * np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta)
    expected = 0.9 * np.exp(-sigma)
    expected[expected < 1 / 255] = 0
    assert np.abs(alpha[0, ..., 0].numpy() - expected).max() < 1e-5
    lit = alpha[0, ..., 0] > 0
    assert torch.allclose(image[0, ..., :3][lit] / alpha[0][lit], torch.tensor([1.0, 0.5, 0.25]))
    assert torch.allclose(image[0, ..., 3][lit], torch.tensor(2.0))


def test_nearer_gaussians_cover_farther_ones_whatever_the_input_order():
    front, back = [0.0, 0.0, 1.0], [0.0, 0.0, 3.0]
    a = draw(blobs([back, front], [0.2, 0.05], [0.8, 0.6], [[0, 0, 1], [1, 0, 0]]))[0]
    b = draw(blobs([front, back], [0.05, 0.2], [0.6, 0.8], [[1, 0, 0], [0, 0, 1]]))[0]
    assert torch.allclose(a, b)

    # At the centre pixel (31.5, 23.5): front alpha 0.6·g1, then the back one behind it.
    near = 0.6 * math.exp(-0.5 * 0.5 / (5.0**2 + 0.3))
    far = 0.8 * math.exp(-0.5 * 0.5 / ((100 * 0.2 / 3) ** 2 + 0.3))
    pixel = a[0, 23, 31]
    assert float(pixel[0]) == pytest.approx(near, rel=1e-5)
    assert float(pixel[2]) == pytest.approx((1 - near) * far, rel=1e-5)


def test_a_pixel_stops_before_its_transmittance_drops_to_1e_4():
    # Opaque (alpha 0.999) red in front of green: one layer leaves 1e-3, a second would
    # leave 1e-6 ≤ 1e-4, so gsplat stops there and no green shows.
    scene = blobs([[0, 0, 1.0], [0, 0, 2.0]], [0.5, 0.5], [1.0, 1.0], [[1, 0, 0], [0, 1, 0]])
    image, alpha, _ = draw(scene)
    assert float(alpha[0, 24, 32]) == pytest.approx(0.999, abs=1e-5)
    assert float(image[0, 24, 32, 1]) == 0.0


def test_chunked_and_whole_renders_agree():
    rng = np.random.default_rng(3)
    count = 400
    means = np.c_[rng.uniform(-0.6, 0.6, (count, 2)), rng.uniform(1, 4, count)]
    scene = blobs(
        means,
        rng.uniform(0.01, 0.2, count),
        rng.uniform(0.2, 1, count),
        rng.uniform(0, 1, (count, 3)),
    )
    background = torch.tensor([[0.2, 0.3, 0.4]])
    whole = draw(scene, backgrounds=background, rasterize_mode="antialiased")
    chunked = draw(scene, backgrounds=background, rasterize_mode="antialiased", pair_floats=4096)
    assert torch.allclose(whole[0], chunked[0], atol=1e-5)
    assert torch.allclose(whole[1], chunked[1], atol=1e-5)


def model_and_camera(device: str):
    """A tilted grid of small coloured Gaussians seen by a 96 × 64 camera."""
    xs, zs = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(0, 1.5, 24))
    means = np.stack((xs.ravel(), 0.3 * xs.ravel() ** 2, zs.ravel()), 1)
    colours = np.stack((0.5 + 0.4 * np.sin(5 * xs), 0.5 + 0.4 * np.cos(4 * zs), 0.4 + 0 * xs), -1)
    count = len(means)
    tensors = {
        "means": torch.tensor(means, dtype=torch.float32),
        "scales": torch.full((count, 3), math.log(0.04)),
        "quats": torch.tensor([[1.0, 0.2, 0, 0]] * count),
        "opacities": torch.full((count,), 1.5),
        "sh0": torch.tensor((colours.reshape(-1, 3) - 0.5) / SH_C0, dtype=torch.float32)[:, None],
        "shn": torch.zeros((count, 3, 3)),
    }
    k = np.array([[80.0, 0, 47.5], [0, 80.0, 31.5], [0, 0, 1]])
    camera = Camera("c", 96, 64, k, look_at(np.array([0.2, -2.5, 0.9]), np.array([0, 0, 0.75])))
    return GaussianModel(tensors, 1).to(device), camera


def test_blending_weights_flow_through_the_cpu_rasterizer():
    model, camera = model_and_camera("cpu")
    weight = blending_weight(model, camera)
    with torch.no_grad():
        _, _, alpha, _ = render_camera(model, camera)
    # Each Gaussian's weight is its share of the image; together they make up the alpha.
    assert float(weight.sum()) == pytest.approx(float(alpha.sum()), rel=1e-4)
    assert float(weight.min()) >= 0
    assert model.params["means"].grad is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cpu_and_gsplat_renders_agree():
    model, camera = model_and_camera("cpu")
    with torch.no_grad():
        rgb, depth, alpha, _ = render_camera(model, camera)
        reference = render_camera(model.cuda(), camera)
    assert float((rgb - reference[0].cpu()).abs().mean()) < 0.01
    assert float((alpha - reference[2].cpu()).abs().mean()) < 0.01
    solid = alpha > 0.5
    assert float((depth - reference[1].cpu())[solid].abs().mean()) < 0.01
