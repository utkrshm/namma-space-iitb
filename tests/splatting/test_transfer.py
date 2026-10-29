"""Splat transfer: local similarities reproduce a known change of gauge, ignore outliers."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.splatting.model import GaussianModel
from src.splatting.transfer import Warp, transfer_model


def _similarity():
    rotation = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix()
    return 1.7, rotation, np.array([0.4, -2.0, 3.0])


def test_warp_reproduces_a_similarity():
    scale, rotation, shift = _similarity()
    old_points = np.random.default_rng(3).uniform(-4, 4, (500, 3))
    new_points = scale * old_points @ rotation.T + shift
    new_points[:10] += 5.0  # mismatched tracks
    warp = Warp(old_points, new_points)
    test = np.random.default_rng(4).uniform(-4, 4, (100, 3))
    positions, linear = warp.blend(test)

    assert np.allclose(positions, scale * test @ rotation.T + shift, atol=1e-6)
    assert np.allclose(linear, scale * rotation, atol=1e-6)


def test_warp_follows_a_bend():
    # Two halves of the old solve differ by a 30% scale about x = 0 in the new one.
    old_points = np.random.default_rng(5).uniform(-4, 4, (4000, 3))
    factor = np.where(old_points[:, :1] < 0, 1.3, 1.0)
    warp = Warp(old_points, old_points * factor)
    test = np.array([[-3.0, 1.0, 0.5], [3.0, -1.0, 0.2]])

    assert np.allclose(warp.blend(test)[0], [[-3.9, 1.3, 0.65], [3.0, -1.0, 0.2]], atol=1e-3)


def test_transfer_maps_covariances():
    scale, rotation, shift = _similarity()
    old_points = np.random.default_rng(3).uniform(-4, 4, (500, 3))
    warp = Warp(old_points, scale * old_points @ rotation.T + shift)

    count = 64
    generator = torch.Generator().manual_seed(0)
    model = GaussianModel(
        {
            "means": torch.rand(count, 3, generator=generator) * 4 - 2,
            "scales": torch.log(torch.rand(count, 3, generator=generator) * 0.1 + 0.01),
            "quats": torch.nn.functional.normalize(
                torch.randn(count, 4, generator=generator), dim=-1
            ),
            "opacities": torch.zeros(count),
            "sh0": torch.zeros(count, 1, 3),
            "shn": torch.zeros(count, 15, 3),
        },
        3,
    )
    moved = transfer_model(model, warp)

    def covariance(m):
        quats = m.params["quats"].detach().numpy().astype(np.float64)
        r = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
        s = np.exp(m.params["scales"].detach().numpy().astype(np.float64))
        return np.einsum("nij,nj,nkj->nik", r, s**2, r)

    expected = scale**2 * np.einsum("ij,njk,lk->nil", rotation, covariance(model), rotation)
    assert np.allclose(covariance(moved), expected, atol=1e-6)
    old_means = model.params["means"].detach().numpy()
    assert np.allclose(
        moved.params["means"].detach().numpy(), scale * old_means @ rotation.T + shift, atol=1e-4
    )
