"""Web `.splat` export: pruning and an exact byte-layout round trip."""

import math

import numpy as np
import torch

from src.splatting.model import SH_C0, GaussianModel
from src.splatting.web_export import prune, read_splat, write_splat


def _model(count=50):
    generator = torch.Generator().manual_seed(0)
    opacity = torch.linspace(0.01, 0.95, count)
    return GaussianModel(
        {
            "means": torch.randn(count, 3, generator=generator),
            "scales": torch.log(torch.rand(count, 3, generator=generator) * 0.1 + 0.01),
            "quats": torch.nn.functional.normalize(
                torch.randn(count, 4, generator=generator), dim=-1
            ),
            "opacities": torch.log(opacity / (1 - opacity)),
            "sh0": (torch.rand(count, 1, 3, generator=generator) - 0.5) / SH_C0,
            "shn": torch.zeros(count, 15, 3),
        },
        3,
    )


def test_prune_drops_only_low_opacity():
    model = _model()

    pruned = prune(model, 0.5)

    kept = model.params["opacities"].sigmoid() >= 0.5
    assert pruned.count == int(kept.sum())
    assert torch.equal(pruned.params["means"], model.params["means"][kept])


def test_splat_file_round_trip(tmp_path):
    model = _model()

    count = write_splat(tmp_path / "scene.splat", model)
    data = read_splat(tmp_path / "scene.splat")

    assert count == 50 and (tmp_path / "scene.splat").stat().st_size == 32 * 50
    # Most important (opacity × volume) first; every Gaussian is present exactly once.
    order = np.lexsort(data["means"].T)
    reference = model.params["means"].detach().numpy()
    assert np.allclose(data["means"][order], reference[np.lexsort(reference.T)])
    volume = data["scales"].prod(axis=1) * data["rgba"][:, 3]
    assert np.all(np.diff(volume) <= 1e-3 * volume.max())
    # 8-bit colour and rotation are exact to quantization.
    first = int(np.argmin(np.abs(reference - data["means"][0]).sum(axis=1)))
    rgb = (0.5 + SH_C0 * model.params["sh0"][first, 0]).clamp(0, 1).detach().numpy()
    assert np.allclose(data["rgba"][0, :3], rgb, atol=1 / 255)
    quat = model.params["quats"][first].detach().numpy()
    assert abs(abs(float(np.dot(data["quats"][0], quat))) - 1) < 1e-3
    assert math.isclose(
        float(data["scales"][0, 0]), float(model.params["scales"][first, 0].exp()), rel_tol=1e-6
    )
