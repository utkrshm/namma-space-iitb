"""Training-view weights: visible Gaussians cover pixels, hidden and off-screen ones do not."""

import numpy as np
import pytest
import torch

from src.splatting.model import GaussianModel


@pytest.mark.cuda
def test_hidden_and_off_screen_gaussians_have_no_weight():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("gsplat")
    from src.splatting.visibility import training_view_weights

    # An opaque wall of 400 discs at z = 4, one Gaussian behind it, one far off-screen.
    # The hidden one only shows through the wall's tiny gaps: far below the 1 px cut.
    grid = np.stack(np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20)), -1)
    wall = np.column_stack((grid.reshape(-1, 2), np.full(400, 4.0)))
    means = np.concatenate((wall, [[0.0, 0.0, 6.0], [50.0, 0.0, 4.0]])).astype(np.float32)
    count = len(means)
    model = GaussianModel(
        {
            "means": torch.from_numpy(means),
            "scales": torch.full((count, 3), np.log(0.15)),
            "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(count, 1),
            "opacities": torch.full((count,), 8.0),  # sigmoid ≈ 1
            "sh0": torch.rand(count, 1, 3),
            "shn": torch.zeros(count, 0, 3),
        },
        0,
    ).to("cuda")
    camera = {
        "width": 64,
        "height": 48,
        "intrinsics": [[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]],
        "world_to_camera": np.eye(4).tolist(),
    }

    peak, total = training_view_weights(model, [camera], scale=1.0)

    assert float(peak[:400].sum()) == pytest.approx(64 * 48, rel=0.02)  # the wall fills the image
    assert float(peak[400]) < 0.5 and float(peak[401]) == 0
    assert torch.equal(peak, total)  # one view
