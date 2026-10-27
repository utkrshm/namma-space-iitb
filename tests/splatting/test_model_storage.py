# Optional-dependency checks must precede imports of torch-dependent modules.
# ruff: noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install the splatting extra on the development server")
pytest.importorskip("plyfile", reason="Install the splatting extra on the development server")

from src.splatting.geometry import InitialPoints
from src.splatting.losses import rgb_depth_loss
from src.splatting.model import GaussianModel
from src.splatting.storage import export_ply, load_checkpoint, load_ply, save_checkpoint


def make_model():
    points = InitialPoints(
        np.array([[0, 0, 2], [0.1, 0, 2], [0, 0.1, 2], [0.1, 0.1, 2]], dtype=np.float32),
        np.full((4, 3), 0.4, dtype=np.float32),
        np.full((4, 3), 0.05, dtype=np.float32),
        1,
        {},
    )
    model = GaussianModel.from_points(points, sh_degree=2, opacity=0.1, device="cpu")
    with torch.no_grad():
        model.params["shn"].copy_(torch.arange(96).reshape(4, 8, 3) / 100)
    return model


def test_ply_preserves_all_sh_channels_and_parameterization(tmp_path):
    model = make_model()
    path = tmp_path / "scene.ply"
    export_ply(path, model, {"scene_units": "arbitrary"})
    restored, metadata = load_ply(path)
    assert restored.sh_degree == 2 and metadata["scene_units"] == "arbitrary"
    for key in model.params:
        torch.testing.assert_close(restored.params[key], model.params[key])


def test_checkpoint_restores_adam_continuation(tmp_path):
    model = make_model()
    optimizer = torch.optim.Adam([model.params["means"]], lr=0.01)
    model.params["means"].square().sum().backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizers={"means": optimizer.state_dict()}, step=1)
    restored, state = load_checkpoint(path)
    other = torch.optim.Adam([restored.params["means"]], lr=0.01)
    other.load_state_dict(state["optimizers"]["means"])
    for candidate, opt in ((model, optimizer), (restored, other)):
        opt.zero_grad(set_to_none=True)
        candidate.params["means"].square().sum().backward()
        opt.step()
    torch.testing.assert_close(restored.params["means"], model.params["means"], rtol=0, atol=0)


def test_invalid_depth_does_not_poison_gradients():
    from src.rendering.renderer import RenderOutput

    rgb = torch.full((16, 16, 3), 0.4, requires_grad=True)
    predicted_depth = torch.full((16, 16), 1.8, requires_grad=True)
    target_depth = torch.full((16, 16), 2.0)
    target_depth[0, 0] = float("nan")
    valid = torch.isfinite(target_depth)
    rendered = RenderOutput(rgb, torch.ones_like(target_depth), predicted_depth, {})
    loss, _ = rgb_depth_loss(
        rendered, torch.full_like(rgb, 0.5), target_depth, torch.ones_like(valid), valid, 0.2, 0.1
    )
    loss.backward()
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(predicted_depth.grad).all()
    assert predicted_depth.grad[0, 0] == 0
