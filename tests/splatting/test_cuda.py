"""Real rasterizer/optimization checks; never replace CUDA with a fake renderer."""

# Optional-dependency checks must precede imports of CUDA-dependent modules.
# ruff: noqa: E402

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")
pytest.importorskip("scipy")
pytest.importorskip("plyfile")

from src.rendering.cli import load_cameras, render_cameras
from src.rendering.renderer import render_gaussians
from src.splatting import SplattingConfig, train_splat
from src.splatting.contracts import Camera
from src.splatting.geometry import InitialPoints
from src.splatting.model import GaussianModel
from src.splatting.storage import load_checkpoint, load_ply

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Real gsplat CUDA checks require the development server",
    ),
]


def test_integer_pixel_center_matches_raster_peak():
    points = InitialPoints(
        np.array([[0, 0, 2]], dtype=np.float32),
        np.ones((1, 3), dtype=np.float32) * 0.5,
        np.ones((1, 3), dtype=np.float32) * 0.01,
        1,
        {},
    )
    model = GaussianModel.from_points(points, 0, 0.8, "cuda")
    camera = Camera("pixel:0", 40, 32, np.array([[45, 0, 19], [0, 45, 15], [0, 0, 1]]), np.eye(4))
    image = render_gaussians(model, camera)
    assert int(image.alpha.argmax()) == 15 * 40 + 19
    assert image.alpha[15, 19] > image.alpha[15, 18]
    assert image.alpha[15, 19] > image.alpha[14, 19]


def test_real_training_export_render_and_completed_resume(plane_scene, tmp_path):
    config = SplattingConfig.from_dict(
        {
            "initialization": {"max_points": 128, "points_per_frame": 128},
            "training": {
                "steps": 6,
                "sh_degree": 1,
                "sh_interval": 2,
                "checkpoint_every": 3,
                "evaluate_every": 3,
                "log_every": 1,
            },
            "refinement": {
                "start_step": 1,
                "stop_step": 5,
                "every": 1,
                "reset_every": 100,
                "max_gaussians": 160,
                "grow_gradient": 1e-12,
            },
        }
    )
    output = tmp_path / "splat"
    result = train_splat(plane_scene, output, config)
    model, state = load_checkpoint(result.checkpoint, "cuda")
    ply, _ = load_ply(result.ply, "cuda")
    assert state["step"] == 6 and 0 < model.count <= 160
    assert all(torch.isfinite(value).all() for value in model.params.values())
    assert "val" in json.loads(result.metrics.read_text())
    camera = load_cameras(plane_scene)[0]
    rendered = render_gaussians(model, camera)
    torch.testing.assert_close(rendered.rgb, render_gaussians(ply, camera).rgb)
    assert rendered.alpha.max() > 0
    rendered.rgb.mean().backward()
    assert model.params["means"].grad is not None
    assert torch.isfinite(model.params["means"].grad).all()

    render_cameras(result.ply, [camera], tmp_path / "renders")
    assert (tmp_path / "renders" / "00000_depth.npy").is_file()
    train_splat(plane_scene, output, config, resume=result.checkpoint)
    restored, _ = load_checkpoint(result.checkpoint, "cuda")
    for key in model.params:
        torch.testing.assert_close(model.params[key], restored.params[key])
