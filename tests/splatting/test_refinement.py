"""Exercise real topology operations and Adam state, without a mock rasterizer."""

# ruff: noqa: E402

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from src.splatting.config import RefinementConfig
from src.splatting.model import GaussianModel
from src.splatting.refinement import BudgetedStrategy


def test_growth_budget_preserves_parameter_optimizer_alignment():
    model = GaussianModel(
        {
            "means": torch.tensor(
                [[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]], dtype=torch.float32
            ),
            "scales": torch.tensor([[math.log(0.001)] * 3, [math.log(0.1)] * 3] * 2),
            "quats": torch.tensor([[1, 0, 0, 0]] * 4, dtype=torch.float32),
            "opacities": torch.full((4,), -2.0),
            "sh0": torch.zeros((4, 1, 3)),
            "shn": torch.zeros((4, 3, 3)),
        },
        1,
    )
    optimizers = {key: torch.optim.Adam([value], lr=0.001) for key, value in model.params.items()}
    sum(value.square().sum() for value in model.params.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    strategy = BudgetedStrategy(RefinementConfig(max_gaussians=6), training_views=2)
    state = strategy.initialize_state(scene_scale=1)
    # Directly supply the documented accumulated projection-gradient statistics.
    # Highest scores select one small clone and one large split (+2 total).
    state["grad2d"] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    state["count"] = torch.ones(4)
    clones, splits = strategy._grow_gs(model.params, optimizers, state, step=600)
    assert (clones, splits) == (1, 1) and model.count == 6
    strategy.check_sanity(model.params, optimizers)
    model.validate()
    for key, optimizer in optimizers.items():
        parameter = model.params[key]
        assert optimizer.param_groups[0]["params"][0] is parameter
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
    assert state["count"].shape == (6,)
    assert strategy._grow_gs(model.params, optimizers, state, step=700) == (0, 0)
