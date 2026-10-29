"""Floor-constrained BA building blocks: trilinear field, rotations and floor-point selection."""

import numpy as np
import torch

from src.sfm.floor_ba import Grid, _exp, floor_points, local_rotation


def test_grid_reproduces_a_linear_field_and_its_rotation():
    grid = Grid(origin=[0.0, 0.0, 0.0], spacing=0.5, shape=(9, 9, 9), device="cpu")
    nodes = torch.stack(
        torch.meshgrid(*(torch.arange(9, dtype=torch.float64) * 0.5,) * 3, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    angle = 0.01
    generator = torch.tensor([[0, -angle, 0], [angle, 0, 0], [0, 0, 0]], dtype=torch.float64)
    delta = nodes @ generator.T  # small rotation about z as a displacement field
    points = torch.rand(50, 3, dtype=torch.float64) * 3 + 0.5

    assert torch.allclose(grid(delta, points), points @ generator.T, atol=1e-12)
    rotation = local_rotation(grid, delta, points, 0.25)
    assert torch.allclose(rotation, torch.tensor([0, 0, angle], dtype=torch.float64).expand(50, 3))


def test_exp_gives_rotations():
    vectors = torch.tensor([[0.0, 0.0, 0.0], [0.3, -0.2, 0.9], [1e-8, 0, 0]], dtype=torch.float64)
    rotations = _exp(vectors)
    eye = torch.eye(3, dtype=torch.float64)

    assert torch.allclose(rotations @ rotations.transpose(1, 2), eye.expand(3, 3, 3), atol=1e-12)
    assert torch.allclose(rotations[0], eye)
    assert torch.allclose(rotations[1] @ vectors[1], vectors[1])  # the axis is fixed


def test_floor_points_take_the_floor_not_furniture():
    rng = np.random.default_rng(0)
    centres = np.column_stack(
        (np.linspace(0, 8, 60), 0.3 * np.sin(np.arange(60)), np.full(60, 1.5))
    )
    floor = np.column_stack((rng.uniform(-1, 9, 6000), rng.uniform(-2, 2, 6000), np.zeros(6000)))
    table = np.column_stack(
        (rng.uniform(3, 4, 1500), rng.uniform(-0.5, 0.5, 1500), np.full(1500, 0.75))
    )
    points = np.concatenate((floor, table))
    selected, camera_height = floor_points(
        points, np.full(len(points), 5), centres, np.array([0, 0, 1.0])
    )

    assert abs(camera_height - 1.5) < 0.05
    assert np.isin(np.arange(6000), selected).mean() > 0.9
    assert not np.isin(np.arange(6000, 7500), selected).any()
