"""Real geometric evaluation and ICP tests on deterministic analytic point clouds."""

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("scipy")

from src.reconstruction.section_evaluation import (
    _accept_candidate,
    correspondence_metrics,
    evaluate_icp,
)
from src.reconstruction.section_registration import Sim3


def test_correspondence_metrics_and_finite_input_validation():
    source = np.array([[0.0, 0, 0], [2, 0, 0]])
    target = np.array([[1.0, 0, 0], [3, 0, 0]])
    metrics = correspondence_metrics(source, target, Sim3.identity(), scene_scale=2)
    assert metrics["count"] == 2
    assert metrics["median"] == pytest.approx(1)
    assert metrics["normalized_rmse"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="finite"):
        correspondence_metrics([[np.nan, 0, 0]], target, Sim3.identity())
    with pytest.raises(ValueError, match="matching correspondence"):
        correspondence_metrics(source, target[:1], Sim3.identity())
    with pytest.raises(ValueError, match="scene_scale"):
        correspondence_metrics(source, target, Sim3.identity(), scene_scale=0)


def test_acceptance_requires_median_gain_and_nonregressing_tail():
    baseline = {"normalized_median": 0.1, "normalized_p90": 0.2}
    assert _accept_candidate(baseline, {"normalized_median": 0.08, "normalized_p90": 0.2})[0]
    assert not _accept_candidate(baseline, {"normalized_median": 0.1, "normalized_p90": 0.2})[0]
    accepted, reason = _accept_candidate(
        baseline, {"normalized_median": 0.08, "normalized_p90": 0.21}
    )
    assert not accepted
    assert "p90" in reason
    accepted, reason = _accept_candidate(
        baseline, {"normalized_median": 0.08, "normalized_p90": 0.205}
    )
    assert not accepted
    assert "p90" in reason


def test_multiscale_icp_recovers_rigid_offset_and_scale_is_preserved():
    x, y = np.meshgrid(np.linspace(-1, 1, 45), np.linspace(-1, 1, 45))
    surface = np.column_stack((x.ravel(), y.ravel(), (0.2 * np.sin(3 * x) * np.cos(2 * y)).ravel()))
    held_out = (np.indices(x.shape)[0].ravel() % 5 == 0) & (np.indices(x.shape)[1].ravel() % 5 == 0)
    source = (surface[~held_out] + [0.025, -0.018, 0.012]) / 1.7
    target = surface[~held_out]
    validation_source = (surface[held_out] + [0.025, -0.018, 0.012]) / 1.7
    validation_target = surface[held_out]
    initial = Sim3(1.7, np.eye(3), np.zeros(3))
    # Inputs live at their original arbitrary scene scale; avoid magnitude-only assertions.
    output, diagnostics = evaluate_icp(
        source,
        target,
        initial,
        validation_source,
        validation_target,
        scene_scale=1.0,
        max_correspondence_distance=0.08,
        max_points=5000,
    )
    assert diagnostics["accepted"]
    assert diagnostics["after"]["median"] < diagnostics["before"]["median"] * 0.5
    assert output.scale == pytest.approx(initial.scale)
    np.testing.assert_allclose(output.apply(validation_source), validation_target, atol=0.003)


def test_worsening_icp_candidate_returns_initial_transform():
    x, y = np.meshgrid(np.linspace(-1, 1, 30), np.linspace(-1, 1, 30))
    target = np.column_stack((x.ravel(), y.ravel(), (0.15 * x**2 + 0.1 * y**2).ravel()))
    source = target + [0.01, 0, 0]
    initial = Sim3.identity()
    output, diagnostics = evaluate_icp(
        source,
        target,
        initial,
        source[::10],
        target[::10] + [0.2, 0, 0],
        scene_scale=1,
        max_correspondence_distance=0.05,
    )
    assert not diagnostics["accepted"]
    assert "improvement" in diagnostics["reason"] or "p90" in diagnostics["reason"]
    np.testing.assert_allclose(output.as_matrix(), initial.as_matrix())
