import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.reconstruction.section_registration import (
    SectionEdge,
    Sim3,
    optimize_section_graph,
    robust_sim3,
)


def test_sim3_composition_inverse_and_matrix() -> None:
    first = Sim3(1.7, Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix(), [1, 2, -1])
    second = Sim3(0.8, Rotation.from_euler("xyz", [-0.1, 0.3, 0.2]).as_matrix(), [-2, 1, 3])
    points = np.array([[0.0, 1.0, 2.0], [3.0, -1.0, 0.5]])

    np.testing.assert_allclose(first.inverse().apply(first.apply(points)), points, atol=1e-10)
    np.testing.assert_allclose(
        first.compose(second).apply(points), first.apply(second.apply(points))
    )
    np.testing.assert_allclose(
        (first.as_matrix() @ np.c_[points, np.ones(len(points))].T).T[:, :3], first.apply(points)
    )


@pytest.mark.parametrize("scale, rotation", [(-1.0, np.eye(3)), (1.0, np.diag([1, 1, -1]))])
def test_sim3_rejects_invalid_similarity(scale: float, rotation: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Sim3(scale, rotation, np.zeros(3))


def test_robust_sim3_recovers_transform_with_outliers_and_planar_points() -> None:
    rng = np.random.default_rng(12)
    source = rng.uniform(-2, 2, (80, 3))
    source[:, 2] = 0.0
    expected = Sim3(1.35, Rotation.from_euler("xyz", [0.15, -0.3, 0.5]).as_matrix(), [2, -1, 0.4])
    target = expected.apply(source) + rng.normal(0, 0.002, source.shape)
    target[:18] = rng.normal(0, 4, (18, 3))

    actual, report = robust_sim3(source, target, threshold=0.02, seed=5)

    assert report["inlier_count"] >= 60
    assert report["inlier_ratio"] >= 0.75
    np.testing.assert_allclose(actual.scale, expected.scale, atol=0.01)
    np.testing.assert_allclose(actual.apply(source[18:]), expected.apply(source[18:]), atol=0.01)


def test_robust_sim3_rejects_collinear_and_bad_input() -> None:
    line = np.c_[np.arange(5), np.zeros((5, 2))]
    with pytest.raises(ValueError, match="span"):
        robust_sim3(line, line)
    with pytest.raises(ValueError, match="shape"):
        robust_sim3(np.zeros((3, 2)), np.zeros((3, 2)))


def test_automatic_threshold_uses_target_units_and_ignores_zero_weight_points() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(30, 3))
    expected = Sim3(3.0, np.eye(3), [0, 0, 0])
    target = expected.apply(source)
    weights = np.ones(len(source))
    weights[:3] = 0

    _, report = robust_sim3(source, target, weights=weights)

    source_extent = np.median(np.linalg.norm(source[3:] - np.median(source[3:], axis=0), axis=1))
    target_extent = np.median(np.linalg.norm(target[3:] - np.median(target[3:], axis=0), axis=1))
    assert report["threshold"] == pytest.approx(target_extent * 0.03)
    assert report["inlier_count"] == 27
    assert report["threshold"] > source_extent * 0.03
    extended_report = robust_sim3(
        np.vstack([source, [1e8, 1e8, 1e8]]),
        np.vstack([target, [-1e12, 1e12, 1e12]]),
        weights=np.r_[weights, 0.0],
    )[1]
    assert extended_report["threshold"] == pytest.approx(report["threshold"])
    source_tiny = source * 1e-10
    target_tiny = target * 1e-10
    tiny_estimate, _ = robust_sim3(source_tiny, target_tiny, threshold=1e-12, seed=3)
    np.testing.assert_allclose(tiny_estimate.scale, expected.scale, rtol=1e-8)


def test_graph_initializes_chain_and_loop_reduces_cost() -> None:
    transforms = [
        Sim3(1.1, Rotation.from_euler("z", 0.2).as_matrix(), [1, 0.2, 0]),
        Sim3(0.9, Rotation.from_euler("y", -0.15).as_matrix(), [0.3, 1, 0.1]),
    ]
    edges = [SectionEdge(0, 1, transforms[0]), SectionEdge(1, 2, transforms[1])]
    chain_poses, _ = optimize_section_graph(3, edges)
    np.testing.assert_allclose(chain_poses[0].as_matrix(), np.eye(4), atol=1e-6)
    for edge in edges:
        predicted = chain_poses[edge.target].inverse().compose(chain_poses[edge.source])
        np.testing.assert_allclose(predicted.as_matrix(), edge.transform.as_matrix(), atol=1e-6)

    loop_transform = transforms[1].compose(transforms[0])
    noisy_loop = Sim3(
        loop_transform.scale * 1.02,
        loop_transform.rotation,
        loop_transform.translation + [0.15, 0, 0],
    )
    poses, report = optimize_section_graph(
        3, edges + [SectionEdge(0, 2, noisy_loop, weight=0.5, kind="loop")]
    )
    assert report["final_cost"] < report["initial_cost"]
    assert report["connected"]
    assert len(poses) == 3
    np.testing.assert_allclose(poses[0].as_matrix(), np.eye(4))


def test_graph_rejects_disconnected_and_invalid_edges() -> None:
    with pytest.raises(ValueError, match="disconnected"):
        optimize_section_graph(3, [SectionEdge(0, 1, Sim3.identity())])
    with pytest.raises(ValueError, match="endpoint"):
        optimize_section_graph(2, [SectionEdge(0, 2, Sim3.identity())])
    with pytest.raises(ValueError, match="endpoint"):
        optimize_section_graph(1, [SectionEdge(0, 1, Sim3.identity())])


def test_graph_reverse_measurement_has_same_pose_and_scene_normalization() -> None:
    measured = Sim3(1.2, Rotation.from_euler("z", 0.3).as_matrix(), [2, 1, -0.5])
    forward, _ = optimize_section_graph(2, [SectionEdge(0, 1, measured, translation_scale=5)])
    reverse, report = optimize_section_graph(
        2, [SectionEdge(1, 0, measured.inverse(), translation_scale=5)]
    )
    np.testing.assert_allclose(forward[1].as_matrix(), reverse[1].as_matrix(), atol=1e-7)
    assert report["edge_residuals_before"] == report["edge_residuals_after"]
    assert report["initial_objective"] == pytest.approx(report["final_objective"])


def test_graph_unit_scaling_and_gross_loop_are_robust() -> None:
    truth = [
        Sim3.identity(),
        Sim3(1.0, np.eye(3), [1, 0, 0]),
        Sim3(1.0, np.eye(3), [2, 0.2, 0]),
        Sim3(1.0, np.eye(3), [3, 0, 0]),
    ]
    edges = []
    for left in range(4):
        for right in range(left + 1, 4):
            measurement = truth[right].inverse().compose(truth[left])
            edges.append(SectionEdge(left, right, measurement, translation_scale=2.0))
    bad = Sim3(1.0, np.eye(3), [100, -50, 20])
    edges.append(SectionEdge(0, 3, bad, weight=0.1, kind="bad-loop", translation_scale=2.0))

    poses, report = optimize_section_graph(4, edges)
    assert report["final_objective"] < report["initial_objective"]
    assert np.linalg.norm(poses[3].translation - truth[3].translation) < 0.5

    unit_factor = 1000.0
    converted_edges = [
        SectionEdge(
            edge.source,
            edge.target,
            Sim3(
                edge.transform.scale,
                edge.transform.rotation,
                edge.transform.translation * unit_factor,
            ),
            edge.weight,
            edge.kind,
            edge.translation_scale * unit_factor,
        )
        for edge in edges
    ]
    converted, _ = optimize_section_graph(4, converted_edges)
    for original_pose, converted_pose in zip(poses, converted, strict=True):
        np.testing.assert_allclose(
            converted_pose.translation / unit_factor, original_pose.translation, atol=1e-5
        )
