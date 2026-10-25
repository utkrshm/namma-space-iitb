"""Held-out evaluation and robust ICP refinement for section alignment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .section_registration import Sim3


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be a finite N x 3 array")
    if len(points) == 0:
        raise ValueError(f"{name} must not be empty")
    return points


def correspondence_metrics(
    source: np.ndarray, target: np.ndarray, transform: Sim3, scene_scale: float = 1.0
) -> dict:
    """Measure index-paired held-out residuals in scene units and normalized units."""
    source = _points(source, "source")
    target = _points(target, "target")
    if source.shape != target.shape:
        raise ValueError("source and target must have matching correspondence shape")
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")

    residuals = transform.apply(source) - target
    if not np.isfinite(residuals).all():
        raise ValueError("transformed correspondence residuals must be finite")
    distances = np.linalg.norm(residuals, axis=1)
    metrics = {
        "count": len(distances),
        "median": float(np.median(distances)),
        "p90": float(np.percentile(distances, 90)),
        "rmse": float(np.sqrt(np.mean(np.square(distances)))),
    }
    metrics.update(
        {
            f"normalized_{key}": value / scene_scale
            for key, value in metrics.items()
            if key != "count"
        }
    )
    return metrics


def _accept_candidate(
    before: dict, after: dict, minimum_improvement: float = 0.01
) -> tuple[bool, str]:
    """Apply deterministic held-out median-improvement and tail-regression checks."""
    baseline = before["normalized_median"]
    candidate = after["normalized_median"]
    tolerance = 1e-9
    if candidate >= baseline * (1 - minimum_improvement) - tolerance:
        return False, "held-out median improvement below required threshold"
    if after["normalized_p90"] > before["normalized_p90"] * 1.02 + tolerance:
        return False, "held-out p90 regression exceeds 2%"
    return True, "held-out median improved and p90 did not regress materially"


def evaluate_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial: Sim3,
    validation_source: np.ndarray,
    validation_target: np.ndarray,
    *,
    scene_scale: float,
    voxel_sizes: tuple[float, ...] | None = None,
    max_correspondence_distance: float = 0.04,
    max_points: int = 30000,
    max_correction_translation: float | None = None,
    max_correction_rotation_degrees: float = 10.0,
) -> tuple[Sim3, dict]:
    """Refine source-to-target Sim3 with multiscale point-to-plane ICP.

    ICP operates on source points already mapped by ``initial`` and only estimates a
    rigid correction. Validation clouds are used exclusively for the final decision.
    """
    source = _points(source_points, "source_points")
    target = _points(target_points, "target_points")
    validation_source = _points(validation_source, "validation_source")
    validation_target = _points(validation_target, "validation_target")
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")
    if max_points < 100 or max_correspondence_distance <= 0:
        raise ValueError("max_points must be at least 100 and correspondence distance positive")
    if max_correction_rotation_degrees <= 0:
        raise ValueError("max_correction_rotation_degrees must be positive")

    from .section_registration import Sim3

    before = correspondence_metrics(validation_source, validation_target, initial, scene_scale)
    source = initial.apply(source)
    if len(source) > max_points:
        source = source[np.linspace(0, len(source) - 1, max_points, dtype=int)]
    if len(target) > max_points:
        target = target[np.linspace(0, len(target) - 1, max_points, dtype=int)]

    import open3d as o3d

    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    sizes = voxel_sizes or tuple(scene_scale * fraction for fraction in (0.02, 0.01, 0.005))
    correction = np.eye(4)
    fitness = 0.0
    inlier_rmse: float | None = None
    support_valid = True
    for voxel in sizes:
        if not np.isfinite(voxel) or voxel <= 0:
            raise ValueError("voxel_sizes must contain finite positive values")
        source_level = source_cloud.voxel_down_sample(voxel)
        target_level = target_cloud.voxel_down_sample(voxel)
        if len(source_level.points) < 6 or len(target_level.points) < 6:
            support_valid = False
            break
        if (
            np.linalg.matrix_rank(
                np.asarray(source_level.points) - np.mean(source_level.points, axis=0)
            )
            < 2
            or np.linalg.matrix_rank(
                np.asarray(target_level.points) - np.mean(target_level.points, axis=0)
            )
            < 2
        ):
            support_valid = False
            break
        radius = voxel * 2.5
        source_level.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, 30))
        target_level.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius, 30))
        loss = o3d.pipelines.registration.TukeyLoss(k=voxel * 2)
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        result = o3d.pipelines.registration.registration_icp(
            source_level,
            target_level,
            max_correspondence_distance * scene_scale,
            correction,
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        correction = result.transformation
        fitness = float(result.fitness)
        inlier_rmse = float(result.inlier_rmse) if np.isfinite(result.inlier_rmse) else None
        if not np.isfinite(fitness) or fitness <= 0 or inlier_rmse is None:
            support_valid = False
            break

    delta = Sim3(1.0, correction[:3, :3], correction[:3, 3])
    candidate = delta.compose(initial)
    after = correspondence_metrics(validation_source, validation_target, candidate, scene_scale)
    accepted, reason = _accept_candidate(before, after)
    translation_limit = max_correction_translation or 0.1 * scene_scale
    correction_translation = float(np.linalg.norm(correction[:3, 3]))
    rotation_cosine = np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1)
    correction_rotation_degrees = float(np.degrees(np.arccos(rotation_cosine)))
    if not support_valid:
        accepted, reason = False, "ICP has insufficient spatial support or correspondences"
    elif correction_translation > translation_limit:
        accepted, reason = False, "rigid correction exceeds translation limit"
    elif correction_rotation_degrees > max_correction_rotation_degrees:
        accepted, reason = False, "rigid correction exceeds rotation limit"
    diagnostics = {
        "accepted": accepted,
        "reason": reason,
        "before": before,
        "after": after,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "correction": correction.tolist(),
        "candidate_transform": candidate.as_matrix().tolist(),
        "transform": (candidate if accepted else initial).as_matrix().tolist(),
    }
    return (candidate if accepted else initial), diagnostics


def render_alignment_diagnostics(
    points: np.ndarray,
    colors: np.ndarray,
    section_indices: np.ndarray,
    output_dir: str | Path,
    cameras: np.ndarray | None = None,
    gravity_axis: int = 1,
    max_points: int = 150000,
) -> dict[str, str]:
    """Save headless orthographic, 3D, and section-boundary diagnostic images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = _points(points, "points")
    colors = np.asarray(colors)
    sections = np.asarray(section_indices)
    if len(colors) != len(points) or len(sections) != len(points):
        raise ValueError("colors and section_indices must match points")
    if gravity_axis not in (0, 1, 2) or max_points < 1:
        raise ValueError("gravity_axis must be 0, 1, or 2 and max_points must be positive")
    if len(points) > max_points:
        selected = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points, colors, sections = points[selected], colors[selected], sections[selected]
    horizontal = [axis for axis in range(3) if axis != gravity_axis]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, axes in (("orthographic", horizontal), ("3d", (0, 1, 2)), ("sections", horizontal)):
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d" if len(axes) == 3 else None)
        if name == "sections":
            plotted_colors = sections
        else:
            plotted_colors = colors / 255 if colors.max(initial=0) > 1 else colors
        if len(axes) == 3:
            axis.scatter(*[points[:, index] for index in axes], c=plotted_colors, s=1)
            axis.set_box_aspect(np.ptp(points, axis=0) + 1e-12)
        else:
            axis.scatter(points[:, axes[0]], points[:, axes[1]], c=plotted_colors, s=1)
        path = output / f"{name}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths[name] = str(path)
    return paths
