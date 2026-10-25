"""Similarity registration and pose-graph optimization for reconstruction sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _rotation(value: ArrayLike) -> NDArray[np.float64]:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("rotation must be a proper rotation matrix")
    return rotation.copy()


@dataclass(frozen=True)
class Sim3:
    """Similarity transform ``X_out = scale * rotation @ X_in + translation``."""

    scale: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        rotation = _rotation(self.rotation)
        translation = np.asarray(self.translation, dtype=float)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("translation must be a finite 3-vector")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation.copy())

    @classmethod
    def identity(cls) -> Sim3:
        return cls(1.0, np.eye(3), np.zeros(3))

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        points_array = np.asarray(points, dtype=float)
        if points_array.shape[-1:] != (3,):
            raise ValueError("points must have a final dimension of 3")
        return self.scale * (points_array @ self.rotation.T) + self.translation

    def inverse(self) -> Sim3:
        inverse_rotation = self.rotation.T
        inverse_scale = 1.0 / self.scale
        return Sim3(
            inverse_scale, inverse_rotation, -inverse_scale * inverse_rotation @ self.translation
        )

    def compose(self, other: Sim3) -> Sim3:
        """Return ``self(other(points))``."""
        return Sim3(
            self.scale * other.scale,
            self.rotation @ other.rotation,
            self.scale * self.rotation @ other.translation + self.translation,
        )

    def as_matrix(self) -> NDArray[np.float64]:
        matrix = np.eye(4)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class SectionEdge:
    """Source-to-target measurement; translation_scale is target-local scene length."""

    source: int
    target: int
    transform: Sim3
    weight: float = 1.0
    kind: str = "overlap"
    translation_scale: float = 1.0


def _fit_sim3(
    source: NDArray[np.float64], target: NDArray[np.float64], weights: NDArray[np.float64]
) -> Sim3:
    weights = weights / weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (target_zero * weights[:, None]).T @ source_zero
    left, singular_values, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    variance = np.sum(weights * np.sum(source_zero**2, axis=1))
    scale = float(np.dot(singular_values, np.diag(correction)) / variance)
    translation = target_center - scale * rotation @ source_center
    return Sim3(scale, rotation, translation)


def _noncollinear(points: NDArray[np.float64]) -> bool:
    centered = points - points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return (
        len(singular_values) >= 2
        and singular_values[0] > 0
        and singular_values[1] > singular_values[0] * 1e-8
    )


def robust_sim3(
    source: ArrayLike,
    target: ArrayLike,
    weights: ArrayLike | None = None,
    threshold: float | None = None,
    seed: int = 0,
) -> tuple[Sim3, dict]:
    """Estimate a robust source-to-target similarity from corresponding 3D points."""
    source_points = np.asarray(source, dtype=float)
    target_points = np.asarray(target, dtype=float)
    if (
        source_points.ndim != 2
        or source_points.shape[1] != 3
        or target_points.shape != source_points.shape
    ):
        raise ValueError("source and target must have matching shape (N, 3)")
    if (
        len(source_points) < 3
        or not np.all(np.isfinite(source_points))
        or not np.all(np.isfinite(target_points))
    ):
        raise ValueError("at least three finite point correspondences are required")
    point_weights = (
        np.ones(len(source_points)) if weights is None else np.asarray(weights, dtype=float)
    )
    if (
        point_weights.shape != (len(source_points),)
        or not np.all(np.isfinite(point_weights))
        or np.any(point_weights < 0)
    ):
        raise ValueError("weights must be finite, nonnegative, and match the points")
    if np.count_nonzero(point_weights) < 3 or point_weights.sum() <= 0:
        raise ValueError("at least three correspondences must have positive weight")
    if not _noncollinear(source_points[point_weights > 0]) or not _noncollinear(
        target_points[point_weights > 0]
    ):
        raise ValueError("correspondences must span at least two spatial dimensions")
    positive_target_points = target_points[point_weights > 0]
    scene_size = float(
        np.median(
            np.linalg.norm(
                positive_target_points - np.median(positive_target_points, axis=0), axis=1
            )
        )
    )
    if threshold is None:
        threshold = max(scene_size * 0.03, np.finfo(float).eps * 100)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(len(source_points), dtype=bool)
    best_score = -1.0
    for _ in range(min(1000, max(100, len(source_points) * 20))):
        sample = rng.choice(
            len(source_points), 3, replace=False, p=point_weights / point_weights.sum()
        )
        if not _noncollinear(source_points[sample]) or not _noncollinear(target_points[sample]):
            continue
        try:
            candidate = _fit_sim3(
                source_points[sample], target_points[sample], point_weights[sample]
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = np.linalg.norm(candidate.apply(source_points) - target_points, axis=1)
        mask = residuals <= threshold
        score = float(point_weights[mask].sum())
        if score > best_score:
            best_score, best_mask = score, mask
    if np.count_nonzero(best_mask) < 3:
        raise ValueError("could not find a nondegenerate consensus")

    best_mask &= point_weights > 0
    if (
        np.count_nonzero(best_mask) < 3
        or not _noncollinear(source_points[best_mask])
        or not _noncollinear(target_points[best_mask])
    ):
        raise ValueError("consensus must span at least two dimensions with positive weight")
    inlier_weights = point_weights * best_mask
    estimate = _fit_sim3(
        source_points[best_mask], target_points[best_mask], inlier_weights[best_mask]
    )
    for _ in range(20):
        residuals = np.linalg.norm(estimate.apply(source_points) - target_points, axis=1)
        median = float(np.median(residuals[best_mask]))
        robust_scale = max(1.4826 * median, threshold * 0.1, np.finfo(float).eps)
        huber = np.minimum(1.0, 1.345 * robust_scale / np.maximum(residuals, np.finfo(float).eps))
        refined_weights = point_weights * best_mask * huber
        refined = _fit_sim3(
            source_points[best_mask], target_points[best_mask], refined_weights[best_mask]
        )
        converged = (
            abs(refined.scale - estimate.scale) < 1e-10
            and np.linalg.norm(refined.rotation - estimate.rotation) < 1e-10
            and np.linalg.norm(refined.translation - estimate.translation) < 1e-10
        )
        if converged:
            estimate = refined
            break
        estimate = refined
        best_mask = (
            np.linalg.norm(estimate.apply(source_points) - target_points, axis=1) <= threshold
        )
        if np.count_nonzero(best_mask) < 3:
            raise ValueError("robust refinement lost its consensus")
    residuals = np.linalg.norm(estimate.apply(source_points) - target_points, axis=1)
    inliers = (residuals <= threshold) & (point_weights > 0)
    report = {
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "threshold": float(threshold),
        "median_residual": float(np.median(residuals[inliers])),
        "p90_residual": float(np.percentile(residuals[inliers], 90)),
        "residuals": residuals.tolist(),
    }
    return estimate, report


def optimize_section_graph(section_count: int, edges: list[SectionEdge]) -> tuple[list[Sim3], dict]:
    """Optimize section-to-global poses using weighted relative similarity constraints."""
    if section_count < 1:
        raise ValueError("section_count must be positive")
    adjacency: list[list[tuple[int, Sim3]]] = [[] for _ in range(section_count)]
    for edge in edges:
        if not 0 <= edge.source < section_count or not 0 <= edge.target < section_count:
            raise ValueError("edge endpoint is outside the section graph")
        if (
            edge.source == edge.target
            or not np.isfinite(edge.weight)
            or edge.weight <= 0
            or not np.isfinite(edge.translation_scale)
            or edge.translation_scale <= 0
        ):
            raise ValueError("edges require distinct endpoints and finite positive weights")
        adjacency[edge.source].append((edge.target, edge.transform))
        adjacency[edge.target].append((edge.source, edge.transform.inverse()))

    if section_count == 1:
        if edges:
            raise ValueError("one-section graph cannot contain edges")
        return [Sim3.identity()], {
            "connected": True,
            "edge_count": 0,
            "initial_cost": 0.0,
            "final_cost": 0.0,
            "initial_poses": [Sim3.identity().as_matrix().tolist()],
            "edge_residuals_before": [],
            "edge_residuals_after": [],
            "initial_objective": 0.0,
            "final_objective": 0.0,
            "success": True,
        }

    poses: list[Sim3 | None] = [None] * section_count
    poses[0] = Sim3.identity()
    queue = [0]
    for current in queue:
        for neighbor, current_to_neighbor in adjacency[current]:
            if poses[neighbor] is None:
                poses[neighbor] = poses[current].compose(current_to_neighbor.inverse())
                queue.append(neighbor)
    if any(pose is None for pose in poses):
        raise ValueError("section graph is disconnected")

    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    initial_poses = [pose for pose in poses if pose is not None]

    def encode(pose: Sim3) -> NDArray[np.float64]:
        return np.r_[
            Rotation.from_matrix(pose.rotation).as_rotvec(), pose.translation, np.log(pose.scale)
        ]

    def decode(values: NDArray[np.float64]) -> list[Sim3]:
        result = [Sim3.identity()]
        for offset in range(0, len(values), 7):
            result.append(
                Sim3(
                    float(np.exp(values[offset + 6])),
                    Rotation.from_rotvec(values[offset : offset + 3]).as_matrix(),
                    values[offset + 3 : offset + 6],
                )
            )
        return result

    initial = np.concatenate([encode(pose) for pose in initial_poses[1:]])
    edge_weights = np.asarray([edge.weight for edge in edges], dtype=float)
    edge_weights /= edge_weights.mean() if len(edge_weights) else 1.0

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        global_poses = decode(values)
        parts = []
        for edge, weight in zip(edges, edge_weights, strict=True):
            predicted = global_poses[edge.target].inverse().compose(global_poses[edge.source])
            error = edge.transform.inverse().compose(predicted)
            rotation_error = Rotation.from_matrix(error.rotation).as_rotvec()
            translation_error = (
                predicted.translation - edge.transform.translation
            ) / edge.translation_scale
            parts.append(
                np.sqrt(weight) * np.r_[rotation_error, translation_error, np.log(error.scale)]
            )
        return np.concatenate(parts) if parts else np.zeros(0)

    before = residual(initial)
    result = least_squares(residual, initial, loss="soft_l1", f_scale=1.0, max_nfev=500)
    after = residual(result.x)
    final_poses = decode(result.x)

    def edge_residuals(values: NDArray[np.float64]) -> list[dict]:
        poses_at_values = decode(values)
        records = []
        for edge in edges:
            predicted = poses_at_values[edge.target].inverse().compose(poses_at_values[edge.source])
            error = edge.transform.inverse().compose(predicted)
            records.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "rotation": float(
                        np.linalg.norm(Rotation.from_matrix(error.rotation).as_rotvec())
                    ),
                    "translation": float(
                        np.linalg.norm(predicted.translation - edge.transform.translation)
                        / edge.translation_scale
                    ),
                    "log_scale": float(abs(np.log(error.scale))),
                }
            )
        return records

    def robust_objective(values: NDArray[np.float64]) -> float:
        normalized = residual(values)
        return float(np.sum(np.sqrt(1.0 + normalized**2) - 1.0))

    return final_poses, {
        "connected": True,
        "edge_count": len(edges),
        "initial_cost": float(0.5 * before @ before),
        "final_cost": float(0.5 * after @ after),
        "initial_poses": [pose.as_matrix().tolist() for pose in initial_poses],
        "edge_residuals_before": edge_residuals(initial),
        "edge_residuals_after": edge_residuals(result.x),
        "initial_objective": robust_objective(initial),
        "final_objective": robust_objective(result.x),
        "success": bool(result.success),
        "message": str(result.message),
    }
