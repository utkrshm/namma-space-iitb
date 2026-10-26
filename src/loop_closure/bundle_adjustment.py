"""Method B: robust bundle adjustment of per-frame poses and depth scales.

Unknowns: for every frame except frame 0 (gauge), a world-frame rotation increment
(rotation vector), a camera-centre increment and a log depth-scale increment; and one 3D
point per feature track. Residuals, each divided by its sigma:
- reprojection of the track point in every observing frame (pixels on the model grid);
- log camera-z of the point versus log(depth_scale * VGGT depth) at that pixel, which fixes
  scale and keeps every frame's dense depth usable after refinement;
- weak priors that keep depth scales and rotations near their initial values.
Intrinsics stay at the owner-section VGGT estimates. A Huber loss limits outliers, and
observations with large reprojection error are removed between rounds.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from .contracts import FrameMatches, FrameModel, LoopConfig
from .manhattan import ManhattanPrior


def build_tracks(
    matches: list[FrameMatches], max_length: int
) -> list[list[tuple[int, np.ndarray]]]:
    """Union matched keypoints into tracks; drop tracks seen twice in one frame."""
    parent: dict[tuple, tuple] = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    coordinates: dict[tuple, np.ndarray] = {}
    for item in matches:
        for xa, xb in zip(item.xy_a, item.xy_b):
            ka = (item.frame_a, round(float(xa[0]) * 4), round(float(xa[1]) * 4))
            kb = (item.frame_b, round(float(xb[0]) * 4), round(float(xb[1]) * 4))
            coordinates[ka], coordinates[kb] = xa, xb
            ra, rb = find(ka), find(kb)
            if ra != rb:
                parent[ra] = rb
    groups: dict[tuple, list[tuple]] = {}
    for key in coordinates:
        groups.setdefault(find(key), []).append(key)
    tracks = []
    for keys in groups.values():
        frames = [key[0] for key in keys]
        if len(keys) < 2 or len(set(frames)) != len(frames) or len(keys) > max_length:
            continue
        tracks.append([(key[0], coordinates[key]) for key in sorted(keys)])
    return tracks


def _depth_at(model: FrameModel, xy: np.ndarray) -> float:
    x = int(np.clip(np.rint(xy[0]), 0, model.depth.shape[1] - 1))
    y = int(np.clip(np.rint(xy[1]), 0, model.depth.shape[0] - 1))
    return float(model.depth[y, x]) if model.valid[y, x] else np.nan


class BundleProblem:
    """Vectorized residuals and Jacobian sparsity for the bundle adjustment."""

    def __init__(
        self,
        models: list[FrameModel],
        tracks,
        config: LoopConfig,
        manhattan: ManhattanPrior | None = None,
    ):
        self.models, self.config, self.manhattan = models, config, manhattan
        frames, points, pixels, depths, initial_points = [], [], [], [], []
        for track in tracks:
            estimates = [models[f].unproject(xy[None])[0] for f, xy in track]
            estimates = [p for p in estimates if np.all(np.isfinite(p))]
            if not estimates:
                continue
            point_index = len(initial_points)
            initial_points.append(np.median(np.stack(estimates), axis=0))
            for frame, xy in track:
                frames.append(frame)
                points.append(point_index)
                pixels.append(xy)
                depths.append(_depth_at(models[frame], xy))
        self.frame = np.asarray(frames, dtype=np.int64)
        self.point = np.asarray(points, dtype=np.int64)
        self.pixel = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        self.depth = np.asarray(depths, dtype=np.float64)
        self.has_depth = np.isfinite(self.depth) & (self.depth > 0)
        self.points0 = np.asarray(initial_points, dtype=np.float64).reshape(-1, 3)
        self.R0 = np.stack([m.camera_to_world[:3, :3] for m in models])
        self.C0 = np.stack([m.camera_to_world[:3, 3] for m in models])
        self.K = np.stack([m.intrinsics for m in models])
        self.s0 = np.asarray([m.depth_scale for m in models], dtype=np.float64)
        self.frame_count = len(models)
        self.active = np.ones(len(self.frame), dtype=bool)

    @property
    def point_count(self) -> int:
        return len(self.points0)

    def x0(self) -> np.ndarray:
        return np.r_[np.zeros(7 * (self.frame_count - 1)), self.points0.ravel()]

    def unpack(self, x: np.ndarray):
        frame_params = np.r_[np.zeros(7), x[: 7 * (self.frame_count - 1)]].reshape(-1, 7)
        rotations = Rotation.from_rotvec(frame_params[:, :3]).as_matrix() @ self.R0
        centers = self.C0 + frame_params[:, 3:6]
        scales = self.s0 * np.exp(frame_params[:, 6])
        points = x[7 * (self.frame_count - 1) :].reshape(-1, 3)
        return frame_params, rotations, centers, scales, points

    def project(self, x: np.ndarray):
        _, rotations, centers, scales, points = self.unpack(x)
        f, p = self.frame[self.active], self.point[self.active]
        camera = np.einsum("nji,nj->ni", rotations[f], points[p] - centers[f])
        z = np.maximum(camera[:, 2], 1e-6)
        k = self.K[f]
        u = k[:, 0, 0] * camera[:, 0] / z + k[:, 0, 2]
        v = k[:, 1, 1] * camera[:, 1] / z + k[:, 1, 2]
        return np.column_stack((u, v)), z, scales[f]

    def residuals(self, x: np.ndarray) -> np.ndarray:
        config = self.config
        frame_params = self.unpack(x)[0]
        projected, z, scales = self.project(x)
        reprojection = (projected - self.pixel[self.active]) / config.reprojection_sigma_px
        depth = self.depth[self.active]
        valid = self.has_depth[self.active]
        depth_residual = np.zeros(len(z))
        depth_residual[valid] = (
            np.log(z[valid]) - np.log(scales[valid] * depth[valid])
        ) / config.depth_sigma_log
        priors = np.column_stack(
            (
                frame_params[1:, :3] / config.rotation_prior_sigma_rad,
                frame_params[1:, 6:7] / config.scale_prior_sigma_log,
            )
        )
        parts = [reprojection.ravel(), depth_residual, priors.ravel()]
        if self.manhattan is not None:
            parts.append(self.manhattan.residuals(self.unpack(x)[1]))
        return np.concatenate(parts)

    def sparsity(self) -> coo_matrix:
        """Each observation row depends on one frame's 7 parameters and one point's 3."""
        f, p = self.frame[self.active], self.point[self.active]
        observations = len(f)
        manhattan_rows = self.frame_count - 1 if self.manhattan is not None else 0
        shape = (
            3 * observations + 4 * (self.frame_count - 1) + manhattan_rows,
            7 * (self.frame_count - 1) + 3 * self.point_count,
        )
        point_offset = 7 * (self.frame_count - 1)
        index = np.arange(observations)
        rows, columns = [], []
        for row_index in (2 * index, 2 * index + 1, 2 * observations + index):
            for j in range(3):
                rows.append(row_index)
                columns.append(point_offset + 3 * p + j)
            moving = f > 0
            for j in range(7):
                rows.append(row_index[moving])
                columns.append(7 * (f[moving] - 1) + j)

        prior_frames = np.arange(1, self.frame_count)
        for j, column in enumerate((0, 1, 2, 6)):
            rows.append(3 * observations + 4 * (prior_frames - 1) + j)
            columns.append(7 * (prior_frames - 1) + column)
        if manhattan_rows:
            # Each frame's square-walls residual depends only on its own rotation.
            for j in range(3):
                rows.append(3 * observations + 4 * (self.frame_count - 1) + prior_frames - 1)
                columns.append(7 * (prior_frames - 1) + j)
        rows, columns = np.concatenate(rows), np.concatenate(columns)
        return coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=shape)

    def parameter_scale(self) -> np.ndarray:
        """Typical step sizes: 0.01 rad, 1% of scene depth, 1% log scale.

        Explicit scales matter: scaling by Jacobian column norms shrinks camera steps,
        because a camera appears in thousands of observations, and the solver then
        stops on `xtol` after moving only the points.
        """
        depth = float(
            np.median(np.linalg.norm(self.points0[self.point] - self.C0[self.frame], axis=1))
        )
        frame = np.r_[np.full(3, 0.01), np.full(3, 0.01 * depth), 0.01]
        return np.r_[
            np.tile(frame, self.frame_count - 1), np.full(3 * self.point_count, 0.01 * depth)
        ]

    def camera_depths(self, x: np.ndarray) -> np.ndarray:
        _, rotations, centers, _, points = self.unpack(x)
        camera = np.einsum(
            "nji,nj->ni", rotations[self.frame], points[self.point] - centers[self.frame]
        )
        return camera[:, 2]

    def reprojection_errors(self, x: np.ndarray) -> np.ndarray:
        saved = self.active
        self.active = np.ones(len(self.frame), dtype=bool)
        projected, _, _ = self.project(x)
        self.active = saved
        return np.linalg.norm(projected - self.pixel, axis=1)


def bundle_adjust(
    models: list[FrameModel],
    matches: list[FrameMatches],
    config: LoopConfig,
    *,
    manhattan: bool = False,
) -> tuple[dict, dict]:
    """Return refined cameras (`cameras.npz` layout) and a report.

    With ``manhattan=True`` a per-frame square-walls yaw prior is added (see `manhattan.py`).
    """
    tracks = build_tracks(matches, config.max_track_length)
    prior = (
        ManhattanPrior(models, config.manhattan_sigma_deg, config.manhattan_min_concentration)
        if manhattan
        else None
    )
    problem = BundleProblem(models, tracks, config, prior)
    manhattan_before = prior.summary(problem.R0) if prior is not None else None
    if problem.point_count < 50:
        raise ValueError("too few feature tracks for bundle adjustment")
    x = problem.x0()
    before = problem.reprojection_errors(x)

    # Track points start from VGGT depth, so a few observations begin far off or behind a
    # camera; they would dominate the first trust-region steps. Exclude them up front.
    problem.active = (before <= config.initial_gate_px) & (problem.camera_depths(x) > 0)
    rounds = []
    for round_index, threshold in enumerate((None, 8.0, 4.0)):
        if threshold is not None:
            problem.active = problem.reprojection_errors(x) <= threshold
        result = least_squares(
            problem.residuals,
            x,
            jac_sparsity=problem.sparsity(),
            method="trf",
            loss="huber",
            f_scale=config.huber_sigma,
            x_scale=problem.parameter_scale(),
            tr_solver="lsmr",
            max_nfev=config.max_ba_iterations,
            verbose=0,
        )
        x = result.x
        errors = problem.reprojection_errors(x)[problem.active]
        rounds.append(
            {
                "round": round_index,
                "outlier_threshold_px": threshold,
                "active_observations": int(problem.active.sum()),
                "cost": float(result.cost),
                "nfev": int(result.nfev),
                "status": int(result.status),
                "message": str(result.message),
                "reprojection_median_px": float(np.median(errors)),
                "reprojection_p90_px": float(np.percentile(errors, 90)),
            }
        )
        print(
            f"BA round {round_index}: active={problem.active.sum()} "
            f"reproj median/p90={np.median(errors):.2f}/{np.percentile(errors, 90):.2f}px "
            f"({result.message})",
            flush=True,
        )
    final_active = problem.active.copy()
    initial_on_final = before[final_active]
    frame_params, rotations, centers, scales, _ = problem.unpack(x)
    camera_to_world = np.tile(np.eye(4), (problem.frame_count, 1, 1))
    camera_to_world[:, :3, :3] = rotations
    camera_to_world[:, :3, 3] = centers
    after = problem.reprojection_errors(x)
    span = float(np.median(np.linalg.norm(problem.C0 - np.median(problem.C0, axis=0), axis=1)))
    report = {
        "manhattan_prior": None
        if prior is None
        else {"before": manhattan_before, "after": prior.summary(rotations)},
        "frames": problem.frame_count,
        "tracks": problem.point_count,
        "observations": len(problem.frame),
        "observations_with_depth": int(problem.has_depth.sum()),
        "matched_pairs": len(matches),
        "reprojection_before_px": {
            "median": float(np.median(before)),
            "p90": float(np.percentile(before, 90)),
        },
        "reprojection_after_px": {
            "median": float(np.median(after)),
            "p90": float(np.percentile(after, 90)),
        },
        "final_inliers": int(final_active.sum()),
        "final_inlier_reprojection_px": {
            "before_median": float(np.median(initial_on_final)),
            "before_p90": float(np.percentile(initial_on_final, 90)),
            "after_median": float(np.median(after[final_active])),
            "after_p90": float(np.percentile(after[final_active], 90)),
        },
        "rounds": rounds,
        "rotation_change_deg": {
            "median": float(np.degrees(np.median(np.linalg.norm(frame_params[:, :3], axis=1)))),
            "max": float(np.degrees(np.max(np.linalg.norm(frame_params[:, :3], axis=1)))),
        },
        "center_change_over_trajectory_spread": {
            "median": float(np.median(np.linalg.norm(frame_params[:, 3:6], axis=1)) / span),
            "max": float(np.max(np.linalg.norm(frame_params[:, 3:6], axis=1)) / span),
        },
        "log_depth_scale_change": {
            "median_abs": float(np.median(np.abs(frame_params[:, 6]))),
            "max_abs": float(np.max(np.abs(frame_params[:, 6]))),
        },
    }
    cameras = {
        "frame_ids": np.asarray([m.frame_id for m in models]),
        "owners": np.asarray([m.owner_section for m in models], dtype=np.int32),
        "camera_to_world": camera_to_world,
        "depth_scale": scales,
        "intrinsics": problem.K,
    }
    return cameras, report
