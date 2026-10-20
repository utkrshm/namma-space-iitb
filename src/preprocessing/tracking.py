"""Grid-distributed KLT with forward/backward checks and explicit unknown motion."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TrackingConfig


def cell_ids(points, shape, grid):
    h, w = shape[:2]
    xy = np.asarray(points).reshape(-1, 2)
    x = np.clip((xy[:, 0] / w * grid).astype(int), 0, grid - 1)
    y = np.clip((xy[:, 1] / h * grid).astype(int), 0, grid - 1)
    return y * grid + x


def detect_corners(gray, config: TrackingConfig):
    h, w = gray.shape
    parts = []
    for y in range(config.grid):
        for x in range(config.grid):
            x0, x1 = x * w // config.grid, (x + 1) * w // config.grid
            y0, y1 = y * h // config.grid, (y + 1) * h // config.grid
            points = cv2.goodFeaturesToTrack(
                gray[y0:y1, x0:x1],
                maxCorners=config.corners_per_cell,
                qualityLevel=config.quality_level,
                minDistance=config.min_distance,
            )
            if points is not None:
                parts.append(points.reshape(-1, 2) + np.array([x0, y0], np.float32))
    return np.concatenate(parts).astype(np.float32) if parts else np.empty((0, 2), np.float32)


@dataclass(frozen=True)
class TrackMeasurement:
    overlap: float | None
    initial_coverage: float
    current_coverage: float
    survival: float
    displacement_px: float | None
    reliable: bool
    tracks: int


class KLTTracker:
    def __init__(self, gray, config: TrackingConfig):
        self.config = config
        self.previous = gray
        self.original = detect_corners(gray, config)
        self.current = self.original.copy()
        self.ids = np.arange(len(self.original))
        self.cells = cell_ids(self.original, gray.shape, config.grid)
        self.initial_cells = len(np.unique(self.cells))
        self.initial_coverage = self.initial_cells / config.grid**2

    def step(self, gray) -> TrackMeasurement:
        cfg = self.config
        if gray.shape != self.previous.shape:
            self.current = np.empty((0, 2), np.float32)
            self.ids = self.ids[:0]
        if len(self.current):
            kwargs = {
                "winSize": (cfg.window, cfg.window),
                "maxLevel": cfg.pyramid_levels,
                "criteria": (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    cfg.iterations,
                    cfg.epsilon,
                ),
            }
            p1, s1, _ = cv2.calcOpticalFlowPyrLK(
                self.previous, gray, self.current[:, None], None, **kwargs
            )
            if p1 is not None:
                p0, s0, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, p1, None, **kwargs)
                if p0 is None:
                    valid = np.zeros(len(p1), bool)
                else:
                    xy = p1.reshape(-1, 2)
                    valid = (s1.ravel() > 0) & (s0.ravel() > 0)
                    valid &= np.linalg.norm(p0.reshape(-1, 2) - self.current, axis=1) < cfg.fb_error
                    valid &= np.isfinite(xy).all(axis=1)
                    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1])
                    valid &= (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])
                self.current = p1.reshape(-1, 2)[valid]
                self.ids = self.ids[valid]
            else:
                self.current = np.empty((0, 2), np.float32)
                self.ids = self.ids[:0]
        self.previous = gray
        count = len(self.current)
        survival = count / len(self.original) if len(self.original) else 0.0
        overlap = (
            len(np.unique(self.cells[self.ids])) / self.initial_cells
            if self.initial_cells
            else None
        )
        coverage = len(np.unique(cell_ids(self.current, gray.shape, cfg.grid))) / cfg.grid**2
        reliable = (
            count >= cfg.min_tracks
            and self.initial_coverage >= cfg.min_coverage
            and coverage >= cfg.min_coverage
            and survival >= cfg.reliable_survival
        )
        displacement = (
            float(np.median(np.linalg.norm(self.current - self.original[self.ids], axis=1)))
            if reliable
            else None
        )
        return TrackMeasurement(
            overlap, self.initial_coverage, coverage, survival, displacement, reliable, count
        )
