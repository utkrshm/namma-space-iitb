"""Typed configuration loaded from TOML; unknown settings fail early."""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DecodeConfig:
    long_side: int = 640
    threads: int = 2


@dataclass(frozen=True)
class ScoreConfig:
    tiles: int = 4
    short_window_s: float = 1.0
    long_window_s: float = 10.0
    reference_percentile: float = 75.0
    relative_blur: float = 0.3
    edge_gradient: float = 10.0
    min_edge_fraction: float = 0.05
    reblur_sigma: float = 1.0
    reblur_loss_blur: float = 0.2
    reblur_loss_context: float = 0.35
    dark_level: int = 16
    blown_level: int = 245
    exposure_fraction: float = 0.65
    low_contrast: float = 8.0
    exposure_jump: float = 30.0
    histogram_cut: float = 0.1
    timing_gap_s: float = 0.066
    high_motion_px_per_s: float = 400.0
    interval_min_s: float = 0.5


@dataclass(frozen=True)
class TrackingConfig:
    grid: int = 4
    corners_per_cell: int = 24
    quality_level: float = 0.01
    global_quality_floor: float = 0.01
    absolute_eigen_floor: float = 0.00001
    min_distance: float = 6.0
    window: int = 21
    pyramid_levels: int = 3
    iterations: int = 20
    epsilon: float = 0.03
    fb_error: float = 1.0
    min_tracks: int = 8
    min_coverage: float = 0.1875


@dataclass(frozen=True)
class SelectionConfig:
    geometry_overlap: float = 0.65
    appearance_overlap: float = 0.85
    max_interval_s: float = 2.0
    min_interval_s: float = 0.3
    buffer_s: float = 0.3
    minimum_edge_overlap: float = 0.25
    appearance: bool = True


@dataclass(frozen=True)
class RevisitConfig:
    enabled: bool = True
    model: str = "facebook/dinov2-base"
    device: str = "auto"
    temporal_exclusion_s: float = 10.0
    top_k: int = 5
    min_similarity: float = 0.75
    image_long_side: int = 1024
    keypoints: int = 1024
    min_inliers: int = 50
    min_inlier_ratio: float = 0.3
    min_coverage: float = 0.25
    min_hull_area: float = 0.05
    ransac_px: float = 1.5
    confidence: float = 0.999
    max_iterations: int = 10000
    feature_cache_size: int = 12


@dataclass(frozen=True)
class ExportConfig:
    jpeg_quality: int = 95
    png_compression: int = 3
    contact_columns: int = 6
    contact_page_size: int = 48
    thumbnail_long_side: int = 220


@dataclass(frozen=True)
class PreprocessConfig:
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    scoring: ScoreConfig = field(default_factory=ScoreConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    revisits: RevisitConfig = field(default_factory=RevisitConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self):
        if self.decode.long_side < 32 or self.decode.threads < 1:
            raise ValueError("Invalid analysis size or thread count")
        s, t, r = self.selection, self.tracking, self.revisits
        if not 0 < s.geometry_overlap <= s.appearance_overlap <= 1:
            raise ValueError("Require 0 < geometry_overlap <= appearance_overlap <= 1")
        if not 0 < s.min_interval_s <= s.max_interval_s or s.buffer_s < 0:
            raise ValueError("Invalid selection intervals")
        if t.grid < 1 or t.corners_per_cell < 1 or t.window % 2 != 1:
            raise ValueError("Invalid tracking grid or LK window")
        if r.top_k < 1 or r.temporal_exclusion_s < 0 or r.feature_cache_size < 1:
            raise ValueError("Invalid revisit configuration")
        if not 1 <= self.export.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be 1..100")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_toml(cls, path: str | Path):
        data = tomllib.loads(Path(path).read_text())
        types = {
            "decode": DecodeConfig,
            "scoring": ScoreConfig,
            "tracking": TrackingConfig,
            "selection": SelectionConfig,
            "revisits": RevisitConfig,
            "export": ExportConfig,
        }
        unknown = set(data) - set(types)
        if unknown:
            raise ValueError(f"Unknown config sections: {unknown}")
        return cls(**{key: types[key](**value) for key, value in data.items()})
