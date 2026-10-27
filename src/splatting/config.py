"""Validated TOML configuration; distances remain in the input scene's units."""

import math
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    max_image_side: int = 1600
    cache_frames: int = 4
    holdout_every: int = 8
    confidence_min: float = 0.0
    depth_min: float = 0.001
    depth_max: float = 1e6
    reprojection_px: float = 1.5
    depth_relative_tolerance: float = 0.05
    consistency_views: int = 4
    min_consistent_views: int = 1
    max_geometry_rejection: float = 0.5


@dataclass(frozen=True)
class InitializationConfig:
    points_per_frame: int = 20000
    max_points: int = 300000
    candidate_multiplier: int = 3
    voxel_fraction: float = 0.001
    neighbor_distance_factor: float = 10.0
    scale_multiplier: float = 0.7
    min_scale_fraction: float = 0.00001
    max_scale_fraction: float = 0.02
    opacity: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 30000
    seed: int = 42
    sh_degree: int = 3
    sh_interval: int = 1000
    ssim_weight: float = 0.2
    depth_weight: float = 0.2
    depth_weight_final: float = 0.02
    scale_regularization: float = 0.001
    means_lr: float = 0.00016
    means_lr_final: float = 0.0000016
    scales_lr: float = 0.005
    quats_lr: float = 0.001
    opacities_lr: float = 0.05
    sh0_lr: float = 0.0025
    shn_lr: float = 0.000125
    checkpoint_every: int = 2000
    evaluate_every: int = 2000
    log_every: int = 100


@dataclass(frozen=True)
class RefinementConfig:
    start_step: int = 500
    stop_step: int = 15000
    every: int = 100
    reset_every: int = 3000
    max_gaussians: int = 1000000
    grow_gradient: float = 0.0008
    grow_scale_fraction: float = 0.01
    prune_opacity: float = 0.005
    prune_scale_fraction: float = 0.1
    absgrad: bool = True


@dataclass(frozen=True)
class RenderConfig:
    near_plane: float = 0.001
    far_plane: float = 1e6
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    packed: bool = True
    rasterize_mode: str = "classic"


@dataclass(frozen=True)
class SplattingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    def __post_init__(self):
        for section in (
            self.data,
            self.initialization,
            self.training,
            self.refinement,
            self.render,
        ):
            for setting in fields(section):
                value, default = getattr(section, setting.name), setting.default
                if type(default) in (int, float):
                    numeric = (
                        type(value) is int if type(default) is int else type(value) in (int, float)
                    )
                    if not numeric or not math.isfinite(value):
                        raise ValueError(
                            f"{setting.name} must be finite and have the expected numeric type"
                        )
                elif isinstance(default, bool) and type(value) is not bool:
                    raise ValueError(f"{setting.name} must be boolean")
                elif isinstance(default, str) and not isinstance(value, str):
                    raise ValueError(f"{setting.name} must be a string")
        if not isinstance(self.render.background, (list, tuple)):
            raise ValueError("background must contain three RGB numbers")
        if any(type(value) not in (float, int) for value in self.render.background):
            raise ValueError("background must contain three RGB numbers")

        d, i, t, f, r = self.data, self.initialization, self.training, self.refinement, self.render
        positive_ints = (
            d.max_image_side,
            d.cache_frames,
            i.points_per_frame,
            i.max_points,
            i.candidate_multiplier,
            t.steps,
            t.sh_interval,
            t.checkpoint_every,
            t.evaluate_every,
            t.log_every,
            f.every,
            f.reset_every,
            f.max_gaussians,
        )
        if any(type(value) is not int or value < 1 for value in positive_ints):
            raise ValueError(
                "Image/cache sizes, point budgets and step intervals must be positive integers"
            )
        if d.max_image_side < 16 or i.max_points < 4:
            raise ValueError("Require max_image_side >= 16 and max_points >= 4")
        if type(d.holdout_every) is not int or d.holdout_every not in (0,) and d.holdout_every < 2:
            raise ValueError("holdout_every must be zero (disabled) or >= 2")
        if not 0 < d.depth_min < d.depth_max or d.reprojection_px <= 0:
            raise ValueError("Invalid depth range or reprojection tolerance")
        if not 0 < d.depth_relative_tolerance < 1 or not 0 <= d.max_geometry_rejection < 1:
            raise ValueError("Invalid geometry consistency thresholds")
        if not 0 <= d.min_consistent_views <= d.consistency_views:
            raise ValueError("Require 0 <= min_consistent_views <= consistency_views")
        if type(d.consistency_views) is not int or type(d.min_consistent_views) is not int:
            raise ValueError("Consistency view counts must be integers")
        if not 0 < i.min_scale_fraction <= i.max_scale_fraction or i.scale_multiplier <= 0:
            raise ValueError("Invalid initial scale limits")
        if i.voxel_fraction <= 0 or i.neighbor_distance_factor <= 1 or not 0 < i.opacity < 1:
            raise ValueError("Invalid voxel, neighbor or opacity settings")
        if type(t.sh_degree) is not int or not 0 <= t.sh_degree <= 3:
            raise ValueError("sh_degree must be an integer in 0..3")
        if type(t.seed) is not int or t.seed < 0 or not 0 <= t.ssim_weight <= 1:
            raise ValueError("Invalid seed or SSIM weight")
        weights = (t.depth_weight, t.depth_weight_final, t.scale_regularization)
        rates = (
            t.means_lr,
            t.means_lr_final,
            t.scales_lr,
            t.quats_lr,
            t.opacities_lr,
            t.sh0_lr,
            t.shn_lr,
        )
        if any(value < 0 for value in weights) or any(value <= 0 for value in rates):
            raise ValueError("Loss weights must be nonnegative and learning rates positive")
        if not 0 <= f.start_step < f.stop_step or f.max_gaussians < i.max_points:
            raise ValueError("Invalid refinement schedule or Gaussian budget")
        if any(type(value) is not int for value in (f.start_step, f.stop_step)):
            raise ValueError("Refinement steps must be integers")
        if not 0 < f.prune_opacity < i.opacity or f.grow_gradient <= 0:
            raise ValueError("Invalid refinement opacity or gradient thresholds")
        if not 0 < f.grow_scale_fraction < f.prune_scale_fraction:
            raise ValueError("Invalid refinement scale thresholds")
        if not 0 < r.near_plane < r.far_plane or r.rasterize_mode not in ("classic", "antialiased"):
            raise ValueError("Invalid clipping planes or rasterize mode")
        if len(r.background) != 3 or any(not 0 <= v <= 1 for v in r.background):
            raise ValueError("background must contain three finite values in [0, 1]")
        if any(type(v) is not bool for v in (r.packed, f.absgrad)):
            raise ValueError("packed and absgrad must be booleans")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SplattingConfig":
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a dictionary of section tables")
        types = {
            "data": DataConfig,
            "initialization": InitializationConfig,
            "training": TrainingConfig,
            "refinement": RefinementConfig,
            "render": RenderConfig,
        }
        if unknown := set(data) - types.keys():
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        sections = {}
        for name, values in data.items():
            if not isinstance(values, dict):
                raise ValueError(f"Config section {name} must be a table")
            if unknown := set(values) - {f.name for f in fields(types[name])}:
                raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
            sections[name] = types[name](**values)
        return cls(**sections)

    @classmethod
    def from_toml(cls, path: str | Path) -> "SplattingConfig":
        return cls.from_dict(tomllib.loads(Path(path).read_text()))
