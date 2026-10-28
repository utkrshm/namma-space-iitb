"""Configuration for Gaussian splatting from a dense SfM (COLMAP) model."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class SfmDataConfig:
    # Training images are downscaled by this integer factor (1 = full resolution).
    downscale: int = 1
    # Held-out views: every Nth geometry frame (the rule `src.splatting.data` uses on the
    # 161 VGGT geometry frames), so both trainers are scored on the same frames.
    holdout_geometry_every: int = 8
    # Also drop training frames this close in time to a held-out frame (0 keeps them).
    holdout_exclusion_s: float = 0.0
    # Optional [start, end] source-time window in seconds; empty trains on every frame.
    time_range_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class SfmTrainingConfig:
    steps: int = 30_000
    seed: int = 42
    sh_degree: int = 3
    sh_interval: int = 1000
    ssim_weight: float = 0.2
    # Coarse-to-fine: (step, downscale) pairs applied on top of data.downscale.
    resolution_schedule: tuple[tuple[int, int], ...] = ((0, 4), (2000, 2), (6000, 1))
    means_lr: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shn_lr: float = 1.25e-4
    # MCMC regularizers (Kheradmand et al. 2024) keep opacity/scale from growing unchecked.
    opacity_regularization: float = 0.01
    scale_regularization: float = 0.01
    # Per-image camera refinement (rotation + translation increments).
    pose_refinement: bool = True
    pose_lr: float = 1e-5
    pose_regularization: float = 1e-6
    pose_start_step: int = 1000
    # Per-image affine colour (exposure / white balance) compensation.
    appearance: bool = True
    appearance_lr: float = 1e-3
    appearance_regularization: float = 1e-3
    log_every: int = 200
    evaluate_every: int = 10_000
    checkpoint_every: int = 5000


@dataclass(frozen=True)
class SfmRefinementConfig:
    max_gaussians: int = 2_500_000
    initial_opacity: float = 0.5
    initial_scale: float = 0.5
    noise_lr: float = 5e5
    start_step: int = 500
    stop_step: int = 25_000
    every: int = 100
    min_opacity: float = 0.005
    # Largest Gaussian extent as a fraction of the scene extent (clamped).
    max_scale_fraction: float = 0.1
    # Gaussians whose screen radius exceeds this fraction of the image side are relocated
    # on refinement steps (3DGS screen-size pruning); 0 disables.
    max_screen_fraction: float = 0.15


@dataclass(frozen=True)
class SfmSplatConfig:
    data: SfmDataConfig = field(default_factory=SfmDataConfig)
    training: SfmTrainingConfig = field(default_factory=SfmTrainingConfig)
    refinement: SfmRefinementConfig = field(default_factory=SfmRefinementConfig)
    rasterize_mode: str = "classic"

    @classmethod
    def from_toml(cls, path: str | Path) -> SfmSplatConfig:
        values = tomllib.loads(Path(path).read_text())
        sections = {f.name: f for f in fields(cls)}
        unknown = set(values) - set(sections)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        built = {}
        for name, value in values.items():
            if not isinstance(value, dict):
                built[name] = value
                continue
            section_type = {
                "data": SfmDataConfig,
                "training": SfmTrainingConfig,
                "refinement": SfmRefinementConfig,
            }[name]
            allowed = {f.name for f in fields(section_type)}
            if set(value) - allowed:
                raise ValueError(f"Unknown settings in [{name}]: {sorted(set(value) - allowed)}")
            if "time_range_s" in value:
                value = dict(value)
                value["time_range_s"] = tuple(value["time_range_s"])
            if "resolution_schedule" in value:
                value = dict(value)
                value["resolution_schedule"] = tuple(map(tuple, value["resolution_schedule"]))
            built[name] = section_type(**value)
        return cls(**built)

    def to_dict(self) -> dict:
        return asdict(self)
