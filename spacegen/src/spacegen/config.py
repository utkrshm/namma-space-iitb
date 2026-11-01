"""Settings for every stage, loadable from one TOML file (``configs/default.toml``).

Every value has a default that works on an unseen indoor video; a TOML file overrides
only what it names. Unknown sections or keys are errors, so a typo never silently
falls back to a default.
"""

from __future__ import annotations

import tomllib
import types
import typing
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class IngestConfig:
    analysis_long_side: int = 480
    max_long_side: int = 1920
    jpeg_quality: int = 95
    # Accumulated median flow (fraction of the image diagonal) between chosen frames.
    motion_step: float = 0.035
    max_gap_s: float = 0.75
    max_frames: int = 1200
    min_relative_sharpness: float = 0.6


@dataclass(frozen=True)
class SfmConfig:
    feature_long_side: int = 1600
    max_keypoints: int = 4096
    detection_threshold: float = 0.005
    sequential_window: int = 8
    descriptor_model: str = "facebook/dinov2-base"
    retrieval_top_k: int = 10
    retrieval_min_similarity: float = 0.4
    # Retrieval partners must be at least this many frames apart (video only).
    retrieval_min_gap: int = 12
    # Retrieval pairs need clear non-planar epipolar support: repeated grilles, tiles and
    # doors otherwise create false loops that fold the trajectory (Coorg).
    retrieval_min_inliers: int = 60
    retrieval_max_planar_fraction: float = 0.7
    ransac_px: float = 2.0
    min_inliers: int = 25
    # Replace COLMAP's focal guess (1.2 × long side) with one estimated from the verified
    # fundamental matrices, then verify again. Phone video is often far wider than the
    # guess (0.55 × long side), and the mappers decompose relative poses with it.
    estimate_focal: bool = True
    # Video: solve with sequential pairs first and keep only loops whose relative rotation
    # agrees with it.
    loop_check: bool = True
    loop_max_rotation_deg: float = 10.0
    camera_model: str = "OPENCV"
    mapper: str = "global"  # "global" (GLOMAP) or "incremental"
    # Retry with the other mapper, then a fisheye lens model, when fewer than this share of
    # frames register or the mean reprojection error is above max_reprojection_px.
    min_registered_fraction: float = 0.9
    max_reprojection_px: float = 1.6
    fisheye_fallback: bool = True


@dataclass(frozen=True)
class DepthConfig:
    enabled: bool = True
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    downscale: int = 4
    # Metric depth (metres) for surfaces nearer than every SfM point of an image, where the
    # fitted relative depth only extrapolates (`spacegen.depth.near_inverse_depth`). Empty:
    # off. Outdoor captures may prefer the Metric-Outdoor variant.
    metric_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"


@dataclass(frozen=True)
class MvsConfig:
    """Plane-sweep multi-view stereo on the depth-map grid (`spacegen.mvs`)."""

    enabled: bool = True
    sources: int = 6
    min_angle_deg: float = 3.0
    max_angle_deg: float = 40.0
    planes: int = 128
    window: int = 7
    # Keep a depth only where NCC ≥ 1 − max_cost and the patch's grey std ≥ min_texture.
    max_cost: float = 0.4
    min_texture: float = 0.02
    # Geometric check against the source depth maps.
    consistent_views: int = 2
    reprojection_px: float = 1.0
    relative_depth: float = 0.01


@dataclass(frozen=True)
class AuditConfig:
    """Self-audit from many viewpoints (`spacegen.audit`) and re-slicing (`spacegen.reslice`)."""

    enabled: bool = True
    long_side: int = 480
    # Probe views around this many training cameras: turned and stepped off the path.
    probe_cameras: int = 60
    probe_yaw_deg: float = 30.0
    probe_pitch_deg: float = 15.0
    probe_shift: float = 0.25  # camera heights
    # A segment between consecutive training cameras is weak past any of these.
    max_unmeasured: float = 0.10
    max_unrendered: float = 0.02
    max_turn_deg: float = 12.0
    max_psnr_drop: float = 2.0  # dB below the median held-out view
    # At most this share of segments is flagged (the worst first).
    max_weak_share: float = 0.25
    sheet_views: int = 8
    # Re-slicing (videos only): a short preview model is audited, then weak segments get
    # frames at motion_step / density, kept at least half a step from held-out frames.
    reslice: bool = True
    preview_steps: int = 5000
    density: int = 3
    max_extra_share: float = 0.5  # extra frames, as a share of the registered ones
    min_inliers: int = 60
    min_inlier_ratio: float = 0.35


@dataclass(frozen=True)
class SplatConfig:
    # Data.
    max_long_side: int = 1920
    # Held-out views: image i with i % holdout_every == 0 (the 3DGS/LLFF rule); 0 = none.
    holdout_every: int = 8
    # Training.
    steps: int = 30_000
    seed: int = 42
    sh_degree: int = 3
    sh_interval: int = 1000
    ssim_weight: float = 0.2
    resolution_schedule: tuple[tuple[int, int], ...] = ((0, 4), (1500, 2), (4000, 1))
    means_lr: float = 1.6e-4
    means_lr_final: float = 1.6e-6
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shn_lr: float = 1.25e-4
    opacity_regularization: float = 0.01
    scale_regularization: float = 0.01
    pose_refinement: bool = True
    pose_lr: float = 1e-5
    pose_start_step: int = 1000
    appearance: bool = True
    appearance_lr: float = 1e-3
    appearance_regularization: float = 1e-3
    # Depth prior: L1 between rendered and fitted metric inverse depth (relative to the
    # image's median), or a Pearson correlation for images without a fit. The weight
    # decays exponentially from the first value to the second over the whole run; it
    # never reaches zero, because MCMC keeps moving Gaussians until refine_stop and blank
    # walls lose their shape again without it (vyas-room).
    depth_prior_weight: float = 0.3
    depth_prior_weight_final: float = 0.03
    # Normal prior: 1 − cos between normals of the rendered and of the prior depth,
    # weighted by exp(−(image gradient / textureless_gradient)²), i.e. mostly on blank
    # walls, ceilings and floors, where the photometric loss cannot place a surface.
    # Decays with the depth prior.
    normal_prior_weight: float = 0.1
    textureless_gradient: float = 0.05
    # Pixels at or above this level in a channel are clipped by the sensor (blown-out
    # windows); there the render is only penalized for being darker. Above 1 disables.
    saturation_level: float = 0.98
    # Extra seed points from monocular depth fitted to the SfM points (0 disables). Off by
    # default: on playroom, per-image fits put seeds in free space, where they grew into
    # blobs in front of other cameras (held-out 11.9 dB against 21.9 dB at 6k steps).
    depth_seed_points: int = 0
    # Extra seed points back-projected from multi-view stereo depths (`spacegen.mvs`).
    # These passed a multi-view check, so unlike monocular seeds they lie on surfaces.
    mvs_seed_points: int = 200_000
    # Seeds from fitted monocular depth confirmed by neighbouring frames, only where MVS
    # measured nothing (`spacegen.mvs.gap_seeds`): near, blank surfaces such as door leaves.
    gap_seed_points: int = 100_000
    # Needle penalty: mean of max(0, log(largest axis / second largest) − log(needle_ratio)).
    # Stretched one-axis splats streak in off-path views; flat discs are unaffected.
    needle_weight: float = 0.01
    needle_ratio: float = 4.0
    # Planar reflections (`spacegen.splat.reflect`): mirrors and glass (and optionally the
    # floor) reflect a mirrored copy of the scene. ``reflect_lr`` is the learning rate of the
    # per-Gaussian strength logits and the per-plane Fresnel f0.
    reflections: bool = True
    # Look for windows, glass doors and mirrors (`spacegen.reflectors`).
    glass_reflectors: bool = True
    # The floor as a reflector too. Off: on vyas-room it cost 1.2 dB held-out PSNR (shoes
    # and chair legs on the floor smeared), and its reflection pass runs on almost every
    # step (a render took 0.15 s instead of 0.063 s). Mirrors and windows alone lost nothing.
    floor_reflector: bool = False
    reflect_lr: float = 0.05
    random_background: bool = False
    rasterize_mode: str = "antialiased"
    # Refinement (MCMC, Kheradmand et al. 2024).
    max_gaussians: int = 1_500_000
    initial_opacity: float = 0.5
    initial_scale: float = 0.5
    noise_lr: float = 5e5
    refine_start: int = 500
    refine_stop: int = 25_000
    refine_every: int = 100
    min_opacity: float = 0.005
    max_scale_fraction: float = 0.1
    # Gaussians wider than this share of the image are relocated, unless they sit on the
    # depth prior's surface (within screen_keep_depth, relative): a near door leaf needs
    # large splats, a floater in front of the camera does not.
    max_screen_fraction: float = 0.15
    # Share of steps that train the view with the largest recent photometric error (drawn
    # ∝ error², `spacegen.splat.train`); the rest go through the views in shuffled order.
    # A door seen in 3 of 344 frames otherwise gets 1% of the steps and never fills in.
    hard_view_share: float = 0.3
    screen_keep_depth: float = 0.1
    # Bookkeeping.
    log_every: int = 200
    evaluate_every: int = 10_000
    checkpoint_every: int = 5000


@dataclass(frozen=True)
class PartitionConfig:
    # Train in spatial blocks when one model would need more Gaussians than this.
    max_images_per_block: int = 450
    overlap: float = 0.15
    finetune_steps: int = 10_000


@dataclass(frozen=True)
class FillConfig:
    enabled: bool = True
    model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    prompt: str = "a photo of the inside of a room, realistic, detailed, natural light"
    negative_prompt: str = "blurry, distorted, text, watermark, people, cartoon"
    # Probe views per standing point: yaw steps × pitches (degrees).
    yaw_steps: int = 12
    pitches_deg: tuple[float, ...] = (-30.0, 0.0, 30.0)
    field_of_view_deg: float = 75.0
    probe_long_side: int = 768
    # A pixel needs this much support (rendered weight of Gaussians seen from a nearby
    # direction) to count as measured.
    min_support: float = 0.5
    # Pixels below this alpha count as unrendered (they show the black background).
    min_alpha: float = 0.9
    # Holes at least this share of a probe view go to SDXL; smaller ones are filled from
    # their border.
    large_hole: float = 0.01
    support_angle_deg: float = 35.0
    finetune_steps: int = 6000
    pseudo_view_weight: float = 0.5


@dataclass(frozen=True)
class TourConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    supersample: float = 1.5
    crf: int = 20
    look_seconds: float = 14.0
    # Look-arounds use the capture cameras within this many camera heights (horizontally)
    # of the standing point, and only views passing the two shares below.
    look_radius: float = 0.4
    max_unmeasured: float = 0.05
    max_unrendered: float = 0.01
    min_look_deg: float = 30.0
    walk_seconds_per_metre: float = 1.2
    # Quality checks (unrendered, unmeasured, near-floater shares) on every Nth frame.
    check_every: int = 10


@dataclass(frozen=True)
class DemoConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    supersample: float = 1.5
    crf: int = 20
    build_seconds: float = 24.0
    clips: int = 8
    clip_seconds: float = 4.5
    # Camera heights travelled per second in highlight clips (about 0.6 m/s).
    walk_speed: float = 0.4
    crossfade_seconds: float = 0.5
    reel_fov_deg: float = 70.0
    # One 360° turn per area after the reel (filled models only). Off by default: a turn
    # faces walls the capture never saw, and generated content looks worst there.
    turns: bool = False
    turn_seconds: float = 12.0


@dataclass(frozen=True)
class PipelineConfig:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    sfm: SfmConfig = field(default_factory=SfmConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    mvs: MvsConfig = field(default_factory=MvsConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    splat: SplatConfig = field(default_factory=SplatConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    fill: FillConfig = field(default_factory=FillConfig)
    tour: TourConfig = field(default_factory=TourConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)

    @classmethod
    def from_toml(cls, path: str | Path | None) -> PipelineConfig:
        if path is None:
            return cls()
        return cls.from_dict(tomllib.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, values: dict) -> PipelineConfig:
        sections = {f.name: f.type for f in fields(cls)}
        unknown = set(values) - set(sections)
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        hints = typing.get_type_hints(cls)
        return cls(**{name: _build(hints[name], value) for name, value in values.items()})

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **sections: dict) -> PipelineConfig:
        """Copy with some keys of some sections overridden, e.g. ``splat={"steps": 7000}``."""
        merged = self.to_dict()
        for name, values in sections.items():
            merged[name].update(values)
        return PipelineConfig.from_dict(merged)


def _build(section_type, values: dict):
    hints = typing.get_type_hints(section_type)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f"Unknown settings in [{section_type.__name__}]: {sorted(unknown)}")
    converted = {}
    for key, value in values.items():
        origin = typing.get_origin(hints[key])
        if origin is tuple:
            inner = typing.get_args(hints[key])
            nested = inner and typing.get_origin(inner[0]) is tuple
            value = tuple(tuple(v) for v in value) if nested else tuple(value)
        elif origin in (typing.Union, types.UnionType):
            pass
        converted[key] = value
    return section_type(**converted)
