"""Versioned, model-independent handoff records. Pixel coordinates are upright x-right/y-down."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeAlias

SCHEMA_VERSION = 2
FrameId: TypeAlias = str  # <clip_id>:<pts>, pts in the source stream time base


class JsonRecord:
    schema_version: ClassVar[int] = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str | dict):
        data = json.loads(value) if isinstance(value, str) else dict(value)
        if data.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unsupported or missing schema_version")
        for f in fields(cls):
            if f.name in data and (str(f.type).startswith("tuple") or f.name == "flags"):
                data[f.name] = tuple(data[f.name])
        if "matrix" in data:
            data["matrix"] = tuple(tuple(row) for row in data["matrix"])
        return cls(**data)


@dataclass(frozen=True)
class ClipInfo(JsonRecord):
    clip_id: str
    path: str
    sha256: str
    width: int
    height: int
    rotation_deg: int
    time_base: str
    duration_s: float
    variable_frame_rate: bool
    color_transfer: str
    is_hdr: bool
    start_pts: int = 0
    color_range: str = "unknown"


@dataclass(frozen=True)
class RotationTransform(JsonRecord):
    rotation_deg: int  # FFmpeg display convention, negative = clockwise
    coded_width: int
    coded_height: int
    matrix: tuple  # 3x3 source-pixel -> upright-pixel homogeneous matrix
    analysis_scale_x: float
    analysis_scale_y: float


@dataclass(frozen=True)
class FrameScores(JsonRecord):
    frame_id: FrameId
    clip_id: str
    source_index: int
    pts: int
    t_s: float
    sharpness: float
    sharpness_rel_short: float
    sharpness_rel_long: float
    flow_px: float | None
    track_survival: float
    tracking_reliable: bool
    dark_frac: float
    blown_frac: float
    luma_mean: float
    luma_std: float
    dt_s: float
    flags: tuple[str, ...] = ()
    edge_fraction: float = 0.0
    reblur_loss: float | None = None


@dataclass(frozen=True)
class SelectedFrame(JsonRecord):
    frame_id: FrameId
    pool: Literal["geometry", "appearance"]
    keyframe_id: FrameId
    reason: tuple[str, ...]
    overlap: float | None
    initial_coverage: float
    displacement_px: float | None
    image_path: str
    width: int
    height: int
    rotation: RotationTransform

    @classmethod
    def from_json(cls, value):
        data = json.loads(value) if isinstance(value, str) else dict(value)
        if data.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unsupported or missing schema_version")
        rotation = dict(data.pop("rotation"))
        rotation["matrix"] = tuple(tuple(row) for row in rotation["matrix"])
        return cls(rotation=RotationTransform(**rotation), reason=tuple(data.pop("reason")), **data)


@dataclass(frozen=True)
class Edge(JsonRecord):
    a: FrameId
    b: FrameId
    kind: Literal["sequential", "revisit"]
    weight: float
    inliers: int
    inlier_coverage: float
    ambiguous_matches: int = 0  # Raw local matches removed by the same-image descriptor guard.


@dataclass(frozen=True)
class GapItem(JsonRecord):
    clip_id: str
    t_start_s: float
    t_end_s: float
    kind: Literal[
        "low_quality_interval",
        "low_texture_interval",
        "tracking_break",
        "weak_link",
        "unlinked_clip",
    ]
    severity: Literal["info", "warn", "review"]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class PreprocessResult(JsonRecord):
    output_dir: str
    artifacts: dict[str, str]
    clips: tuple[ClipInfo, ...]
    scores: tuple[FrameScores, ...]
    selected: tuple[SelectedFrame, ...]
    edges: tuple[Edge, ...]
    gaps: tuple[GapItem, ...]

    @property
    def geometry(self):
        return tuple(f for f in self.selected if f.pool == "geometry")

    @property
    def appearance(self):
        return tuple(f for f in self.selected if f.pool == "appearance")

    @classmethod
    def from_json(cls, value):
        data = json.loads(value) if isinstance(value, str) else dict(value)
        if data.pop("schema_version", None) != SCHEMA_VERSION:
            raise ValueError("Unsupported or missing schema_version")
        for key, record in (
            ("clips", ClipInfo),
            ("scores", FrameScores),
            ("selected", SelectedFrame),
            ("edges", Edge),
            ("gaps", GapItem),
        ):
            data[key] = tuple(
                record.from_json({"schema_version": SCHEMA_VERSION, **r}) for r in data[key]
            )
        return cls(**data)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({**data, "schema_version": SCHEMA_VERSION}, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported or missing schema_version: {path}")
    return data
