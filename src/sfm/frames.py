"""Dense frame selection from preprocessing scores and full-resolution export."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import cv2

from ..preprocessing import load_result
from ..preprocessing.decode import upright
from .contracts import SfmConfig


def select_frames(scores, geometry_ids: set[str], config: SfmConfig) -> list[dict]:
    """One sharpest frame per 1/fps window, avoiding flagged frames, plus all geometry frames.

    Sharpness is compared only inside a window, so slow lighting changes along the video do
    not bias the choice. A window whose frames are all flagged still contributes its
    sharpest frame: a gap in coverage hurts SfM more than one soft frame.
    """
    window = 1.0 / config.frames_per_second
    chosen: dict[str, object] = {}
    groups: dict[tuple[str, int], list] = {}
    for record in scores:
        groups.setdefault((record.clip_id, int(record.t_s // window)), []).append(record)

    for records in groups.values():
        clean = [r for r in records if not set(r.flags) & set(config.avoid_flags)]
        best = max(clean or records, key=lambda r: r.sharpness)
        chosen[best.frame_id] = best
    for record in scores:
        if record.frame_id in geometry_ids:
            chosen[record.frame_id] = record

    ordered = sorted(chosen.values(), key=lambda r: (r.clip_id, r.t_s))
    return [
        {
            "name": f"{index:05d}.jpg",
            "frame_id": record.frame_id,
            "clip_id": record.clip_id,
            "timestamp_s": float(record.t_s),
            "source_index": int(record.source_index),
            "sharpness": float(record.sharpness),
            "flags": list(record.flags),
            "geometry": record.frame_id in geometry_ids,
        }
        for index, record in enumerate(ordered)
    ]


def decode_selected(clip, wanted: set[str], threads: int):
    """Yield ``(frame_id, upright RGB)`` for the wanted frames of one clip.

    Frame IDs and orientation follow `src.preprocessing.decode.decode_frames` exactly. Frame
    threading and converting only the wanted frames make this several times faster than a
    full-resolution conversion of every frame.
    """
    remaining = set(wanted)
    with av.open(clip.path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = threads
        for frame in container.decode(stream):
            pts = int(Fraction(frame.pts) * frame.time_base / Fraction(clip.time_base))
            frame_id = f"{clip.clip_id}:{pts}"
            if frame_id in remaining:
                remaining.remove(frame_id)
                yield frame_id, upright(frame.to_ndarray(format="rgb24"), clip.rotation_deg)
                if not remaining:
                    return
    if remaining:
        raise ValueError(f"Missing source frames: {sorted(remaining)[:5]}")


def export_frames(preprocess_dir: str | Path, output_dir: str | Path, config: SfmConfig) -> list:
    """Write ``frames.json`` and upright full-resolution JPEGs; reuse a finished export."""
    output = Path(output_dir)
    listing = output / "frames.json"
    if listing.exists():
        frames = json.loads(listing.read_text())
        if all((output / "images" / f["name"]).exists() for f in frames):
            return frames

    result = load_result(preprocess_dir)
    geometry = {f.frame_id for f in result.selected if f.pool == "geometry"}
    frames = select_frames(result.scores, geometry, config)
    by_id = {f["frame_id"]: f for f in frames}

    (output / "images").mkdir(parents=True, exist_ok=True)
    for clip in result.clips:
        wanted = {f["frame_id"] for f in frames if f["clip_id"] == clip.clip_id}
        for frame_id, image in decode_selected(clip, wanted, config.decode_threads):
            record = by_id[frame_id]
            path = output / "images" / record["name"]
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]):
                raise OSError(f"Could not write {path}")
            record["width"], record["height"] = int(bgr.shape[1]), int(bgr.shape[0])

    listing.write_text(json.dumps(frames, indent=1) + "\n")
    return frames
