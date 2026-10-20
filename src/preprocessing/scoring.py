"""Per-frame measurements and conservative quality flags at a common analysis scale."""

import csv
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import cv2
import numpy as np

from .config import PreprocessConfig
from .contracts import FrameScores
from .decode import decode_frames
from .tracking import KLTTracker


def tiled_sharpness(gray, tiles=4) -> float:
    values = [
        float(cv2.Laplacian(tile, cv2.CV_64F).var())
        for row in np.array_split(gray, tiles)
        for tile in np.array_split(row, tiles, axis=1)
    ]
    return float(np.percentile(values, 75))


def rolling_reference(times, values, radius, percentile):
    """Timestamp-based centered windows, not nominal frame-count windows."""
    times, values = np.asarray(times), np.asarray(values)
    left = np.searchsorted(times, times - radius)
    right = np.searchsorted(times, times + radius, side="right")
    return np.array([np.percentile(values[a:b], percentile) for a, b in zip(left, right)])


def score_clip(clip, config: PreprocessConfig):
    cfg = config.scoring
    rows, previous, previous_hist, previous_t = [], None, None, None
    for frame in decode_frames(clip, config.decode):
        gray = frame.image
        track = KLTTracker(previous, config.tracking).step(gray) if previous is not None else None
        histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram)
        flags = []
        if (
            previous_hist is not None
            and cv2.compareHist(previous_hist, histogram, cv2.HISTCMP_CORREL) < cfg.histogram_cut
        ):
            flags.append("possible_cut")
        dt = frame.t_s - previous_t if previous_t is not None else 0.0
        # A normal 33ms VFR step is not a warning; >=67ms is an info-level flag.
        if dt >= cfg.timing_gap_s:
            flags.append("timing_gap_info")
        rows.append(
            FrameScores(
                frame.frame_id,
                clip.clip_id,
                frame.source_index,
                frame.pts,
                frame.t_s,
                tiled_sharpness(gray, cfg.tiles),
                1.0,
                1.0,
                track.displacement_px if track else None,
                track.survival if track else 0.0,
                track.reliable if track else False,
                float(np.mean(gray < cfg.dark_level)),
                float(np.mean(gray > cfg.blown_level)),
                float(gray.mean()),
                float(gray.std()),
                dt,
                tuple(flags),
            )
        )
        previous, previous_hist, previous_t = gray, histogram, frame.t_s
    times = [row.t_s for row in rows]
    sharpness = np.array([row.sharpness for row in rows])
    short = rolling_reference(times, sharpness, cfg.short_window_s, 50)
    long = rolling_reference(times, sharpness, cfg.long_window_s, cfg.reference_percentile)
    luma_ref = rolling_reference(times, [r.luma_mean for r in rows], cfg.short_window_s, 50)
    result = []
    for row, a, b, luma in zip(rows, short, long, luma_ref):
        flags = list(row.flags)
        rel_short, rel_long = row.sharpness / max(a, 1e-6), row.sharpness / max(b, 1e-6)
        if row.sharpness < cfg.absolute_blur:
            flags.append("low_detail_or_blur")
        if rel_long < cfg.relative_blur:
            flags.append("suspected_blur")
        if (
            row.flow_px is not None
            and row.dt_s > 0
            and row.flow_px / row.dt_s >= cfg.high_motion_px_per_s
        ):
            flags.append("high_motion")
        if max(row.dark_frac, row.blown_frac) > cfg.exposure_fraction:
            flags.append("exposure")
        if row.luma_std < cfg.low_contrast:
            flags.append("low_contrast")
        if abs(row.luma_mean - luma) >= cfg.exposure_jump:
            flags.append("exposure_jump")
        result.append(
            replace(
                row,
                sharpness_rel_short=float(rel_short),
                sharpness_rel_long=float(rel_long),
                flags=tuple(flags),
            )
        )
    return result


def write_scores(path: Path, records):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(FrameScores)])
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["flags"] = json.dumps(row["flags"])
            row["flow_px"] = "" if row["flow_px"] is None else row["flow_px"]
            writer.writerow(row)
    temporary.replace(path)


def read_scores(path: Path):
    records = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("source_index", "pts"):
                row[key] = int(row[key])
            row["tracking_reliable"] = row["tracking_reliable"] == "True"
            row["flags"] = tuple(json.loads(row["flags"]))
            for key in (
                "t_s",
                "sharpness",
                "sharpness_rel_short",
                "sharpness_rel_long",
                "flow_px",
                "track_survival",
                "dark_frac",
                "blown_frac",
                "luma_mean",
                "luma_std",
                "dt_s",
            ):
                row[key] = float(row[key]) if row[key] != "" else None
            records.append(FrameScores(**row))
    return records
