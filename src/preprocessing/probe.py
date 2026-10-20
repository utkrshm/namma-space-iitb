"""Probe only whitelisted metadata; never export container location tags."""

import hashlib
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .contracts import ClipInfo


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_clip(path: str | Path, clip_id: str | None = None) -> ClipInfo:
    path = Path(path).resolve()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        rotation = round(first.rotation)
        if rotation % 90:
            raise ValueError("Only right-angle display rotations are supported")
        transfer = {1: "bt709", 6: "smpte170m", 16: "smpte2084", 18: "arib-std-b67"}.get(
            int(stream.codec_context.color_trc), "unknown"
        )
        width, height = first.width, first.height
        time_base = Fraction(stream.time_base)
        start = stream.start_time if stream.start_time is not None else first.pts
        duration = float(stream.duration * time_base) if stream.duration else 0.0
        color_range = {0: "unknown", 1: "limited", 2: "full"}.get(int(first.color_range))
    # Packet PTS are inexpensive to inspect; sort for presentation order (B-frame safe).
    with av.open(str(path)) as container:
        pts = sorted(p.pts for p in container.demux(video=0) if p.pts is not None)
    deltas = np.diff(pts)
    # One time-base tick of rounding jitter does not constitute VFR.
    variable = bool(len(deltas) and np.ptp(deltas) > 1)
    return ClipInfo(
        clip_id or path.stem,
        str(path),
        file_hash(path),
        width,
        height,
        rotation,
        str(time_base),
        duration,
        variable,
        transfer,
        transfer in {"smpte2084", "arib-std-b67"},
        int(start),
        color_range,
    )
