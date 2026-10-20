"""Bounded-memory decoding. PyAV/libswscale respects the frame color range."""

from dataclasses import dataclass
from fractions import Fraction

import av
import cv2
import numpy as np

from .config import DecodeConfig
from .contracts import ClipInfo, RotationTransform


@dataclass(frozen=True)
class DecodedFrame:
    frame_id: str
    source_index: int
    pts: int
    t_s: float
    image: np.ndarray


def upright(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, k=(rotation_deg // 90) % 4))


def rotation_transform(clip: ClipInfo, long_side: int) -> RotationTransform:
    w, h = clip.width, clip.height
    angle = clip.rotation_deg % 360
    matrices = {
        0: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        90: ((0, 1, 0), (-1, 0, w - 1), (0, 0, 1)),
        180: ((-1, 0, w - 1), (0, -1, h - 1), (0, 0, 1)),
        270: ((0, -1, h - 1), (1, 0, 0), (0, 0, 1)),
    }
    uw, uh = (h, w) if angle in {90, 270} else (w, h)
    scale = min(1.0, long_side / max(uw, uh))
    aw, ah = max(1, round(uw * scale)), max(1, round(uh * scale))
    return RotationTransform(clip.rotation_deg, w, h, matrices[angle], aw / uw, ah / uh)


def decode_frames(clip: ClipInfo, config: DecodeConfig, *, full_resolution=False, rgb=False):
    if clip.is_hdr:
        raise ValueError("HDR input requires an explicit consistent SDR conversion before ingest")
    with av.open(clip.path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = config.threads
        previous_pts = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f"Missing presentation timestamp in {clip.clip_id}, frame {index}")
            pts = int(Fraction(frame.pts) * frame.time_base / Fraction(clip.time_base))
            if previous_pts is not None and pts <= previous_pts:
                raise ValueError(f"Non-increasing PTS in {clip.clip_id}: {pts}")
            previous_pts = pts
            if (frame.width, frame.height) != (clip.width, clip.height):
                raise ValueError("Mid-clip resolution changes require separate clips")
            # Conversion to rgb24 uses the declared range, including yuvj full range.
            # Resize in libswscale before conversion to avoid full-size RGB allocations.
            scale = (
                1 if full_resolution else min(1, config.long_side / max(frame.width, frame.height))
            )
            resized = frame.reformat(
                width=round(frame.width * scale), height=round(frame.height * scale), format="rgb24"
            )
            image = upright(resized.to_ndarray(), clip.rotation_deg)
            if not rgb:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            yield DecodedFrame(
                f"{clip.clip_id}:{pts}",
                index,
                pts,
                float((pts - clip.start_pts) * Fraction(clip.time_base)),
                image,
            )
