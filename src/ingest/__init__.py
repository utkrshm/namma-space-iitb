"""Ingest: a video (or a folder of images) → sharp, well-overlapping frames.

Input
-----
A video file readable by PyAV/FFmpeg (any container; display rotation is honoured), or a
directory of ``.jpg``/``.jpeg``/``.png`` images in capture order (sorted by name).

Output (``<out>/frames/``)
--------------------------
- ``images/<00000>.jpg``: upright RGB, long side at most ``max_long_side``.
- ``frames.json``: ``{"source", "kind": "video"|"images", "fps", "frames": [...]}``. Each
  frame has ``name``, ``index`` (source frame number or image position), ``position``
  (order among the chosen frames), ``timestamp_s``
  (``null`` for images), ``source_name`` (images only), ``sharpness``, ``width`` and
  ``height``.
- ``selection.png``: motion, sharpness and the chosen frames along the video.

Frame selection (video)
-----------------------
Frames are chosen by *camera motion*, not at a fixed rate. A slow pan then gets few
frames and a fast one many, so neighbouring frames always overlap. Each frame's motion is
the median optical-flow displacement of tracked corners, as a fraction of the image
diagonal. The timeline is cut wherever the accumulated motion reaches ``motion_step``
(or ``max_gap_s`` passes), and each piece contributes its sharpest frame. This is the
Coorg lesson that a sharp frame next to a blurred one is worth more than an evenly spaced
blurred one.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from ..config import IngestConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sharpness(gray: np.ndarray) -> float:
    """75th percentile of tile Laplacian variances (robust to plain walls in part of a frame)."""
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    height, width = gray.shape
    tiles = [
        laplacian[y : y + height // 4, x : x + width // 4].var()
        for y in range(0, height - height // 4 + 1, height // 4)
        for x in range(0, width - width // 4 + 1, width // 4)
    ]
    return float(np.percentile(tiles, 75))


def _upright(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(image, k=(rotation_deg // 90) % 4))


def _rotation(path: Path) -> int:
    with av.open(str(path)) as container:
        first = next(container.decode(container.streams.video[0]))
        rotation = round(first.rotation)
    if rotation % 90:
        raise ValueError("Only right-angle display rotations are supported")
    return rotation


def _decode(path: Path, long_side: int | None, rotation: int, wanted: set[int] | None = None):
    """Yield ``(index, seconds, upright RGB)``; resizing happens in libswscale."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        start = stream.start_time or 0
        for index, frame in enumerate(container.decode(stream)):
            if wanted is not None and index not in wanted:
                continue
            scale = (
                1.0 if long_side is None else min(1.0, long_side / max(frame.width, frame.height))
            )
            rgb = frame.reformat(
                width=round(frame.width * scale) // 2 * 2,
                height=round(frame.height * scale) // 2 * 2,
                format="rgb24",
            ).to_ndarray()
            seconds = float((frame.pts - start) * Fraction(frame.time_base)) if frame.pts else 0.0
            yield index, seconds, _upright(rgb, rotation)
            if wanted is not None and index == max(wanted):
                return


def measure_video(path: Path, config: IngestConfig) -> dict[str, np.ndarray]:
    """Per-frame time, sharpness and motion at ``analysis_long_side`` (one decode)."""
    rotation = _rotation(path)
    times, sharp, motion = [], [], []
    previous = corners = None
    for _, seconds, rgb in _decode(path, config.analysis_long_side, rotation):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        diagonal = float(np.hypot(*gray.shape))
        moved = np.nan
        if previous is not None and corners is not None and len(corners) >= 8:
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, corners, None)
            good = status.ravel() == 1
            if good.sum() >= 8:
                moved = float(np.median(np.linalg.norm(tracked[good] - corners[good], axis=-1)))
                moved /= diagonal
        corners = cv2.goodFeaturesToTrack(gray, 400, 0.01, 8)
        previous = gray
        times.append(seconds)
        sharp.append(sharpness(gray))
        motion.append(moved)
    motion = np.array(motion)
    # Untrackable frames (blank wall, blur) move like their neighbours, never "not at all".
    known = np.isfinite(motion)
    if known.any():
        motion[~known] = np.interp(np.flatnonzero(~known), np.flatnonzero(known), motion[known])
    else:
        motion[:] = 0.0
    return {
        "times": np.array(times),
        "sharpness": np.array(sharp),
        "motion": motion,
        "rotation": np.array(rotation),
    }


def choose_frames(times, sharp, motion, config: IngestConfig, step: float) -> list[int]:
    """Sharpest frame of each piece of the timeline holding ``step`` of motion."""
    chosen, start = [], 0
    accumulated = 0.0
    for index in range(1, len(times) + 1):
        if index < len(times):
            accumulated += motion[index]
        end_of_piece = (
            index == len(times)
            or accumulated >= step
            or times[index] - times[start] >= config.max_gap_s
        )
        if not end_of_piece:
            continue
        piece = np.arange(start, index)
        # Ignore frames far blurrier than the local median when the piece has any choice.
        local = np.median(sharp[max(0, start - 30) : index + 30])
        sharp_enough = piece[sharp[piece] >= config.min_relative_sharpness * local]
        candidates = sharp_enough if len(sharp_enough) else piece
        chosen.append(int(candidates[np.argmax(sharp[candidates])]))
        start, accumulated = index, 0.0
    return chosen


def select_video_frames(measures: dict, config: IngestConfig) -> tuple[list[int], float]:
    """Frame indices and the motion step used; the step grows until ``max_frames`` fits."""
    times, sharp, motion = measures["times"], measures["sharpness"], measures["motion"]
    step = config.motion_step
    while True:
        chosen = choose_frames(times, sharp, motion, config, step)
        if len(chosen) <= config.max_frames:
            return chosen, step
        step *= len(chosen) / config.max_frames * 1.05


def _natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _resize(rgb: np.ndarray, long_side: int) -> np.ndarray:
    scale = long_side / max(rgb.shape[:2])
    if scale >= 1:
        return rgb
    size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _write(path: Path, rgb: np.ndarray, quality: int) -> None:
    if not cv2.imwrite(str(path), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"Could not write {path}")


def ingest(source: str | Path, output_dir: str | Path, config: IngestConfig | None = None) -> dict:
    """Write ``frames/``; a finished ingest of the same source is reused."""
    config = config or IngestConfig()
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    listing_path = output / "frames.json"
    if listing_path.exists():
        listing = json.loads(listing_path.read_text())
        if listing["source"] == str(source) and all(
            (output / "images" / f["name"]).exists() for f in listing["frames"]
        ):
            return listing
    (output / "images").mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        listing = _ingest_images(source, output, config)
    elif source.is_file():
        listing = _ingest_video(source, output, config)
    else:
        raise FileNotFoundError(source)
    if len(listing["frames"]) < 3:
        raise ValueError(f"Only {len(listing['frames'])} usable frames in {source}")
    listing_path.write_text(json.dumps(listing, indent=1) + "\n")
    return listing


def _ingest_images(source: Path, output: Path, config: IngestConfig) -> dict:
    paths = sorted(
        (p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key
    )
    frames = []
    for position, path in enumerate(paths):
        rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise OSError(f"Unreadable image {path}")
        rgb = _resize(rgb[..., ::-1], config.max_long_side)
        name = f"{position:05d}.jpg"
        _write(output / "images" / name, rgb, config.jpeg_quality)
        gray = cv2.cvtColor(_resize(rgb, config.analysis_long_side), cv2.COLOR_RGB2GRAY)
        frames.append(
            {
                "name": name,
                "index": position,
                "position": position,
                "timestamp_s": None,
                "source_name": path.name,
                "sharpness": sharpness(gray),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            }
        )
    return {"source": str(source), "kind": "images", "fps": None, "frames": frames}


def _ingest_video(source: Path, output: Path, config: IngestConfig) -> dict:
    measures = measure_video(source, config)
    chosen, step = select_video_frames(measures, config)
    rotation = int(measures["rotation"])
    frames, by_index = [], {index: number for number, index in enumerate(chosen)}
    for index, _, rgb in _decode(source, config.max_long_side, rotation, set(chosen)):
        number = by_index[index]
        name = f"{number:05d}.jpg"
        _write(output / "images" / name, rgb, config.jpeg_quality)
        frames.append(
            {
                "name": name,
                "index": index,
                "timestamp_s": float(measures["times"][index]),
                "sharpness": float(measures["sharpness"][index]),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            }
        )
    frames.sort(key=lambda f: f["index"])
    for position, frame in enumerate(frames):
        frame["position"] = position
    duration = float(measures["times"][-1]) if len(measures["times"]) else 0.0
    fps = (len(measures["times"]) - 1) / duration if duration > 0 else None
    _plot_selection(measures, chosen, output / "selection.png")
    return {
        "source": str(source),
        "kind": "video",
        "fps": fps,
        "rotation_deg": rotation,
        "decoded_frames": len(measures["times"]),
        "motion_step": step,
        "frames": frames,
    }


def _plot_selection(measures: dict, chosen: list[int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = measures["times"]
    figure, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(times, measures["motion"], lw=0.6)
    axes[0].set_ylabel("motion / diagonal")
    axes[1].semilogy(times, measures["sharpness"], lw=0.6)
    axes[1].semilogy(times[chosen], measures["sharpness"][chosen], "r.", ms=3)
    axes[1].set_ylabel("sharpness")
    axes[1].set_xlabel("seconds")
    figure.suptitle(f"{len(chosen)} frames chosen of {len(times)}")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
