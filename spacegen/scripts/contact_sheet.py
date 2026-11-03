"""Contact sheets of a run's videos, for checking them by eye (CPU only).

    uv run --no-sync python scripts/contact_sheet.py RUN [--per-section 4] [--demo 12]

Writes ``RUN/tour/sheet.jpg`` (``--per-section`` frames from each section in
``tour.json``, one row per section, labelled) and ``RUN/demo/sheet.jpg`` (``--demo``
evenly spaced frames), each frame at 480 × 270.
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

SIZE = (480, 270)


def read_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    frames = {}
    with av.open(str(video)) as container:
        for number, frame in enumerate(container.decode(video=0)):
            if number in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                frames[number] = cv2.resize(rgb, SIZE, interpolation=cv2.INTER_AREA)
            if number >= max(wanted):
                break
    return frames


def caption(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, SIZE[1] - 20), (SIZE[0], SIZE[1]), (0, 0, 0), -1)
    cv2.putText(image, text, (6, SIZE[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return image


def grid(rows: list[list[np.ndarray]]) -> np.ndarray:
    columns = max(len(row) for row in rows)
    blank = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    return np.vstack([np.hstack(row + [blank] * (columns - len(row))) for row in rows])


def tour_sheet(run: Path, per_section: int) -> Path | None:
    video, report = run / "tour" / "tour.mp4", run / "tour" / "tour.json"
    if not video.exists() or not report.exists():
        return None
    sections = [s for s in json.loads(report.read_text())["sections"] if s["frames"]]
    starts = np.cumsum([0] + [s["frames"] for s in sections])
    picks = [
        [int(start + (count - 1) * t) for t in np.linspace(0.05, 0.95, per_section)]
        for start, count in zip(starts[:-1], [s["frames"] for s in sections], strict=True)
    ]
    frames = read_frames(video, {p for row in picks for p in row})
    rows = [
        [caption(frames[p], f"{section['name']} #{p}") for p in row if p in frames]
        for section, row in zip(sections, picks, strict=True)
    ]
    path = run / "tour" / "sheet.jpg"
    cv2.imwrite(str(path), grid(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def demo_sheet(run: Path, count: int) -> Path | None:
    video = run / "demo" / "demo.mp4"
    if not video.exists():
        return None
    with av.open(str(video)) as container:
        total = container.streams.video[0].frames
    wanted = [int(t) for t in np.linspace(0, max(total - 1, 0), count)]
    frames = read_frames(video, set(wanted))
    images = [caption(frames[p], f"demo #{p}") for p in wanted if p in frames]
    rows = [images[i : i + 4] for i in range(0, len(images), 4)]
    path = run / "demo" / "sheet.jpg"
    cv2.imwrite(str(path), grid(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--per-section", type=int, default=4)
    parser.add_argument("--demo", type=int, default=12)
    args = parser.parse_args()

    for path in (tour_sheet(args.run, args.per_section), demo_sheet(args.run, args.demo)):
        if path is not None:
            print(path)


if __name__ == "__main__":
    main()
