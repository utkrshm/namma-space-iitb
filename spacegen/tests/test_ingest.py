"""Frame selection on a real encoded video: a camera panning over a photo-like texture."""

import json

import av
import cv2
import numpy as np

from spacegen.config import IngestConfig
from spacegen.ingest import ingest, sharpness


def write_pan(path, speeds, blurred=(), size=(320, 240), fps=30):
    """Pan a window across a random texture; ``speeds`` gives pixels per frame; frames in
    ``blurred`` get a heavy box blur."""
    rng = np.random.default_rng(0)
    texture = cv2.resize(
        rng.integers(0, 255, (300, 1200, 3), dtype=np.uint8),
        (12000, 3000),
        interpolation=cv2.INTER_CUBIC,
    )
    texture = cv2.GaussianBlur(texture, (0, 0), 3)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    x = 0.0
    for number, speed in enumerate(speeds):
        x += speed
        frame = texture[1000 : 1000 + size[1], int(x) : int(x) + size[0]].copy()
        if number in blurred:
            frame = cv2.blur(frame, (15, 15))
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_frames_follow_motion_and_avoid_blur(tmp_path):
    # 60 slow frames (1 px/frame), then 60 fast frames (12 px/frame).
    speeds = [1.0] * 60 + [12.0] * 60
    blurred = set(range(70, 120, 2))  # every other fast frame is blurred
    write_pan(tmp_path / "pan.mp4", speeds, blurred)
    config = IngestConfig(analysis_long_side=320, motion_step=0.05, max_gap_s=10.0)
    listing = ingest(tmp_path / "pan.mp4", tmp_path / "frames", config)

    indices = np.array([f["index"] for f in listing["frames"]])
    slow, fast = (indices < 60).sum(), (indices >= 60).sum()
    assert fast > 4 * slow, (slow, fast)
    # Blurred frames are skipped whenever a sharp neighbour exists.
    chosen_blurred = [i for i in indices if i in blurred]
    assert len(chosen_blurred) <= 1, chosen_blurred
    assert [f["position"] for f in listing["frames"]] == list(range(len(indices)))
    image = cv2.imread(str(tmp_path / "frames" / "images" / listing["frames"][0]["name"]))
    assert image.shape[:2] == (240, 320)
    # A second call reuses the finished ingest.
    assert ingest(tmp_path / "pan.mp4", tmp_path / "frames", config) == json.loads(
        (tmp_path / "frames" / "frames.json").read_text()
    )


def test_image_folder_keeps_natural_order(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    rng = np.random.default_rng(1)
    for number in (10, 2, 1):
        cv2.imwrite(
            str(folder / f"IMG_{number}.jpg"), rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        )
    listing = ingest(folder, tmp_path / "frames")
    assert [f["source_name"] for f in listing["frames"]] == ["IMG_1.jpg", "IMG_2.jpg", "IMG_10.jpg"]
    assert listing["kind"] == "images"


def test_sharpness_drops_with_blur():
    image = np.random.default_rng(0).integers(0, 255, (240, 320), dtype=np.uint8)
    assert sharpness(cv2.blur(image, (9, 9))) < 0.1 * sharpness(image)
