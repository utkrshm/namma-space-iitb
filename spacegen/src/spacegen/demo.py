"""Demo video: a top-down build-up camera by camera, then a highlight reel from inside.

1. **Build-up (sped up):** a fixed view from above with the ceiling cut away. Gaussians
   appear when the capture camera that first saw them arrives, so the space assembles
   camera pose by camera pose. Generated (filled) Gaussians are left out: seen from
   above they add haze, and this view is meant to show what was captured. The walked
   path is drawn as it grows, and the current camera is drawn as a small frustum.
2. **Highlight reel:** eye-level clips along the smoothed capture path (`tour.WalkPath`:
   walking pace, slower where the capture turned on the spot). Candidate windows
   are scored on low-resolution renders by sharpness × measured share × rendered share,
   and the best non-overlapping ones are joined with crossfades. With ``demo.turns`` a
   filled model (`spacegen.fill`) adds one 360° turn per area.

Output: ``<run>/demo/demo.mp4`` (H.264) and ``demo.json`` (chosen clips and their scores).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import DemoConfig
from .contracts import Camera, look_at
from .scene_frame import run_frame
from .splat.model import load_ply, render_camera, write_json
from .splat.train import load_cameras
from .tour import WalkPath, Writer, cutaway, ease, label, measured_support


def first_seen(means: torch.Tensor, cameras: list[Camera], max_depth: float) -> torch.Tensor:
    """Index (in ``cameras`` order) of the first camera whose view holds each Gaussian."""
    seen = torch.full((len(means),), len(cameras), device=means.device, dtype=torch.long)
    for index, camera in enumerate(cameras):
        w2c = torch.tensor(camera.world_to_camera, dtype=torch.float32, device=means.device)
        k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=means.device)
        local = means @ w2c[:3, :3].T + w2c[:3, 3]
        depth = local[:, 2]
        pixels = local[:, :2] / depth.clamp_min(1e-6)[:, None] * k.diagonal()[:2] + k[:2, 2]
        inside = (depth > 0.02) & (depth < max_depth)
        inside &= (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
        inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
        seen[inside & (seen == len(cameras))] = index
    return seen


def _project(points: np.ndarray, camera: Camera, scale: float) -> np.ndarray:
    local = points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    pixels = local[:, :2] / np.maximum(local[:, 2:3], 1e-6) * np.diag(camera.intrinsics)[:2]
    return (pixels + camera.intrinsics[:2, 2]) * scale


def _draw_path(image, view: Camera, scale, centres, current: Camera, camera_height):
    if len(centres) > 1:
        points = _project(centres, view, scale).round().astype(np.int32)
        cv2.polylines(image, [points], False, (255, 200, 40), 2, cv2.LINE_AA)
    # Current camera: a small frustum (apex at the camera, four far corners).
    k = current.intrinsics
    depth = 0.35 * camera_height
    corners = [
        ((x - k[0, 2]) / k[0, 0] * depth, (y - k[1, 2]) / k[1, 1] * depth, depth)
        for x, y in (
            (0, 0),
            (current.width, 0),
            (current.width, current.height),
            (0, current.height),
        )
    ]
    camera_to_world = current.camera_to_world
    world = np.array(corners) @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    apex = _project(camera_to_world[:3, 3][None], view, scale)[0]
    far = _project(world, view, scale)
    for corner in far:
        cv2.line(
            image,
            tuple(apex.round().astype(int)),
            tuple(corner.round().astype(int)),
            (255, 80, 60),
            2,
            cv2.LINE_AA,
        )
    cv2.polylines(image, [far.round().astype(np.int32)], True, (255, 80, 60), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(apex.round().astype(int)), 5, (255, 80, 60), -1, cv2.LINE_AA)


def _crossfade(writer: Writer, previous: np.ndarray | None, frames: list[np.ndarray], count: int):
    for number, image in enumerate(frames):
        if previous is not None and number < count:
            weight = (number + 1) / (count + 1)
            image = (image * weight + previous * (1 - weight)).astype(np.uint8)
        writer.write(image)
    return frames[-1] if frames else previous


def render_demo(run: Path, config: DemoConfig, device: str, log) -> dict:
    run = Path(run)
    output = run / "demo"
    output.mkdir(exist_ok=True)
    filled = (run / "fill" / "scene.ply").exists()
    model = load_ply(run / ("fill" if filled else "splat") / "scene.ply", device)
    cameras, splits = load_cameras(run / "splat" / "cameras.json")
    train = [c for c, split in zip(cameras, splits, strict=True) if split == "train"]
    if "support" not in model.extras:
        from .support import compute_support

        model.extras["support"] = compute_support(model, train)
    frame = run_frame(run, cameras)
    up, height_unit = frame.up, frame.camera_height
    width = round(config.width * config.supersample)
    height = round(config.height * config.supersample)
    writer = Writer(output / "demo.mp4", config)
    report: dict = {"model": "filled" if filled else "splat"}

    # 1. Build-up from above, camera by camera.
    centres_aligned = frame.to_aligned(np.stack([c.centre for c in train]))
    overview = model
    if "filled" in model.extras:
        overview = model.subset(~model.extras["filled"].reshape(-1).bool())
    overview = cutaway(overview, frame, centres_aligned)
    seen = first_seen(overview.params["means"].detach(), train, 3.0 * height_unit)
    low = np.percentile(centres_aligned[:, :2], 1, axis=0) - 0.6
    high = np.percentile(centres_aligned[:, :2], 99, axis=0) + 0.6
    middle = np.r_[(low + high) / 2, 0.2]
    fov = 45.0
    focal = (width / 2) / math.tan(math.radians(fov) / 2)
    k = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    # Fit the footprint, then tilt 20° off vertical so walls read as walls.
    span = max((high - low)[0] / width * height, (high - low)[1]) * 0.5
    distance = 1.08 * span / math.tan(math.atan(math.tan(math.radians(fov) / 2) * height / width))
    tilt = math.radians(70)
    eye = middle + distance * np.array([0.0, -math.cos(tilt), math.sin(tilt)])
    view = Camera(
        "top",
        width,
        height,
        k,
        look_at(frame.to_world(eye[None])[0], frame.to_world(middle[None])[0], up),
    )
    background = torch.tensor((0.12, 0.12, 0.14), device=device)
    count = int(config.build_seconds * config.fps)
    path_world = np.stack([c.centre for c in train])
    for number in range(count + int(1.5 * config.fps)):
        index = min(len(train) - 1, int(number / max(count - 1, 1) * (len(train) - 1)))
        visible = overview.subset(seen <= index)
        with torch.no_grad():
            rgb, _, _, _ = render_camera(visible, view, background=background)
        image = (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()
        image = cv2.resize(image, (config.width, config.height), interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(image)
        _draw_path(
            image, view, 1 / config.supersample, path_world[: index + 1], train[index], height_unit
        )
        writer.write(label(image, f"Building the space: camera {index + 1} / {len(train)}"))
    report["build_up"] = {"frames": count, "cameras": len(train)}
    del overview
    torch.cuda.empty_cache()

    # 2. Highlight reel along the smoothed capture path.
    walk = WalkPath(train, frame, 1 / config.walk_speed, config.max_turn_deg_per_s)
    pose = walk.pose
    focal = (width / 2) / math.tan(math.radians(config.reel_fov_deg) / 2)
    k = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    near = 0.12 * height_unit

    clip_length = config.clip_seconds  # seconds of the walk per clip
    starts = np.arange(0, max(walk.duration - clip_length, 0) + 1e-6, clip_length / 2)
    scores = []
    for start in starts:
        values = []
        # Seven moments including both ends: smears near walls often last under a second.
        for t in ease(np.linspace(0, 1, 7)):
            small = Camera(
                "probe",
                320,
                180,
                k * np.array([[320 / width], [180 / height], [1]]),
                pose(start + t * clip_length),
            )
            with torch.no_grad():
                rgb, _, alpha, _ = render_camera(model, small, near_plane=near)
                fraction, small_alpha, _ = measured_support(model, small)
            gray = cv2.cvtColor((rgb.clamp(0, 1) * 255).byte().cpu().numpy(), cv2.COLOR_RGB2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            rendered = float((alpha > 0.9).float().mean())
            measured = float(((fraction > 0.5) & (small_alpha > 0.5)).float().mean())
            values.append(sharp * rendered**4 * measured**2)
        scores.append(float(np.min(values)))  # a clip is as good as its worst moment
    chosen: list[float] = []
    for index in np.argsort(scores)[::-1]:
        if all(abs(starts[index] - other) >= clip_length for other in chosen):
            chosen.append(float(starts[index]))
        if len(chosen) == config.clips:
            break
    chosen.sort()
    log(f"demo: {len(chosen)} highlight clips from {len(starts)} candidates")
    previous = None
    fade = int(config.crossfade_seconds * config.fps)
    frames_per_clip = int(config.clip_seconds * config.fps)
    for number, start in enumerate(chosen):
        frames = []
        for t in ease(np.linspace(0, 1, frames_per_clip)):
            camera = Camera("reel", width, height, k, pose(start + t * clip_length))
            with torch.no_grad():
                rgb, _, _, _ = render_camera(model, camera, near_plane=near)
            image = cv2.resize(
                (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy(),
                (config.width, config.height),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(
                label(np.ascontiguousarray(image), f"Inside the splat: highlight {number + 1}")
            )
        previous = _crossfade(writer, previous, frames, fade)

    # 3. Optionally, with a filled model, one 360° turn per area.
    turns = []
    if filled and config.turns:
        from .fill import standing_points

        for stand in standing_points(frame, cameras):
            eye = frame.to_world(np.array(stand["eye_aligned"])[None])[0]
            frames = []
            for yaw in 2 * math.pi * ease(np.linspace(0, 1, int(config.turn_seconds * config.fps))):
                direction = np.array([math.cos(yaw), math.sin(yaw), math.sin(math.radians(-8))])
                camera = Camera(
                    "turn", width, height, k, look_at(eye, eye + frame.rotation.T @ direction, up)
                )
                with torch.no_grad():
                    rgb, _, _, _ = render_camera(model, camera, near_plane=near)
                image = cv2.resize(
                    (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy(),
                    (config.width, config.height),
                    interpolation=cv2.INTER_AREA,
                )
                frames.append(label(np.ascontiguousarray(image), f"Standing in {stand['area']}"))
            previous = _crossfade(writer, previous, frames, fade)
            turns.append(stand["area"])
    writer.close()

    report.update(
        {
            "video": str(output / "demo.mp4"),
            "seconds": writer.frames / config.fps,
            "clips": [{"start_seconds": s, "score": scores[list(starts).index(s)]} for s in chosen],
            "turns": turns,
        }
    )
    write_json(output / "demo.json", report)
    log(f"demo: {report['seconds']:.0f}s written to {report['video']}")
    return report
