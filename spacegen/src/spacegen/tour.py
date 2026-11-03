"""Tour video of a finished space: overview orbit, a look-around in each area, a walk.

1. **Overview:** a slow orbit from above with the ceiling cut away (a "dollhouse"
   view). It drops Gaussians above 1.15 camera heights, below the floor, and far outside
   the camera path, which are never seen from above, and is framed on what it keeps
   (`framing_distance`).
2. **Look-around:** at each area's standing point, a turn through the directions the
   capture filmed from near there, at the phone's own tilt (`filmed_arc`). Every 5° view
   is checked before rendering (`_checked_arc`); only the longest passing part is played,
   out and back, and an area with too little is skipped. A full 360° turn over generated
   content smeared and invented walls in the first areas of the vyas-room tour.
3. **Walkthrough:** eye level along the smoothed capture path, at walking pace.

Frames are rendered at ``supersample`` × the output size and area-downsampled, so fine
patterns (tiles, grilles) do not flicker. Output: ``<run>/tour/tour.mp4`` (H.264,
yuv420p) and ``tour.json`` (sections, frame counts and the quality checks of `Renderer`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from .config import DemoConfig, TourConfig
from .contracts import Camera, look_at, made_from
from .scene_frame import SceneFrame, run_frame
from .splat.model import GaussianModel, load_ply, render_camera, write_json
from .splat.train import load_cameras
from .support import support_map

# A pixel is a hole below this alpha. Trained on a black background, dark objects (black
# clothes, shadowed shelves) settle at alpha 0.6–0.9 and still render correctly, so 0.9
# flagged them as holes (vyas-room look-arounds); real holes are far more transparent.
HOLE_ALPHA = 0.5


def _intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    return np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])


class Writer:
    def __init__(self, path: Path, config: TourConfig | DemoConfig):
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=config.fps)
        self.stream.width, self.stream.height = config.width, config.height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(config.crf), "preset": "medium"}
        self.frames = 0

    def write(self, rgb: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def label(rgb: np.ndarray, text: str) -> np.ndarray:
    """White text on a translucent dark box in the top-left corner."""
    scale = rgb.shape[0] / 1080
    font, size, thickness = cv2.FONT_HERSHEY_DUPLEX, 1.0 * scale, max(1, round(2 * scale))
    (width, height), base = cv2.getTextSize(text, font, size, thickness)
    x, y, pad = round(36 * scale), round(36 * scale), round(14 * scale)
    box = rgb[y - pad : y + height + base + pad, x - pad : x + width + pad]
    box[:] = (box * 0.45).astype(np.uint8)
    cv2.putText(rgb, text, (x, y + height), font, size, (255, 255, 255), thickness, cv2.LINE_AA)
    return rgb


class Renderer:
    """Supersampled frames plus quality checks on every ``check_every``-th frame:

    - ``unrendered``: share of pixels with alpha < ``HOLE_ALPHA`` (holes showing the black
      background);
    - ``unmeasured``: share of rendered pixels that no capture camera saw from a nearby
      direction, i.e. generated (filled) or unsupported content (needs ``support``);
    - ``near``: share of pixels closer than 0.1 camera heights (floaters on the lens).
    """

    def __init__(
        self,
        model,
        config: TourConfig,
        fov_deg: float,
        near: float,
        camera_height: float,
        background=None,
    ):
        self.model, self.config, self.near = model, config, near
        self.background = (
            None
            if background is None
            else torch.tensor(background, dtype=torch.float32, device=model.device)
        )
        self.camera_height = camera_height
        self.width = round(config.width * config.supersample)
        self.height = round(config.height * config.supersample)
        self.k = _intrinsics(self.width, self.height, fov_deg)
        self.frames = 0
        self.checks: dict[str, list[float]] = {"unrendered": [], "unmeasured": [], "near": []}

    def reset(self) -> None:
        self.checks = {key: [] for key in self.checks}

    def summary(self) -> dict:
        return {key: float(np.mean(v)) if v else None for key, v in self.checks.items()}

    @torch.no_grad()
    def __call__(self, world_to_camera: np.ndarray) -> np.ndarray:
        camera = Camera("tour", self.width, self.height, self.k, world_to_camera)
        rgb, depth, alpha, _ = render_camera(
            self.model, camera, near_plane=self.near, background=self.background
        )
        if self.frames % self.config.check_every == 0 and self.background is None:
            self.checks["unrendered"].append(float((alpha < HOLE_ALPHA).float().mean()))
            near = (depth < 0.1 * self.camera_height) & (alpha > 0.5)
            self.checks["near"].append(float(near.float().mean()))
            if "support" in self.model.extras:
                small = camera.resized(self.width // 4, self.height // 4)
                fraction, small_alpha, _ = measured_support(self.model, small)
                rendered = small_alpha > 0.5
                if rendered.any():
                    self.checks["unmeasured"].append(
                        float((fraction[rendered] < 0.5).float().mean())
                    )
        self.frames += 1
        image = (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()
        return cv2.resize(
            image, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA
        )


def measured_support(model, camera):
    """Support map counting only captured (not ``filled``) Gaussians."""
    support = model.extras["support"]
    filled = model.extras.get("filled")
    if filled is not None:
        model.extras["support"] = torch.where(filled.bool(), torch.zeros_like(support), support)
    try:
        return support_map(model, camera, 35.0)
    finally:
        model.extras["support"] = support


def cutaway(model: GaussianModel, frame: SceneFrame, centres_aligned: np.ndarray) -> GaussianModel:
    from scipy.spatial import cKDTree

    aligned = frame.to_aligned(model.params["means"].detach().cpu().numpy())
    distance, _ = cKDTree(centres_aligned[:, :2]).query(aligned[:, :2])
    scale = model.params["scales"].detach().exp().max(1).values.cpu().numpy() / frame.camera_height
    keep = (aligned[:, 2] < 1.15) & (aligned[:, 2] > -0.3) & (distance < 1.6) & (scale < 0.5)
    return model.subset(torch.from_numpy(keep).to(model.device))


def overview_points(model: GaussianModel, frame: SceneFrame, count: int = 20_000) -> np.ndarray:
    """Up to ``count`` aligned centres of the model's opaque Gaussians (opacity > 0.3, or
    all when there are few): what an overview has to show."""
    means = model.params["means"].detach().cpu().numpy()
    solid = model.params["opacities"].detach().sigmoid().cpu().numpy() > 0.3
    points = frame.to_aligned(means[solid] if solid.sum() >= 100 else means)
    rng = np.random.default_rng(0)
    return points[rng.choice(len(points), min(count, len(points)), replace=False)]


def orbit_eye(middle: np.ndarray, distance: float, elevation: float, angle: float) -> np.ndarray:
    """Aligned eye ``distance`` from ``middle`` at ``elevation`` above the floor plane."""
    return middle + distance * np.array(
        [
            math.cos(elevation) * math.cos(angle),
            math.cos(elevation) * math.sin(angle),
            math.sin(elevation),
        ]
    )


def framing_distance(
    points: np.ndarray,
    middle: np.ndarray,
    elevation: float,
    angles,
    k: np.ndarray,
    width: int,
    height: int,
    margin: float = 0.94,
    share: float = 98.0,
) -> float:
    """Smallest distance (camera heights) at which ``share``% of the aligned ``points``
    project inside ``margin`` of a ``width`` × ``height`` view looking at ``middle`` from
    every orbit angle in ``angles``."""
    centre = np.array([(width - 1) / 2, (height - 1) / 2])
    limit = margin * np.array([width / 2, height / 2])

    def fits(distance: float) -> bool:
        for angle in angles:
            eye = orbit_eye(middle, distance, elevation, angle)
            world_to_camera = look_at(eye, middle, (0.0, 0.0, 1.0))
            local = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            if np.percentile(local[:, 2], 100 - share) <= 0.05:
                return False
            pixels = local[:, :2] / np.maximum(local[:, 2:3], 1e-6) * np.diag(k)[:2] + k[:2, 2]
            if (np.percentile(np.abs(pixels - centre), share, axis=0) > limit).any():
                return False
        return True

    low, high = 0.1, 1000.0
    if not len(points) or not fits(high):
        return high
    for _ in range(40):
        middle_distance = (low + high) / 2
        low, high = (low, middle_distance) if fits(middle_distance) else (middle_distance, high)
    return high


def ease(t: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


class WalkPath:
    """The smoothed capture path at walking pace, as a function of time in seconds.

    Positions and view directions come from ``cameras`` in capture order, Gaussian-smoothed.
    Each step between two cameras takes the longer of its walking time
    (``seconds_per_height`` per camera height) and its turning time at
    ``max_turn_deg_per_s``: spaced by distance alone, the view whipped round wherever the
    capture turned on the spot (up to 700°/s on coorg and new_home). The view keeps the
    phone's heading and 40% of its tilt, between 35° down and 15° up (much of the m-block
    video looks straight down at the floor).
    """

    def __init__(self, cameras, frame: SceneFrame, seconds_per_height, max_turn_deg_per_s):
        path = np.stack([c.centre for c in cameras])
        forward = np.stack([c.camera_to_world[:3, 2] for c in cameras])
        sigma = max(2.0, len(path) / 150)
        path = gaussian_filter1d(path, sigma, axis=0, mode="nearest")
        forward = gaussian_filter1d(forward, 2 * sigma, axis=0, mode="nearest")
        up = frame.up

        vertical = forward @ up
        level = forward - vertical[:, None] * up
        size = np.linalg.norm(level, axis=1)
        pitch = np.clip(np.arctan2(0.4 * vertical, size), math.radians(-35), math.radians(15))
        # Where the phone looked straight down its heading is noise: keep the last good one.
        valid = size > 0.1 * np.linalg.norm(forward, axis=1)
        if valid.any():
            last = np.maximum.accumulate(np.where(valid, np.arange(len(path)), -1))
            last = np.where(last < 0, np.flatnonzero(valid)[0], last)
            heading = level[last] / size[last, None]
        else:
            heading = np.tile(frame.rotation[0], (len(path), 1))
        views = np.cos(pitch)[:, None] * heading + np.sin(pitch)[:, None] * up

        distance = np.linalg.norm(np.diff(path, axis=0), axis=1) / frame.camera_height
        turn = np.degrees(np.arccos(np.clip((views[1:] * views[:-1]).sum(1), -1, 1)))
        step = np.maximum(distance * seconds_per_height, turn / max_turn_deg_per_s)
        self.times = np.r_[0, np.cumsum(step)]
        self.duration = float(self.times[-1])
        self.path, self.views, self.up = path, views, up

    def pose(self, time: float) -> np.ndarray:
        """World-to-camera matrix at ``time`` seconds (clamped to the path's ends)."""
        position = np.array([np.interp(time, self.times, self.path[:, a]) for a in range(3)])
        look = np.array([np.interp(time, self.times, self.views[:, a]) for a in range(3)])
        return look_at(position, position + look / np.linalg.norm(look), self.up)


def scene_model(run: Path, log=print) -> tuple[Path, bool]:
    """``(path, filled)``: ``fill/scene.ply`` when it was made from the current splat, else
    ``splat/scene.ply``."""
    splat = run / "splat" / "scene.ply"
    record = run / "fill" / "fill.json"
    if (run / "fill" / "scene.ply").exists() and record.exists():
        if made_from(json.loads(record.read_text()), record, splat):
            return run / "fill" / "scene.ply", True
        log("the filled model is from an older splat; rendering the splat itself")
    return splat, False


def render_tour(run: Path, config: TourConfig, device: str, log) -> dict:
    run = Path(run)
    output = run / "tour"
    output.mkdir(exist_ok=True)
    path, filled = scene_model(run, log)
    model = load_ply(path, device)
    cameras, splits = load_cameras(run / "splat" / "cameras.json")
    if "support" not in model.extras:
        from .support import compute_support

        train_cameras = [c for c, split in zip(cameras, splits, strict=True) if split == "train"]
        model.extras["support"] = compute_support(model, train_cameras)
    frame = run_frame(run, cameras)
    centres = frame.to_aligned(np.stack([c.centre for c in cameras]))
    up = frame.up
    fps = config.fps
    writer = Writer(output / "tour.mp4", config)
    sections = []

    # 1. Overview orbit with the ceiling cut away.
    overview = cutaway(model, frame, centres)
    fov = 50.0
    renderer = Renderer(overview, config, fov, 0.01, frame.camera_height, (0.13, 0.13, 0.15))
    # Frame what the cutaway keeps (walls up to 1.6 camera heights beyond the path), at one
    # distance for the whole orbit: framed on the path, the orbit camera sat just above the
    # far walls of a room filmed from its middle (m-block).
    points = overview_points(overview, frame)
    low, high = np.percentile(points[:, :2], 2, axis=0), np.percentile(points[:, :2], 98, axis=0)
    middle = np.r_[(low + high) / 2, 0.3]
    elevation = math.radians(55)
    distance = framing_distance(
        points,
        middle,
        elevation,
        np.linspace(0, 2 * math.pi, 24, endpoint=False),
        _intrinsics(config.width, config.height, fov),
        config.width,
        config.height,
    )
    count = int(config.look_seconds * 1.4 * fps)
    for number in range(count):
        angle = 2 * math.pi * number / count
        eye = orbit_eye(middle, distance, elevation, angle)
        world_to_camera = look_at(frame.to_world(eye[None])[0], frame.to_world(middle[None])[0], up)
        writer.write(label(renderer(world_to_camera), "Overview"))
    sections.append(
        {
            "name": "overview",
            "frames": count,
            "distance_camera_heights": distance,
            "checks": renderer.summary(),
        }
    )
    del overview
    torch.cuda.empty_cache()

    # 2. Look-around in every area.
    renderer = Renderer(model, config, 75.0, 0.15 * frame.camera_height, frame.camera_height)
    train = [i for i, split in enumerate(splits) if split == "train"]
    for stand in filmed_standing_points(frame, cameras, train, config):
        renderer.reset()
        arc = stand["arc"]
        checked = None
        if arc is not None:
            eye = frame.to_world(np.array(stand["eye_aligned"])[None])[0]
            checked = _checked_arc(model, frame, eye, arc, config)
        if checked is None:
            sections.append(
                {"name": f"look-around {stand['area']}", "frames": 0, "skipped": arc or "no arc"}
            )
            continue
        (first, last), pitch, scores = checked
        frames = int(config.look_seconds * fps * min(1.0, (last - first) / math.radians(90)))
        half = ease(np.linspace(0, 1, max(frames // 2, 2)))
        yaws = np.concatenate((first + (last - first) * half, last + (first - last) * half))
        for yaw in yaws:
            direction = np.array(
                [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
            )
            world_to_camera = look_at(eye, eye + frame.rotation.T @ direction, up)
            writer.write(label(renderer(world_to_camera), f"Standing in {stand['area']}"))
        sections.append(
            {
                "name": f"look-around {stand['area']}",
                "frames": len(yaws),
                "sweep_deg": math.degrees(last - first),
                "pitch_deg": math.degrees(pitch),
                "nearby_cameras": arc["cameras"],
                "planned_checks": scores,
                "checks": renderer.summary(),
            }
        )

    # 3. Walkthrough along the smoothed capture path (training cameras, capture order).
    renderer.reset()
    walk = WalkPath(
        [cameras[i] for i in train],
        frame,
        config.walk_seconds_per_metre * 1.5,
        config.max_turn_deg_per_s,
    )
    frames = int(max(10.0, walk.duration) * fps)
    for time in np.linspace(0, walk.duration, frames):
        writer.write(label(renderer(walk.pose(time)), "Walkthrough"))
    sections.append(
        {
            "name": "walkthrough",
            "frames": frames,
            "checks": renderer.summary(),
        }
    )
    writer.close()

    report = {
        "video": str(output / "tour.mp4"),
        "model": "filled" if filled else "splat",
        "seconds": writer.frames / fps,
        "frames": writer.frames,
        "sections": sections,
    }
    write_json(output / "tour.json", report)
    log(
        f"tour: {report['seconds']:.0f}s, checks per section "
        f"{[(s['name'], s.get('checks', 'skipped')) for s in sections]}"
    )
    return report


def _heading(camera: Camera, frame: SceneFrame) -> float:
    forward = frame.rotation @ camera.camera_to_world[:3, 2]
    return math.atan2(forward[1], forward[0])


def filmed_standing_points(frame: SceneFrame, cameras, train: list[int], config: TourConfig):
    """Per area, the training camera position with the widest filmed arc around it.

    The area's middle is often a spot the phone only passed through, facing one way (in
    vyas-room the camera nearest each area's centre had a 10° arc, against 80–110° at the
    best spot). Ties go to more nearby cameras. The eye keeps that camera's height.
    """
    centres = frame.to_aligned(np.stack([c.centre for c in cameras]))
    allowed = set(train)
    stands = []
    for area in frame.areas:
        best, best_key = None, (-1.0, -1)
        for index in area["cameras"]:
            if index not in allowed:
                continue
            arc = filmed_arc(cameras, train, frame, centres[index], config)
            if arc is None:
                continue
            key = (round(arc["yaws"][1] - arc["yaws"][0], 3), arc["cameras"])
            if key > best_key:
                best, best_key = (index, arc), key
        if best is None:
            stands.append({"area": area["name"], "camera": None, "eye_aligned": None, "arc": None})
            continue
        index, arc = best
        stands.append(
            {
                "area": area["name"],
                "camera": index,
                "eye_aligned": centres[index].tolist(),
                "arc": arc,
            }
        )
    return stands


def filmed_arc(cameras, train: list[int], frame: SceneFrame, eye_aligned, config: TourConfig):
    """Yaw range and tilt that the capture filmed from near a standing point.

    Training cameras within ``look_radius`` camera heights (horizontally) of the eye vote
    for 10° heading bins. The longest circular run of filmed bins (gaps of one bin
    allowed) is the arc, trimmed by 10° at each end. The tilt is their median pitch, so
    the turn looks where the phone looked. Returns None with fewer than 5 such cameras.
    """
    centres = frame.to_aligned(np.stack([cameras[i].centre for i in train]))
    distance = np.linalg.norm(centres[:, :2] - np.asarray(eye_aligned)[:2], axis=1)
    nearby = [train[j] for j in np.flatnonzero(distance < config.look_radius)]
    if len(nearby) < 5:
        return None
    forward = np.stack([frame.rotation @ cameras[i].camera_to_world[:3, 2] for i in nearby])
    headings = np.degrees(np.arctan2(forward[:, 1], forward[:, 0])) % 360
    pitch = float(np.median(np.degrees(np.arcsin(np.clip(forward[:, 2], -1, 1)))))
    bins = 36
    filmed = np.zeros(bins, bool)
    filmed[(headings // 10).astype(int) % bins] = True
    filmed |= np.roll(filmed, 1) & np.roll(filmed, -1)  # bridge one-bin gaps
    if filmed.all():
        first, count = 0, bins
    else:
        first, count, run = 0, 0, 0
        for b in range(2 * bins):
            run = run + 1 if filmed[b % bins] else 0
            if count < run <= bins:
                first, count = b - run + 1, run
    if count == 0:
        return None
    low = math.radians(10 * first + 10)
    high = math.radians(10 * (first + count) - 10)
    if high <= low:
        return None
    return {
        "yaws": (low, high),
        "pitch_deg": float(np.clip(pitch, -40, 10)),
        "cameras": len(nearby),
    }


def _checked_arc(model, frame: SceneFrame, eye, arc: dict, config: TourConfig):
    """The longest part of ``arc`` whose views pass the checks, before anything is rendered.

    Each 5° view (320 × 180) must have at most ``max_unrendered`` pixels with alpha < HOLE_ALPHA
    and at most ``max_unmeasured`` rendered pixels that no capture camera saw from a
    nearby direction (filled Gaussians count as unmeasured). Returns ``((first, last),
    pitch, scores)`` in radians, or None when less than ``min_look_deg`` passes.
    """
    low, high = arc["yaws"]
    pitch = math.radians(arc["pitch_deg"])
    yaws = np.arange(low, high + 1e-6, math.radians(5))
    k = _intrinsics(320, 180, 75.0)
    passed, scores = [], []
    for yaw in yaws:
        direction = np.array(
            [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
        )
        camera = Camera(
            "probe", 320, 180, k, look_at(eye, eye + frame.rotation.T @ direction, frame.up)
        )
        fraction, alpha, _ = measured_support(model, camera)
        unrendered = float((alpha < HOLE_ALPHA).float().mean())
        rendered = alpha > 0.5
        unmeasured = float((fraction[rendered] < 0.5).float().mean()) if rendered.any() else 1.0
        scores.append(
            {"yaw_deg": math.degrees(yaw), "unrendered": unrendered, "unmeasured": unmeasured}
        )
        passed.append(unrendered <= config.max_unrendered and unmeasured <= config.max_unmeasured)
    best, best_first, run = 0, 0, 0
    for j, ok in enumerate(passed):
        run = run + 1 if ok else 0
        if run > best:
            best, best_first = run, j - run + 1
    if best == 0 or 5 * (best - 1) < config.min_look_deg:
        return None
    return (yaws[best_first], yaws[best_first + best - 1]), pitch, scores
