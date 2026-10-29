"""Video tour of a whole splat: build-up, cut-away overview, rooms and walkthrough.

A capture-path fly-through (`flythrough.py`) only shows what the phone saw. This tour also
looks at the model from outside, the way a 3D viewer would:

1. **Build-up:** top-down plan. Gaussians appear when a capture camera first sees them, so
   the model fills in as the camera path is drawn.
2. **Dollhouse orbit:** a 360° orbit from above with the ceiling cut away.
3. **Rooms:** a slow arc above each room, given as capture-time ranges.
4. **Walkthrough:** eye-level render along the smoothed capture path (the full, unfiltered
   model), beside the plan with the current position.

Scene frame: SfM world units. "Up" is the normal of a plane fitted to the camera centres,
signed to agree with the cameras' up direction (−y in OpenCV camera axes). Heights are
measured from the mean camera centre; the floor is the densest Gaussian height below the
cameras. Cut-away views drop Gaussians above the cut height, below the floor, far from
the camera path, very large, faint, or needle-shaped: these are never seen from above,
and render as streaks from outside the capture views.

Output: H.264 MP4 (yuv420p) written with PyAV.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

from .flythrough import smooth_path
from .model import GaussianModel
from .sfm_training import render
from .storage import load_model

ROOMS = (("Entrance", 0.0, 74.5), ("Living hall", 74.5, 180.0), ("Dining room", 180.0, 1e9))


@dataclass(frozen=True)
class SceneFrame:
    origin: np.ndarray  # mean camera centre
    up: np.ndarray
    across: np.ndarray  # main horizontal direction of the camera path
    side: np.ndarray  # up × across
    floor: float  # floor height relative to origin (negative)

    def height(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.up

    def plan(self, points: np.ndarray) -> np.ndarray:
        """Horizontal (across, side) coordinates relative to the origin."""
        relative = points - self.origin
        return np.stack((relative @ self.across, relative @ self.side), axis=-1)

    def point(self, across: float, side: float, height: float) -> np.ndarray:
        return self.origin + across * self.across + side * self.side + height * self.up


def scene_frame(camera_to_world: np.ndarray, means: np.ndarray, opacity: np.ndarray):
    centres = camera_to_world[:, :3, 3]
    origin = centres.mean(0)
    _, _, axes = np.linalg.svd(centres - origin, full_matrices=False)
    up = axes[2] * np.sign(axes[2] @ -camera_to_world[:, :3, 1].mean(0))
    across = axes[0] - (axes[0] @ up) * up
    across /= np.linalg.norm(across)

    # Floor: the densest height band of opaque Gaussians below the cameras.
    heights = (means - origin) @ up
    spread = np.ptp(centres @ up) + 1e-6
    below = heights[(opacity > 0.3) & (heights < -spread)]
    low, high = np.percentile(below, [1, 99]) if len(below) > 100 else (below.min(), below.max())
    counts, edges = np.histogram(below, bins=100, range=(low - 1e-3, high + 1e-3))
    floor = float(edges[np.argmax(counts)] + (edges[1] - edges[0]) / 2)
    return SceneFrame(origin, up, across, np.cross(up, across), floor)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV world-to-camera (x right, y down, z forward) looking from eye to target."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    down = -up - (-up @ forward) * forward
    down /= np.linalg.norm(down)
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = np.stack((np.cross(down, forward), down, forward))
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ eye
    return world_to_camera


def local_floor(frame: SceneFrame, means: np.ndarray, opacity: np.ndarray, cell: float = 0.25):
    """Floor height under each point, from a plan grid (cell size in camera heights).

    A floor that is not one plane in the reconstruction (a step, or drift in the camera
    solve) would otherwise be cut away where it lies below the global floor estimate.
    Each cell takes the densest height band of opaque Gaussians well below the cameras; a
    3 × 3 median removes cells dominated by furniture, and empty cells take the nearest.
    """
    camera_height = -frame.floor
    size = cell * camera_height
    heights = frame.height(means)
    low = (opacity > 0.3) & (heights < -0.5 * camera_height)
    keys = np.floor(frame.plan(means[low]) / size).astype(int)
    cells, inverse = np.unique(keys, axis=0, return_inverse=True)
    bins = np.arange(
        heights[low].min(), heights[low].max() + 0.03 * camera_height, 0.03 * camera_height
    )
    floors = {}
    for index, key in enumerate(map(tuple, cells)):
        values = heights[low][inverse.ravel() == index]
        if len(values) >= 10:
            counts, edges = np.histogram(values, bins=bins)
            floors[key] = edges[np.argmax(counts)] + (edges[1] - edges[0]) / 2
    smoothed = {}
    for (i, j), value in floors.items():
        near = [
            floors[(i + a, j + b)]
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            if (i + a, j + b) in floors
        ]
        smoothed[(i, j)] = float(np.median(near)) if len(near) >= 3 else value
    centres = (np.array(list(smoothed)) + 0.5) * size
    values = np.array(list(smoothed.values()))
    tree = cKDTree(centres)

    def floor_at(points: np.ndarray) -> np.ndarray:
        return values[tree.query(frame.plan(points))[1]]

    return floor_at


def cutaway_mask(
    frame: SceneFrame,
    means: np.ndarray,
    scales: np.ndarray,
    opacity: np.ndarray,
    centres: np.ndarray,
    cut_fraction: float = 1.25,
    floor_at=None,
) -> np.ndarray:
    """Gaussians kept in views from above; sizes are fractions of the camera height."""
    camera_height = -frame.floor
    floor = floor_at(means) if floor_at else frame.floor
    heights = frame.height(means)
    distance, _ = cKDTree(frame.plan(centres)).query(frame.plan(means))
    ordered = np.sort(scales, axis=1)
    needle = ordered[:, 2] / np.maximum(ordered[:, 1], 1e-9)
    keep = (
        (heights < floor + cut_fraction * camera_height)
        & (heights > floor - 0.05 * camera_height)  # reflections sit below glossy floors
        & (distance < 1.3 * camera_height)
        & (ordered[:, 2] < 0.08 * camera_height)
        & (opacity > 0.1)
        & (needle < 8)
    )

    # Floaters: Gaussians whose 8th-nearest kept neighbour is far away.
    kept = np.flatnonzero(keep)
    if len(kept) > 9:
        spacing, _ = cKDTree(means[kept]).query(means[kept], k=9)
        keep[kept[spacing[:, 8] > 0.06 * camera_height]] = False
    return keep


@torch.no_grad()
def first_seen(means: torch.Tensor, world_to_camera, intrinsics, sizes, times, max_depth):
    """Capture time at which each Gaussian centre first falls inside a camera's view."""
    seen = torch.full((len(means),), float("inf"), device=means.device)
    for index in np.argsort(times):
        w2c = torch.as_tensor(world_to_camera[index], dtype=torch.float32, device=means.device)
        k = torch.as_tensor(intrinsics[index], dtype=torch.float32, device=means.device)
        camera = means @ w2c[:3, :3].T + w2c[:3, 3]
        depth = camera[:, 2]
        pixels = camera[:, :2] / depth.clamp_min(1e-6)[:, None] * k.diagonal()[:2] + k[:2, 2]
        width, height = sizes[index]
        inside = (depth > 0.05) & (depth < max_depth)
        inside &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        inside &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        seen[inside & torch.isinf(seen)] = float(times[index])
    return seen


class _Video:
    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int):
        import av

        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "medium"}
        self.frames = 0

    def write(self, rgb: np.ndarray):
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _caption(image: np.ndarray, lines: list[str], origin=(40, 70), scale=1.1):
    x, y = origin
    for index, line in enumerate(lines):
        size = scale if index == 0 else scale * 0.7
        position = (x, y + int(index * 48 * scale))
        cv2.putText(
            image, line, position, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 6, cv2.LINE_AA
        )
        cv2.putText(
            image, line, position, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2, cv2.LINE_AA
        )
    return image


def _intrinsics(focal: float, width: int, height: int, device: str) -> torch.Tensor:
    k = [[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]]
    return torch.tensor(k, dtype=torch.float32, device=device)


def _to_image(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()


@torch.no_grad()
def render_tour(
    model_path: str | Path,
    run_dir: str | Path,
    output: str | Path,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    crf: int = 23,
    build_s: float = 14.0,
    orbit_s: float = 18.0,
    room_s: float = 7.0,
    keyframe_stride: int = 4,
    frames_per_key: int = 6,
    rooms=ROOMS,
    device: str = "cuda:0",
) -> dict:
    # Cameras (refined poses written by the trainer) and capture times.
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    cameras = json.loads((run_dir / "cameras.json").read_text())["cameras"]
    frames = json.loads((Path(run["sfm_dir"]) / "frames.json").read_text())
    time_of = {f["frame_id"]: f["timestamp_s"] for f in frames}
    world_to_camera = np.array([c["world_to_camera"] for c in cameras])
    camera_to_world = np.linalg.inv(world_to_camera)
    intrinsics = np.array([c["intrinsics"] for c in cameras])
    sizes = [(c["width"], c["height"]) for c in cameras]
    times = np.array([time_of[c["frame_id"]] for c in cameras])
    order = np.argsort(times)
    centres = camera_to_world[:, :3, 3]

    # Model, scene frame and the cut-away subset.
    model, metadata = load_model(model_path, device)
    degree = metadata.get("active_sh_degree", model.sh_degree)
    params = {key: value.detach() for key, value in model.params.items()}
    means = params["means"].cpu().numpy()
    opacity = params["opacities"].sigmoid().cpu().numpy()
    frame = scene_frame(camera_to_world, means, opacity)
    floor_at = local_floor(frame, means, opacity)
    keep = cutaway_mask(
        frame, means, params["scales"].exp().cpu().numpy(), opacity, centres, floor_at=floor_at
    )
    keep_t = torch.from_numpy(keep).to(device)
    cutaway = GaussianModel({k: v[keep_t] for k, v in params.items()}, model.sh_degree)
    camera_height = -frame.floor
    seen = first_seen(
        cutaway.params["means"], world_to_camera, intrinsics, sizes, times, 1.6 * camera_height
    )
    seen = torch.where(torch.isinf(seen), torch.tensor(times.max(), device=device), seen)
    logits = cutaway.params["opacities"].data.clone()

    def draw(model_, world_to_camera_, k, w, h, mode="antialiased"):
        w2c = torch.as_tensor(world_to_camera_, dtype=torch.float32, device=device)
        rgb, _, _, _ = render(model_, w2c, k, w, h, degree, mode)
        return _to_image(rgb)

    # Plan camera: high above the middle of the kept Gaussians, across-axis horizontal.
    plan = frame.plan(means[keep])
    low, high = np.percentile(plan, 3, axis=0), np.percentile(plan, 97, axis=0)
    middle = (low + high) / 2

    def plan_camera(w, h, margin=1.06):
        altitude = 12 * camera_height
        focal = altitude / margin * min(w / (high[0] - low[0]), h / (high[1] - low[1]))
        eye = frame.point(middle[0], middle[1], altitude)
        target = frame.point(middle[0], middle[1], frame.floor)
        w2c = look_at(eye, target, frame.side)  # image up = side, image right = across
        return w2c, _intrinsics(focal, w, h, device)

    def project(w2c, k, points):
        camera = points @ w2c[:3, :3].T + w2c[:3, 3]
        k = k.cpu().numpy()
        return camera[:, :2] / camera[:, 2:] * k.diagonal()[:2] + k[:2, 2]

    video = _Video(Path(output), width, height, fps, crf)
    sections = {}
    start_time, end_time = times.min(), times.max()

    # 1. Build-up on the plan.
    plan_w2c, plan_k = plan_camera(width, height)
    path_pixels = project(plan_w2c, plan_k, centres[order])
    count = round(build_s * fps)
    for index in range(count):
        now = start_time + (end_time - start_time) * min(1.0, index / (0.85 * count))
        cutaway.params["opacities"].data = torch.where(seen <= now, logits, logits.new_tensor(-30))
        image = draw(cutaway, plan_w2c, plan_k, width, height)
        visible = np.searchsorted(times[order], now, side="right")
        if visible > 1:
            points = path_pixels[:visible].round().astype(np.int32)
            cv2.polylines(image, [points], False, (255, 210, 0), 2, cv2.LINE_AA)
            cv2.circle(image, tuple(points[-1]), 9, (255, 60, 60), -1, cv2.LINE_AA)
        video.write(_caption(image, ["1. Building the 3D model", f"capture time {now:5.1f} s"]))
    cutaway.params["opacities"].data = logits
    sections["build_up"] = count

    # 2. Dollhouse orbit around the whole house.
    centre = frame.point(middle[0], middle[1], frame.floor)
    reach = 0.5 * np.linalg.norm(high - low)
    orbit_focal = 0.5 * height / np.tan(np.radians(26))
    count = round(orbit_s * fps)
    k = _intrinsics(orbit_focal, width, height, device)
    for index in range(count):
        angle = 2 * np.pi * index / count - np.pi / 2
        elevation = np.radians(52 + 10 * np.sin(2 * np.pi * index / count))
        distance = 0.75 * reach / np.tan(np.radians(26))
        offset = np.cos(angle) * frame.across + np.sin(angle) * frame.side
        eye = centre + distance * (np.cos(elevation) * offset + np.sin(elevation) * frame.up)
        image = draw(cutaway, look_at(eye, centre, frame.up), k, width, height)
        video.write(_caption(image, ["2. Whole house, ceiling cut away"]))
    sections["orbit"] = count

    # 3. Each room: a slow arc from above, centred on what the room's cameras looked at
    #    (one camera height ahead of each camera, projected on the plan).
    count = round(room_s * fps)
    forward = camera_to_world[:, :3, 2] - np.outer(camera_to_world[:, :3, 2] @ frame.up, frame.up)
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    looked_at = frame.plan(centres + camera_height * forward)
    for name, t0, t1 in rooms:
        inside = (times >= t0) & (times < t1)
        if inside.sum() < 10:
            continue
        room_low = np.percentile(looked_at[inside], 10, axis=0)
        room_high = np.percentile(looked_at[inside], 90, axis=0)
        room_middle = (room_low + room_high) / 2
        target = frame.point(room_middle[0], room_middle[1], frame.floor + 0.3 * camera_height)
        room_reach = 0.5 * np.linalg.norm(room_high - room_low) + 0.5 * camera_height
        distance = max(1.1 * room_reach / np.tan(np.radians(26)), 4 * camera_height)
        base = np.arctan2(room_middle[1] - middle[1], room_middle[0] - middle[0])
        t1_label = min(t1, end_time)
        for index in range(count):
            angle = base + np.radians(-50 + 100 * index / max(count - 1, 1))
            elevation = np.radians(70)
            offset = np.cos(angle) * frame.across + np.sin(angle) * frame.side
            eye = target + distance * (np.cos(elevation) * offset + np.sin(elevation) * frame.up)
            image = draw(cutaway, look_at(eye, target, frame.up), k, width, height)
            lines = [f"3. {name}", f"seen at {t0:.0f}-{t1_label:.0f} s of the video"]
            video.write(_caption(image, lines))
        sections[f"room:{name}"] = count

    # 4. Walkthrough at eye level (full model) beside the plan.
    side_h = height
    side_w = round(sizes[0][0] * side_h / sizes[0][1]) // 2 * 2
    map_w = width - side_w
    map_w2c, map_k = plan_camera(map_w, height, margin=1.04)
    plan_image = draw(cutaway, map_w2c, map_k, map_w, height)
    map_path = project(map_w2c, map_k, centres[order]).round().astype(np.int32)
    cv2.polylines(plan_image, [map_path], False, (255, 210, 0), 2, cv2.LINE_AA)
    first = cameras[order[0]]
    focal = first["intrinsics"][0][0] * side_h / first["height"]
    side_k = _intrinsics(focal, side_w, side_h, device)

    keys = order[::keyframe_stride]
    path = smooth_path(camera_to_world[keys], frames_per_key, window=5)
    path_times = np.interp(
        np.linspace(0, len(keys) - 1, len(path)), np.arange(len(keys)), times[keys]
    )
    for pose, now in zip(path, path_times):
        eye_image = draw(model, np.linalg.inv(pose), side_k, side_w, side_h, "classic")
        panel = plan_image.copy()
        here = project(map_w2c, map_k, np.stack((pose[:3, 3], pose[:3, 3] + 0.6 * pose[:3, 2])))
        here = here.round().astype(np.int32)
        cv2.arrowedLine(
            panel, tuple(here[0]), tuple(here[1]), (255, 60, 60), 4, cv2.LINE_AA, 0, 0.4
        )
        cv2.circle(panel, tuple(here[0]), 10, (255, 60, 60), -1, cv2.LINE_AA)
        room = next((name for name, t0, t1 in rooms if t0 <= now < t1), "")
        _caption(panel, ["4. Walkthrough (eye level)", f"{room}   {now:5.1f} s"])
        video.write(np.concatenate((eye_image, panel), axis=1))
    sections["walkthrough"] = len(path)

    video.close()
    report = {
        "output": str(output),
        "frames": video.frames,
        "seconds": video.frames / fps,
        "sections": sections,
        "gaussians": int(model.count),
        "cutaway_gaussians": int(keep.sum()),
        "floor_height": frame.floor,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render a video tour of a whole splat")
    parser.add_argument("model", type=Path, help="scene.ply or checkpoint.pt")
    parser.add_argument("run_dir", type=Path, help="Training run with run.json and cameras.json")
    parser.add_argument("--output", type=Path, required=True, help=".mp4 path")
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality (lower is better)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    report = render_tour(
        args.model, args.run_dir, args.output, crf=args.crf, fps=args.fps, device=args.device
    )
    print(report)
    return 0
