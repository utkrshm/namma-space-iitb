"""Fly-through video of a splat along the capture trajectory.

The path follows the SfM cameras in time order. It is smoothed (moving average of
centres; spherical interpolation of orientations between keyframes) so the video shows
the reconstruction rather than the hand-held shake of the capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, Slerp

from .sfm_scene import load_sfm_scene
from .sfm_training import render
from .storage import load_model


def smooth_path(camera_to_world: np.ndarray, frames_per_key: int, window: int) -> np.ndarray:
    """Interpolated, smoothed camera-to-world poses between consecutive keyframes."""
    centres = uniform_filter1d(camera_to_world[:, :3, 3], size=window, axis=0, mode="nearest")
    keys = np.arange(len(camera_to_world))
    rotations = Rotation.from_matrix(camera_to_world[:, :3, :3])
    times = np.linspace(0, len(keys) - 1, (len(keys) - 1) * frames_per_key + 1)
    path = np.tile(np.eye(4), (len(times), 1, 1))
    path[:, :3, :3] = Slerp(keys, rotations)(times).as_matrix()
    for axis in range(3):
        path[:, axis, 3] = np.interp(times, keys, centres[:, axis])
    return path


@torch.no_grad()
def render_flythrough(
    model_path: str | Path,
    sfm_dir: str | Path,
    output: str | Path,
    *,
    keyframe_stride: int = 4,
    frames_per_key: int = 6,
    window: int = 5,
    height: int = 960,
    fps: int = 30,
    time_range_s: tuple[float, float] | None = None,
    device: str = "cuda:0",
) -> int:
    model, metadata = load_model(model_path, device)
    degree = metadata.get("active_sh_degree", model.sh_degree)
    scene = load_sfm_scene(sfm_dir, time_range_s=time_range_s)
    keys = np.stack([c.camera_to_world for c in scene.cameras[::keyframe_stride]])
    path = smooth_path(keys, frames_per_key, window)

    reference = scene.cameras[0]
    width = round(reference.width * height / reference.height) // 2 * 2
    camera = reference.resized(width, height)
    k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for pose in path:
        world_to_camera = torch.tensor(np.linalg.inv(pose), dtype=torch.float32, device=device)
        rgb, _, _, _ = render(model, world_to_camera, k, width, height, degree, "classic")
        frame = (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()
        writer.write(np.ascontiguousarray(frame[..., ::-1]))
    writer.release()
    return len(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render a smoothed fly-through video")
    parser.add_argument("model", type=Path, help="scene.ply or checkpoint.pt")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help=".mp4 path")
    parser.add_argument("--keyframe-stride", type=int, default=4)
    parser.add_argument("--frames-per-key", type=int, default=6)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--time-range", type=float, nargs=2, help="Source seconds [start end]")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    count = render_flythrough(
        args.model, args.sfm_dir, args.output,
        keyframe_stride=args.keyframe_stride, frames_per_key=args.frames_per_key,
        height=args.height, fps=args.fps,
        time_range_s=tuple(args.time_range) if args.time_range else None, device=args.device,
    )  # fmt: skip
    print(f"Wrote {count} frames to {args.output}")
    return 0
