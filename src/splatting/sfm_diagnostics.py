"""Where a splat fails: per-time-segment fit and Gaussian statistics from a checkpoint.

A splat can look fine on average and still be broken in one part of a capture (a bad
camera stretch, moving people, a window) or everywhere (fog of large transparent
Gaussians). This renders training views at low resolution with their learned camera and
colour corrections, reports PSNR per source-time segment, and summarizes Gaussian
extents and opacities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import GaussianModel
from .sfm_scene import load_sfm_scene, read_rgb
from .sfm_training import CameraCorrections, render


@torch.no_grad()
def diagnose(
    checkpoint: str | Path,
    sfm_dir: str | Path,
    *,
    downscale: int = 8,
    stride: int = 3,
    segment_s: float = 10.0,
    device: str = "cuda:0",
) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GaussianModel(state["params"], state["sh_degree"]).to(device)
    scene = load_sfm_scene(sfm_dir)
    corrections = CameraCorrections(len(scene.cameras)).to(device)
    corrections.load_state_dict(state["corrections"])
    degree = min(model.sh_degree, int(state["step"]) // 1000)

    rows = []
    for index in scene.train_indices[::stride]:
        target = torch.from_numpy(read_rgb(scene.image_paths[index], downscale)).to(device)
        target = target.float() / 255
        height, width = target.shape[:2]
        camera = scene.cameras[index].resized(width, height)
        base = torch.tensor(camera.camera_to_world, dtype=torch.float32, device=device)
        world_to_camera = torch.linalg.inv(corrections.camera_to_world(index, base))
        k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
        rgb, _, _, _ = render(model, world_to_camera, k, width, height, degree, "classic")
        rgb = corrections.apply_colour(index, rgb).clamp(0, 1)
        psnr = float(-10 * torch.log10((rgb - target).square().mean().clamp_min(1e-12)))
        rows.append((float(scene.timestamps[index]), psnr))

    times, psnrs = np.array(rows).T
    segments = []
    for start in np.arange(0, times.max() + segment_s, segment_s):
        inside = (times >= start) & (times < start + segment_s)
        if inside.any():
            segments.append(
                {
                    "start_s": float(start),
                    "views": int(inside.sum()),
                    "psnr_mean": float(psnrs[inside].mean()),
                    "psnr_min": float(psnrs[inside].min()),
                }
            )
    extent = model.params["scales"].exp().max(dim=-1).values
    quantiles = torch.tensor([0.5, 0.9, 0.99, 0.999], device=device)
    return {
        "step": int(state["step"]),
        "gaussians": model.count,
        "psnr_mean": float(psnrs.mean()),
        "segments": segments,
        "extent_quantiles": dict(
            zip(("q50", "q90", "q99", "q999"), extent.quantile(quantiles).tolist())
        ),
        "opacity_mean": float(model.params["opacities"].sigmoid().mean()),
        "opacity_below_0.05": float((model.params["opacities"].sigmoid() < 0.05).float().mean()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Per-segment fit and Gaussian statistics")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    report = diagnose(
        args.checkpoint, args.sfm_dir, downscale=args.downscale, stride=args.stride,
        device=args.device,
    )  # fmt: skip
    for segment in report.pop("segments"):
        print(
            f"{segment['start_s']:6.0f}s  views={segment['views']:3d}  "
            f"psnr={segment['psnr_mean']:5.1f}  min={segment['psnr_min']:5.1f}"
        )
    print(json.dumps(report, indent=1))
    return 0
