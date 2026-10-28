"""Monocular depth prior for textureless regions (blank ceilings, plain walls).

Photometric loss cannot place Gaussians on a surface without texture: any depth renders
the same colour. A monocular network (Depth Anything V2 Small, Apache-2.0) predicts relative
inverse depth for every frame. Training adds ``1 − Pearson(rendered inverse depth,
predicted)`` over covered pixels, which is invariant to the unknown scale and shift of the
prediction (as in FSGS / DNGaussian).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .sfm_scene import read_rgb

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@torch.no_grad()
def compute_mono_depth(sfm_dir: str | Path, downscale: int = 4, device: str = "cuda:0") -> Path:
    """Write ``<sfm_dir>/mono_depth/<image stem>.npy`` (float16 inverse depth, image grid / downscale)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    root = Path(sfm_dir).resolve()
    images = sorted((root / "undistorted" / "images").glob("*.jpg"))
    target = root / "mono_depth"
    target.mkdir(exist_ok=True)
    processor = model = None
    for path in images:
        output = target / (path.stem + ".npy")
        if output.exists():
            continue
        if model is None:
            processor = AutoImageProcessor.from_pretrained(MODEL)
            model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval().to(device)
        rgb = read_rgb(path, downscale)
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        predicted = model(**inputs).predicted_depth[None]
        resized = torch.nn.functional.interpolate(
            predicted, size=rgb.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
        np.save(output, resized.cpu().numpy().astype(np.float16))
    return target


def pearson_loss(rendered_depth: torch.Tensor, prior: torch.Tensor, valid: torch.Tensor):
    """1 − correlation between rendered inverse depth and the prior over valid pixels."""
    if valid.sum() < 16:
        return rendered_depth.new_zeros(())
    inverse = 1 / rendered_depth.clamp_min(1e-3)
    x, y = inverse[valid], prior[valid]
    x, y = x - x.mean(), y - y.mean()
    correlation = (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)
    return 1 - correlation


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Precompute monocular depth for an SfM dir")
    parser.add_argument("sfm_dir", type=Path)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    print(compute_mono_depth(args.sfm_dir, args.downscale, args.device))
    return 0
