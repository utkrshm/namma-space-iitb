"""Monocular depth for every undistorted image (Depth Anything V2 Small, Apache-2.0).

Photometric loss cannot place a surface without texture (blank walls, ceilings, glossy
floors): any depth renders the same colour. The network's *relative inverse depth* gives
those regions a shape. Training compares it with the render through a Pearson
correlation, which ignores the prediction's unknown scale and shift (as in FSGS and
DNGaussian).

Output: ``<out>/depth/<image stem>.npy``, float16 relative inverse depth on the image grid
divided by ``downscale``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .config import DepthConfig


def read_rgb(path: Path, long_side: int | None = None, downscale: int = 1) -> np.ndarray:
    """uint8 RGB, area-resized to ``long_side`` and/or by an integer factor."""
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = 1.0
    if long_side is not None and max(height, width) > long_side:
        scale = long_side / max(height, width)
    scale /= downscale
    if scale < 1:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


@torch.no_grad()
def compute_depth(image_paths: list[Path], output_dir, config: DepthConfig, device="cuda:0"):
    """Write one ``.npy`` per image; existing files are kept."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    processor = model = None
    for path in image_paths:
        output = target / (Path(path).stem + ".npy")
        if output.exists():
            continue
        if model is None:
            processor = AutoImageProcessor.from_pretrained(config.model)
            model = AutoModelForDepthEstimation.from_pretrained(config.model).eval().to(device)
        rgb = read_rgb(path, downscale=config.downscale)
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        predicted = model(**inputs).predicted_depth[None]
        resized = torch.nn.functional.interpolate(
            predicted, size=rgb.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
        np.save(output, resized.cpu().numpy().astype(np.float16))
    del model
    return target


_MODELS: dict[str, tuple] = {}


@torch.no_grad()
def predict_depth(rgb: np.ndarray, model_name: str, device="cuda:0") -> np.ndarray:
    """Relative inverse depth of one RGB image (used by the fill stage)."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if model_name not in _MODELS:
        _MODELS[model_name] = (
            AutoImageProcessor.from_pretrained(model_name),
            AutoModelForDepthEstimation.from_pretrained(model_name).eval().to(device),
        )
    processor, model = _MODELS[model_name]
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    predicted = model(**inputs).predicted_depth[None]
    return (
        torch.nn.functional.interpolate(
            predicted, size=rgb.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
    )


def pearson_loss(rendered_depth: torch.Tensor, prior: torch.Tensor, valid: torch.Tensor):
    """1 − correlation between rendered inverse depth and the prior over valid pixels."""
    if valid.sum() < 16:
        return rendered_depth.new_zeros(())
    inverse = 1 / rendered_depth.clamp_min(1e-6)
    x, y = inverse[valid], prior[valid]
    x, y = x - x.mean(), y - y.mean()
    correlation = (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)
    return 1 - correlation
