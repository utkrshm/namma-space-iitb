"""Monocular depth for every undistorted image (Depth Anything V2 Small, Apache-2.0).

Photometric loss cannot place a surface without texture (blank walls, ceilings, glossy
floors): any depth renders the same colour. The network's *relative inverse depth* gives
those regions a shape. The prediction has an unknown scale and shift per image, so each
map is fitted to the SfM points its image observes (`align_depth_maps`). Training then
compares the render with *metric* inverse depth, and with the normals of that depth.
Images without a usable fit fall back to a Pearson correlation, which ignores scale and
shift (as in FSGS and DNGaussian).

Outputs:
- ``<out>/depth/<image stem>.npy``: float16 relative inverse depth on the image grid
  divided by ``downscale``;
- ``<out>/depth/alignment.json``: per image stem, ``a`` and ``b`` with inverse depth in
  SfM units = a · mono + b, the number of SfM points used and the median relative error.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..config import DepthConfig


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


def fit_inverse_depth(mono: np.ndarray, inverse: np.ndarray, rounds: int = 2):
    """``a, b`` with inverse depth ≈ a · mono + b, and the median relative residual.

    Least squares, refitted ``rounds`` times on the best 80% of the points: monocular
    depth is wrong at object edges and on some materials, and those points must not tilt
    the fit.
    """
    a, b = np.polyfit(mono, inverse, 1)
    for _ in range(rounds):
        residual = np.abs(a * mono + b - inverse)
        keep = residual <= np.percentile(residual, 80)
        a, b = np.polyfit(mono[keep], inverse[keep], 1)
    error = np.abs(a * mono + b - inverse) / np.maximum(inverse, 1e-9)
    return float(a), float(b), float(np.median(error))


def observed_points(camera, points: np.ndarray, size: tuple[int, int]):
    """Pixels (x, y) and depths of world ``points`` that land inside an image of ``size`` (w, h)."""
    width, height = size
    local = points @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
    z = local[:, 2]
    k = camera.resized(width, height).intrinsics
    uv = local[:, :2] / np.maximum(z[:, None], 1e-9) * np.diag(k)[:2] + k[:2, 2]
    pixel = np.round(uv).astype(int)
    inside = (
        (z > 0)
        & (pixel[:, 0] >= 0)
        & (pixel[:, 0] < width)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < height)
    )
    return pixel[inside], z[inside]


def align_depth_maps(solution, depth_dir, min_points: int = 30, max_error: float = 0.15) -> dict:
    """Fit each image's monocular inverse depth to the SfM points that image observes.

    Output: ``<depth_dir>/alignment.json``, ``{image stem: {"a", "b", "points", "error"}}``
    with inverse depth (SfM units) = a · mono + b. Images with fewer than ``min_points``
    points, a ≤ 0, or a median relative error above ``max_error`` get no entry; training
    falls back to a scale-free comparison for them. An existing file is kept.
    """
    target = Path(depth_dir) / "alignment.json"
    if target.exists():
        return json.loads(target.read_text())
    per_camera: list[list[int]] = [[] for _ in solution.cameras]
    for point, views in enumerate(solution.point_views):
        for view in views:
            per_camera[view].append(point)

    fits = {}
    for index, path in enumerate(solution.image_paths):
        depth_path = Path(depth_dir) / (path.stem + ".npy")
        if not depth_path.exists() or len(per_camera[index]) < min_points:
            continue
        mono = np.load(depth_path).astype(np.float32)
        height, width = mono.shape
        pixel, z = observed_points(
            solution.cameras[index], solution.points[per_camera[index]], (width, height)
        )
        if len(z) < min_points:
            continue
        a, b, error = fit_inverse_depth(mono[pixel[:, 1], pixel[:, 0]], 1 / z)
        if a > 0 and error <= max_error:
            fits[path.stem] = {"a": a, "b": b, "points": len(z), "error": error}
    target.write_text(json.dumps(fits, indent=1) + "\n")
    return fits


def depth_normals(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unit normals [H,W,3] (camera frame, facing the camera) of a z-depth map [H,W].

    Back-projects every pixel and crosses the central differences along x and y; border
    pixels repeat their neighbour.
    """
    height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    points = torch.stack(
        (
            (xs - intrinsics[0, 2]) / intrinsics[0, 0] * depth,
            (ys - intrinsics[1, 2]) / intrinsics[1, 1] * depth,
            depth,
        ),
        dim=-1,
    )
    padded = torch.nn.functional.pad(points.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[
        0
    ]
    along_x = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]).permute(1, 2, 0)
    along_y = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]).permute(1, 2, 0)
    normals = torch.nn.functional.normalize(torch.cross(along_x, along_y, dim=-1), dim=-1)
    facing = (normals * points).sum(-1, keepdim=True) > 0
    return torch.where(facing, -normals, normals)


def pearson_loss(rendered_depth: torch.Tensor, prior: torch.Tensor, valid: torch.Tensor):
    """1 − correlation between rendered inverse depth and the prior over valid pixels."""
    if valid.sum() < 16:
        return rendered_depth.new_zeros(())
    inverse = 1 / rendered_depth.clamp_min(1e-6)
    x, y = inverse[valid], prior[valid]
    x, y = x - x.mean(), y - y.mean()
    correlation = (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-8)
    return 1 - correlation
