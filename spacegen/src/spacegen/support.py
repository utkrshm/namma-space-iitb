"""View-direction support: from which directions did the capture see each Gaussian?

A splat is only measured from the directions its training cameras looked from. Seen from
elsewhere, a Gaussian fitted to a wall's front can smear across the room (Coorg: 360°
turns from a room's middle). So each Gaussian gets a 64-bit mask over 64 near-uniform
directions (a Fibonacci sphere). Bit d is set when a training camera in direction d saw
the Gaussian with at least ``min_pixels`` of blending weight. The direction is
camera → Gaussian, in world axes.

For any new view, a Gaussian is *supported* when some set bit lies within ``angle_deg``
of the current camera → Gaussian direction. Rendering that 0/1 flag with the splat's own
blending weights gives a per-pixel *support* map. Pixels with low support (or low alpha)
show content the capture never measured from here: the fill stage replaces them, and
the tour can mark them.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .contracts import Camera
from .splat.model import GaussianModel, render

DIRECTIONS = 64


def fibonacci_directions(count: int = DIRECTIONS) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * index / count)
    azimuth = np.pi * (1 + 5**0.5) * index
    return np.stack(
        (np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)), axis=1
    )


def _bit_masks(directions: torch.Tensor, angle_deg: float) -> torch.Tensor:
    """For each direction bin, an int64 mask of the bins within ``angle_deg`` of it."""
    near = (directions @ directions.T) >= math.cos(math.radians(angle_deg))
    weights = torch.tensor(
        [1 << b if b < 63 else -(1 << 63) for b in range(DIRECTIONS)], dtype=torch.int64
    )
    return (near.cpu().long() * weights).sum(1).to(directions.device)


def _direction_bins(model: GaussianModel, centre: torch.Tensor, directions: torch.Tensor):
    offset = model.params["means"].detach() - centre
    offset = offset / offset.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (offset @ directions.T).argmax(1)


def blending_weight(model: GaussianModel, camera: Camera, mode="antialiased") -> torch.Tensor:
    """Per-Gaussian total blending weight in pixels (gradient of a unit colour)."""
    unit = torch.ones((model.count, 1), device=model.device, requires_grad=True)
    rgb, _, _, _ = _render_feature(model, camera, unit, mode)
    # Only d/d(unit): backward() would also add gradients to the model's parameters, which
    # a later optimizer step (fill's distillation) would then apply.
    (weight,) = torch.autograd.grad(rgb.sum(), unit)
    return weight[:, 0].detach()


def _render_feature(model, camera: Camera, feature: torch.Tensor, mode="antialiased"):
    device = model.device
    world_to_camera = torch.tensor(camera.world_to_camera, dtype=torch.float32, device=device)
    k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
    return render(model, world_to_camera, k, camera.width, camera.height, mode=mode, colors=feature)


def compute_support(
    model: GaussianModel,
    cameras: list[Camera],
    long_side: int = 480,
    min_pixels: float = 0.5,
    max_cameras: int = 400,
) -> torch.Tensor:
    """int64 direction mask per Gaussian (stored as ``model.extras["support"]``)."""
    device = model.device
    directions = torch.tensor(fibonacci_directions(), dtype=torch.float32, device=device)
    weights = torch.tensor(
        [1 << b if b < 63 else -(1 << 63) for b in range(DIRECTIONS)],
        dtype=torch.int64,
        device=device,
    )
    mask = torch.zeros(model.count, dtype=torch.int64, device=device)
    chosen = np.linspace(0, len(cameras) - 1, min(max_cameras, len(cameras))).round().astype(int)
    for index in np.unique(chosen):
        camera = cameras[index]
        scale = long_side / max(camera.width, camera.height)
        small = camera.resized(round(camera.width * scale), round(camera.height * scale))
        seen = blending_weight(model, small) >= min_pixels
        centre = torch.tensor(camera.centre, dtype=torch.float32, device=device)
        bins = _direction_bins(model, centre, directions)
        mask |= torch.where(seen, weights[bins], torch.zeros_like(mask))
    return mask


@torch.no_grad()
def supported(model: GaussianModel, camera: Camera, angle_deg: float) -> torch.Tensor:
    """0/1 per Gaussian: seen by the capture from within ``angle_deg`` of this view."""
    device = model.device
    directions = torch.tensor(fibonacci_directions(), dtype=torch.float32, device=device)
    centre = torch.tensor(camera.centre, dtype=torch.float32, device=device)
    bins = _direction_bins(model, centre, directions)
    neighbours = _bit_masks(directions, angle_deg)[bins]
    return ((model.extras["support"] & neighbours) != 0).float()


@torch.no_grad()
def support_map(model: GaussianModel, camera: Camera, angle_deg: float, mode="antialiased"):
    """Per-pixel (support fraction, alpha, depth): support is the share of rendered weight
    that comes from supported Gaussians (0 where nothing renders)."""
    flag = supported(model, camera, angle_deg)[:, None]
    value, depth, alpha, _ = _render_feature(model, camera, flag, mode)
    fraction = value[..., 0] / alpha.clamp_min(1e-6)
    return fraction.clamp(0, 1) * (alpha > 1e-3), alpha, depth
