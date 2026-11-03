"""Rasterization on CUDA (gsplat) or on the CPU (PyTorch, same rules).

`rasterization` takes the keyword arguments of `gsplat.rasterization` that the pipeline
uses and returns what gsplat returns: colours [1,H,W,D] (plus the expected depth as a last
channel with ``render_mode="RGB+ED"``), alphas [1,H,W,1] and an info dict with ``radii``
[1,N,2], ``means2d`` [1,N,2] and ``depths`` [1,N]. CUDA tensors go to gsplat unchanged.

CPU tensors follow gsplat 1.5.3's rules, so a CPU render can stand in for a GPU one when
checking tours, doors or mirrors by eye:
- projection by gsplat's own PyTorch code (`gsplat.cuda._torch_impl`): the 0.3-pixel
  low-pass filter, the antialiasing compensation and radii of 3.33 σ per axis;
- per pixel, front to back by camera depth, at pixel centres (u + ½, v + ½);
- alpha = min(0.999, opacity · exp(−σ)); pairs with alpha below 1/255 are skipped;
- a pixel stops before the Gaussian that would take its transmittance to 1e-4 or below.

The CPU path is plain PyTorch, so gradients flow (`spacegen.support.blending_weight`
uses them), but it is far too slow to train on: a 480 × 270 frame of a 1.5M-Gaussian
room takes tens of seconds. Pixel–Gaussian pairs are processed in depth-ordered chunks
of at most ``PAIR_FLOATS`` values.
"""

from __future__ import annotations

import math

import torch

ALPHA_MIN = 1 / 255
ALPHA_MAX = 0.999
TRANSMITTANCE_MIN = 1e-4
PAIR_FLOATS = 2**27


def rasterization(**options):
    """`gsplat.rasterization` for CUDA tensors; the PyTorch rasterizer for CPU ones."""
    if options["means"].is_cuda:
        from gsplat import rasterization as gsplat_rasterization

        return gsplat_rasterization(**options)
    return cpu_rasterization(**options)


def spherical_harmonics(degree: int, directions: torch.Tensor, coefficients: torch.Tensor):
    """`gsplat.spherical_harmonics` (CUDA) or gsplat's PyTorch version of it (CPU)."""
    if directions.is_cuda:
        from gsplat import spherical_harmonics as gsplat_spherical_harmonics

        return gsplat_spherical_harmonics(degree, directions, coefficients)
    from gsplat.cuda._torch_impl import _spherical_harmonics

    return _spherical_harmonics(degree, directions, coefficients)


def cpu_rasterization(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,  # gsplat's argument name
    width: int,
    height: int,
    sh_degree: int | None = None,
    packed: bool = False,
    rasterize_mode: str = "classic",
    render_mode: str = "RGB",
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    backgrounds: torch.Tensor | None = None,
    pair_floats: int = PAIR_FLOATS,
):
    """One camera, as `gsplat.rasterization` (unpacked; RGB or RGB+ED). ``pair_floats``
    bounds the memory of one chunk of pixel–Gaussian pairs."""
    from gsplat.cuda._torch_impl import _fully_fused_projection, _quat_scale_to_covar_preci

    if viewmats.shape[0] != 1:
        raise ValueError("the CPU rasterizer renders one camera at a time")
    if render_mode not in ("RGB", "RGB+ED"):
        raise ValueError(f"render_mode {render_mode!r} is not supported on the CPU")

    covariances, _ = _quat_scale_to_covar_preci(quats, scales, compute_preci=False)
    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means,
        covariances,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=rasterize_mode == "antialiased",
    )
    strength = opacities if compensations is None else opacities * compensations[0]
    if sh_degree is None:
        features = colors
    else:
        centre = torch.linalg.inv(viewmats[0])[:3, 3]
        features = spherical_harmonics(sh_degree, means - centre, colors)
        features = torch.clamp_min(features + 0.5, 0.0)
    channels = features.shape[-1]
    if render_mode == "RGB+ED":
        features = torch.cat((features, depths[0][:, None]), -1)

    pixels = width * height
    total = features.new_zeros((pixels, features.shape[-1]))
    weight = features.new_zeros(pixels)
    log_transmittance = torch.zeros(pixels, dtype=torch.float64, device=means.device)
    chunks = _depth_chunks(
        radii[0], means2d[0], depths[0], width, height, features.shape[-1], pair_floats
    )
    for chunk in chunks:
        pixel, x, y, gaussian = _pairs(*chunk, width)
        dx = means2d[0, gaussian, 0] - (x.to(means2d.dtype) + 0.5)
        dy = means2d[0, gaussian, 1] - (y.to(means2d.dtype) + 0.5)
        conic = conics[0, gaussian]
        sigma = 0.5 * (conic[:, 0] * dx * dx + conic[:, 2] * dy * dy) + conic[:, 1] * dx * dy
        alpha = torch.clamp_max(strength[gaussian] * torch.exp(-sigma), ALPHA_MAX)
        keep = (sigma >= 0) & (alpha >= ALPHA_MIN)
        pixel, gaussian, alpha = pixel[keep], gaussian[keep], alpha[keep]
        if not len(pixel):
            continue

        # Pairs come in depth order; a stable sort by pixel keeps that order per pixel.
        pixel, order = torch.sort(pixel, stable=True)
        gaussian, alpha = gaussian[order], alpha[order]
        log_keep = torch.log1p(-alpha.double())
        inclusive = torch.cumsum(log_keep, 0)
        starts = torch.ones_like(pixel, dtype=torch.bool)
        starts[1:] = pixel[1:] != pixel[:-1]
        segment = torch.cumsum(starts, 0) - 1
        base = (inclusive - log_keep)[starts][segment]
        before = log_transmittance[pixel] + (inclusive - log_keep - base)
        after = log_transmittance[pixel] + (inclusive - base)
        composited = after > math.log(TRANSMITTANCE_MIN)
        visibility = torch.where(composited, torch.exp(before).to(alpha.dtype) * alpha, 0.0)

        total = total.index_add(0, pixel, features[gaussian] * visibility[:, None])
        weight = weight.index_add(0, pixel, visibility)
        log_transmittance = log_transmittance.index_add(0, pixel, log_keep)

    if backgrounds is not None:
        background = (1 - weight)[:, None] * backgrounds[0][None, :channels]
        total = torch.cat((total[:, :channels] + background, total[:, channels:]), -1)
    if render_mode == "RGB+ED":
        expected = total[:, channels:] / weight.clamp_min(1e-10)[:, None]
        total = torch.cat((total[:, :channels], expected), -1)
    info = {
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": strength[None],
        "width": width,
        "height": height,
    }
    return total.reshape(1, height, width, -1), weight.reshape(1, height, width, 1), info


def _depth_chunks(radii, means2d, depths, width: int, height: int, channels: int, floats: int):
    """Pixel boxes ``(gaussians, low corner, size, pixel count)`` of the projected
    Gaussians, nearest first, in chunks of at most ``floats`` values (one Gaussian's box
    may exceed it)."""
    visible = (radii > 0).all(-1).nonzero()[:, 0]
    visible = visible[torch.argsort(depths[visible].detach(), stable=True)]
    centre = means2d[visible].detach()
    reach = radii[visible].to(centre.dtype)
    low = (centre - reach).floor().clamp_min(0).long()
    high = (centre + reach).ceil().long()
    high[:, 0] = high[:, 0].clamp_max(width - 1)
    high[:, 1] = high[:, 1].clamp_max(height - 1)
    size = (high - low + 1).clamp_min(0)
    counts = size[:, 0] * size[:, 1]
    ends = torch.cumsum(counts, 0)
    budget = max(floats // (channels + 8), 1)
    start = 0
    while start < len(visible):
        limit = (int(ends[start - 1]) if start else 0) + budget
        end = max(int(torch.searchsorted(ends, limit, right=True)), start + 1)
        yield visible[start:end], low[start:end], size[start:end], counts[start:end]
        start = end


def _pairs(gaussians, low, size, counts, width: int):
    """Every (pixel, x, y, Gaussian) in the chunk's boxes, Gaussian by Gaussian."""
    owner = torch.repeat_interleave(torch.arange(len(gaussians), device=counts.device), counts)
    first = torch.cumsum(counts, 0) - counts
    local = torch.arange(len(owner), device=counts.device) - first[owner]
    columns = size[owner, 0]
    x = low[owner, 0] + local % columns
    y = low[owner, 1] + torch.div(local, columns, rounding_mode="floor")
    return y * width + x, x, y, gaussians[owner]
