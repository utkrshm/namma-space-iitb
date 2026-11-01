"""Planar reflections: a mirrored copy of the scene seen through floors, mirrors and glass.

Without this layer a splat explains a reflection on a polished floor or a mirror with
geometry *behind* the surface (a virtual room under the floor). That breaks the floor
as soon as a depth prior holds it in place, and the reflection is then painted onto the
surface: it no longer moves with the viewer. Following MirrorGaussian (Liu et al., ECCV
2024) and TR-Gaussians (Liu et al., 2025), a reflector is a plane, and what it reflects
is the scene's own Gaussians mirrored across that plane:

    image = T + Σ_k S_k · F_k(θ) · R_k

- ``T``: the ordinary render (the surface itself, and whatever is seen through glass);
- ``S_k``: rendered with ``T``'s blending weights from each Gaussian's reflection
  strength for plane ``k``, ``sigmoid(reflect) · exp(−(distance to plane / tau)²)``. Only
  Gaussians on the plane can carry it, and anything standing in front of the reflector
  hides it as it hides the surface;
- ``F_k(θ)``: Schlick's Fresnel term, ``f0 + (1 − f0)(1 − cos θ)⁵``, with θ the angle
  between the viewing ray and the plane normal and ``f0`` learned per plane: polished
  stone reflects more at grazing angles, a mirror reflects everything;
- ``R_k``: the Gaussians in front of the plane (farther than ``tau``), mirrored across
  it, rendered at half resolution (reflections on stone and tiles are soft).

The rendered depth is ``T``'s, so depth priors keep holding the surface on its plane.

Planes are ``(n, d)`` with ``n · x + d = 0`` in world units and ``n`` pointing to the side
the cameras are on. A model's planes live in ``GaussianModel.reflectors`` (``planes``
[K,4], ``f0`` logits [K], ``tau``) and its per-Gaussian logits in ``params["reflect"]``
[N,K]; on disk, the logits are ``reflect_<k>`` PLY properties and the planes a
``<ply stem>.reflectors.json`` file next to the PLY.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Initial per-Gaussian reflection-strength logit and per-plane Fresnel f0.
INITIAL_LOGIT = -2.0
INITIAL_F0 = 0.1
# Logit given to Gaussians that join a model without a reflection strength (fill).
ABSENT_LOGIT = -10.0


def make_reflectors(
    planes, tau: float, device, axes=None, bounds=None, kinds=None, opaque=None
) -> dict:
    """Reflectors on ``device``. Optional, per plane: ``axes`` [K,2,3] (two in-plane unit
    vectors) with ``bounds`` [K,4] (u min, u max, v min, v max, world units), outside of
    which the plane reflects nothing; ``kinds`` (names); ``opaque`` (nothing real lies
    behind it: a floor or a mirror, not a window)."""
    planes = torch.as_tensor(planes, dtype=torch.float32, device=device).reshape(-1, 4)
    count = len(planes)
    f0 = torch.full((count,), math.log(INITIAL_F0 / (1 - INITIAL_F0)), device=device)
    if axes is None:
        axes = torch.zeros((count, 2, 3))
        bounds = torch.tensor([[-math.inf, math.inf, -math.inf, math.inf]] * count)
    return {
        "planes": planes,
        "f0": f0,
        "tau": float(tau),
        "axes": torch.as_tensor(axes, dtype=torch.float32, device=device).reshape(-1, 2, 3),
        "bounds": torch.as_tensor(bounds, dtype=torch.float32, device=device).reshape(-1, 4),
        "kinds": list(kinds) if kinds is not None else ["floor"] * count,
        "opaque": torch.as_tensor(
            [True] * count if opaque is None else opaque, dtype=torch.bool, device=device
        ),
    }


def reflectors_to_dict(reflectors: dict) -> dict:
    def finite(values):
        return [
            [v if math.isfinite(v) else (1e30 if v > 0 else -1e30) for v in row] for row in values
        ]

    return {
        "planes": reflectors["planes"].detach().cpu().tolist(),
        "f0": torch.sigmoid(reflectors["f0"].detach()).cpu().tolist(),
        "tau": reflectors["tau"],
        "axes": reflectors["axes"].cpu().tolist(),
        "bounds": finite(reflectors["bounds"].cpu().tolist()),
        "kinds": reflectors["kinds"],
        "opaque": reflectors["opaque"].cpu().tolist(),
    }


def reflectors_from_dict(value: dict, device) -> dict:
    reflectors = make_reflectors(
        value["planes"],
        value["tau"],
        device,
        value.get("axes"),
        value.get("bounds"),
        value.get("kinds"),
        value.get("opaque"),
    )
    if "f0" in value:
        f0 = torch.tensor(value["f0"], dtype=torch.float32, device=device).clamp(1e-4, 1 - 1e-4)
        reflectors["f0"] = torch.log(f0 / (1 - f0))
    return reflectors


def within_bounds(points: torch.Tensor, reflectors: dict, margin: float = 0.0) -> torch.Tensor:
    """[N,K]: the point's projection onto each plane lies inside that plane's bounds."""
    axes, bounds = reflectors["axes"], reflectors["bounds"]
    u = torch.einsum("nc,kc->nk", points, axes[:, 0])
    v = torch.einsum("nc,kc->nk", points, axes[:, 1])
    return (
        (u >= bounds[:, 0] - margin)
        & (u <= bounds[:, 1] + margin)
        & (v >= bounds[:, 2] - margin)
        & (v <= bounds[:, 3] + margin)
    )


def behind_opaque(points: torch.Tensor, reflectors: dict) -> torch.Tensor:
    """[N]: the point lies behind (beyond ``tau``) a floor or mirror, inside its bounds:
    a reflection seen as a room behind the mirror, not a real surface."""
    planes = reflectors["planes"]
    behind = (points @ planes[:, :3].T + planes[:, 3]) < -reflectors["tau"]
    return (behind & within_bounds(points, reflectors) & reflectors["opaque"]).any(-1)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions (broadcasting)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def mirror_quats(quats: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Rotations of Gaussians mirrored across a plane with unit ``normal``.

    The mirrored covariance is M R S² Rᵀ M with M = I − 2nnᵀ. M R has determinant −1, but
    M R F with F = diag(1, 1, −1) is a rotation with the same covariance (S² is diagonal).
    As M = −Rot(n, π) and F = −Rot(z, π), M R F = Rot(n, π) R Rot(z, π): in quaternions
    (0, n) ⊗ q ⊗ (0, 0, 0, 1). Elementwise, unlike a batched 3 × 3 product, which ran
    13× slower than the whole render on 1.5M Gaussians.
    """
    along_normal = torch.cat((normal.new_zeros(1), normal))
    about_z = normal.new_tensor([0.0, 0.0, 0.0, 1.0])
    return quat_multiply(quat_multiply(along_normal, quats), about_z)


def mirror(points: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    """``points`` [..., 3] reflected across ``plane`` (n, d)."""
    normal, offset = plane[:3], plane[3]
    return points - 2 * ((points @ normal) + offset)[..., None] * normal


def strengths(params: dict, reflectors: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """``(strength [N,K], signed distance [N,K])`` of each Gaussian for each plane."""
    planes = reflectors["planes"]
    distance = params["means"] @ planes[:, :3].T + planes[:, 3]
    gate = torch.exp(-((distance / reflectors["tau"]) ** 2))
    gate = gate * within_bounds(params["means"], reflectors, reflectors["tau"])
    return torch.sigmoid(params["reflect"]) * gate, distance


def sh_colours(params: dict, degree: int, viewpoint: torch.Tensor, keep=None) -> torch.Tensor:
    """RGB [N,3] of the spherical harmonics seen from ``viewpoint``, as gsplat computes them
    (only for the Gaussians in ``keep``, when given)."""
    from gsplat import spherical_harmonics

    coeffs = torch.cat((params["sh0"], params["shn"][:, : (degree + 1) ** 2 - 1]), dim=1)
    means = params["means"]
    if keep is not None:
        means, coeffs = means[keep], coeffs[keep]
    colours = spherical_harmonics(degree, means - viewpoint, coeffs)
    return torch.clamp_min(colours + 0.5, 0.0)


def in_view(reflectors: dict, world_to_camera, k, width: int, height: int) -> list[int]:
    """Planes whose extent may appear in the image (unbounded planes always may)."""
    chosen = []
    for index, (plane, axes, bounds) in enumerate(
        zip(reflectors["planes"], reflectors["axes"], reflectors["bounds"], strict=True)
    ):
        if not bool(torch.isfinite(bounds).all()) or float(bounds.abs().max()) > 1e20:
            chosen.append(index)
            continue
        base = -plane[3] * plane[:3]
        corners = torch.stack(
            [base + u * axes[0] + v * axes[1] for u in bounds[:2] for v in bounds[2:]]
        )
        local = corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        if bool((local[:, 2] <= 0).all()):
            continue
        if bool((local[:, 2] <= 0).any()):
            chosen.append(index)  # straddles the camera plane: keep, to be safe
            continue
        x = local[:, 0] / local[:, 2] * k[0, 0] + k[0, 2]
        y = local[:, 1] / local[:, 2] * k[1, 1] + k[1, 2]
        if x.max() >= 0 and x.min() <= width and y.max() >= 0 and y.min() <= height:
            chosen.append(index)
    return chosen


def render_with_reflections(
    model,
    world_to_camera: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    degree: int,
    mode: str,
    near_plane: float,
    background,
):
    """`spacegen.splat.render` for a model with reflectors; ``k`` is already in gsplat's
    pixel convention. Returns RGB, depth, alpha and the main pass's info."""
    from gsplat import rasterization

    params = model.params
    reflectors = model.reflectors
    planes = reflectors["planes"]
    count = len(planes)
    camera_to_world = torch.linalg.inv(world_to_camera)
    centre = camera_to_world[:3, 3]

    strength, distance = strengths(params, reflectors)
    visible = in_view(reflectors, world_to_camera, k, width, height)
    strength = strength[:, visible]
    count = len(visible)
    colours = sh_colours(params, degree, centre)
    opacities = params["opacities"].sigmoid()
    scales = params["scales"].exp()
    output, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=scales,
        opacities=opacities,
        colors=torch.cat((colours, strength), -1),
        viewmats=world_to_camera[None],
        Ks=k[None],
        width=width,
        height=height,
        sh_degree=None,
        packed=False,
        rasterize_mode=mode,
        render_mode="RGB+ED",
        near_plane=near_plane,
        backgrounds=None
        if background is None
        else torch.cat((background, background.new_zeros(count)))[None],
    )
    output, alpha = output[0], alpha[0, ..., 0]
    rgb, maps, depth = output[..., :3], output[..., 3 : 3 + count], output[..., -1]

    # Per-pixel cos θ between the viewing ray and each plane normal.
    ys, xs = torch.meshgrid(
        torch.arange(height, device=k.device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=k.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    rays = torch.stack(
        ((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], torch.ones_like(xs)), -1
    )
    rays = F.normalize(rays @ camera_to_world[:3, :3].T, dim=-1)

    for channel, index in enumerate(visible):
        plane = planes[index]
        side = (centre @ plane[:3]) + plane[3]
        region = maps[..., channel].detach() > 1e-3
        if side <= 0 or not region.any():
            continue
        # Render the reflection at half resolution, only in the box around the reflector.
        rows = region.any(1).nonzero()[:, 0]
        columns = region.any(0).nonzero()[:, 0]
        top, bottom = int(rows[0]) // 2, (int(rows[-1]) + 2) // 2
        left, right = int(columns[0]) // 2, (int(columns[-1]) + 2) // 2
        box_k = k.clone()
        box_k[:2] /= 2
        box_k[0, 2] -= left
        box_k[1, 2] -= top
        front = distance[:, index] > reflectors["tau"]
        if not front.any():
            continue
        # The mirrored Gaussian's colour is the original's, seen from the mirrored camera.
        # Gaussians behind the plane get zero opacity (cheaper than gathering the rest).
        mirrored_colours = sh_colours(params, degree, mirror(centre, plane))
        mirrored, _, _ = rasterization(
            means=mirror(params["means"], plane),
            quats=mirror_quats(params["quats"], plane[:3]),
            scales=scales,
            opacities=opacities * front,
            colors=mirrored_colours,
            viewmats=world_to_camera[None],
            Ks=box_k[None],
            width=right - left,
            height=bottom - top,
            sh_degree=None,
            packed=False,
            rasterize_mode=mode,
            near_plane=near_plane,
        )
        half = F.pad(
            mirrored[0].permute(2, 0, 1),
            (left, max(0, width // 2 - right), top, max(0, height // 2 - bottom)),
        )
        mirrored = F.interpolate(half[None], size=(height, width), mode="bilinear")[0]
        mirrored = mirrored.permute(1, 2, 0)
        cosine = (rays @ plane[:3]).abs()
        f0 = torch.sigmoid(reflectors["f0"][index])
        fresnel = f0 + (1 - f0) * (1 - cosine) ** 5
        rgb = rgb + (maps[..., channel] * fresnel)[..., None] * mirrored
    return rgb, depth, alpha, info
