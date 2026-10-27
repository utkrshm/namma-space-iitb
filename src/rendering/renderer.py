"""Differentiable Gaussian renderer shared by optimization, export and viewing."""

from dataclasses import dataclass

import torch

from ..splatting.config import RenderConfig
from ..splatting.contracts import Camera
from ..splatting.model import GaussianModel


@dataclass
class RenderOutput:
    rgb: torch.Tensor  # [H,W,3], floating RGB in the training image's colour space
    alpha: torch.Tensor  # [H,W], accumulated opacity
    depth: torch.Tensor  # [H,W], expected camera-Z; undefined where alpha is zero
    info: dict  # gsplat intermediates needed by the densification strategy


def require_cuda(device: str) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training/rendering requires CUDA on the development server")
    return target


def render_gaussians(
    model: GaussianModel,
    camera: Camera,
    config: RenderConfig | None = None,
    *,
    sh_degree: int | None = None,
    absgrad: bool = False,
) -> RenderOutput:
    """Render one undistorted camera without detaching the computation graph.

    Camera-Z depth and coordinates use the input scene's units. RGB is composited
    against the configured background; depth is opacity-normalized, not range.
    """
    from gsplat import rasterization

    config = config or RenderConfig()
    if model.device.type != "cuda":
        raise RuntimeError("gsplat rendering requires a CUDA model")
    degree = model.sh_degree if sh_degree is None else sh_degree
    if not 0 <= degree <= model.sh_degree:
        raise ValueError("Requested SH degree exceeds the model's stored coefficients")
    params = model.params
    intrinsics = torch.tensor(camera.intrinsics, dtype=torch.float32, device=model.device)
    # The file contract/VGGT uses integer-centre pixels; gsplat 1.5.3 samples
    # raster pixels at (u + .5, v + .5). Translate K only at this boundary.
    intrinsics[0, 2] += 0.5
    intrinsics[1, 2] += 0.5
    colors, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat((params["sh0"], params["shn"]), dim=1),
        viewmats=torch.tensor(camera.world_to_camera, dtype=torch.float32, device=model.device)[
            None
        ],
        Ks=intrinsics[None],
        width=camera.width,
        height=camera.height,
        near_plane=config.near_plane,
        far_plane=config.far_plane,
        sh_degree=degree,
        packed=config.packed,
        absgrad=absgrad,
        rasterize_mode=config.rasterize_mode,
        render_mode="RGB+ED",
    )
    opacity = alpha[0, ..., 0]
    background = colors.new_tensor(config.background)
    rgb = colors[0, ..., :3] + (1 - opacity[..., None]) * background
    return RenderOutput(rgb, opacity, colors[0, ..., 3], info)
