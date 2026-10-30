"""Gaussians, rendering and the standard 3DGS PLY.

Parameterization (the graphdeco PLY layout): world means, log standard deviations, wxyz
rotations, opacity logits and channel-major spherical harmonics. Two extra per-Gaussian
fields are written as additional PLY properties, which standard viewers ignore:

- ``filled`` (0/1): the Gaussian was created or changed mainly by generated content
  (`spacegen.fill`), not by the capture;
- ``support_<k>`` (uint32 × 2): bitmask of the 64 view directions from which training
  cameras saw the Gaussian (`spacegen.support`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

SH_C0 = 0.28209479177387814
PARAMETERS = ("means", "scales", "quats", "opacities", "sh0", "shn")


class GaussianModel(nn.Module):
    def __init__(self, tensors: dict[str, torch.Tensor], sh_degree: int, extras=None):
        super().__init__()
        if not 0 <= sh_degree <= 3:
            raise ValueError("SH degree must be 0..3")
        self.sh_degree = sh_degree
        self.params = nn.ParameterDict(
            {key: nn.Parameter(tensors[key].detach().float().contiguous()) for key in PARAMETERS}
        )
        count = len(self.params["means"])
        # Non-trainable per-Gaussian fields (``filled``, ``support``) travel with the model.
        self.extras: dict[str, torch.Tensor] = {
            key: value.to(self.params["means"].device) for key, value in (extras or {}).items()
        }
        expected = {
            "means": (count, 3),
            "scales": (count, 3),
            "quats": (count, 4),
            "opacities": (count,),
            "sh0": (count, 1, 3),
            "shn": (count, (sh_degree + 1) ** 2 - 1, 3),
        }
        for key, shape in expected.items():
            if tuple(self.params[key].shape) != shape:
                raise ValueError(f"{key}: expected {shape}, got {tuple(self.params[key].shape)}")

    def _apply(self, fn, recurse=True):
        # Keep the non-parameter fields on the same device as the parameters.
        super()._apply(fn, recurse)
        self.extras = {key: fn(value) for key, value in self.extras.items()}
        return self

    @property
    def count(self) -> int:
        return len(self.params["means"])

    @property
    def device(self) -> torch.device:
        return self.params["means"].device

    def subset(self, keep: torch.Tensor) -> GaussianModel:
        return GaussianModel(
            {k: v[keep] for k, v in self.params.items()},
            self.sh_degree,
            {k: v[keep] for k, v in self.extras.items()},
        )

    @staticmethod
    def concatenate(models: list[GaussianModel]) -> GaussianModel:
        degree = models[0].sh_degree
        tensors = {k: torch.cat([m.params[k].detach() for m in models]) for k in PARAMETERS}
        extras = {}
        for key in set().union(*(m.extras for m in models)):
            parts = []
            for m in models:
                if key in m.extras:
                    parts.append(m.extras[key])
                else:
                    shape = next(x.extras[key] for x in models if key in x.extras).shape[1:]
                    parts.append(torch.zeros((m.count, *shape), dtype=torch.int64, device=m.device))
            extras[key] = torch.cat(parts)
        return GaussianModel(tensors, degree, extras)


def render(
    model: GaussianModel,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int | None = None,
    mode: str = "antialiased",
    near_plane: float = 0.01,
    colors: torch.Tensor | None = None,
    background: torch.Tensor | None = None,
):
    """RGB [H,W,3], expected depth [H,W], alpha [H,W] and gsplat's info dict.

    ``intrinsics`` use integer pixel centres; gsplat samples at (u + ½, v + ½), so the
    principal point moves by +½ here. ``colors`` replaces the SH colours with any
    per-Gaussian feature (e.g. support), rendered with the same blending weights.
    """
    from gsplat import rasterization

    params = model.params
    k = intrinsics.clone()
    k[0, 2] += 0.5
    k[1, 2] += 0.5
    if colors is None:
        colors = torch.cat((params["sh0"], params["shn"]), dim=1)
        degree = model.sh_degree if sh_degree is None else sh_degree
    else:
        degree = None
    output, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=colors,
        viewmats=world_to_camera[None],
        Ks=k[None],
        width=width,
        height=height,
        sh_degree=degree,
        packed=False,
        rasterize_mode=mode,
        render_mode="RGB+ED",
        near_plane=near_plane,
        backgrounds=None if background is None else background[None],
    )
    channels = output.shape[-1] - 1
    return output[0, ..., :channels], output[0, ..., channels], alpha[0, ..., 0], info


def render_camera(model, camera, device=None, **options):
    """`render` for a `spacegen.contracts.Camera`."""
    device = device or model.device
    world_to_camera = torch.tensor(camera.world_to_camera, dtype=torch.float32, device=device)
    k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
    return render(model, world_to_camera, k, camera.width, camera.height, **options)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def save_ply(path: str | Path, model: GaussianModel, metadata: dict | None = None) -> None:
    from plyfile import PlyData, PlyElement

    params = {key: value.detach().cpu().numpy() for key, value in model.params.items()}
    count = model.count
    shn = params["shn"].transpose(0, 2, 1).reshape(count, -1)
    columns = [
        ("x", params["means"][:, 0]),
        ("y", params["means"][:, 1]),
        ("z", params["means"][:, 2]),
    ]
    columns += [(n, np.zeros(count, np.float32)) for n in ("nx", "ny", "nz")]
    columns += [(f"f_dc_{i}", params["sh0"][:, 0, i]) for i in range(3)]
    columns += [(f"f_rest_{i}", shn[:, i]) for i in range(shn.shape[1])]
    columns += [("opacity", params["opacities"])]
    columns += [(f"scale_{i}", params["scales"][:, i]) for i in range(3)]
    columns += [(f"rot_{i}", params["quats"][:, i]) for i in range(4)]
    dtype = [(name, "<f4") for name, _ in columns]
    extras = {k: v.detach().cpu().numpy() for k, v in model.extras.items()}
    if "filled" in extras:
        dtype.append(("filled", "u1"))
    if "support" in extras:
        dtype += [("support_0", "<u4"), ("support_1", "<u4")]
    vertices = np.empty(count, dtype=dtype)
    for name, value in columns:
        vertices[name] = value
    if "filled" in extras:
        vertices["filled"] = extras["filled"].astype(np.uint8)
    if "support" in extras:
        bits = extras["support"].astype(np.int64).view(np.uint64)
        vertices["support_0"] = (bits & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        vertices["support_1"] = (bits >> np.uint64(32)).astype(np.uint32)

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(temporary)
    temporary.replace(path)
    if metadata is not None:
        write_json(path.with_suffix(".json"), metadata)


def load_ply(path: str | Path, device: str = "cuda:0") -> GaussianModel:
    from plyfile import PlyData

    vertices = PlyData.read(str(path))["vertex"]
    names = vertices.data.dtype.names
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    degree = math.isqrt(len(rest) // 3 + 1) - 1

    def stack(keys):
        return torch.from_numpy(np.column_stack([vertices[k] for k in keys]).astype(np.float32))

    count = len(vertices)
    shn = stack(rest).reshape(count, 3, -1).transpose(1, 2) if rest else torch.zeros((count, 0, 3))
    tensors = {
        "means": stack(["x", "y", "z"]),
        "scales": stack([f"scale_{i}" for i in range(3)]),
        "quats": stack([f"rot_{i}" for i in range(4)]),
        "opacities": torch.from_numpy(np.array(vertices["opacity"], dtype=np.float32)),
        "sh0": stack([f"f_dc_{i}" for i in range(3)])[:, None, :],
        "shn": shn,
    }
    extras = {}
    if "filled" in names:
        extras["filled"] = torch.from_numpy(np.asarray(vertices["filled"]).astype(np.int64))
    if "support_0" in names:
        low = np.asarray(vertices["support_0"]).astype(np.uint64)
        high = np.asarray(vertices["support_1"]).astype(np.uint64)
        extras["support"] = torch.from_numpy((low | (high << np.uint64(32))).view(np.int64))
    return GaussianModel(tensors, degree, extras).to(device)
