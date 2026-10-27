"""Learnable 3D Gaussians; checkpoint and PLY use the same parameterization."""

import math

import torch
from torch import nn

from .geometry import InitialPoints

SH_C0 = 0.28209479177387814


class GaussianModel(nn.Module):
    """World means, log standard deviations, wxyz rotations, opacity logits, SH."""

    def __init__(self, tensors: dict[str, torch.Tensor], sh_degree: int):
        super().__init__()
        self.sh_degree = sh_degree
        self.params = nn.ParameterDict(
            {
                key: nn.Parameter(value.detach().to(dtype=torch.float32).contiguous())
                for key, value in tensors.items()
            }
        )
        self.validate()

    def validate(self):
        if type(self.sh_degree) is not int or not 0 <= self.sh_degree <= 3:
            raise ValueError("Supported SH degree is 0..3")
        if "means" not in self.params or len(self.params["means"]) == 0:
            raise ValueError("Gaussian model is empty")
        count = len(self.params["means"])
        expected = {
            "means": (count, 3),
            "scales": (count, 3),
            "quats": (count, 4),
            "opacities": (count,),
            "sh0": (count, 1, 3),
            "shn": (count, (self.sh_degree + 1) ** 2 - 1, 3),
        }
        if set(self.params) != set(expected):
            raise ValueError(f"Gaussian parameters must be {set(expected)}")
        for key, shape in expected.items():
            value = self.params[key]
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid Gaussian parameter {key}: expected finite {shape}")
        if (self.params["quats"].norm(dim=-1) < 1e-8).any():
            raise ValueError("Gaussian rotations cannot be zero quaternions")
        if not torch.isfinite(self.params["scales"].exp()).all():
            raise ValueError("Gaussian scales overflow")
        if (self.params["scales"].exp() <= 0).any():
            raise ValueError("Gaussian scales must be positive")

    @classmethod
    def from_points(cls, points: InitialPoints, sh_degree: int, opacity: float, device: str):
        means = torch.as_tensor(points.means, device=device)
        count = len(means)
        quats = torch.zeros((count, 4), device=device)
        quats[:, 0] = 1
        colors = torch.as_tensor(points.colors, device=device)
        return cls(
            {
                "means": means,
                "scales": torch.as_tensor(points.scales, device=device).log(),
                "quats": quats,
                "opacities": torch.full((count,), math.log(opacity / (1 - opacity)), device=device),
                "sh0": ((colors - 0.5) / SH_C0)[:, None, :],
                "shn": torch.zeros((count, (sh_degree + 1) ** 2 - 1, 3), device=device),
            },
            sh_degree,
        )

    @property
    def device(self) -> torch.device:
        return self.params["means"].device

    @property
    def count(self) -> int:
        return len(self.params["means"])

    @torch.no_grad()
    def constrain(self, scene_scale: float):
        self.params["scales"].clamp_(math.log(scene_scale * 1e-6), math.log(scene_scale * 0.5))
        self.params["opacities"].clamp_(-12, 12)
        quats = self.params["quats"]
        zero = quats.norm(dim=-1) < 1e-8
        quats[zero] = quats.new_tensor([1, 0, 0, 0])
        quats.div_(quats.norm(dim=-1, keepdim=True))
