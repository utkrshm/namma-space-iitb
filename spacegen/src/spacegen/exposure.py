"""The capture's own exposure near any view, for renders that should look like the video.

Phones auto-expose and white-balance every frame. Training absorbs that in one colour
map per training image (`spacegen.splat.train.Corrections`, zero mean over the images),
so the splat's own colours sit at the video's average exposure. The phone exposes down
whenever a bright window is in view, so a render at the average exposure is far brighter
there: the window and the curtain next to it blow out to white. vyas-room 00208 scores
12.1 dB against its frame that way.

`CaptureExposure` blends the colour maps of the training cameras near a view. The weights
are exp(−½ (distance / ``radius``)²) · exp(−½ (angle / ``cone_deg``)²) over camera position
(camera heights) and viewing direction, plus ``prior`` on the identity map, so a view far
from every capture camera keeps the average exposure. Tested on held-out views, whose own
maps never enter, at radius 0.15 and cone 10°: vyas-room went from 22.06 to 26.15 dB
(00208 from 12.1 to 28.2) and m-block from 20.71 to 26.75. `ExposureFilter` smooths the
blended map over the frames of a video, like a camera's auto-exposure.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .splat.train import Corrections, bilateral_slice, load_cameras


class CaptureExposure:
    """Blended colour maps of training cameras (``maps`` [T,12] affine residuals, or
    [T,12,D,H,W] bilateral grids) at their ``centres`` and viewing ``forwards``."""

    def __init__(
        self,
        centres: np.ndarray,
        forwards: np.ndarray,
        maps: torch.Tensor,
        camera_height: float,
        radius: float = 0.15,
        cone_deg: float = 10.0,
        prior: float = 0.05,
    ):
        self.centres, self.forwards, self.maps = centres, forwards, maps
        self.scale = radius * camera_height
        self.cone = math.radians(cone_deg)
        self.prior = prior

    @classmethod
    def from_run(cls, run, camera_height: float, device, radius=0.15, cone_deg=10.0, log=print):
        """From ``splat/cameras.json`` and the colour maps in ``splat/checkpoint.pt``; None
        when the checkpoint is missing (renders then keep the average exposure)."""
        run = Path(run)
        path = run / "splat" / "checkpoint.pt"
        if not path.exists():
            log("exposure: no splat/checkpoint.pt; rendering at the average exposure")
            return None
        state = torch.load(path, map_location="cpu", weights_only=True)["corrections"]
        corrections = Corrections.for_state(state)
        corrections.load_state_dict(state)
        cameras, splits = load_cameras(run / "splat" / "cameras.json")
        if len(cameras) != len(corrections.pose):
            log("exposure: the checkpoint's cameras differ from cameras.json; not used")
            return None
        train = [i for i, split in enumerate(splits) if split == "train"]
        maps = corrections.colour_parameter.detach()[train].to(device)
        centres = np.stack([cameras[i].centre for i in train])
        forwards = np.stack([cameras[i].camera_to_world[:3, 2] for i in train])
        return cls(centres, forwards, maps, camera_height, radius, cone_deg)

    def weights(self, world_to_camera: np.ndarray) -> np.ndarray:
        """Per training camera, its share of the blend (the identity map takes the rest)."""
        camera_to_world = np.linalg.inv(world_to_camera)
        distance = np.linalg.norm(self.centres - camera_to_world[:3, 3], axis=1) / self.scale
        cosine = np.clip(self.forwards @ camera_to_world[:3, 2], -1.0, 1.0)
        angle = np.arccos(cosine) / self.cone
        weight = np.exp(-0.5 * distance**2) * np.exp(-0.5 * angle**2)
        return weight / (weight.sum() + self.prior)

    def residual(self, world_to_camera: np.ndarray) -> torch.Tensor:
        """The blended colour map (residual from identity) for a view."""
        weight = torch.tensor(self.weights(world_to_camera), dtype=self.maps.dtype)
        weight = weight.to(self.maps.device).reshape(-1, *[1] * (self.maps.dim() - 1))
        return (weight * self.maps).sum(0)


def apply_residual(residual: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """``rgb`` [H,W,3] through a colour map residual, as `Corrections.apply_colour`."""
    if residual.dim() > 1:
        return bilateral_slice(residual, rgb)
    matrix = torch.eye(3, device=rgb.device, dtype=rgb.dtype) + residual[:9].view(3, 3)
    return rgb @ matrix.T + residual[9:]


class ExposureFilter:
    """`CaptureExposure` for the consecutive frames of a video: the map follows an
    exponential moving average with ``seconds`` time constant; `reset` starts a new shot."""

    def __init__(self, exposure: CaptureExposure, fps: float, seconds: float = 0.25):
        self.exposure = exposure
        self.keep = math.exp(-1.0 / max(fps * seconds, 1e-6))
        self.state: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None

    def __call__(self, world_to_camera: np.ndarray, rgb: torch.Tensor) -> torch.Tensor:
        target = self.exposure.residual(world_to_camera).to(rgb.device)
        if self.state is None:
            self.state = target
        else:
            self.state = self.keep * self.state + (1 - self.keep) * target
        return apply_residual(self.state, rgb)
