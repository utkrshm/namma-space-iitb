"""How much each Gaussian shows in the training views, and pruning of the ones that never do.

The weight of a Gaussian in one view is the sum over pixels of its alpha-blending weight
(transmittance × alpha), i.e. how many pixels' worth of colour it contributes. It is read
off as a gradient: rendering a constant colour ``c_i = 1`` per Gaussian gives
``image = Σ_i w_i c_i`` per pixel, so ``∂ Σ image / ∂ c_i = Σ_pixels w_i``.

Weights are in full-resolution pixels of the cameras in ``cameras.json`` (refined training
poses written by the trainer); held-out cameras are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import GaussianModel
from .storage import export_ply, load_model


def training_view_weights(model: GaussianModel, cameras: list[dict], scale: float = 0.5):
    """Per Gaussian: (largest weight in any one view, total weight), full-resolution px."""
    from gsplat import rasterization

    params = {key: value.detach() for key, value in model.params.items()}
    count = model.count
    device = model.device
    peak = torch.zeros(count, device=device)
    total = torch.zeros(count, device=device)
    for camera in cameras:
        width, height = round(camera["width"] * scale), round(camera["height"] * scale)
        k = torch.tensor(camera["intrinsics"], dtype=torch.float32, device=device)
        k[:2] *= scale
        k[:2, 2] += 0.5  # integer pixel centres → gsplat's (u + 0.5, v + 0.5)
        view = torch.tensor(camera["world_to_camera"], dtype=torch.float32, device=device)
        unit = torch.ones(count, 1, device=device, requires_grad=True)
        image, _, _ = rasterization(
            params["means"],
            params["quats"],
            params["scales"].exp(),
            params["opacities"].sigmoid(),
            unit,
            view[None],
            k[None],
            width,
            height,
            sh_degree=None,
            packed=False,
        )
        image.sum().backward()
        weight = unit.grad[:, 0] / scale**2
        peak = torch.maximum(peak, weight)
        total += weight
    return peak, total


def prune_invisible(
    model_path: str | Path,
    run_dir: str | Path,
    output: str | Path,
    min_pixels: float = 1.0,
    device: str = "cuda:0",
) -> dict:
    """Keep Gaussians covering at least ``min_pixels`` in some training view."""
    cameras = json.loads((Path(run_dir) / "cameras.json").read_text())["cameras"]
    cameras = [camera for camera in cameras if camera.get("split", "train") == "train"]
    model, metadata = load_model(model_path, device)

    peak, _ = training_view_weights(model, cameras)
    keep = peak >= min_pixels
    kept = GaussianModel({k: v.detach()[keep] for k, v in model.params.items()}, model.sh_degree)
    export_ply(output, kept, metadata)

    report = {
        "input": str(model_path),
        "output": str(output),
        "training_views": len(cameras),
        "min_pixels": min_pixels,
        "gaussians_in": int(model.count),
        "gaussians_kept": int(keep.sum()),
        "peak_weight_px_percentiles": {
            str(q): float(v)
            for q, v in zip((10, 50, 90), np.percentile(peak.cpu().numpy(), (10, 50, 90)))
        },
    }
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Drop Gaussians unseen in every training view")
    parser.add_argument("model", type=Path, help="scene.ply or checkpoint.pt")
    parser.add_argument("run_dir", type=Path, help="Training run with cameras.json")
    parser.add_argument("--output", type=Path, required=True, help="Pruned .ply")
    parser.add_argument("--min-pixels", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    print(prune_invisible(args.model, args.run_dir, args.output, args.min_pixels, args.device))
    return 0
