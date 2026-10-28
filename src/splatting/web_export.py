"""Compact web export: prune near-transparent Gaussians and write the 32-byte `.splat` format.

A full-SH 3DGS PLY stores 62 floats (248 bytes) per Gaussian; a multi-million Gaussian
house is about a gigabyte. Browser viewers (antimatter15/splat, SuperSplat import, many
three.js viewers) read the `.splat` layout, 32 bytes per Gaussian:

    position  3 × float32   world coordinates (scene units)
    scale     3 × float32   standard deviations (exp of the stored log scales)
    colour    4 × uint8     RGB = 0.5 + C0·SH_dc (view-independent), A = opacity
    rotation  4 × uint8     unit quaternion (w, x, y, z) mapped by q·128 + 128

Gaussians are sorted by importance (opacity × volume), as those viewers expect. Pruning
keeps Gaussians above an opacity threshold; `prune_report` measures what it costs on
held-out views before you rely on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import SH_C0, GaussianModel


def importance(model: GaussianModel) -> torch.Tensor:
    params = model.params
    return params["opacities"].sigmoid() * params["scales"].exp().prod(dim=-1)


def prune(model: GaussianModel, min_opacity: float) -> GaussianModel:
    """Copy of ``model`` without Gaussians whose opacity is below ``min_opacity``."""
    keep = model.params["opacities"].sigmoid() >= min_opacity
    return GaussianModel(
        {key: value.detach()[keep] for key, value in model.params.items()}, model.sh_degree
    )


def write_splat(path: str | Path, model: GaussianModel) -> int:
    """Write the `.splat` file; returns the number of Gaussians written."""
    params = {key: value.detach().cpu() for key, value in model.params.items()}
    order = torch.argsort(importance(model).cpu(), descending=True)

    means = params["means"][order].numpy().astype(np.float32)
    scales = params["scales"][order].exp().numpy().astype(np.float32)
    rgb = (0.5 + SH_C0 * params["sh0"][order, 0, :]).clamp(0, 1)
    alpha = params["opacities"][order].sigmoid()[:, None]
    colour = (torch.cat((rgb, alpha), dim=1) * 255).round().clamp(0, 255).byte().numpy()
    quats = torch.nn.functional.normalize(params["quats"][order], dim=-1)
    rotation = (quats * 128 + 128).round().clamp(0, 255).byte().numpy()

    record = np.dtype(
        [("position", "<f4", 3), ("scale", "<f4", 3), ("colour", "u1", 4), ("rotation", "u1", 4)]
    )
    data = np.empty(len(means), dtype=record)
    data["position"], data["scale"] = means, scales
    data["colour"], data["rotation"] = colour, rotation
    Path(path).write_bytes(data.tobytes())
    return len(data)


def read_splat(path: str | Path) -> dict[str, np.ndarray]:
    """Inverse of `write_splat` (for checks): positions, scales, rgba 0..1, wxyz quaternions."""
    record = np.dtype(
        [("position", "<f4", 3), ("scale", "<f4", 3), ("colour", "u1", 4), ("rotation", "u1", 4)]
    )
    data = np.frombuffer(Path(path).read_bytes(), dtype=record)
    quats = (data["rotation"].astype(np.float32) - 128) / 128
    return {
        "means": data["position"],
        "scales": data["scale"],
        "rgba": data["colour"].astype(np.float32) / 255,
        "quats": quats / np.linalg.norm(quats, axis=1, keepdims=True),
    }


@torch.no_grad()
def held_out_psnr(model, scene, indices, downscale: int, sh_degree: int) -> float:
    """Mean colour-corrected PSNR of held-out views (cameras as solved, no corrections)."""
    from .sfm_scene import read_rgb
    from .sfm_training import fit_colour, render

    device = model.device
    values = []
    for index in indices:
        target = torch.from_numpy(read_rgb(scene.image_paths[index], downscale)).to(device)
        target = target.float() / 255
        height, width = target.shape[:2]
        camera = scene.cameras[index].resized(width, height)
        rgb, _, _, _ = render(
            model,
            torch.tensor(camera.world_to_camera, dtype=torch.float32, device=device),
            torch.tensor(camera.intrinsics, dtype=torch.float32, device=device),
            width,
            height,
            sh_degree,
            "classic",
        )
        corrected = fit_colour(rgb.clamp(0, 1), target)
        values.append(float(-10 * torch.log10((corrected - target).square().mean())))
    return float(np.mean(values))


def main(argv=None) -> int:
    import argparse
    import json

    from .sfm_scene import load_sfm_scene
    from .storage import export_ply, load_model

    parser = argparse.ArgumentParser(description="Prune a splat and write a web `.splat` file")
    parser.add_argument("model", type=Path, help="scene.ply or checkpoint.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-opacity", type=float, default=0.02)
    parser.add_argument("--sfm-dir", type=Path, help="Measure held-out PSNR before/after")
    parser.add_argument("--downscale", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    model, metadata = load_model(args.model, args.device)
    pruned = prune(model, args.min_opacity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = write_splat(args.output_dir / "scene.splat", pruned)
    export_ply(args.output_dir / "scene_pruned.ply", pruned, metadata)
    report = {
        "min_opacity": args.min_opacity,
        "gaussians_before": model.count,
        "gaussians_after": count,
        "splat_megabytes": (args.output_dir / "scene.splat").stat().st_size / 2**20,
        "pruned_ply_megabytes": (args.output_dir / "scene_pruned.ply").stat().st_size / 2**20,
    }
    if args.sfm_dir:
        scene = load_sfm_scene(args.sfm_dir)
        degree = metadata.get("active_sh_degree", model.sh_degree)
        report["val_psnr_cc_db"] = {
            "full": held_out_psnr(model, scene, scene.val_indices, args.downscale, degree),
            "pruned": held_out_psnr(pruned, scene, scene.val_indices, args.downscale, degree),
            "pruned_view_independent": held_out_psnr(
                pruned, scene, scene.val_indices, args.downscale, 0
            ),
        }
    (args.output_dir / "web_export.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0
