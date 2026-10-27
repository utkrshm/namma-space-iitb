"""Offline RGB/alpha/depth rendering from saved splats and posed cameras."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..splatting.config import RenderConfig, SplattingConfig
from ..splatting.contracts import Camera
from ..splatting.storage import load_model, write_json
from .renderer import render_gaussians, require_cuda


def load_cameras(path: str | Path) -> list[Camera]:
    """Read {schema_version: 1, cameras: [...]} or a refined-scene manifest."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported camera-list schema")
    if "frames" in value and value.get("camera_convention") != "opencv_world_to_camera":
        raise ValueError("Expected OpenCV world-to-camera extrinsics")
    if "cameras" in value and value.get("camera_convention") != "opencv_world_to_camera":
        raise ValueError("Declare camera_convention='opencv_world_to_camera'")
    entries = value.get("cameras", value.get("frames"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Camera file needs a nonempty cameras or frames array")
    cameras = [Camera.from_dict(entry, path.parent) for entry in entries]
    if len({camera.frame_id for camera in cameras}) != len(cameras):
        raise ValueError("Camera frame IDs must be unique")
    return cameras


def rendering_settings(metadata: dict) -> tuple[RenderConfig, int | None]:
    config = SplattingConfig.from_dict(metadata["config"]) if "config" in metadata else None
    render = config.render if config else RenderConfig()
    degree = metadata.get("active_sh_degree")
    if degree is None and config and "step" in metadata:
        degree = min(config.training.sh_degree, metadata["step"] // config.training.sh_interval)
    return render, degree


@torch.no_grad()
def render_cameras(
    model_path: str | Path,
    cameras: list[Camera],
    output_dir: str | Path,
    *,
    device: str = "cuda",
    max_image_side: int | None = None,
):
    require_cuda(device)
    if not cameras:
        raise ValueError("Supply at least one rendering camera")
    if max_image_side is not None and max_image_side < 1:
        raise ValueError("max_image_side must be positive")
    model, metadata = load_model(model_path, device)
    config, degree = rendering_settings(metadata)
    directory = Path(output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Render into an empty directory to avoid overwriting results")
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, camera in enumerate(cameras):
        if max_image_side and max(camera.width, camera.height) > max_image_side:
            factor = max_image_side / max(camera.width, camera.height)
            camera = camera.resized(
                max(1, round(camera.width * factor)), max(1, round(camera.height * factor))
            )
        rendered = render_gaussians(model, camera, config, sh_degree=degree)
        rgb = rendered.rgb.clamp(0, 1).mul(255).round().byte().cpu().numpy()
        alpha = rendered.alpha.clamp(0, 1).mul(255).round().byte().cpu().numpy()
        depth = rendered.depth.cpu().numpy().copy()
        # Camera-Z is undefined in empty pixels. Preserve this in the numeric output.
        depth[rendered.alpha.cpu().numpy() <= 1e-6] = np.nan
        stem = f"{index:05d}"
        if not cv2.imwrite(str(directory / f"{stem}_rgb.png"), rgb[..., ::-1]):
            raise OSError("Failed to save rendered RGB")
        if not cv2.imwrite(str(directory / f"{stem}_alpha.png"), alpha):
            raise OSError("Failed to save rendered alpha")
        np.save(directory / f"{stem}_depth.npy", depth, allow_pickle=False)
        rows.append(
            {
                **camera.to_dict(),
                "rgb": f"{stem}_rgb.png",
                "alpha": f"{stem}_alpha.png",
                "depth": f"{stem}_depth.npy",
            }
        )
    write_json(
        directory / "renders.json",
        {
            "schema_version": 1,
            "model": str(Path(model_path).resolve()),
            "camera_convention": "opencv_world_to_camera",
            "depth_convention": "camera_z",
            "pixel_convention": "integer_centers",
            "scene_units": metadata.get("scene_units", "unspecified"),
            "frames": rows,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Gaussian checkpoint .pt or standard SH .ply")
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-image-side", type=int)
    args = parser.parse_args()
    render_cameras(
        args.model,
        load_cameras(args.cameras),
        args.output,
        device=args.device,
        max_image_side=args.max_image_side,
    )
