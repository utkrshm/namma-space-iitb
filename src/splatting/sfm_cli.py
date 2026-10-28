"""Command line entry point: Gaussian splatting from a dense SfM model."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .sfm_config import SfmSplatConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train a Gaussian splat from src.sfm outputs")
    parser.add_argument("sfm_dir", type=Path, help="Directory with frames.json and undistorted/")
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--config", type=Path, help="TOML with [data], [training], [refinement]")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-gaussians", type=int)
    parser.add_argument("--downscale", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-ply", type=Path, help="Start from this Gaussian PLY")
    parser.add_argument(
        "--resume", action="store_true", help="Continue from <output>/checkpoint.pt"
    )
    args = parser.parse_args(argv)

    from .sfm_training import train_sfm_splat

    config = SfmSplatConfig.from_toml(args.config) if args.config else SfmSplatConfig()
    if args.steps is not None:
        config = replace(config, training=replace(config.training, steps=args.steps))
    if args.max_gaussians is not None:
        refinement = replace(config.refinement, max_gaussians=args.max_gaussians)
        config = replace(config, refinement=refinement)
    if args.downscale is not None:
        config = replace(config, data=replace(config.data, downscale=args.downscale))
    metrics = train_sfm_splat(
        args.sfm_dir,
        args.output,
        config,
        device=args.device,
        resume=args.resume,
        initial_ply=args.initial_ply,
    )
    print(json.dumps({k: metrics[k] for k in ("gaussians", "train", "val")}, indent=1))
    print(f"PLY: {args.output / 'scene.ply'}; cameras: {args.output / 'cameras.json'}")
    return 0
