"""Train a Gaussian splat from a refined VGGT scene manifest."""

import argparse
from pathlib import Path

from .config import SplattingConfig


def main():
    from .training import train_splat

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = SplattingConfig.from_toml(args.config) if args.config else SplattingConfig()
    result = train_splat(args.manifest, args.output, config, device=args.device, resume=args.resume)
    print(f"Checkpoint: {result.checkpoint}\nPLY: {result.ply}\nMetrics: {result.metrics}")
