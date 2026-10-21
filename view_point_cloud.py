"""View a saved cloud: uv run --extra viewer python view_point_cloud.py PATH.ply."""

import argparse
from pathlib import Path

from src.reconstruction import show_point_cloud


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="View a saved point cloud with Open3D.")
    parser.add_argument("point_cloud", type=Path, help="Point-cloud file, e.g. point_cloud.ply")
    parser.add_argument("--title", help="Optional window title")
    args = parser.parse_args(argv)

    if not args.point_cloud.is_file():
        parser.error(f"Point-cloud file does not exist: {args.point_cloud}")

    show_point_cloud(args.point_cloud, title=args.title or f"Namma Space — {args.point_cloud}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
