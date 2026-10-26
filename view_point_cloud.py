"""View a saved cloud: uv run --extra viewer python view_point_cloud.py PATH.ply."""

import argparse
from pathlib import Path

from src.reconstruction import show_point_cloud, show_point_cloud_with_camera_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="View a saved point cloud with Open3D.")
    parser.add_argument("point_cloud", type=Path, help="Point-cloud file, e.g. point_cloud.ply")
    parser.add_argument("--title", help="Optional window title")
    parser.add_argument(
        "--follow-cameras",
        nargs="?",
        const="auto",
        metavar="CAMERA_SOURCE",
        help=(
            "Traverse the cloud along VGGT camera poses. Optionally provide a camera NPZ, "
            "point-cloud NPZ or directory; by default cameras are found beside the cloud."
        ),
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=3.0,
        help="Camera-traversal playback rate in poses per second (default: 3)",
    )
    parser.add_argument(
        "--no-camera-loop",
        action="store_true",
        help="Stop at the final camera pose instead of looping",
    )
    parser.add_argument(
        "--camera-fov",
        type=float,
        default=60.0,
        help="Vertical field of view in degrees for camera traversal (default: 60)",
    )
    args = parser.parse_args(argv)

    if not args.point_cloud.is_file():
        parser.error(f"Point-cloud file does not exist: {args.point_cloud}")
    if args.camera_fps <= 0:
        parser.error("--camera-fps must be positive")
    if not 1.0 < args.camera_fov < 179.0:
        parser.error("--camera-fov must be between 1 and 179 degrees")

    title = args.title or f"Namma Space — {args.point_cloud}"
    if args.follow_cameras is not None:
        camera_source = None if args.follow_cameras == "auto" else Path(args.follow_cameras)
        if camera_source is not None and not camera_source.exists():
            parser.error(f"Camera source does not exist: {camera_source}")
        show_point_cloud_with_camera_path(
            args.point_cloud,
            camera_source,
            title=title,
            fps=args.camera_fps,
            loop=not args.no_camera_loop,
            fov_deg=args.camera_fov,
        )
    else:
        show_point_cloud(args.point_cloud, title=title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
