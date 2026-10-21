"""Open3D export and interactive preview, usable without Torch or VGGT weights."""

from pathlib import Path

import numpy as np

from .contracts import PointCloud


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        raise ImportError("Install Open3D: uv sync --extra viewer") from error
    return o3d


def to_open3d(cloud: PointCloud):
    """Convert RGB uint8 observations to an Open3D point cloud without axis flips."""
    if (
        cloud.points.ndim != 2
        or cloud.points.shape[1] != 3
        or cloud.colors.shape != cloud.points.shape
    ):
        raise ValueError("Cloud points and RGB colors must both have shape N,3")
    if not len(cloud.points):
        raise ValueError("Point cloud is empty; review valid pixels and confidence filters")
    if not np.isfinite(cloud.points).all() or cloud.colors.dtype != np.uint8:
        raise ValueError("Cloud points must be finite and RGB colors must be uint8")

    o3d = _open3d()
    geometry = o3d.geometry.PointCloud()
    geometry.points = o3d.utility.Vector3dVector(np.asarray(cloud.points, dtype=np.float64))
    geometry.colors = o3d.utility.Vector3dVector(cloud.colors.astype(np.float64) / 255)
    return geometry


def save_point_cloud(cloud: PointCloud, path: str | Path) -> Path:
    """Write a coloured PLY and report I/O failure instead of silently continuing."""
    path = Path(path)
    if path.suffix.lower() != ".ply":
        raise ValueError("Point-cloud export path must end in .ply")
    geometry = to_open3d(cloud)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _open3d().io.write_point_cloud(str(path), geometry):
        raise OSError(f"Open3D could not write point cloud: {path}")
    return path


def show_point_cloud(cloud: PointCloud | str | Path, *, title: str = "Namma Space — VGGT") -> None:
    """Open the local interactive viewer, or view an exported cloud without inference."""
    o3d = _open3d()
    if isinstance(cloud, (str, Path)):
        path = Path(cloud)
        if not path.is_file():
            raise FileNotFoundError(path)
        geometry = o3d.io.read_point_cloud(str(path))
        if geometry.is_empty():
            raise ValueError(f"No readable points in: {path}")
    else:
        geometry = to_open3d(cloud)
    o3d.visualization.draw_geometries([geometry], window_name=title, width=1280, height=800)
