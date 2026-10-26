"""Open3D export and interactive preview, usable without Torch or VGGT weights."""

import time
from pathlib import Path

import numpy as np

from .camera_path import CameraPath, find_camera_path, load_camera_path
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


def _load_geometry(cloud: PointCloud | str | Path):
    o3d = _open3d()
    if isinstance(cloud, (str, Path)):
        path = Path(cloud)
        if not path.is_file():
            raise FileNotFoundError(path)
        geometry = o3d.io.read_point_cloud(str(path))
        if geometry.is_empty():
            raise ValueError(f"No readable points in: {path}")
        return geometry
    return to_open3d(cloud)


def show_point_cloud(cloud: PointCloud | str | Path, *, title: str = "Namma Space — VGGT") -> None:
    """Open the local interactive viewer, or view an exported cloud without inference."""
    o3d = _open3d()
    geometry = _load_geometry(cloud)
    o3d.visualization.draw_geometries([geometry], window_name=title, width=1280, height=800)


def _display_intrinsic(o3d, width: int, height: int, fov_deg: float):
    if not np.isfinite(fov_deg) or not 1.0 < fov_deg < 179.0:
        raise ValueError("camera_fov must be between 1 and 179 degrees")
    focal = (height / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return o3d.camera.PinholeCameraIntrinsic(width, height, focal, focal, width / 2.0, height / 2.0)


def _apply_camera_pose(visualizer, intrinsic, world_to_camera: np.ndarray) -> None:
    o3d = _open3d()
    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = intrinsic
    parameters.extrinsic = np.asarray(world_to_camera, dtype=np.float64)
    visualizer.get_view_control().convert_from_pinhole_camera_parameters(
        parameters, allow_arbitrary=True
    )


def show_point_cloud_with_camera_path(
    cloud: PointCloud | str | Path,
    camera_path: CameraPath | str | Path | None = None,
    *,
    title: str = "Namma Space — VGGT camera traversal",
    fps: float = 3.0,
    loop: bool = True,
    fov_deg: float = 60.0,
    width: int = 1280,
    height: int = 800,
) -> None:
    """Traverse a cloud along saved VGGT camera poses in capture order."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Camera traversal fps must be positive")

    o3d = _open3d()
    geometry = _load_geometry(cloud)
    if camera_path is None:
        if not isinstance(cloud, (str, Path)):
            raise ValueError("Auto-discovering cameras requires a saved point-cloud path")
        camera_path = find_camera_path(cloud)
    path = camera_path if isinstance(camera_path, CameraPath) else load_camera_path(camera_path)
    world_to_camera = path.world_to_camera
    intrinsic = _display_intrinsic(o3d, width, height, fov_deg)

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(window_name=title, width=width, height=height):
        raise RuntimeError("Open3D could not create a visualization window")
    visualizer.add_geometry(geometry)

    index = 0
    _apply_camera_pose(visualizer, intrinsic, world_to_camera[index])
    last_update = time.monotonic()
    interval = 1.0 / fps

    def animate(vis):
        nonlocal index, last_update
        now = time.monotonic()
        if now - last_update < interval:
            return False
        last_update = now
        index += 1
        if index >= len(world_to_camera):
            if not loop:
                vis.register_animation_callback(None)
                return False
            index = 0
        _apply_camera_pose(vis, intrinsic, world_to_camera[index])
        return False

    print(
        f"Following {len(world_to_camera)} VGGT camera poses at {fps:g} fps"
        f"{' (looping)' if loop else ''}. Close the window to stop.",
        flush=True,
    )
    visualizer.register_animation_callback(animate)
    visualizer.run()
    visualizer.destroy_window()


def follow_cameras(
    cloud: PointCloud | str | Path,
    camera_path: CameraPath | str | Path | None = None,
    **kwargs,
) -> None:
    """Alias for :func:`show_point_cloud_with_camera_path`."""
    show_point_cloud_with_camera_path(cloud, camera_path, **kwargs)
