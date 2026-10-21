"""NumPy camera/depth processing, independent of Torch and model availability."""

import numpy as np

from .contracts import CameraParameters, ImageTransform


def decode_cameras(
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    transforms: tuple[ImageTransform, ...],
) -> CameraParameters:
    """Build homogeneous inverse poses and intrinsics in the upright source grid."""
    extrinsics = np.asarray(world_to_camera, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    size = len(transforms)
    if extrinsics.shape != (size, 3, 4) or intrinsics.shape != (size, 3, 3):
        raise ValueError("Camera arrays must be S,3,4 and S,3,3 and match image transforms")
    if not np.isfinite(extrinsics).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Camera parameters must be finite")

    poses = np.tile(np.eye(4, dtype=np.float32), (size, 1, 1))
    poses[:, :3, :3] = extrinsics[:, :3, :3].transpose(0, 2, 1)
    poses[:, :3, 3] = -np.einsum("sij,sj->si", poses[:, :3, :3], extrinsics[:, :3, 3])
    original_intrinsics = np.stack(
        [
            np.linalg.solve(transform.source_to_model, k)
            for transform, k in zip(transforms, intrinsics)
        ]
    )
    return CameraParameters(intrinsics, original_intrinsics, extrinsics, poses)


def unproject_depth(depth: np.ndarray, cameras: CameraParameters) -> np.ndarray:
    """S,H,W camera z-depth -> S,H,W,3 world XYZ, without changing scene scale."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 3 or depth.shape[0] != len(cameras.intrinsics):
        raise ValueError("Depth must have shape S,H,W and match the camera count")

    size, height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    points = np.empty((size, height, width, 3), dtype=np.float32)
    for i, (k, pose) in enumerate(zip(cameras.intrinsics, cameras.camera_to_world)):
        rays = pixels @ np.linalg.inv(k).T
        camera_points = rays * depth[i, ..., None]
        points[i] = camera_points @ pose[:3, :3].T + pose[:3, 3]
    return points
