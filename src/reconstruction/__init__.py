"""VGGT perception and Open3D previews; heavy dependencies are imported lazily."""

from .contracts import (
    CameraParameters,
    FrameProvenance,
    ImageBatch,
    ImageTransform,
    PointCloud,
    ReconstructionResult,
)
from .geometry import decode_cameras, unproject_depth
from .images import load_images
from .preprocessed import load_preprocessed_frames, reconstruct_preprocessed
from .vggt import VGGTInference
from .visualization import save_point_cloud, show_point_cloud, to_open3d

__all__ = [
    "CameraParameters",
    "FrameProvenance",
    "ImageBatch",
    "ImageTransform",
    "PointCloud",
    "ReconstructionResult",
    "VGGTInference",
    "decode_cameras",
    "load_images",
    "load_preprocessed_frames",
    "reconstruct_preprocessed",
    "save_point_cloud",
    "show_point_cloud",
    "to_open3d",
    "unproject_depth",
]
