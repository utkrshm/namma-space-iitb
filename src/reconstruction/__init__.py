"""VGGT perception and Open3D previews; heavy dependencies are imported lazily."""

from .contracts import (
    CameraParameters,
    FrameProvenance,
    ImageBatch,
    ImageTransform,
    PointCloud,
    ReconstructionResult,
)
from .floor import (
    FloorAlignment,
    estimate_floor_alignment,
    segment_floor_images,
    transform_observations,
)
from .geometry import decode_cameras, unproject_depth
from .images import load_images
from .preprocessed import load_preprocessed_frames, reconstruct_preprocessed
from .vggt import VGGTInference
from .visualization import save_point_cloud, show_point_cloud, to_open3d

__all__ = [
    "CameraParameters",
    "FloorAlignment",
    "FrameProvenance",
    "ImageBatch",
    "ImageTransform",
    "PointCloud",
    "ReconstructionResult",
    "VGGTInference",
    "decode_cameras",
    "estimate_floor_alignment",
    "load_images",
    "load_preprocessed_frames",
    "reconstruct_preprocessed",
    "save_point_cloud",
    "segment_floor_images",
    "show_point_cloud",
    "to_open3d",
    "transform_observations",
    "unproject_depth",
]
