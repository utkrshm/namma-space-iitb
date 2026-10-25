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
from .section_benchmark import build_benchmark_section_alignment, build_revisit_edges
from .section_evaluation import correspondence_metrics, evaluate_icp
from .section_registration import SectionEdge, Sim3, optimize_section_graph, robust_sim3
from .sections import (
    SectionSpec,
    align_sections,
    create_sections,
    run_section_alignment,
    run_section_inference,
)
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
    "SectionEdge",
    "SectionSpec",
    "Sim3",
    "VGGTInference",
    "align_sections",
    "build_benchmark_section_alignment",
    "build_revisit_edges",
    "correspondence_metrics",
    "create_sections",
    "decode_cameras",
    "estimate_floor_alignment",
    "evaluate_icp",
    "load_images",
    "load_preprocessed_frames",
    "optimize_section_graph",
    "reconstruct_preprocessed",
    "robust_sim3",
    "run_section_alignment",
    "run_section_inference",
    "save_point_cloud",
    "segment_floor_images",
    "show_point_cloud",
    "to_open3d",
    "transform_observations",
    "unproject_depth",
]
