"""Stage contract for dense full-resolution structure from motion (SfM).

Input
-----
A preprocessing output directory (``manifest.json``, ``all_frames.csv``) and the source
video it references. Optionally, a camera file in the loop-closure ``cameras.npz``
schema (``frame_ids``, ``camera_to_world``, model-grid ``intrinsics``) is used only as an
independent check of the solved cameras and as the initial focal length.

Output (``<output_dir>/``)
--------------------------
- ``frames.json``: selected frames in time order: ``name`` (COLMAP image name),
  ``frame_id`` (``<clip_id>:<pts>``), ``timestamp_s``, ``source_index``, ``sharpness`` and
  ``geometry`` (true for preprocessing geometry frames).
- ``images/``: upright full-resolution RGB JPEGs exactly as decoded from the video.
- ``features/``, ``pairs.json``, ``matches.npz``: ALIKED keypoints (pixel coordinates in
  ``images/``, integer-centre convention), match pairs with their source
  (``sequential``/``retrieval``) and raw LightGlue matches.
- ``database.db``, ``sparse/<k>/``: COLMAP database and raw models (one shared camera).
- ``undistorted/{images,sparse}``: the largest model undistorted to one PINHOLE camera.
  This is the hand-off to Gaussian splatting.
- ``report.json`` and ``inspection/``: registration statistics, reprojection errors and
  the comparison with VGGT cameras.

Conventions
-----------
COLMAP stores world-to-camera rotations and translations (OpenCV axes: x right,
y down, z forward). Pixel coordinates follow COLMAP, where the centre of the top-left
pixel is (0.5, 0.5); ALIKED keypoints (integer-centre) are shifted by +0.5 when written to
the database. Scene units are arbitrary (SfM gauge).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SfmConfig:
    # Frame selection: one sharpest frame per window, plus every geometry frame.
    frames_per_second: float = 4.0
    avoid_flags: tuple[str, ...] = ("suspected_blur", "high_motion")
    jpeg_quality: int = 95
    decode_threads: int = 8

    # Features and matching.
    feature_long_side: int = 1600
    max_keypoints: int = 4096
    detection_threshold: float = 0.005
    sequential_window: int = 8
    descriptor_model: str = "facebook/dinov2-base"
    retrieval_top_k: int = 8
    retrieval_min_similarity: float = 0.45
    retrieval_exclusion_s: float = 4.0

    # Retrieval pairs are kept only with clear, non-planar epipolar support; repeated
    # grilles and floor inlays otherwise create false loops (see LOOP_CLOSURE.md).
    retrieval_min_inliers: int = 60
    retrieval_max_planar_fraction: float = 0.7
    ransac_px: float = 2.0

    # Mapping.
    camera_model: str = "SIMPLE_RADIAL"
    mapper: str = "global"  # "global" (GLOMAP) or "incremental"
    min_inliers: int = 25
