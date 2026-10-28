"""Dense SfM pipeline: frames → features → pairs → matches → mapping → undistortion → report."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import SfmConfig
from .evaluation import (
    camera_poses,
    compare_with_vggt,
    model_report,
    render_inspection,
    trajectory_continuity,
)
from .frames import export_frames
from .loops import consistent_loops
from .mapping import build_database, initial_camera_params, run_mapper, undistort
from .matching import build_pairs, extract_features, global_descriptors, load_features, match_pairs


def check_loops(matches, listing, features, frames, config: SfmConfig):
    """Filter retrieval pairs against the reference model; returns matches and a summary."""
    import pycolmap

    reference = pycolmap.Reconstruction(config.loop_reference)
    focal, cx, cy = next(iter(reference.cameras.values())).params[:3]
    intrinsics = np.array([[focal, 0, cx - 0.5], [0, focal, cy - 0.5], [0, 0, 1.0]])
    kept, report = consistent_loops(
        matches,
        listing,
        features,
        frames,
        camera_poses(reference, frames),
        intrinsics,
        config.loop_max_rotation_deg,
        config.ransac_px,
    )
    errors = [r["rotation_error_deg"] for r in report if "rotation_error_deg" in r]
    summary = {
        "reference": config.loop_reference,
        "retrieval_checked": len(report),
        "retrieval_accepted": sum(r["accepted"] for r in report),
        "rotation_error_deg_percentiles": {
            str(q): float(np.percentile(errors, q)) for q in (10, 50, 75, 90, 99)
        }
        if errors
        else {},
    }
    return kept, summary


def run_sfm(
    preprocess_dir: str | Path,
    output_dir: str | Path,
    *,
    config: SfmConfig | None = None,
    vggt_cameras: str | Path | None = None,
    device: str = "cuda:0",
) -> dict:
    """Solve full-resolution cameras for densely selected video frames.

    Every step caches its result in ``output_dir``; mapping always reruns because it is
    cheap compared with decoding, feature extraction and matching.
    """
    config = config or SfmConfig()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    vggt_cameras = Path(vggt_cameras) if vggt_cameras else None
    timings = {}

    start = time.perf_counter()
    frames = export_frames(preprocess_dir, output, config)
    timings["frames_s"] = time.perf_counter() - start

    start = time.perf_counter()
    extract_features(output, frames, config, device)
    features = load_features(output, frames)
    descriptors = global_descriptors(output, frames, config, device)
    timestamps = np.array([frame["timestamp_s"] for frame in frames])
    pairs = build_pairs(timestamps, descriptors, config)
    matches, listing = match_pairs(output, features, pairs, config, device)
    loop_check = None
    if config.loop_reference:
        matches, loop_check = check_loops(matches, listing, features, frames, config)
    timings["matching_s"] = time.perf_counter() - start

    start = time.perf_counter()
    camera_params = initial_camera_params(frames, vggt_cameras, config)
    database = build_database(output, frames, features, matches, camera_params, config)
    models = run_mapper(output, database, config)
    if not models:
        raise RuntimeError("SfM produced no model")
    timings["mapping_s"] = time.perf_counter() - start

    start = time.perf_counter()
    undistorted = undistort(output, output / "sparse" / "0")
    timings["undistortion_s"] = time.perf_counter() - start

    model = models[0]
    poses = camera_poses(model, frames)
    vggt = (
        compare_with_vggt(poses, vggt_cameras)
        if vggt_cameras is not None and vggt_cameras.exists()
        else None
    )
    retrieval = [p for p in listing if p["kind"] == "retrieval"]
    report = {
        "config": asdict(config),
        "initial_camera_params": camera_params,
        "pairs": {
            "sequential": sum(p["kind"] == "sequential" for p in listing),
            "retrieval_candidates": len(retrieval),
            "retrieval_accepted": sum(p["accepted"] for p in retrieval),
            "matched": len(matches),
        },
        "loop_check": loop_check,
        "models": [m.num_reg_images() for m in models],
        "continuity": trajectory_continuity(poses, frames),
        "largest_model": model_report(model, frames),
        "vggt_comparison": None if vggt is None else {k: v for k, v in vggt.items()},
        "undistorted": str(undistorted),
        "timings": timings,
    }
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    render_inspection(
        model, frames, poses, vggt, vggt_cameras, output / "inspection" / "topdown.png"
    )
    return report
