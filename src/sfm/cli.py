"""Command line entry point for dense full-resolution SfM."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .contracts import SfmConfig
from .pipeline import run_sfm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve full-resolution cameras for densely selected video frames"
    )
    parser.add_argument("preprocess_dir", type=Path, help="Preprocessing output directory")
    parser.add_argument("--output-dir", type=Path, help="Default: <preprocess_dir>/../sfm")
    parser.add_argument(
        "--vggt-cameras", type=Path, help="cameras.npz used to check SfM cameras and seed focal"
    )
    parser.add_argument("--fps", type=float, help="Selected frames per second of video")
    parser.add_argument("--mapper", choices=("global", "incremental"))
    parser.add_argument("--camera-model", choices=("SIMPLE_RADIAL", "OPENCV"))
    parser.add_argument("--retrieval-top-k", type=int, help="0 uses sequential pairs only")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    overrides = {
        key: value
        for key, value in {
            "frames_per_second": args.fps,
            "mapper": args.mapper,
            "camera_model": args.camera_model,
            "retrieval_top_k": args.retrieval_top_k,
        }.items()
        if value is not None
    }
    config = replace(SfmConfig(), **overrides)
    output = args.output_dir or args.preprocess_dir.parent / "sfm"
    report = run_sfm(
        args.preprocess_dir,
        output,
        config=config,
        vggt_cameras=args.vggt_cameras,
        device=args.device,
    )
    summary = {key: report[key] for key in ("pairs", "models", "timings")}
    summary["largest_model"] = {
        k: v for k, v in report["largest_model"].items() if k != "unregistered_timestamps_s"
    }
    if report["vggt_comparison"]:
        summary["vggt_comparison"] = {
            k: v
            for k, v in report["vggt_comparison"].items()
            if k not in ("per_frame", "alignment")
        }
    print(json.dumps(summary, indent=1))
    print(f"Report: {output / 'report.json'}; splat input: {report['undistorted']}")
    return 0
