"""Command line for real image-track based camera/depth refinement."""

import argparse
import json
from pathlib import Path

from .contracts import RefinementConfig
from .pipeline import refine_multiview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--tracks", type=Path, help="Reuse fingerprint-checked whole-track evidence"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pose-steps", type=int, default=800)
    parser.add_argument("--field-steps", type=int, default=800)
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--geometry-prior-weight", type=float, default=0.02)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument(
        "--flow-anchors",
        action="store_true",
        help="Add verified texture tracks outside learned-feature neighborhoods",
    )
    parser.add_argument(
        "--dense-anchors",
        action="store_true",
        help="Match weak observation groups with real indoor LoFTR",
    )
    parser.add_argument("--matching-only", action="store_true")
    parser.add_argument(
        "--export-only", action="store_true", help="Re-export fingerprint-checked saved corrections"
    )
    parser.add_argument("--no-inspect", action="store_true")
    args = parser.parse_args(argv)
    config = RefinementConfig(
        device=args.device,
        pose_steps=args.pose_steps,
        field_steps=args.field_steps,
        grid_size=args.grid_size,
        keypoints=args.keypoints,
        max_points=args.max_points,
        learning_rate=args.learning_rate,
        retrieval_top_k=0 if args.no_retrieval else 5,
        geometry_prior_weight=args.geometry_prior_weight,
    )
    from dataclasses import replace

    config = replace(config, flow_anchors=args.flow_anchors, dense_anchors=args.dense_anchors)
    result = refine_multiview(
        args.input_dir,
        args.output_dir,
        config=config,
        tracks_path=args.tracks,
        matching_only=args.matching_only,
        export_only=args.export_only,
        inspect=not args.no_inspect,
    )
    print(json.dumps(result, indent=2))
    return 0
