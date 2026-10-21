"""Large-GPU reconstruction command and CPU-only saved-cloud viewer."""

import argparse
import json
from pathlib import Path

import numpy as np

from .preprocessed import reconstruct_preprocessed
from .vggt import VGGTInference
from .visualization import save_point_cloud, show_point_cloud


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Joint VGGT inference on preprocessed frames, with Open3D PLY export/viewing."
    )
    parser.add_argument(
        "preprocessing_dir", type=Path, nargs="?", help="Frozen preprocessing directory"
    )
    parser.add_argument(
        "--view", type=Path, metavar="PLY", help="View an existing cloud; no model loading"
    )
    parser.add_argument("--output-dir", type=Path, help="Default: sibling reconstruction directory")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:0")
    parser.add_argument(
        "--model-id", default="facebook/VGGT-1B", help="Hugging Face ID or local model directory"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--image-size", type=int, default=518, help="Padded side length, multiple of 14"
    )
    parser.add_argument("--pool", choices=("geometry", "appearance"), default="geometry")
    parser.add_argument(
        "--stride", type=int, default=1, help="Take every Nth frame in capture order"
    )
    parser.add_argument(
        "--max-frames", type=int, help="Use first N selected frames (partial-scene preview)"
    )
    parser.add_argument("--source", choices=("depth", "point_head"), default="depth")
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=25.0,
        help="Discard bottom percentile [0,100]",
    )
    parser.add_argument(
        "--min-confidence", type=float, help="Optional absolute, uncalibrated confidence cutoff"
    )
    parser.add_argument(
        "--max-points", type=int, default=1_000_000, help="Cap exported/displayed points"
    )
    parser.add_argument(
        "--no-view", action="store_true", help="Save artifacts without opening a GUI"
    )
    args = parser.parse_args(argv)

    if args.view is not None:
        if args.preprocessing_dir is not None or args.no_view:
            parser.error("--view cannot be combined with preprocessing_dir or --no-view")
        show_point_cloud(args.view)
        return 0
    if args.preprocessing_dir is None:
        parser.error("Provide preprocessing_dir, or --view POINT_CLOUD.ply")
    if (
        args.stride < 1
        or (args.max_frames is not None and args.max_frames < 1)
        or args.max_points < 1
    ):
        parser.error("stride, max-frames and max-points must be positive")
    if not np.isfinite(args.confidence_percentile) or not 0 <= args.confidence_percentile <= 100:
        parser.error("confidence-percentile must be in [0,100]")
    if args.min_confidence is not None and not np.isfinite(args.min_confidence):
        parser.error("min-confidence must be finite")

    source_dir = args.preprocessing_dir.resolve()
    output_dir = (args.output_dir or source_dir.parent / "reconstruction").resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        parser.error("Reconstruction outputs must be outside the frozen preprocessing directory")

    model = VGGTInference(
        args.model_id,
        device=args.device,
        precision=args.precision,
        image_size=args.image_size,
        local_files_only=args.local_files_only,
    )
    print(
        f"Joint VGGT inference: {source_dir} ({args.device}); no independent chunking", flush=True
    )
    try:
        result = reconstruct_preprocessed(
            source_dir, model, pool=args.pool, stride=args.stride, max_frames=args.max_frames
        )
    finally:
        model.unload()

    cloud = result.point_cloud(
        source=args.source,
        confidence_percentile=args.confidence_percentile,
        min_confidence=args.min_confidence,
        max_points=args.max_points,
    )
    if not len(cloud.points):
        raise ValueError("No points survived filtering; lower the confidence cutoffs")
    result.save(output_dir)
    path = save_point_cloud(cloud, output_dir / "point_cloud.ply")
    np.savez_compressed(
        output_dir / "point_cloud.npz",
        points=cloud.points,
        colors=cloud.colors,
        confidence=cloud.confidence,
        frame_indices=cloud.frame_indices,
        pixel_xy=cloud.pixel_xy,
    )
    (output_dir / "preview.json").write_text(
        json.dumps(
            {
                "source": args.source,
                "confidence_percentile": args.confidence_percentile,
                "min_confidence": args.min_confidence,
                "max_points": args.max_points,
                "pool": args.pool,
                "stride": args.stride,
                "max_frames": args.max_frames,
                "frame_count": len(result.frames),
                "point_count": len(cloud.points),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(f"Saved {len(cloud.points):,} points from {len(result.frames)} views to {path}")
    if not args.no_view:
        show_point_cloud(cloud, title=f"Namma Space — {source_dir.parent.name}")
    return 0
