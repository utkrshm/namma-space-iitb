"""Command line entry point for overlapping reconstruction sections."""

import argparse
from pathlib import Path

from .sections import align_sections


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Infer overlapping VGGT reconstruction sections")
    parser.add_argument("preprocessing_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--section-size", type=int, default=36)
    parser.add_argument("--overlap", type=int, default=11)
    parser.add_argument(
        "--tail-section-size",
        type=int,
        help="Replace the final full window with smaller overlapping tail windows",
    )
    parser.add_argument(
        "--tail-context-frames",
        type=int,
        default=0,
        help="Extend tail windows earlier for a stronger alignment bridge",
    )
    parser.add_argument(
        "--tail-start-section",
        type=int,
        help="Zero-based normal-window index to replace with tail-sized windows",
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--model-id", default="facebook/VGGT-1B")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument(
        "--skip-inference", action="store_true", help="Reuse completed inference and resume"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-ply", action="store_true")
    parser.add_argument("--no-icp", action="store_true", help="Skip optional held-out-checked ICP")
    parser.add_argument("--no-revisits", action="store_true", help="Adjacent-only diagnostic graph")
    parser.add_argument("--max-points", type=int, default=1_000_000)
    args = parser.parse_args(argv)
    source = args.preprocessing_dir.resolve()
    output = (args.output_dir or source.parent / "section_alignment").resolve()
    align_sections(
        source,
        output,
        inference_only=args.inference_only,
        skip_inference=args.skip_inference,
        section_size=args.section_size,
        overlap=args.overlap,
        tail_section_size=args.tail_section_size,
        tail_context_frames=args.tail_context_frames,
        tail_start_section=args.tail_start_section,
        image_size=args.image_size,
        device=args.device,
        precision=args.precision,
        resume=not args.no_resume,
        local_files_only=args.local_files_only,
        model_id=args.model_id,
        save_ply=not args.no_ply,
        try_icp=not args.no_icp,
        use_revisits=not args.no_revisits,
        max_points=args.max_points,
    )
    return 0
