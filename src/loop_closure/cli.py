"""Command line entry point for loop closure on a section-alignment export."""

import argparse
from dataclasses import replace
from pathlib import Path

from .contracts import LoopConfig
from .pipeline import DEFAULT_SLICE_TIMES, run_loop_closure, summary_table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect loops and close them with VGGT loop chunks (A) and/or bundle adjustment (B)"
    )
    parser.add_argument(
        "sections_dir", type=Path, help="Existing section_alignment directory (read-only)"
    )
    parser.add_argument("--output-dir", type=Path, help="Default: <sections_dir>/../loop_closure")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("a", "b", "m"),
        default=["a", "b", "m"],
        help="a: VGGT loop chunks; b: bundle adjustment; m: bundle adjustment + square-walls prior",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--allow-download", action="store_true", help="Allow fetching model weights"
    )
    parser.add_argument("--temporal-exclusion-s", type=float)
    parser.add_argument("--min-similarity", type=float)
    parser.add_argument("--chunk-half-window", type=int)
    parser.add_argument("--max-loop-chunks", type=int)
    parser.add_argument("--slice-times", type=float, nargs="*", default=list(DEFAULT_SLICE_TIMES))
    args = parser.parse_args(argv)

    overrides = {
        key: value
        for key, value in {
            "temporal_exclusion_s": args.temporal_exclusion_s,
            "min_similarity": args.min_similarity,
            "chunk_half_window": args.chunk_half_window,
            "max_loop_chunks": args.max_loop_chunks,
        }.items()
        if value is not None
    }
    config = replace(LoopConfig(), **overrides)
    output = args.output_dir or args.sections_dir.parent / "loop_closure"

    comparison = run_loop_closure(
        args.sections_dir,
        output,
        methods=tuple(args.methods),
        config=config,
        device=args.device,
        precision=args.precision,
        local_files_only=not args.allow_download,
        slice_times=tuple(args.slice_times),
    )
    print(summary_table(comparison))
    print(f"Results: {output / 'comparison.json'}; renders: {output / 'inspection'}")
    return 0
