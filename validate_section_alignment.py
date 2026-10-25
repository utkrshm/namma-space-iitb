"""Build CPU-verified section revisit constraints and validate section outputs."""

import argparse
import json
from pathlib import Path

from src.reconstruction.section_benchmark import (
    build_benchmark_section_alignment,
    build_revisit_edges,
)
from src.reconstruction.sections import run_section_alignment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--preprocessing-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--build-revisits-only", action="store_true")
    parser.add_argument(
        "--benchmark-only", action="store_true", help="Evaluate existing final graph and PLY"
    )
    args = parser.parse_args(argv)

    if args.benchmark_only:
        result = build_benchmark_section_alignment(
            args.output_dir, args.baseline_dir, args.preprocessing_dir
        )
        print(json.dumps(result, indent=2))
        return 0

    edges, revisit_report = build_revisit_edges(args.output_dir, args.preprocessing_dir)
    if not args.build_revisits_only:
        run_section_alignment(args.output_dir, extra_edges=edges)
        result = build_benchmark_section_alignment(
            args.output_dir, args.baseline_dir, args.preprocessing_dir
        )
        print(json.dumps({"revisits": revisit_report, "benchmark": result}, indent=2))
    else:
        print(json.dumps(revisit_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
