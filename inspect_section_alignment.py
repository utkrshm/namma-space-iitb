"""Render headless visual diagnostics for a saved section-alignment result."""

import argparse
import json
from pathlib import Path

from src.reconstruction.section_inspection import inspect_section_cloud


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="section_alignment output directory")
    parser.add_argument(
        "--baseline-dir", type=Path, required=True, help="original reconstruction directory"
    )
    args = parser.parse_args(argv)
    report = inspect_section_cloud(args.output_dir, args.baseline_dir)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
