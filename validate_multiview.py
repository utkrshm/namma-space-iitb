"""Audit every exported multi-view refinement point against its source evidence."""

import argparse
import json
from pathlib import Path

from src.refinement.validation import audit_refinement


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(audit_refinement(args.output_dir, args.input_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
