"""Inspect actual multi-view-refined PLYs with common-view and camera diagnostics."""

import argparse
import json
from pathlib import Path

from src.refinement.inputs import load_refinement_scene
from src.refinement.inspection import inspect_refinement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--input-dir", type=Path, help="Override the saved input path after moving machines"
    )
    args = parser.parse_args()
    scene = load_refinement_scene(args.input_dir) if args.input_dir else None
    print(json.dumps(inspect_refinement(args.output_dir, scene=scene), indent=2))
