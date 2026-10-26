"""Run real LoFTR positive/negative controls and audit held-out flow protection."""

import argparse
import json
from pathlib import Path

from src.refinement.contracts import RefinementConfig
from src.refinement.controls import run_refinement_controls
from src.refinement.inputs import load_refinement_scene
from src.refinement.matching import load_observation_tracks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--negative-input-dir", type=Path)
    args = parser.parse_args(argv)

    scene = load_refinement_scene(args.input_dir)
    tracks = load_observation_tracks(args.tracks, scene)
    negative_scene = (
        load_refinement_scene(args.negative_input_dir) if args.negative_input_dir else None
    )
    result = run_refinement_controls(
        scene, tracks, args.output_dir, RefinementConfig(device=args.device), negative_scene
    )
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
