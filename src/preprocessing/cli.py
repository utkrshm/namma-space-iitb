"""CLI entry point: uv run namma-preprocess VIDEO [VIDEO ...]."""

import argparse
from dataclasses import replace
from pathlib import Path

from .config import PreprocessConfig
from .pipeline import preprocess_capture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/preprocessing.toml"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-revisits", action="store_true")
    args = parser.parse_args(argv)
    config = PreprocessConfig.from_toml(args.config)
    if args.no_revisits:
        config = replace(config, revisits=replace(config.revisits, enabled=False))
    output = args.output_dir or Path("outputs") / args.clips[0].stem / "preprocessing"
    result = preprocess_capture(args.clips, output, config, resume=not args.no_resume)
    print(
        f"{len(result.scores)} frames; {len(result.geometry)} geometry; "
        f"{len(result.appearance)} appearance; {len(result.edges)} edges\n{result.output_dir}"
    )
    return 0
