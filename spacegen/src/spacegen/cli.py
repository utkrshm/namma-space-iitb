"""Command line: ``spacegen run <video|image folder> --out <dir>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import STAGES, run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="spacegen", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    runner = commands.add_parser("run", help="Run the whole pipeline (resumable)")
    runner.add_argument("source", type=Path, help="Video file or folder of images")
    runner.add_argument("--out", type=Path, required=True, help="Run directory")
    runner.add_argument("--config", type=Path, help="TOML overriding configs/default.toml")
    runner.add_argument("--until", choices=STAGES, default="demo")
    runner.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one setting, e.g. --set splat.steps=7000 (value parsed as JSON)",
    )
    runner.add_argument("--device", default="cuda:0")

    bench = commands.add_parser("bench", help="Run and score the benchmark scenes")
    bench.add_argument("--data", type=Path, required=True, help="Benchmark data folder")
    bench.add_argument("--out", type=Path, required=True)
    bench.add_argument("--scenes", nargs="*", help="Subset of scene names")
    bench.add_argument("--config", type=Path)
    bench.add_argument("--set", action="append", default=[])
    bench.add_argument("--device", default="cuda:0")

    args = parser.parse_args(argv)
    config = PipelineConfig.from_toml(args.config)
    overrides: dict[str, dict] = {}
    for item in args.set:
        key, value = item.split("=", 1)
        section, name = key.split(".", 1)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides.setdefault(section, {})[name] = parsed
    if overrides:
        config = config.replace(**overrides)

    if args.command == "run":
        report = run(args.source, args.out, config, until=args.until, device=args.device)
        print(
            json.dumps(
                {k: report.get(k) for k in ("splat", "fill", "tour", "demo")}, indent=1, default=str
            )
        )
        return 0
    from .bench import run_benchmarks

    run_benchmarks(args.data, args.out, config, args.scenes, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
