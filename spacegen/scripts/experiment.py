"""Short A/B splat trainings on a finished run, and per-frame comparisons.

    uv run --no-sync python scripts/experiment.py train RUN NAME [--steps N] [--set splat.x=y]
    uv run --no-sync python scripts/experiment.py compare RUN BASE NAME [NAME ...]

``train`` reuses RUN's frames, cameras (with re-sliced extra frames), depth and MVS, and
writes ``RUN/exp/NAME/splat``; the configuration is RUN's ``config.json`` with ``steps``
(``refine_stop`` scaled with it) and the overrides applied. ``compare`` prints held-out
means (of the metrics all runs have) and, per frame, the PSNR change against BASE (same
cameras and held-out frames, or it refuses).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from spacegen.config import PipelineConfig, parse_overrides


def train(run: Path, name: str, steps: int, overrides: list[str]) -> None:
    from spacegen.reslice import run_solution
    from spacegen.splat_stage import build_splat

    config = PipelineConfig.from_dict(json.loads((run / "config.json").read_text()))
    sections = parse_overrides(overrides)
    splat = sections.setdefault("splat", {})
    splat["steps"] = steps
    # Densification stops at the same share of the run as in the full training (25k of
    # 30k), not at 25k: a 10k-step test would otherwise densify until its last step.
    splat.setdefault("refine_stop", round(config.splat.refine_stop * steps / config.splat.steps))
    config = config.replace(**sections)
    output = run / "exp" / name
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=1) + "\n")
    (output / "overrides.json").write_text(json.dumps({"steps": steps, "set": overrides}) + "\n")
    metrics = build_splat(run_solution(run), output, config, run / "depth", "cuda:0", print)
    print(json.dumps(metrics.get("val"), indent=1))


def held_out(run: Path, name: str) -> dict:
    folder = run if name == "." else run / "exp" / name
    path = folder / "splat" / "evaluation" / "val" / "metrics.json"
    return json.loads(path.read_text())


def compare(run: Path, base: str, names: list[str]) -> None:
    reference = held_out(run, base)
    frames = {r["name"]: r for r in reference["frames"]}
    reports = {name: held_out(run, name) for name in [base, *names]}
    # Only the metrics every run has (older runs lack psnr_nb).
    keys = [k for k in reference["mean"] if all(k in r["mean"] for r in reports.values())]
    print(f"{'run':>20} " + " ".join(f"{k:>8}" for k in keys))
    for name, report in reports.items():
        if {r["name"] for r in report["frames"]} != set(frames):
            sys.exit(f"{name}: different held-out frames from {base}; not comparable")
        print(f"{name:>20} " + " ".join(f"{report['mean'][k]:8.3f}" for k in keys))
    for name in names:
        delta = {
            r["name"]: r["psnr"] - frames[r["name"]]["psnr"] for r in held_out(run, name)["frames"]
        }
        values = np.array(list(delta.values()))
        worst = sorted(delta, key=delta.get)[:5]
        print(
            f"{name}: PSNR change per frame median {np.median(values):+.2f}, "
            f"{(values < -1).sum()} frames lose > 1 dB, {(values > 1).sum()} gain > 1 dB; "
            f"worst {', '.join(f'{w} {delta[w]:+.1f}' for w in worst)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train")
    trainer.add_argument("run", type=Path)
    trainer.add_argument("name")
    trainer.add_argument("--steps", type=int, default=10_000)
    trainer.add_argument("--set", action="append", default=[])
    comparer = commands.add_parser("compare")
    comparer.add_argument("run", type=Path)
    comparer.add_argument("base", help='an experiment name, or "." for the run itself')
    comparer.add_argument("names", nargs="+")
    args = parser.parse_args()

    if args.command == "train":
        train(args.run.resolve(), args.name, args.steps, args.set)
    else:
        compare(args.run.resolve(), args.base, args.names)


if __name__ == "__main__":
    main()
