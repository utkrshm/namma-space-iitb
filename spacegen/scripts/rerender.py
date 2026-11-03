"""Render a finished run's tour and demo again with the current code (needs the GPU).

    uv run --no-sync python scripts/rerender.py RUN [--only tour|demo]

Uses RUN's ``config.json`` as it is (unlike ``spacegen run``, which rewrites it) and its
existing splat, fill and scene frame. The previous ``tour/`` and ``demo/`` folders are
kept as ``tour.prev/`` and ``demo.prev/`` (replacing older ones), and ``report.json``
gets the new summaries.
"""

import argparse
import json
import shutil
from pathlib import Path

from spacegen.config import PipelineConfig
from spacegen.contracts import write_text_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--only", choices=("tour", "demo"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run = args.run.resolve()
    config = PipelineConfig.from_dict(json.loads((run / "config.json").read_text()))
    stages = [args.only] if args.only else ["tour", "demo"]
    report_path = run / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    for stage in stages:
        folder, previous = run / stage, run / f"{stage}.prev"
        if folder.exists():
            shutil.rmtree(previous, ignore_errors=True)
            folder.rename(previous)
        if stage == "tour":
            from spacegen.tour import render_tour

            report["tour"] = render_tour(run, config.tour, args.device, print)
        else:
            from spacegen.demo import render_demo

            report["demo"] = render_demo(run, config.demo, args.device, print)
        write_text_atomic(report_path, json.dumps(report, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
