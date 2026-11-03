"""Low-resolution tour and demo of a finished run, rendered on the CPU (no GPU needed).

    uv run --no-sync python scripts/cpu_preview.py RUN [--only tour|demo] [--width 480] [--fps 4]

Runs the pipeline's own `render_tour` / `render_demo` with RUN's ``config.json`` at
``--width`` (16:9), ``--fps`` and no supersampling, on the CPU rasterizer
(`spacegen.splat.raster`), into ``RUN/cpu_preview/{tour,demo}``. The run's own videos are
left alone; ``scripts/contact_sheet.py RUN/cpu_preview`` makes sheets of the previews.
A 480-wide frame of a 1.5M-Gaussian room takes 5–15 s, so expect tens of minutes.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from spacegen.config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--only", choices=("tour", "demo"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    run = args.run.resolve()
    config = PipelineConfig.from_dict(json.loads((run / "config.json").read_text()))
    size = {"width": args.width, "height": round(args.width * 9 / 16 / 2) * 2}
    preview = {**size, "fps": args.fps, "supersample": 1.0}
    started = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - started:7.0f}s] {message}", flush=True)

    for stage in [args.only] if args.only else ["tour", "demo"]:
        output = run / "cpu_preview" / stage
        if stage == "tour":
            from spacegen.tour import render_tour

            render_tour(run, dataclasses.replace(config.tour, **preview), "cpu", log, output)
        else:
            from spacegen.demo import render_demo

            render_demo(run, dataclasses.replace(config.demo, **preview), "cpu", log, output)
        log(f"{stage} written to {output}")


if __name__ == "__main__":
    main()
