"""The whole pipeline: video or images → frames → cameras → depth → splat → fill → tour → demo.

Every stage writes into its own folder under the run directory and is skipped when its
output already exists, so an interrupted run resumes where it stopped. ``report.json``
collects each stage's summary and the quality checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import PipelineConfig
from .depth import compute_depth
from .ingest import ingest
from .sfm import load_solution, solve_cameras

STAGES = ("ingest", "sfm", "depth", "splat", "fill", "tour", "demo")


class RunLog:
    """Print and append to ``<run>/log.txt`` with elapsed time."""

    def __init__(self, path: Path):
        self.path, self.started = path, time.perf_counter()

    def __call__(self, message: str) -> None:
        line = f"[{time.perf_counter() - self.started:8.0f}s] {message}"
        print(line, flush=True)
        with self.path.open("a") as stream:
            stream.write(line + "\n")


def run(
    source,
    output_dir,
    config: PipelineConfig | None = None,
    *,
    until: str = "demo",
    device: str = "cuda:0",
) -> dict:
    """Run the stages up to and including ``until``; returns the report."""
    config = config or PipelineConfig()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = RunLog(output / "log.txt")
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=1) + "\n")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["source"] = str(Path(source).resolve())
    last = STAGES.index(until)

    def save():
        report_path.write_text(json.dumps(report, indent=1, default=str) + "\n")

    def timed(stage, function):
        log(f"stage {stage}")
        start = time.perf_counter()
        value = function()
        report.setdefault("timings_s", {})[stage] = time.perf_counter() - start
        save()
        return value

    listing = timed("ingest", lambda: ingest(source, output / "frames", config.ingest))
    report["ingest"] = {
        "kind": listing["kind"],
        "frames": len(listing["frames"]),
        "decoded_frames": listing.get("decoded_frames"),
        "motion_step": listing.get("motion_step"),
    }
    log(f"{len(listing['frames'])} frames")
    if last < 1:
        return report

    sfm = timed("sfm", lambda: solve_cameras(output / "frames", output / "sfm", config.sfm, device))
    report["sfm"] = {
        k: sfm[k] for k in ("frames", "pairs", "loop_check", "models", "largest_model")
    }
    log(f"registered {sfm['largest_model']['registered']}/{sfm['frames']} frames")
    solution = load_solution(output / "sfm", output / "frames")
    if last < 2:
        return report

    depth_dir = None
    if config.depth.enabled:
        depth_dir = timed(
            "depth",
            lambda: compute_depth(solution.image_paths, output / "depth", config.depth, device),
        )
    if last < 3:
        return report

    from .splat_stage import build_splat

    splat = timed("splat", lambda: build_splat(solution, output, config, depth_dir, device, log))
    report["splat"] = splat
    log(f"splat: {json.dumps(splat.get('val'))}")
    if last < 4:
        return report

    if config.fill.enabled:
        from .fill import fill_gaps

        report["fill"] = timed("fill", lambda: fill_gaps(output, config, device, log))
    if last < 5:
        return report

    from .tour import render_tour

    report["tour"] = timed("tour", lambda: render_tour(output, config.tour, device, log))
    if last < 6:
        return report

    from .demo import render_demo

    report["demo"] = timed("demo", lambda: render_demo(output, config.demo, device, log))
    save()
    return report
