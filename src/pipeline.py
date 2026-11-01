"""The whole pipeline: video or images → frames → cameras → depth → splat → tour → demo.

Every stage writes into its own folder under the run directory and is skipped when its
output already exists, so an interrupted run resumes where it stopped. ``report.json``
collects each stage's summary and the quality checks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import PipelineConfig
from .depth import align_depth_maps, compute_depth
from .ingest import IMAGE_SUFFIXES, ingest
from .sfm import load_solution, solve_cameras

STAGES = ("ingest", "sfm", "depth", "splat", "tour", "demo")


def _validate_run(source: Path, output: Path, config: PipelineConfig) -> dict:
    """Refuse to reuse stage caches for a different capture or configuration."""
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(source)

    settings = json.loads(json.dumps(config.to_dict()))
    identity = {
        "source": str(source),
        "inputs": [
            {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in paths
        ],
        "config": settings,
    }
    identity_path = output / "run.json"
    mismatch = identity_path.exists() and json.loads(identity_path.read_text()) != identity
    config_path = output / "config.json"
    if config_path.exists():
        mismatch |= json.loads(config_path.read_text()) != settings
    listing_path = output / "frames" / "frames.json"
    if listing_path.exists():
        mismatch |= json.loads(listing_path.read_text()).get("source") != str(source)
    if mismatch:
        raise ValueError("The run's source or configuration changed; choose a new --out directory")
    return identity


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
    last = STAGES.index(until)
    output = Path(output_dir).resolve()
    source = Path(source).resolve()
    identity = _validate_run(source, output, config)

    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(identity, indent=1) + "\n")
    log = RunLog(output / "log.txt")
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=1) + "\n")
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report["source"] = str(Path(source).resolve())

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
    save()
    if last < 1:
        return report

    sfm = timed("sfm", lambda: solve_cameras(output / "frames", output / "sfm", config.sfm, device))
    report["sfm"] = {
        k: sfm[k] for k in ("frames", "pairs", "loop_check", "models", "largest_model")
    }
    log(f"registered {sfm['largest_model']['registered']}/{sfm['frames']} frames")
    save()
    solution = load_solution(output / "sfm", output / "frames")
    if last < 2:
        return report

    depth_dir = None
    if config.depth.enabled:

        def depth_stage():
            folder = compute_depth(solution.image_paths, output / "depth", config.depth, device)
            fits = align_depth_maps(solution, folder)
            report["depth"] = {"images": len(solution.image_paths), "metric_fits": len(fits)}
            return folder

        depth_dir = timed("depth", depth_stage)
        save()
    if last < 3:
        return report

    from .splat_stage import build_splat

    splat = timed("splat", lambda: build_splat(solution, output, config, depth_dir, device, log))
    report["splat"] = splat
    log(f"splat: {json.dumps(splat.get('val'))}")
    save()
    if last < 4:
        return report

    from .tour import render_tour

    report["tour"] = timed("tour", lambda: render_tour(output, config.tour, device, log))
    save()
    if last < 5:
        return report

    from .demo import render_demo

    report["demo"] = timed("demo", lambda: render_demo(output, config.demo, device, log))
    save()
    return report
