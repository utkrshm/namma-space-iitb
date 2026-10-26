"""End-to-end loop closure: detect loops, run methods A/B, compare on held-out evidence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bundle_adjustment import bundle_adjust
from .contracts import FrameMatches, LoopConfig, LoopPair
from .detection import FeatureBank, MatchStore, detect_loops
from .evaluation import correspondence_disagreement, render_camera_slices, render_comparison
from .loop_chunks import build_loop_edges
from .store import SectionCache, build_frame_models, export_cloud, load_graph_cameras, save_cameras

DEFAULT_SLICE_TIMES = (83.2, 117.35, 122.4, 158.6, 171.0, 184.4, 209.0)


def _check_output(sections_dir: Path, output_dir: Path) -> None:
    sections = sections_dir.resolve()
    output = output_dir.resolve()
    if output == sections or output.is_relative_to(sections):
        raise ValueError("Loop-closure outputs must be outside the read-only section directory")


def run_loop_closure(
    sections_dir: str | Path,
    output_dir: str | Path,
    *,
    methods: tuple[str, ...] = ("a", "b", "m"),
    config: LoopConfig | None = None,
    device: str = "cuda:0",
    precision: str = "bfloat16",
    local_files_only: bool = True,
    slice_times: tuple[float, ...] = DEFAULT_SLICE_TIMES,
) -> dict:
    """Run the requested methods and write `comparison.json` plus inspection renders.

    Method "a" adds VGGT loop-chunk edges to the section graph. Method "b" bundle-adjusts
    from the baseline graph and, when "a" also ran, from method A's graph as well. Method
    "m" is bundle adjustment plus a per-frame square-walls yaw prior, started from method A
    when available.
    """
    config = config or LoopConfig()
    sections_dir, output_dir = Path(sections_dir), Path(output_dir)
    _check_output(sections_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = SectionCache(sections_dir)
    baseline = load_graph_cameras(sections_dir, sections_dir)
    owners = [int(o) for o in baseline["owners"]]
    save_cameras(output_dir / "baseline_cameras.npz", baseline)
    matches = MatchStore(
        output_dir, cache, owners, config, lambda: FeatureBank(cache, owners, config, device)
    )
    timings = {}

    started = time.perf_counter()
    loops, loop_report = detect_loops(cache, owners, matches, config, output_dir, device)
    timings["detection_s"] = time.perf_counter() - started
    print(
        f"Loops: {loop_report['candidate_count']} candidates, {loop_report['verified_count']} verified, "
        f"{sum(p.role == 'chunk' for p in loops)} chunk centres, "
        f"{sum(p.role == 'holdout' for p in loops)} held out",
        flush=True,
    )

    def loop_matches(pair: LoopPair) -> FrameMatches:
        return matches.get(pair.frame_a, pair.frame_b, "loop")

    sequential = []
    for frame in range(len(cache.frame_ids)):
        for offset in config.sequential_offsets:
            if frame + offset < len(cache.frame_ids):
                sequential.append(matches.get(frame, frame + offset, "sequential"))
    used_loops = [loop_matches(p) for p in loops if p.role in ("chunk", "train")]
    held_out = [loop_matches(p) for p in loops if p.role == "holdout"]

    solutions = {"baseline": baseline}
    clouds = {"baseline": sections_dir / "point_cloud.npz"}
    reports: dict[str, dict] = {}

    if "a" in methods:
        from ..reconstruction.sections import run_section_alignment

        started = time.perf_counter()
        method_dir = output_dir / "method_a"
        edges, edge_reports = build_loop_edges(
            cache,
            loops,
            loop_matches,
            config,
            method_dir,
            device=device,
            precision=precision,
            local_files_only=local_files_only,
        )
        (method_dir / "loop_edges.json").write_text(json.dumps(edge_reports, indent=2) + "\n")
        run_section_alignment(
            sections_dir,
            try_icp=False,
            extra_edges=edges,
            export_dir=method_dir,
            strict_post_graph=False,
        )
        solutions["method_a"] = load_graph_cameras(sections_dir, method_dir)
        save_cameras(method_dir / "cameras.npz", solutions["method_a"])
        clouds["method_a"] = method_dir / "point_cloud.npz"
        graph = json.loads((method_dir / "graph.json").read_text())
        reports["method_a"] = {
            "loop_chunks": len(edge_reports),
            "accepted_loop_edges": len(edges),
            "rejections": [r.get("reason") for r in edge_reports if not r.get("accepted")],
            "post_graph_failures": [
                {k: e[k] for k in ("source", "target", "normalized_median")}
                for e in graph["post_graph_validation"]
                if not e["absolute_gate_passed"]
            ],
            "optimizer": {
                k: graph["optimizer"][k]
                for k in ("initial_objective", "final_objective", "success")
            },
        }
        timings["method_a_s"] = time.perf_counter() - started

    arms = []
    loop_init = ["method_a"] if "method_a" in solutions else []
    if "b" in methods:
        arms += [(f"method_b_from_{init}", init, False) for init in ["baseline", *loop_init]]
    if "m" in methods:
        arms += [(f"method_m_from_{init}", init, True) for init in (loop_init or ["baseline"])]
    for name, init, manhattan in arms:
        started = time.perf_counter()
        method_dir = output_dir / name
        method_dir.mkdir(parents=True, exist_ok=True)
        models = build_frame_models(sections_dir, solutions[init])
        cameras, report = bundle_adjust(
            models, sequential + used_loops, config, manhattan=manhattan
        )
        save_cameras(method_dir / "cameras.npz", cameras)
        refined = build_frame_models(sections_dir, cameras)
        report["export"] = export_cloud(refined, method_dir)
        (method_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        solutions[name], clouds[name], reports[name] = (
            cameras,
            method_dir / "point_cloud.npz",
            report,
        )
        timings[f"{name}_s"] = time.perf_counter() - started

    models = {name: build_frame_models(sections_dir, cams) for name, cams in solutions.items()}
    evaluation = {
        name: {
            "held_out_loops": correspondence_disagreement(frame_models, held_out),
            "used_loops": correspondence_disagreement(frame_models, used_loops),
            "sequential": correspondence_disagreement(frame_models, sequential),
        }
        for name, frame_models in models.items()
    }
    images = render_comparison(clouds, models, output_dir / "inspection")
    images += render_camera_slices(clouds, models, list(slice_times), output_dir / "inspection")

    comparison = {
        "sections_dir": str(cache.root),
        "config": asdict(config),
        "methods": list(solutions),
        "loops": {
            "candidates": loop_report["candidate_count"],
            "verified": loop_report["verified_count"],
            "roles": {
                role: sum(p.role == role for p in loops) for role in ("chunk", "train", "holdout")
            },
            "held_out_pairs": [[m.frame_a, m.frame_b] for m in held_out],
        },
        "evaluation_note": (
            "Normalized 3D disagreement of matched pixels (distance / mean camera distance). "
            "held_out_loops were never used by any method; used_loops and sequential matches "
            "were inputs to bundle adjustment (method B) and only the chunk centres' matches "
            "gated method A."
        ),
        "evaluation": evaluation,
        "reports": reports,
        "timings_s": timings,
        "images": [str(p) for p in images],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, default=float) + "\n"
    )
    return comparison


def summary_table(comparison: dict) -> str:
    """Plain-text table of the evaluation for terminal output."""
    lines = [
        f"{'method':28s} {'held-out loops med/p90':>24s} {'used loops med':>15s} {'sequential med':>15s}"
    ]
    for name, values in comparison["evaluation"].items():
        held = values["held_out_loops"]
        lines.append(
            f"{name:28s} "
            f"{held.get('median', np.nan):11.4f}/{held.get('p90', np.nan):<11.4f} "
            f"{values['used_loops'].get('median', np.nan):15.4f} "
            f"{values['sequential'].get('median', np.nan):15.4f}"
        )
    return "\n".join(lines)
