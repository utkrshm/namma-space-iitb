"""Artifact-based orchestration with content-validated, section-specific resume keys."""

import hashlib
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2

from .config import PreprocessConfig
from .contracts import (
    SCHEMA_VERSION,
    ClipInfo,
    Edge,
    GapItem,
    PreprocessResult,
    SelectedFrame,
    read_json,
    write_json,
)
from .export import export_frames, write_manifest
from .graph import build_graph
from .probe import file_hash, probe_clip
from .report import build_report
from .revisits import find_revisits, revisits_available
from .scoring import read_scores, score_clip, write_scores
from .selection import select_clip
from .storage import artifact_path, safe_clip_id
from .visualize import visualize


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _versions():
    result = {"python": platform.python_version()}
    for name in (
        "av",
        "opencv-python-headless",
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "transformers",
        "lightglue",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def preprocess_capture(clips, output_dir, config=None, resume=True) -> PreprocessResult:
    config = config or PreprocessConfig()
    if isinstance(config, (str, Path)):
        config = PreprocessConfig.from_toml(config)
    sources = [Path(p).resolve() for p in clips]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Provide one or more distinct video paths")
    ids = [safe_clip_id(p.stem) for p in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("Input clip names must yield unique clip IDs")
    root = Path(output_dir).resolve()
    if any(root == p or root in p.parents for p in sources):
        raise ValueError("Source videos must live outside the generated output directory")
    root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(config.decode.threads)
    versions = _versions()
    # A code change invalidates artifacts too, even if the config was not edited.
    code_hash = _hash({p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})
    run_path = root / "run.json"
    try:
        previous = read_json(run_path) if resume and run_path.exists() else {}
    except ValueError:
        # A schema upgrade requires fresh passes, but prior export ownership is retained
        # solely to remove stale managed images after a successful new export.
        previous = json.loads(run_path.read_text())
        for entry in previous.get("passes", {}).values():
            entry.pop("key", None)
    run = {
        "started_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "inputs": [{"path": str(p), "sha256": file_hash(p)} for p in sources],
        "versions": versions,
        "code_hash": code_hash,
        "passes": {},
    }

    def stage(name, dependencies, settings, action):
        key = _hash(
            {
                "inputs": run["inputs"],
                "config": settings,
                "dependencies": dependencies,
                "code": code_hash,
                "versions": versions,
                "schema": SCHEMA_VERSION,
            }
        )
        old = previous.get("passes", {}).get(name, {})
        started = time.perf_counter()
        valid = bool(old.get("key") == key and old.get("artifacts"))
        if valid:
            valid = all(
                artifact_path(root, path).is_file()
                and file_hash(artifact_path(root, path)) == digest
                for path, digest in old["artifacts"].items()
            )
        if valid:
            entry = {**old, "resumed": True, "validation_s": time.perf_counter() - started}
            print(f"{name}: resumed", flush=True)
        else:
            print(f"{name}: running", flush=True)
            artifacts = action()
            entry = {
                "key": key,
                "resumed": False,
                "duration_s": time.perf_counter() - started,
                "artifacts": {str(path.relative_to(root)): file_hash(path) for path in artifacts},
            }
            print(f"{name}: {entry['duration_s']:.2f}s", flush=True)
        run["passes"][name] = entry
        write_json(run_path, run)
        return entry["artifacts"]

    def probe():
        records = [probe_clip(p, clip_id) for p, clip_id in zip(sources, ids)]
        write_json(root / "clips.json", {"clips": [r.to_dict() for r in records]})
        return [root / "clips.json"]

    probe_artifacts = stage("probe", {}, {}, probe)
    clip_records = [ClipInfo.from_json(c) for c in read_json(root / "clips.json")["clips"]]

    def score():
        records = [row for clip in clip_records for row in score_clip(clip, config)]
        write_scores(root / "all_frames.csv", records)
        return [root / "all_frames.csv"]

    score_artifacts = stage(
        "score",
        probe_artifacts,
        {k: asdict(getattr(config, k)) for k in ("decode", "scoring", "tracking")},
        score,
    )
    scores = read_scores(root / "all_frames.csv")

    def select():
        selected, edges, measurements, links = [], [], [], []
        for clip in clip_records:
            a, b, c, d = select_clip(clip, [s for s in scores if s.clip_id == clip.clip_id], config)
            selected.extend(a)
            edges.extend(b)
            measurements.extend(c)
            links.extend(d)
        write_json(
            root / "selection.json",
            {
                "frames": [r.to_dict() for r in selected],
                "sequential_edges": [r.to_dict() for r in edges],
                "measurements": measurements,
                "links": links,
                "candidate_pool": [s.frame_id for s in scores],
            },
        )
        return [root / "selection.json"]

    select_artifacts = stage(
        "select",
        score_artifacts,
        {k: asdict(getattr(config, k)) for k in ("decode", "tracking", "selection")},
        select,
    )
    selection = read_json(root / "selection.json")
    selected = [SelectedFrame.from_json(r) for r in selection["frames"]]
    sequential = [Edge.from_json(r) for r in selection["sequential_edges"]]

    def link():
        revisits, status, matches = find_revisits(clip_records, selected, scores, config, root)
        graph = build_graph(selected, sequential, revisits)
        write_json(root / "covisibility.json", {**graph, "debug_matches": matches})
        write_json(
            root / "report.json",
            build_report(
                clip_records,
                scores,
                selection["links"],
                sequential + revisits,
                graph,
                status,
                config,
            ),
        )
        return [
            root / "covisibility.json",
            root / "report.json",
            *sorted((root / ".cache/revisits").glob("*.png")),
        ]

    link_artifacts = stage(
        "link",
        {**score_artifacts, **select_artifacts},
        {
            "revisits": asdict(config.revisits),
            "available": revisits_available(),
            "report": asdict(config.scoring),
        },
        link,
    )

    def export():
        exported = export_frames(clip_records, selected, config, root)
        write_manifest(root, exported, clip_records, selection["candidate_pool"])
        debug_paths = visualize(
            root,
            clip_records,
            scores,
            exported,
            selection["measurements"],
            read_json(root / "covisibility.json")["debug_matches"],
            config,
        )
        artifacts = [
            root / "manifest.json",
            *[root / f.image_path for f in exported],
            *debug_paths,
        ]
        # Remove only files owned by a previous successful export, never unrelated user files.
        current = {str(p.relative_to(root)) for p in artifacts}
        for old in previous.get("passes", {}).get("export", {}).get("artifacts", {}):
            if old not in current:
                artifact_path(root, old).unlink(missing_ok=True)
        return artifacts

    stage("export", {**select_artifacts, **link_artifacts}, asdict(config.export), export)
    run["completed_at"] = datetime.now(UTC).isoformat()
    write_json(run_path, run)
    return load_result(root)


def load_result(output_dir) -> PreprocessResult:
    root = Path(output_dir).resolve()
    manifest = read_json(root / "manifest.json")
    graph, report = read_json(root / "covisibility.json"), read_json(root / "report.json")
    selected = tuple(SelectedFrame.from_json(f) for f in manifest["frames"])
    for frame in selected:
        if not artifact_path(root, frame.image_path).is_file():
            raise FileNotFoundError(frame.image_path)
    return PreprocessResult(
        str(root),
        {
            key: str(root / filename)
            for key, filename in {
                "manifest": "manifest.json",
                "clips": "clips.json",
                "scores": "all_frames.csv",
                "selection": "selection.json",
                "graph": "covisibility.json",
                "report": "report.json",
                "run": "run.json",
                "debug": "debug",
            }.items()
        },
        tuple(ClipInfo.from_json(c) for c in manifest["clips"]),
        tuple(read_scores(root / "all_frames.csv")),
        selected,
        tuple(Edge.from_json(e) for e in graph["edges"]),
        tuple(GapItem.from_json(g) for g in report["gaps"]),
    )
