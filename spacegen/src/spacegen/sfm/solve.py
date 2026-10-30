"""Camera solve: frames → features → pairs → matches → (loop check) → mapping → undistortion.

Input: an ingest directory (``frames.json`` + ``images/``).

Output (``<out>/sfm/``):
- ``features/``: ALIKED keypoints, DINOv2 descriptors, LightGlue matches and pair records;
- ``reference/``: the sequential-only solve used to check loops (video only);
- ``database.db`` and ``sparse/<k>/``: the COLMAP database and models, largest first;
- ``undistorted/{images,sparse}``: the largest model undistorted to PINHOLE cameras,
  which is the hand-off to splatting;
- ``report.json``: registration, reprojection error, loop check and timings.

COLMAP stores world-to-camera poses (OpenCV axes). Scene units are arbitrary.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ..config import SfmConfig
from ..contracts import Camera
from .features import build_pairs, extract_features, global_descriptors, load_features, match_pairs
from .loops import check_loops
from .mapping import build_database, run_mapper, undistort


def _solve(folder, images, frames, features, matches, config, mapper):
    database = build_database(folder, images, frames, features, matches, config)
    return run_mapper(folder, database, images, mapper)


def _quality(models, total: int) -> tuple[float, float]:
    """Sort key: registered share (to 1%), then lower reprojection error."""
    if not models:
        return (0.0, 0.0)
    errors = [p.error for p in models[0].points3D.values()]
    error = float(np.mean(errors)) if errors else 0.0
    return (round(models[0].num_reg_images() / total, 2), -error)


def _map_with_fallbacks(output: Path, images, frames, features, matches, config: SfmConfig):
    """Map with the configured lens model and mapper; when too few frames register (or the
    reprojection error says the lens model is wrong), try the other mapper and then a
    fisheye lens model, and keep the best result in ``<output>/sparse``.

    Action cameras and phone ultra-wide lenses bend lines more than OPENCV's two radial
    terms describe; COLMAP's OPENCV_FISHEYE model fits them.
    """
    other = "incremental" if config.mapper == "global" else "global"
    attempts = [(config.camera_model, config.mapper), (config.camera_model, other)]
    if config.fisheye_fallback and config.camera_model != "OPENCV_FISHEYE":
        attempts.append(("OPENCV_FISHEYE", config.mapper))
    total = len(frames)
    best, best_key, tried = None, None, {}
    for number, (lens, mapper) in enumerate(attempts):
        folder = output if number == 0 else output / f"retry_{lens.lower()}_{mapper}"
        attempt_config = replace(config, camera_model=lens)
        models = _solve(folder, images, frames, features, matches, attempt_config, mapper)
        key = _quality(models, total)
        tried[f"{lens}/{mapper}"] = {
            "models": [m.num_reg_images() for m in models],
            "mean_reprojection_px": -key[1],
        }
        if best_key is None or key > best_key:
            best, best_key, best_folder = models, key, folder
        good = key[0] >= config.min_registered_fraction and -key[1] <= config.max_reprojection_px
        if good:
            break
    if best and best_folder != output:
        _fresh(output / "sparse")
        for index, model in enumerate(best):
            model.write(_fresh(output / "sparse" / str(index)))
    return best or [], tried


def _model_summary(model, total: int) -> dict:
    errors = [p.error for p in model.points3D.values()]
    return {
        "registered": model.num_reg_images(),
        "registered_fraction": model.num_reg_images() / total,
        "points": model.num_points3D(),
        "mean_reprojection_px": float(np.mean(errors)) if errors else None,
        "mean_track_length": float(model.compute_mean_track_length()),
        "cameras": [
            {"model": c.model.name, "params": [float(v) for v in c.params]}
            for c in model.cameras.values()
        ],
    }


def solve_cameras(frames_dir, output_dir, config: SfmConfig | None = None, device="cuda:0"):
    """Solve cameras; a finished solve with the same config is reused."""
    config = config or SfmConfig()
    frames_dir, output = Path(frames_dir).resolve(), Path(output_dir).resolve()
    report_path = output / "report.json"
    if report_path.exists() and (output / "undistorted" / "sparse").exists():
        report = json.loads(report_path.read_text())
        if report.get("config") == asdict(config):
            return report
    output.mkdir(parents=True, exist_ok=True)
    listing = json.loads((frames_dir / "frames.json").read_text())
    frames, images = listing["frames"], frames_dir / "images"
    names = [frame["name"] for frame in frames]
    timings = {}

    start = time.perf_counter()
    cache = output / "features"
    extract_features(images, names, cache, config, device)
    features = load_features(cache, names)
    descriptors = global_descriptors(images, names, cache, config, device)
    pairs = build_pairs(descriptors, config)
    matches, pair_records = match_pairs(cache, features, pairs, config, device)
    timings["matching_s"] = time.perf_counter() - start

    start = time.perf_counter()
    loop_summary = None
    kinds = {(p["i"], p["j"]): p["kind"] for p in pair_records}
    if config.loop_check and listing["kind"] == "video":
        sequential = {k: v for k, v in matches.items() if kinds[k] == "sequential"}
        reference = _solve(
            output / "reference", images, frames, features, sequential, config, "global"
        )
        matches, loop_summary = check_loops(
            matches,
            pair_records,
            features,
            names,
            reference,
            config.loop_max_rotation_deg,
            config.ransac_px,
            config.retrieval_min_inliers,
        )
        loop_summary["reference_models"] = [m.num_reg_images() for m in reference]
    timings["loop_check_s"] = time.perf_counter() - start

    start = time.perf_counter()
    models, tried = _map_with_fallbacks(output, images, frames, features, matches, config)
    if not models:
        raise RuntimeError("SfM registered no images; the video may lack texture or overlap")
    timings["mapping_s"] = time.perf_counter() - start

    start = time.perf_counter()
    undistort(output, output / "sparse" / "0", images)
    timings["undistortion_s"] = time.perf_counter() - start

    report = {
        "config": asdict(config),
        "frames": len(frames),
        "pairs": {
            "sequential": sum(p["kind"] == "sequential" for p in pair_records),
            "retrieval_candidates": sum(p["kind"] == "retrieval" for p in pair_records),
            "retrieval_passed_gate": sum(
                p["kind"] == "retrieval" and p["accepted"] for p in pair_records
            ),
            "used": len(matches),
        },
        "loop_check": loop_summary,
        "mappers_tried": tried,
        "models": [m.num_reg_images() for m in models],
        "largest_model": _model_summary(models[0], len(frames)),
        "timings": timings,
    }
    report_path.write_text(json.dumps(report, indent=1) + "\n")
    return report


def _fresh(path: Path) -> Path:
    import shutil

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


@dataclass
class Solution:
    """Undistorted cameras in capture order, their images, and the SfM points."""

    cameras: list[Camera]
    image_paths: list[Path]
    points: np.ndarray  # [N, 3] world
    colors: np.ndarray  # [N, 3] in 0..1
    point_views: list[np.ndarray]  # per point, indices into ``cameras``
    frames: list[dict]  # ingest records, same order as ``cameras``


def load_solution(sfm_dir, frames_dir, min_points: int = 30) -> Solution:
    """Read ``undistorted/``; principal points move to integer pixel centres (−0.5).

    Images that observe fewer than ``min_points`` triangulated points are left out: the
    global mapper can "register" an image with no supporting points, and its pose is then
    arbitrary (playroom: one camera 870 units from a scene 8 units wide). One such camera
    once inflated the scene radius, and with it every learning rate, by 100×.
    """
    import pycolmap

    sfm_dir, frames_dir = Path(sfm_dir).resolve(), Path(frames_dir).resolve()
    records = {f["name"]: f for f in json.loads((frames_dir / "frames.json").read_text())["frames"]}
    model = pycolmap.Reconstruction(sfm_dir / "undistorted" / "sparse")
    entries = []
    for image in model.images.values():
        if not image.has_pose or image.num_points3D < min_points:
            continue
        camera = model.cameras[image.camera_id]
        fx, fy, cx, cy = camera.params
        intrinsics = np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1.0]])
        world_to_camera = np.eye(4)
        world_to_camera[:3] = image.cam_from_world().matrix()
        entries.append(
            (
                records[image.name]["index"],
                image.image_id,
                Camera(image.name, camera.width, camera.height, intrinsics, world_to_camera),
            )
        )
    entries.sort(key=lambda e: e[0])
    position = {image_id: k for k, (_, image_id, _) in enumerate(entries)}
    points = list(model.points3D.values())
    return Solution(
        [e[2] for e in entries],
        [sfm_dir / "undistorted" / "images" / e[2].name for e in entries],
        np.array([p.xyz for p in points], np.float32).reshape(-1, 3),
        np.array([p.color for p in points], np.float32).reshape(-1, 3) / 255.0,
        [
            np.array([position[e.image_id] for e in p.track.elements if e.image_id in position])
            for p in points
        ],
        [records[e[2].name] for e in entries],
    )
