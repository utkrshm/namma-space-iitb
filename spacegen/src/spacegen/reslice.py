"""Re-slicing: denser frames from the video where a preview model is weak.

Ingest spaces frames by camera motion, the same everywhere. Some moments need more: fast
turns, frames that failed to register, the stretch behind a poor held-out view. Halving
the step for the whole video would double the camera solve (Coorg: 3 h). Instead:

1. Train a short preview splat (``preview_steps``) on the solved cameras and audit it
   (`spacegen.audit`). The audit names the weak *segments* (stretches between two
   consecutive training cameras).
2. In each weak segment, choose extra frames at ``motion_step / density`` with ingest's
   rule (sharpest frame per piece of motion). Frames already chosen are skipped, and so
   is any frame within half a motion step of a held-out frame, so that held-out views stay
   held out. The total is capped at ``max_extra_share`` of the registered frames.
3. Undistort each to the solved pinhole camera with an exact remap: every output pixel's
   ray is sent through COLMAP's lens model (``img_from_cam``).
4. Register by PnP. ALIKED + LightGlue matches to the segment's two training cameras.
   Their keypoints get 3D positions from the MVS depth, or else from the preview's
   rendered depth (opaque pixels only). Then RANSAC + Levenberg–Marquardt on the inliers.
   A frame is kept with at least ``min_inliers`` inliers making up ``min_inlier_ratio``
   of the matches, and a pose near its neighbours.
5. Monocular depth for the kept frames, fitted to the preview's depth (the SfM points do
   not know these frames); the fits join ``depth/alignment.json``.

Outputs (``<run>/reslice/``)
----------------------------
- ``preview/splat/``: the preview model; ``audit/``: its audit;
- ``images/<x000123>.jpg``: undistorted extra frames (``123`` = source frame number);
- ``extra.json``: ``{"summary", "frames": [...]}``. Each frame has ``name``, ``index``,
  ``timestamp_s``, ``extra: true``, ``segment``, ``inliers``, ``camera`` (a
  `spacegen.contracts.Camera` dict), and ``registered``.

Extra frames only ever train (`spacegen.splat_stage`). `run_solution` returns the solved
cameras with the registered extra frames merged in capture order.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import PipelineConfig
from .contracts import Camera
from .sfm import Solution, load_solution


def run_solution(run) -> Solution:
    """The run's camera solve plus the registered re-sliced frames, in capture order."""
    run = Path(run)
    return with_extra_frames(load_solution(run / "sfm", run / "frames"), run)


def with_extra_frames(solution: Solution, run) -> Solution:
    path = Path(run) / "reslice" / "extra.json"
    if not path.exists():
        return solution
    extra = [f for f in json.loads(path.read_text())["frames"] if f["registered"]]
    if not extra:
        return solution
    # (frame index, original position or None, camera, image, frame record)
    entries = [
        (f["index"], k, camera, image, f)
        for k, (camera, image, f) in enumerate(
            zip(solution.cameras, solution.image_paths, solution.frames, strict=True)
        )
    ]
    folder = Path(run) / "reslice" / "images"
    for record in extra:
        frame = {k: v for k, v in record.items() if k != "camera"}
        camera = Camera.from_dict(record["camera"])
        entries.append((record["index"], None, camera, folder / record["name"], frame))
    entries.sort(key=lambda e: e[0])
    moved = {e[1]: k for k, e in enumerate(entries) if e[1] is not None}
    return replace(
        solution,
        cameras=[e[2] for e in entries],
        image_paths=[e[3] for e in entries],
        frames=[e[4] for e in entries],
        point_views=[
            np.array([moved[v] for v in views], dtype=int) for views in solution.point_views
        ],
    )


def undistort_map(raw: dict, pinhole: Camera) -> tuple[np.ndarray, np.ndarray]:
    """cv2.remap maps from the raw lens to ``pinhole``.

    ``raw`` is ``{"model", "params", "width", "height"}`` in COLMAP's terms.
    """
    import pycolmap

    camera = pycolmap.Camera(
        model=raw["model"], width=raw["width"], height=raw["height"], params=raw["params"]
    )
    k = pinhole.intrinsics
    ys, xs = np.mgrid[0 : pinhole.height, 0 : pinhole.width].astype(np.float64)
    rays = np.stack(
        ((xs - k[0, 2]) / k[0, 0], (ys - k[1, 2]) / k[1, 1], np.ones_like(xs)), -1
    ).reshape(-1, 3)
    # COLMAP pixel centres sit at +0.5; ours (and OpenCV's remap) at integers.
    pixels = camera.img_from_cam(rays) - 0.5
    pixels = pixels.reshape(pinhole.height, pinhole.width, 2).astype(np.float32)
    return pixels[..., 0], pixels[..., 1]


def choose_extra(
    measures: dict, segments: list[dict], taken: set[int], held_out: list[int], step: float, config
) -> list[tuple[int, int]]:
    """``(video frame index, segment number)`` of extra frames for weak segments."""
    from .ingest import choose_frames

    times, sharp, motion = measures["times"], measures["sharpness"], measures["motion"]
    travelled = np.cumsum(motion)
    guarded = travelled[held_out] if held_out else np.zeros(0)
    chosen = []
    for number, segment in enumerate(segments):
        inside = np.flatnonzero((times > segment["start_s"]) & (times < segment["end_s"]))
        if len(inside) < 2:
            continue
        picks = choose_frames(
            times[inside], sharp[inside], motion[inside], config.ingest, step / config.audit.density
        )
        for pick in inside[picks]:
            if pick in taken:
                continue
            if len(guarded) and np.min(np.abs(guarded - travelled[pick])) < 0.5 * step:
                continue
            chosen.append((int(pick), number))
    return chosen


class Matcher:
    """ALIKED keypoints + LightGlue matches (as in `spacegen.sfm.features`)."""

    def __init__(self, config, device):
        from lightglue import ALIKED, LightGlue

        self.device, self.long_side = device, config.sfm.feature_long_side
        self.extractor = ALIKED(max_num_keypoints=config.sfm.max_keypoints).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(device)
        self.cache: dict[str, dict] = {}

    @torch.no_grad()
    def features(self, key: str, rgb: np.ndarray) -> dict:
        if key not in self.cache:
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
            self.cache[key] = self.extractor.extract(tensor, resize=self.long_side)
        return self.cache[key]

    @torch.no_grad()
    def __call__(self, a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
        result = self.matcher({"image0": a, "image1": b})
        pairs = result["matches"][0].cpu().numpy()
        return (
            a["keypoints"][0].cpu().numpy()[pairs[:, 0]],
            b["keypoints"][0].cpu().numpy()[pairs[:, 1]],
        )


def lift(xy: np.ndarray, camera: Camera, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points of pixels ``xy`` (``camera``'s grid) using ``depth`` (any grid); 0 = none."""
    height, width = depth.shape
    sx, sy = width / camera.width, height / camera.height
    columns = np.clip(np.round((xy[:, 0] + 0.5) * sx - 0.5).astype(int), 0, width - 1)
    rows = np.clip(np.round((xy[:, 1] + 0.5) * sy - 0.5).astype(int), 0, height - 1)
    z = depth[rows, columns]
    keep = z > 0
    k = camera.intrinsics
    local = np.stack(((xy[:, 0] - k[0, 2]) / k[0, 0] * z, (xy[:, 1] - k[1, 2]) / k[1, 1] * z, z), 1)
    world = local @ camera.camera_to_world[:3, :3].T + camera.camera_to_world[:3, 3]
    return world[keep], keep


def register(
    rgb: np.ndarray,
    template: Camera,
    neighbours: list[tuple[str, Camera, np.ndarray, np.ndarray]],
    matcher: Matcher,
    config,
) -> tuple[Camera | None, dict]:
    """PnP pose of an undistorted image; ``neighbours`` are (name, camera, rgb, depth)."""
    query = matcher.features("query", rgb)
    matcher.cache.pop("query")
    world, pixels, matched = [], [], 0
    for name, camera, image, depth in neighbours:
        xy_query, xy_neighbour = matcher(query, matcher.features(name, image))
        matched += len(xy_query)
        points, keep = lift(xy_neighbour, camera, depth)
        world.append(points)
        pixels.append(xy_query[keep])
    world, pixels = np.concatenate(world), np.concatenate(pixels)
    info = {"matches": int(matched), "lifted": len(world), "inliers": 0}
    if not len(world):
        return None, info
    if len(world) < config.audit.min_inliers:
        return None, info
    found, rvec, tvec, inliers = cv2.solvePnPRansac(
        world.astype(np.float64),
        pixels.astype(np.float64),
        template.intrinsics,
        None,
        iterationsCount=2000,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not found or inliers is None:
        return None, info
    inliers = inliers[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(
        world[inliers], pixels[inliers], template.intrinsics, None, rvec, tvec
    )
    info["inliers"] = len(inliers)
    info["inlier_ratio"] = float(len(inliers) / max(1, len(world)))
    if (
        len(inliers) < config.audit.min_inliers
        or info["inlier_ratio"] < config.audit.min_inlier_ratio
    ):
        return None, info
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = cv2.Rodrigues(rvec)[0]
    world_to_camera[:3, 3] = tvec[:, 0]
    camera = template.moved(world_to_camera)

    # The pose must lie near the two training cameras it sits between: within the gap
    # between them (or 5% of the scene depth) of their midpoint, and turned no more than
    # 10° beyond them.
    from .audit import turn_deg

    a, b = neighbours[0][1], neighbours[-1][1]
    depth = neighbours[0][3]
    tolerance = max(
        float(np.linalg.norm(a.centre - b.centre)), 0.05 * float(np.median(depth[depth > 0]))
    )
    info["offset"] = float(np.linalg.norm(camera.centre - (a.centre + b.centre) / 2))
    info["turn_deg"] = float(min(turn_deg(camera, a), turn_deg(camera, b)))
    if info["offset"] > tolerance or info["turn_deg"] > turn_deg(a, b) + 10:
        return None, info
    return camera, info


def reslice(
    run, solution: Solution, depth_dir, config: PipelineConfig, device="cuda:0", log=print
) -> dict:
    """Write ``<run>/reslice/``; a finished one (``extra.json``) is reused."""
    run = Path(run)
    out = run / "reslice"
    if (out / "extra.json").exists():
        return json.loads((out / "extra.json").read_text())["summary"]
    listing = json.loads((run / "frames" / "frames.json").read_text())
    if listing["kind"] != "video":
        return {"skipped": "not a video"}
    (out / "images").mkdir(parents=True, exist_ok=True)

    # 1. Preview model and its audit.
    from .audit import audit_model
    from .splat import load_ply, render_camera
    from .splat_stage import build_splat

    steps = config.audit.preview_steps
    preview = config.replace(
        splat={
            "steps": steps,
            "refine_stop": int(steps * 0.8),
            "evaluate_every": 10**9,
            "checkpoint_every": 10**9,
        },
        partition={"finetune_steps": steps // 2},
    )
    build_splat(solution, out / "preview", preview, depth_dir, device, log)
    audit_model(run, out / "preview" / "splat", out / "audit", config.audit, device, log)
    audit = json.loads((out / "audit" / "audit.json").read_text())
    weak = [s for s in audit["segments"] if s["reasons"]]
    log(f"reslice: {len(weak)} weak segments of {len(audit['segments'])}")

    # 2. Extra frames in the weak segments.
    from .ingest import _decode, _rotation, measure_video

    source = Path(listing["source"])
    measures_path = run / "frames" / "measures.npz"
    if measures_path.exists():
        measures = dict(np.load(measures_path))
    else:
        measures = measure_video(source, config.ingest)
        np.savez(measures_path, **measures)
    taken = {f["index"] for f in listing["frames"]}
    every = config.splat.holdout_every
    held_out = [f["index"] for f in listing["frames"] if every and f["position"] % every == 0]
    picks = choose_extra(measures, weak, taken, held_out, listing["motion_step"], config)
    limit = int(config.audit.max_extra_share * len(solution.cameras))
    if len(picks) > limit:
        picks = [picks[i] for i in np.linspace(0, len(picks) - 1, limit).round().astype(int)]
    log(f"reslice: {len(picks)} extra frames chosen")

    # 3–4. Undistort and register.
    sfm_report = json.loads((run / "sfm" / "report.json").read_text())
    lens = sfm_report["largest_model"]["cameras"][0]
    template = solution.cameras[0]
    first = listing["frames"][0]
    raw = {
        "model": lens["model"],
        "params": lens["params"],
        "width": first["width"],
        "height": first["height"],
    }
    map_x, map_y = undistort_map(raw, template)
    by_name = {c.name: k for k, c in enumerate(solution.cameras)}
    model = load_ply(out / "preview" / "splat" / "scene.ply", device)
    matcher = Matcher(config, device)
    depths: dict[str, np.ndarray] = {}

    def neighbour(name: str):
        index = by_name[name]
        camera = solution.cameras[index]
        if name not in depths:
            mvs_path = Path(depth_dir) / "mvs" / f"{Path(name).stem}.npy" if depth_dir else None
            with torch.no_grad():
                small = camera.resized(camera.width // 2, camera.height // 2)
                _, rendered, alpha, _ = render_camera(model, small)
            rendered = torch.where(alpha > 0.95, rendered, 0.0).cpu().numpy()
            if mvs_path is not None and mvs_path.exists():
                measured = np.load(mvs_path).astype(np.float32)
                measured = cv2.resize(
                    measured, rendered.shape[::-1], interpolation=cv2.INTER_NEAREST
                )
                rendered = np.where(measured > 0, measured, rendered)
            depths[name] = rendered
        rgb = cv2.cvtColor(cv2.imread(str(solution.image_paths[index])), cv2.COLOR_BGR2RGB)
        return name, camera, rgb, depths[name]

    records = []
    wanted = {index for index, _ in picks}
    segment_of = dict(picks)
    rotation = _rotation(source)
    for index, seconds, rgb in (
        _decode(source, config.ingest.max_long_side, rotation, wanted) if wanted else []
    ):
        segment = weak[segment_of[index]]
        undistorted = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        name = f"x{index:06d}.jpg"
        camera, info = register(
            undistorted,
            Camera(
                name, template.width, template.height, template.intrinsics, template.world_to_camera
            ),
            [neighbour(n) for n in segment["cameras"] if n in by_name],
            matcher,
            config,
        )
        record = {
            "name": name,
            "index": index,
            "timestamp_s": float(seconds),
            "extra": True,
            "segment": segment["cameras"],
            "reasons": segment["reasons"],
            "registered": camera is not None,
            **info,
        }
        if camera is not None:
            cv2.imwrite(
                str(out / "images" / name), undistorted[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            record["camera"] = camera.to_dict()
        records.append(record)
    registered = [r for r in records if r["registered"]]
    log(f"reslice: {len(registered)}/{len(records)} extra frames registered")

    # 5. Depth for the registered frames, fitted to the preview's depth.
    if depth_dir is not None and registered:
        from .depth import compute_depth, fit_inverse_depth

        paths = [out / "images" / r["name"] for r in registered]
        compute_depth(paths, depth_dir, config.depth, device)
        fits_path = Path(depth_dir) / "alignment.json"
        fits = json.loads(fits_path.read_text()) if fits_path.exists() else {}
        for record in registered:
            stem = Path(record["name"]).stem
            mono = np.load(Path(depth_dir) / f"{stem}.npy").astype(np.float32)
            camera = Camera.from_dict(record["camera"]).resized(mono.shape[1], mono.shape[0])
            with torch.no_grad():
                _, rendered, alpha, _ = render_camera(model, camera)
            rendered, alpha = rendered.cpu().numpy(), alpha.cpu().numpy()
            usable = alpha > 0.95
            if usable.sum() < 100:
                continue
            a, b, error = fit_inverse_depth(mono[usable], 1 / np.maximum(rendered[usable], 1e-9))
            if a > 0 and error <= 0.15:
                fits[stem] = {
                    "a": a,
                    "b": b,
                    "points": int(usable.sum()),
                    "error": error,
                    "source": "preview",
                }
        fits_path.write_text(json.dumps(fits, indent=1) + "\n")

    summary = {
        "weak_segments": len(weak),
        "reasons": audit["summary"]["reasons"],
        "chosen": len(records),
        "registered": len(registered),
        "preview_held_out_psnr_cc": audit["summary"]["held_out_psnr_cc_median"],
    }
    (out / "extra.json").write_text(
        json.dumps({"summary": summary, "frames": records}, indent=1) + "\n"
    )
    log(f"reslice: {json.dumps(summary)}")
    return summary
