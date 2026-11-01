"""Self-audit: render a model from many viewpoints and find what is missing where.

Training views say little about quality: the model can memorise them. The audit looks
from elsewhere:

- **Held-out views:** frames never trained on. PSNR after an affine colour fit (exposure
  is not a geometry problem), and the error map.
- **In-between views:** the pose halfway between consecutive training cameras
  (interpolated). The phone passed through it, so the video has frames there. If the
  model renders it with holes or with content no camera measured from that side, then
  frames are missing at that moment of the video, not only from the model.
- **Probe views** around sampled training cameras: turning ±``probe_yaw_deg`` and
  ±``probe_pitch_deg``, and stepping ±``probe_shift`` camera heights sideways and
  forward. These show what a visitor would see off the filmed path: holes, unmeasured
  content and floaters close to the lens.

Every in-between view, and every held-out view, belongs to a *segment*: the stretch of
video between two consecutive training cameras. A segment is weak when:
- its in-between view has more than ``max_unmeasured`` of pixels unmeasured, or more
  than ``max_unrendered`` of pixels unrendered;
- the camera turned more than ``max_turn_deg`` across it;
- a held-out view in it scores ``max_psnr_drop`` dB below the median;
- a frame chosen at ingest failed to register there.

`spacegen.reslice` takes extra frames from weak segments.

Outputs (``<out>/``)
--------------------
- ``audit.json``: ``summary``, ``held_out`` (per view), ``segments`` (per segment: times,
  turn, checks, ``reasons``), ``probes`` (mean checks per probe kind, and the worst views);
- ``sheet.jpg``: the worst held-out views (truth | render | error) and the worst in-between
  and probe views (render | unmeasured in red);
- ``timeline.png``: the checks along the video, with weak segments shaded.

A pixel is *unrendered* when alpha < 0.5, and *unmeasured* when it renders but less than
half of its blending weight comes from Gaussians that a training camera saw from within
35° of this direction (`spacegen.support`).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from .config import AuditConfig
from .contracts import Camera
from .depth import read_rgb
from .scene_frame import SceneFrame, estimate_frame
from .splat import load_cameras, load_ply, render_camera
from .splat.model import write_json
from .splat.train import fit_colour
from .support import compute_support, support_map

SUPPORT_ANGLE_DEG = 35.0


def small(camera: Camera, long_side: int) -> Camera:
    scale = long_side / max(camera.width, camera.height)
    return camera.resized(round(camera.width * scale), round(camera.height * scale))


def between(a: Camera, b: Camera, t: float = 0.5) -> Camera:
    """The pose a fraction ``t`` of the way from ``a`` to ``b`` (slerp and lerp)."""
    rotations = Rotation.from_matrix([a.camera_to_world[:3, :3], b.camera_to_world[:3, :3]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = Slerp([0, 1], rotations)(t).as_matrix()
    camera_to_world[:3, 3] = (1 - t) * a.centre + t * b.centre
    return a.moved(np.linalg.inv(camera_to_world))


def turn_deg(a: Camera, b: Camera) -> float:
    relative = a.world_to_camera[:3, :3] @ b.camera_to_world[:3, :3]
    return math.degrees(math.acos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))


def probe_views(camera: Camera, frame: SceneFrame, config: AuditConfig) -> list[tuple[str, Camera]]:
    """Turned and stepped copies of ``camera``: ``(kind, camera)`` pairs."""
    camera_to_world = camera.camera_to_world
    up = frame.up
    views = []
    for kind, axis, angle in (
        ("yaw-", up, -config.probe_yaw_deg),
        ("yaw+", up, config.probe_yaw_deg),
        ("pitch-", camera_to_world[:3, 0], -config.probe_pitch_deg),
        ("pitch+", camera_to_world[:3, 0], config.probe_pitch_deg),
    ):
        turned = camera_to_world.copy()
        turned[:3, :3] = (
            Rotation.from_rotvec(axis * math.radians(angle)).as_matrix() @ turned[:3, :3]
        )
        views.append((kind, camera.moved(np.linalg.inv(turned))))
    right = camera_to_world[:3, 0] - (camera_to_world[:3, 0] @ up) * up
    forward = camera_to_world[:3, 2] - (camera_to_world[:3, 2] @ up) * up
    for kind, direction in (("left", -right), ("right", right), ("forward", forward)):
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        moved = camera_to_world.copy()
        moved[:3, 3] += direction / norm * config.probe_shift * frame.camera_height
        views.append((kind, camera.moved(np.linalg.inv(moved))))
    return views


@torch.no_grad()
def view_checks(model, camera: Camera, near: float) -> tuple[dict, dict]:
    """``(checks, images)`` of one view: unrendered, unmeasured and near shares."""
    rgb, depth, alpha, _ = render_camera(model, camera)
    fraction, _, _ = support_map(model, camera, SUPPORT_ANGLE_DEG)
    rendered = alpha > 0.5
    unmeasured = rendered & (fraction < 0.5)
    checks = {
        "unrendered": float((~rendered).float().mean()),
        "unmeasured": float(unmeasured.float().sum() / rendered.float().sum().clamp_min(1)),
        "near": float(((depth < near) & rendered).float().mean()),
    }
    return checks, {"rgb": rgb.clamp(0, 1), "unmeasured": unmeasured, "rendered": rendered}


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.clamp(0, 1) * 255).round().byte().cpu().numpy()


def _overlay(images: dict) -> np.ndarray:
    """Render with unmeasured pixels tinted red and unrendered ones black."""
    rgb = images["rgb"].clone()
    red = torch.tensor([1.0, 0.0, 0.0], device=rgb.device)
    rgb[images["unmeasured"]] = 0.4 * rgb[images["unmeasured"]] + 0.6 * red
    rgb[~images["rendered"]] = 0
    return _to_uint8(rgb)


def _image_path(run: Path, name: str) -> Path:
    for folder in (run / "sfm" / "undistorted" / "images", run / "reslice" / "images"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(name)


def _tile(images: list[np.ndarray], caption: str, height: int = 320) -> np.ndarray:
    """Images side by side at ``height`` under a caption band."""
    resized = [
        cv2.resize(i, (round(i.shape[1] * height / i.shape[0]), height), cv2.INTER_AREA)
        for i in images
    ]
    tile = np.hstack(resized)
    band = np.zeros((22, tile.shape[1], 3), np.uint8)
    scale = min(0.45, 0.45 * tile.shape[1] / (8 * len(caption)))
    cv2.putText(band, caption, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1)
    return np.vstack((band, tile))


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    width = max(t.shape[1] for t in tiles)
    height = max(t.shape[0] for t in tiles)
    padded = [
        np.pad(t, ((0, height - t.shape[0] + 4), (0, width - t.shape[1] + 4), (0, 0)))
        for t in tiles
    ]
    padded += [np.zeros_like(padded[0])] * (-len(padded) % columns)
    return np.vstack([np.hstack(padded[i : i + columns]) for i in range(0, len(padded), columns)])


def _sheet(groups: list[tuple[list[np.ndarray], int]], path: Path) -> None:
    """Groups of tiles, each a grid with its own number of columns, stacked."""
    grids = [_grid(tiles, columns) for tiles, columns in groups if tiles]
    if not grids:
        return
    width = max(g.shape[1] for g in grids)
    grids = [np.pad(g, ((0, 8), (0, width - g.shape[1]), (0, 0))) for g in grids]
    cv2.imwrite(str(path), np.vstack(grids)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


def _timeline(segments: list[dict], held_out: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not segments:
        return
    middle = [(s["start_s"] + s["end_s"]) / 2 for s in segments]
    figure, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(middle, [s["between"]["unmeasured"] for s in segments], lw=0.8, label="unmeasured")
    axes[0].plot(middle, [s["between"]["unrendered"] for s in segments], lw=0.8, label="unrendered")
    axes[0].set_ylabel("in-between view")
    axes[0].legend(loc="upper right")
    axes[1].plot(middle, [s["turn_deg"] for s in segments], lw=0.8)
    axes[1].set_ylabel("turn (°)")
    axes[2].plot([v["timestamp_s"] for v in held_out], [v["psnr_cc"] for v in held_out], "o-", ms=3)
    axes[2].set_ylabel("held-out PSNR (dB)")
    axes[2].set_xlabel("video time (s)")
    for segment in segments:
        if segment["reasons"]:
            for axis in axes:
                axis.axvspan(segment["start_s"], segment["end_s"], color="red", alpha=0.15, lw=0)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def audit_model(run, model_dir, out, config: AuditConfig, device="cuda:0", log=print) -> dict:
    """Audit ``<model_dir>/scene.ply`` (with ``cameras.json``); writes ``<out>/``."""
    run, model_dir, out = Path(run), Path(model_dir), Path(out)
    if (out / "audit.json").exists():
        return json.loads((out / "audit.json").read_text())["summary"]
    out.mkdir(parents=True, exist_ok=True)

    model = load_ply(model_dir / "scene.ply", device)
    cameras, splits = load_cameras(model_dir / "cameras.json")
    train = [i for i, s in enumerate(splits) if s == "train"]
    val = [i for i, s in enumerate(splits) if s == "val"]
    model.extras["support"] = compute_support(model, [cameras[i] for i in train])
    if (run / "scene_frame.json").exists():
        frame = SceneFrame.load(run / "scene_frame.json")
    else:
        means = model.params["means"].detach().cpu().numpy()
        sample = means[np.random.default_rng(0).choice(len(means), min(200_000, len(means)))]
        frame = estimate_frame([cameras[i] for i in train], sample)
    near = 0.1 * frame.camera_height

    listing = json.loads((run / "frames" / "frames.json").read_text())["frames"]
    records = {f["name"]: f for f in listing}
    extra_path = run / "reslice" / "extra.json"
    if extra_path.exists():
        records.update({f["name"]: f for f in json.loads(extra_path.read_text())["frames"]})

    def seconds(index: int) -> float:
        record = records.get(cameras[index].name, {})
        value = record.get("timestamp_s")
        return float(value) if value is not None else float(record.get("index", index))

    # Held-out views.
    held_out = []
    for index in val:
        camera = small(cameras[index], config.long_side)
        with torch.no_grad():
            rgb, _, _, _ = render_camera(model, camera)
        truth = read_rgb(_image_path(run, cameras[index].name))
        truth = cv2.resize(truth, (camera.width, camera.height), interpolation=cv2.INTER_AREA)
        target = torch.from_numpy(truth).to(device).float() / 255
        corrected = fit_colour(rgb.clamp(0, 1), target)
        error = (corrected - target).abs().mean(-1)
        psnr_cc = float(-10 * torch.log10(((corrected - target) ** 2).mean()))
        held_out.append(
            {
                "name": cameras[index].name,
                "timestamp_s": seconds(index),
                "psnr_cc": psnr_cc,
                "_tiles": (
                    truth,
                    _to_uint8(corrected),
                    _to_uint8((error * 4)[..., None].expand(-1, -1, 3)),
                ),
            }
        )
    median_psnr = float(np.median([v["psnr_cc"] for v in held_out])) if held_out else None

    # Segments between consecutive training cameras (capture order), with their
    # in-between views.
    order = sorted(train, key=seconds)
    registered = {cameras[i].name for i in range(len(cameras))}
    dropped = [f for f in listing if f["name"] not in registered]
    segments, between_tiles = [], []
    for a, b in itertools.pairwise(order):
        start, end = seconds(a), seconds(b)
        view = small(between(cameras[a], cameras[b]), config.long_side)
        checks, images = view_checks(model, view, near)
        segment = {
            "start_s": start,
            "end_s": end,
            "cameras": [cameras[a].name, cameras[b].name],
            "turn_deg": turn_deg(cameras[a], cameras[b]),
            "step": float(np.linalg.norm(cameras[a].centre - cameras[b].centre))
            / frame.camera_height,
            "between": checks,
            "held_out": [v["name"] for v in held_out if start <= v["timestamp_s"] <= end],
            "dropped": [
                f["name"]
                for f in dropped
                if f.get("timestamp_s") is not None and start < f["timestamp_s"] < end
            ],
        }
        reasons = []
        if checks["unmeasured"] > config.max_unmeasured:
            reasons.append("unmeasured in-between")
        if checks["unrendered"] > config.max_unrendered:
            reasons.append("holes in-between")
        if segment["turn_deg"] > config.max_turn_deg:
            reasons.append("fast turn")
        low = [
            v
            for v in held_out
            if v["name"] in segment["held_out"]
            and v["psnr_cc"] < median_psnr - config.max_psnr_drop
        ]
        if low:
            reasons.append("poor held-out view")
        if segment["dropped"]:
            reasons.append("dropped frames")
        segment["reasons"] = reasons
        segment["score"] = (
            checks["unmeasured"] / config.max_unmeasured
            + checks["unrendered"] / config.max_unrendered
            + segment["turn_deg"] / config.max_turn_deg
            + len(low)
            + len(segment["dropped"])
        )
        segments.append(segment)
        between_tiles.append(
            (checks["unmeasured"] + checks["unrendered"], segment, _overlay(images))
        )

    # Keep only the worst ``max_weak_share`` of segments flagged.
    flagged = sorted((s for s in segments if s["reasons"]), key=lambda s: -s["score"])
    allowed = math.ceil(config.max_weak_share * len(segments))
    for segment in flagged[allowed:]:
        segment["reasons"] = []
    weak = [s for s in segments if s["reasons"]]

    # Probe views off the filmed path.
    probe_rows, probe_tiles = {}, []
    for index in order[:: max(1, len(order) // config.probe_cameras)]:
        for kind, view in probe_views(small(cameras[index], config.long_side // 2), frame, config):
            checks, images = view_checks(model, view, near)
            probe_rows.setdefault(kind, []).append(checks)
            badness = checks["unmeasured"] + checks["unrendered"] + checks["near"]
            probe_tiles.append((badness, f"{cameras[index].name} {kind}", _overlay(images), checks))
    probes = {
        kind: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        for kind, rows in probe_rows.items()
    }

    # Contact sheet of the worst views.
    worst_held_out = [
        _tile(list(v["_tiles"]), f"held-out {v['name']} {v['psnr_cc']:.1f} dB")
        for v in sorted(held_out, key=lambda v: v["psnr_cc"])[: config.sheet_views]
    ]
    worst_between = []
    for _, segment, image in sorted(between_tiles, key=lambda t: -t[0])[: config.sheet_views]:
        checks = segment["between"]
        caption = (
            f"between {segment['cameras'][0]} {segment['cameras'][1]}: "
            f"{checks['unmeasured']:.0%} unmeasured, {checks['unrendered']:.0%} holes"
        )
        worst_between.append(_tile([image], caption))
    worst_probes = []
    for _, name, image, checks in sorted(probe_tiles, key=lambda t: -t[0])[: config.sheet_views]:
        caption = (
            f"{name}: {checks['unmeasured']:.0%} unm., {checks['unrendered']:.0%} holes, "
            f"{checks['near']:.0%} near"
        )
        worst_probes.append(_tile([image], caption))
    _sheet([(worst_held_out, 2), (worst_between, 4), (worst_probes, 4)], out / "sheet.jpg")
    for view in held_out:
        del view["_tiles"]
    _timeline(segments, held_out, out / "timeline.png")

    summary = {
        "held_out_views": len(held_out),
        "held_out_psnr_cc_median": median_psnr,
        "segments": len(segments),
        "weak_segments": len(weak),
        "weak_seconds": float(sum(s["end_s"] - s["start_s"] for s in weak)),
        "reasons": {
            reason: sum(reason in s["reasons"] for s in segments)
            for reason in sorted({r for s in segments for r in s["reasons"]})
        },
        "between_unmeasured_mean": float(np.mean([s["between"]["unmeasured"] for s in segments]))
        if segments
        else None,
        "between_unrendered_mean": float(np.mean([s["between"]["unrendered"] for s in segments]))
        if segments
        else None,
        "probes": probes,
    }
    write_json(
        out / "audit.json",
        {"summary": summary, "held_out": held_out, "segments": segments, "probes": probes},
    )
    log(f"audit: {json.dumps(summary)}")
    return summary
