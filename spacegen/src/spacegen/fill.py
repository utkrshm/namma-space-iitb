"""Fill and mark: replace what the capture never saw with generated content, and say so.

Problem: from a standing point inside a room, many directions were never filmed. There
the splat shows black gaps (nothing rendered) or smears (Gaussians fitted from other
directions, see `spacegen.support`). Training on the capture cannot fix that.

Method (a per-scene version of the "inpaint novel views, then distil" family, e.g.
G4Splat 2025 and GSFix3D 2025):

1. **Probe views:** at each area's standing point (the area camera nearest its centre,
   so it is free space), a ring of ``yaw_steps`` × ``pitches_deg`` views at eye height
   with a ``field_of_view_deg`` lens.
2. **Holes:** pixels with alpha < ``min_alpha`` (0.9), or with support < ``min_support``
   (seen from no
   nearby direction).
3. **Inpaint:** small holes and slivers are filled from their border (OpenCV Telea);
   regions of at least ``large_hole`` of the view are painted by SDXL inpainting. Known
   pixels are pasted back exactly, so only holes change.
4. **Depth:** monocular depth of the inpainted view, fitted (scale and shift in inverse
   depth) to the render's depth on known pixels. It is used where the holes are.
5. **Seed Gaussians** on a pixel grid in the holes at that depth, flagged ``filled`` and
   marked as supported from the probe's direction. Later, overlapping probes then see
   them as content and do not invent a second, contradictory version. This is what keeps
   the fill consistent across views.
6. **Distil:** fine-tune on the real training images (full loss, with the trainer's
   per-image corrections) and on the probe views (loss only inside holes, plus depth).
   Real views keep measured content correct, and probe views push smears out of the way.
   Densification is off, so the per-Gaussian flags stay aligned.

Output (``<run>/fill/``): ``scene.ply`` (with ``filled`` and ``support``), ``fill.json``
(per probe: hole share before and after, generated share), ``probes/`` (render | holes |
inpainted for each probe) and ``scene_frame.json`` in the run directory.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import FillConfig, PipelineConfig
from .contracts import Camera, look_at
from .depth import predict_depth
from .scene_frame import SceneFrame, estimate_frame
from .splat.model import SH_C0, GaussianModel, load_ply, render, render_camera, save_ply, write_json
from .splat.train import Corrections, Images, evaluate, load_cameras, scene_radius, ssim
from .support import compute_support, fibonacci_directions, support_map


def standing_points(frame: SceneFrame, cameras: list[Camera]) -> list[dict]:
    """Per area: the area camera nearest the area centre (free space), at its own height."""
    centres = frame.to_aligned(np.stack([c.centre for c in cameras]))
    stands = []
    for area in frame.areas:
        members = np.array(area["cameras"])
        plan = centres[members, :2]
        nearest = members[np.linalg.norm(plan - np.array(area["centre"]), axis=1).argmin()]
        eye = centres[nearest].copy()
        eye[2] = float(np.median(centres[members, 2]))
        stands.append({"area": area["name"], "camera": int(nearest), "eye_aligned": eye.tolist()})
    return stands


def probe_camera(frame: SceneFrame, eye_aligned, yaw_deg, pitch_deg, width, height, fov_deg, name):
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    direction = np.array(
        [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    )
    eye = frame.to_world(np.asarray(eye_aligned)[None])[0]
    target = eye + frame.rotation.T @ direction
    focal = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    k = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    return Camera(name, width, height, k, look_at(eye, target, frame.up))


def split_holes(rgb: np.ndarray, hole: np.ndarray, large_share: float):
    """Fill every hole from its border (Telea); return that image and the mask of holes
    covering at least ``large_share`` of the view, which need a generative model."""
    base = cv2.inpaint(rgb, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8))
    big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= large_share * hole.size]
    return base, np.isin(labels, big)


class Inpainter:
    """SDXL inpainting at about 1024 px; known pixels are pasted back with a soft seam."""

    def __init__(self, config: FillConfig, device: str):
        from diffusers import AutoPipelineForInpainting

        self.config = config
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            config.model, torch_dtype=torch.float16, variant="fp16"
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.generator = torch.Generator(device).manual_seed(0)

    def __call__(self, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """Small holes and slivers: classical inpainting from the border (diffusion paints
        flat grey into thin masks). Regions of at least ``large_hole`` of the view: SDXL."""
        from PIL import Image

        base, large = split_holes(rgb, hole, self.config.large_hole)
        if not large.any():
            return base
        region = cv2.dilate(large.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        height, width = hole.shape
        scale = 1024 / max(width, height)
        size = (round(width * scale / 8) * 8, round(height * scale / 8) * 8)
        image = Image.fromarray(cv2.resize(base, size, interpolation=cv2.INTER_CUBIC))
        mask = Image.fromarray(
            cv2.resize(region.astype(np.uint8) * 255, size, interpolation=cv2.INTER_NEAREST)
        )
        result = self.pipe(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            image=image,
            mask_image=mask,
            width=size[0],
            height=size[1],
            num_inference_steps=30,
            strength=0.99,
            guidance_scale=6.0,
            generator=self.generator,
        ).images[0]
        generated = cv2.resize(np.asarray(result), (width, height), interpolation=cv2.INTER_AREA)
        soft = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 3.0)
        soft = np.maximum(soft, large.astype(np.float32))[..., None]
        return (generated * soft + base * (1 - soft)).round().astype(np.uint8)


def align_depth(mono_inverse: np.ndarray, rendered: np.ndarray, known: np.ndarray, fallback: float):
    """Metric depth for every pixel: inverse depth = a·mono + b fitted on ``known`` pixels."""
    x, y = mono_inverse[known], 1 / np.maximum(rendered[known], 1e-6)
    if known.sum() < 500:
        scale = 1 / fallback / max(np.median(mono_inverse), 1e-6)
        return 1 / np.maximum(mono_inverse * scale, 1e-6)
    a, b = np.polyfit(x, y, 1)
    for _ in range(2):  # trim the worst 10% and refit
        residual = np.abs(a * x + b - y)
        keep = residual <= np.percentile(residual, 90)
        a, b = np.polyfit(x[keep], y[keep], 1)
    if a <= 0:
        scale = np.median(y) / max(np.median(x), 1e-6)
        a, b = scale, 0.0
    inverse = a * mono_inverse + b
    return 1 / np.clip(inverse, 1 / (50 * fallback), 1 / (0.15 * fallback))


def seed_gaussians(
    camera: Camera, rgb, depth, hole, stride: int, degree: int, device
) -> GaussianModel | None:
    """Gaussians on a pixel grid inside ``hole``, back-projected at ``depth``."""
    ys, xs = np.mgrid[stride // 2 : hole.shape[0] : stride, stride // 2 : hole.shape[1] : stride]
    inside = hole[ys, xs]
    if not inside.any():
        return None
    ys, xs = ys[inside], xs[inside]
    z = depth[ys, xs]
    k = camera.intrinsics
    local = np.stack(((xs - k[0, 2]) / k[0, 0] * z, (ys - k[1, 2]) / k[1, 1] * z, z), axis=1)
    camera_to_world = camera.camera_to_world
    means = local @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    colours = rgb[ys, xs].astype(np.float32) / 255
    size = z * stride / k[0, 0] * 0.8
    count = len(means)
    quats = np.zeros((count, 4), np.float32)
    quats[:, 0] = 1
    tensors = {
        "means": torch.from_numpy(means.astype(np.float32)),
        "scales": torch.from_numpy(np.log(np.repeat(size[:, None], 3, 1)).astype(np.float32)),
        "quats": torch.from_numpy(quats),
        "opacities": torch.full((count,), math.log(0.8 / 0.2)),
        "sh0": torch.from_numpy((colours - 0.5) / SH_C0)[:, None, :],
        "shn": torch.zeros((count, (degree + 1) ** 2 - 1, 3)),
    }
    return GaussianModel(tensors, degree).to(device)


def _direction_bits(means: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    directions = torch.tensor(fibonacci_directions(), dtype=torch.float32, device=means.device)
    offset = F.normalize(means - centre, dim=1)
    bins = (offset @ directions.T).argmax(1)
    weights = torch.tensor(
        [1 << b if b < 63 else -(1 << 63) for b in range(64)],
        dtype=torch.int64,
        device=means.device,
    )
    return weights[bins]


def hole_mask(alpha: np.ndarray, fraction: np.ndarray, config: FillConfig) -> np.ndarray:
    hole = (alpha < config.min_alpha) | (fraction < config.min_support)
    hole = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hole = cv2.dilate(hole, np.ones((7, 7), np.uint8))
    return hole.astype(bool)


@torch.no_grad()
def measured_share(model: GaussianModel, camera: Camera, config: FillConfig) -> dict:
    """Shares of a view that are unrendered, generated, and measured by the capture."""
    filled = model.extras.get("filled")
    support = model.extras["support"]
    if filled is not None:
        model.extras["support"] = torch.where(filled.bool(), torch.zeros_like(support), support)
    fraction, alpha, _ = support_map(model, camera, config.support_angle_deg)
    model.extras["support"] = support
    unrendered = float((alpha < config.min_alpha).float().mean())
    measured = float(
        ((alpha >= config.min_alpha) & (fraction >= config.min_support)).float().mean()
    )
    return {
        "unrendered": unrendered,
        "measured": measured,
        "generated_or_unsupported": 1 - unrendered - measured,
    }


def fill_gaps(run: Path, config: PipelineConfig, device: str, log) -> dict:
    fill = config.fill
    output = Path(run) / "fill"
    if (output / "fill.json").exists() and (output / "scene.ply").exists():
        return json.loads((output / "fill.json").read_text())
    (output / "probes").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    model = load_ply(Path(run) / "splat" / "scene.ply", device)
    cameras, splits = load_cameras(Path(run) / "splat" / "cameras.json")
    train = [i for i, split in enumerate(splits) if split == "train"]
    if "support" not in model.extras:
        model.extras["support"] = compute_support(model, [cameras[i] for i in train])
    log(f"fill: direction support computed ({time.perf_counter() - started:.0f}s)")
    model.extras["filled"] = torch.zeros(model.count, dtype=torch.int64, device=device)

    from .reslice import run_solution

    solution = run_solution(run)
    frame = estimate_frame([cameras[i] for i in train], solution.points)
    write_json(Path(run) / "scene_frame.json", frame.to_dict())
    stands = standing_points(frame, [cameras[i] for i in range(len(cameras))])
    log(
        f"fill: {len(stands)} standing points, camera height {frame.camera_height:.3f}, "
        f"floor found {frame.floor_found}"
    )

    width = fill.probe_long_side
    height = round(width * 0.75)
    probes = [
        probe_camera(
            frame,
            stand["eye_aligned"],
            yaw,
            pitch,
            width,
            height,
            fill.field_of_view_deg,
            f"{s}_{p}_{y}",
        )
        for s, stand in enumerate(stands)
        for p, pitch in enumerate(fill.pitches_deg)
        for y, yaw in enumerate(np.arange(fill.yaw_steps) * 360 / fill.yaw_steps)
    ]
    before = [measured_share(model, probe, fill) for probe in probes]
    log(f"fill: {len(probes)} probes scored ({time.perf_counter() - started:.0f}s)")

    inpainter = Inpainter(fill, device)
    pseudo, records = [], []
    for number, probe in enumerate(probes):
        fraction, alpha, depth = support_map(model, probe, fill.support_angle_deg)
        with torch.no_grad():  # the reflection pass would otherwise keep its graph on the GPU
            rgb, _, _, _ = render_camera(model, probe)
        rgb_np = (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()
        alpha_np, fraction_np, depth_np = (t.cpu().numpy() for t in (alpha, fraction, depth))
        hole = hole_mask(alpha_np, fraction_np, fill)
        share = float(hole.mean())
        record = {"probe": probe.name, "hole_share": share}
        if share < 0.003:
            records.append(record)
            continue
        composite = inpainter(rgb_np, hole)
        known = (~hole) & (alpha_np >= fill.min_alpha) & (depth_np > 0)
        mono = predict_depth(composite, config.depth.model, device)
        fallback = 2.0 * frame.camera_height
        metric = align_depth(mono, depth_np / np.maximum(alpha_np, 1e-6), known, fallback)
        seeded = seed_gaussians(probe, composite, metric, hole, 3, model.sh_degree, device)
        if seeded is not None:
            centre = torch.tensor(probe.centre, dtype=torch.float32, device=device)
            seeded.extras = {
                "support": _direction_bits(seeded.params["means"].detach(), centre),
                "filled": torch.ones(seeded.count, dtype=torch.int64, device=device),
            }
            model = GaussianModel.concatenate([model, seeded])
            record["seeded"] = seeded.count
        pseudo.append({"camera": probe, "rgb": composite, "hole": hole, "depth": metric})
        records.append(record)
        sheet = np.concatenate(
            (rgb_np, np.where(hole[..., None], [255, 0, 255], rgb_np).astype(np.uint8), composite),
            1,
        )
        cv2.imwrite(str(output / "probes" / f"{number:03d}.jpg"), sheet[..., ::-1])
        if number % 6 == 0:
            log(f"fill: probe {number}/{len(probes)} hole {share:.1%} gaussians {model.count}")
    del inpainter
    torch.cuda.empty_cache()

    log(f"fill: distilling {len(pseudo)} probe views into {model.count} Gaussians")
    model = distil(model, cameras, train, pseudo, run, config, device, log)
    after = [measured_share(model, probe, fill) for probe in probes]
    save_ply(output / "scene.ply", model, {"filled_gaussians": int(model.extras["filled"].sum())})

    # Measured content must not get worse: score the held-out views again.
    val = [i for i, split in enumerate(splits) if split == "val"]
    held_out = None
    if val:
        images = Images(
            [solution.image_paths[i] for i in range(len(cameras))], config.splat.max_long_side, {1}
        )
        held_out = evaluate(model, cameras, images, val, None, config.splat, output, "val", False)[
            "mean"
        ]

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))

    report = {
        "standing_points": stands,
        "probes": len(probes),
        "inpainted_probes": len(pseudo),
        "filled_gaussians": int(model.extras["filled"].sum()),
        "gaussians": model.count,
        "before": {k: mean(before, k) for k in before[0]},
        "after": {k: mean(after, k) for k in after[0]},
        "held_out_after_fill": held_out,
        "per_probe": [
            {**r, "before": b, "after": a} for r, b, a in zip(records, before, after, strict=True)
        ],
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "fill.json", report)
    log(f"fill: before {report['before']} after {report['after']}")
    return report


def distil(
    model, cameras, train, pseudo, run, config: PipelineConfig, device, log
) -> GaussianModel:
    """Fine-tune on real views (full loss) and probe views (holes only); no densification."""
    fill, splat = config.fill, config.splat
    state = torch.load(
        Path(run) / "splat" / "checkpoint.pt", map_location=device, weights_only=True
    )
    corrections = Corrections(len(cameras)).to(device)
    corrections.load_state_dict(state["corrections"])
    from .reslice import run_solution

    solution = run_solution(run)
    images = Images(solution.image_paths, splat.max_long_side, {2})
    radius = scene_radius([cameras[i] for i in train])
    rates = {
        "means": splat.means_lr_final * 10 * radius,
        "scales": splat.scales_lr,
        "quats": splat.quats_lr,
        "opacities": splat.opacities_lr,
        "sh0": splat.sh0_lr,
        "shn": splat.shn_lr,
    }
    optimizer = torch.optim.Adam(
        [{"params": [model.params[k]], "lr": r} for k, r in rates.items()], eps=1e-15
    )
    targets = [
        {
            "camera": p["camera"],
            "rgb": torch.from_numpy(p["rgb"]).to(device).float() / 255,
            "hole": torch.from_numpy(p["hole"]).to(device),
            "inverse": torch.from_numpy(1 / p["depth"]).to(device).float(),
        }
        for p in pseudo
    ]
    rng = np.random.default_rng(0)
    for step in range(fill.finetune_steps):
        if targets and rng.random() < 0.4:
            target = targets[rng.integers(len(targets))]
            rgb, depth, alpha, _ = render_camera(model, target["camera"], mode=splat.rasterize_mode)
            hole = target["hole"]
            weight = torch.where(hole, fill.pseudo_view_weight, 0.1)[..., None]
            loss = (weight * (rgb - target["rgb"]).abs()).mean()
            inverse = alpha / depth.clamp_min(1e-6)
            loss = loss + 0.05 * ((inverse - target["inverse"]).abs() * hole).mean() * radius
            loss = loss + 0.1 * ((1 - alpha) * hole).mean()
        else:
            index = int(train[rng.integers(len(train))])
            target_rgb = images.get(index, 2, device)
            h, w = target_rgb.shape[:2]
            camera = cameras[index].resized(w, h)
            rgb, _, _, _ = render(
                model,
                torch.tensor(camera.world_to_camera, dtype=torch.float32, device=device),
                torch.tensor(camera.intrinsics, dtype=torch.float32, device=device),
                w,
                h,
                mode=splat.rasterize_mode,
            )
            if splat.appearance:
                with torch.no_grad():
                    matrix = torch.eye(3, device=device) + corrections.colour[index, :9].view(3, 3)
                    offset = corrections.colour[index, 9:]
                rgb = rgb @ matrix.T + offset
            loss = 0.8 * (rgb - target_rgb).abs().mean() + 0.2 * (1 - ssim(rgb, target_rgb))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.params["quats"].data = F.normalize(model.params["quats"].data, dim=1)
        if step % 500 == 0:
            log(f"fill distil step {step} loss {float(loss):.4f}")
    keep = model.params["opacities"].detach().sigmoid() > 0.005
    return model.subset(keep)
