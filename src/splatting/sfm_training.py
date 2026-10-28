"""Gaussian splatting from a dense SfM model: MCMC densification, camera refinement and
per-image colour compensation.

Inputs are `src.sfm` outputs (``frames.json`` + ``undistorted/``). Outputs use the layout of
`src.splatting.training`, so `view_gaussian_splat.py` and `render_gaussian_splat.py` read
them unchanged:

    scene.ply (+ .json sidecar)   standard 3DGS PLY, SH degree 3, SfM world coordinates
    cameras.json                  train cameras with refined poses, val cameras as solved
    checkpoint.pt                 parameters, camera/colour corrections, step
    training.jsonl                losses, PSNR, Gaussian count, speed, GPU memory
    metrics.json                  final train/val metrics
    evaluation/<step>/<split>/    reference | render previews and per-view metrics

Held-out (val) views never contribute gradients. Their colour correction is fitted after
rendering (``psnr_cc_db``, an affine colour fit per image, as in Zip-NeRF), because
exposure is a property of the capture, not of the scene.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import SH_C0, GaussianModel
from .sfm_config import SfmSplatConfig
from .sfm_scene import SfmScene, load_sfm_scene, read_rgb
from .storage import export_ply, write_json


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of [H,W,3] images with an 11×11 Gaussian window (separable)."""
    x, y = prediction.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    axis = torch.arange(11, dtype=x.dtype, device=x.device) - 5
    kernel = torch.exp(-(axis**2) / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    horizontal = kernel.view(1, 1, 1, 11).expand(3, 1, 1, 11)
    vertical = kernel.view(1, 1, 11, 1).expand(3, 1, 11, 1)

    def blur(value):
        return F.conv2d(F.conv2d(value, horizontal, groups=3), vertical, groups=3)

    mean_x, mean_y = blur(x), blur(y)
    var_x = blur(x * x) - mean_x.square()
    var_y = blur(y * y) - mean_y.square()
    covariance = blur(x * y) - mean_x * mean_y
    numerator = (2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)
    return (numerator / denominator).mean()


def rotation_6d(value: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt rotation from the first two columns (Zhou et al. 2019)."""
    a, b = value[..., :3], value[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


class CameraCorrections(torch.nn.Module):
    """Per-image pose increment (6D rotation + translation) and affine colour transform.

    Both start at the identity. The pose increment right-multiplies camera-to-world, i.e.
    it moves the camera in its own frame; the colour transform maps rendered RGB to the
    image's exposure and white balance.
    """

    def __init__(self, count: int):
        super().__init__()
        self.pose = torch.nn.Parameter(torch.zeros(count, 9))
        self.colour = torch.nn.Parameter(torch.zeros(count, 12))
        self.register_buffer("identity", torch.tensor([1.0, 0, 0, 0, 1, 0]))

    def camera_to_world(self, index: int, base: torch.Tensor) -> torch.Tensor:
        delta = torch.eye(4, device=base.device)
        delta[:3, :3] = rotation_6d(self.pose[index, :6] + self.identity)
        delta[:3, 3] = self.pose[index, 6:]
        return base @ delta

    def apply_colour(self, index: int, rgb: torch.Tensor) -> torch.Tensor:
        matrix = torch.eye(3, device=rgb.device) + self.colour[index, :9].view(3, 3)
        return rgb @ matrix.T + self.colour[index, 9:]


def fit_colour(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Least-squares affine colour map rendered → target, applied to ``rendered``."""
    x = torch.cat((rendered.reshape(-1, 3), torch.ones_like(rendered[..., :1]).reshape(-1, 1)), 1)
    y = target.reshape(-1, 3)
    # Normal equations (4×4) keep memory constant at full resolution.
    gram = (x.T @ x).double() + 1e-6 * torch.eye(4, device=x.device, dtype=torch.float64)
    solution = torch.linalg.solve(gram, (x.T @ y).double()).float()
    return (x @ solution).reshape(rendered.shape).clamp(0, 1)


def initial_model(scene: SfmScene, config: SfmSplatConfig, device: str) -> GaussianModel:
    """Seed Gaussians at SfM points; scale = mean distance to the 3 nearest neighbours."""
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(scene.points).query(scene.points, k=4)
    spacing = np.sqrt((distances[:, 1:] ** 2).mean(1)).clip(1e-7)
    count = len(scene.points)
    quats = np.zeros((count, 4), np.float32)
    quats[:, 0] = 1
    opacity = config.refinement.initial_opacity
    degree = config.training.sh_degree
    tensors = {
        "means": torch.from_numpy(scene.points),
        "scales": torch.from_numpy(
            np.log(spacing * config.refinement.initial_scale).repeat(3).reshape(-1, 3)
        ).float(),
        "quats": torch.from_numpy(quats),
        "opacities": torch.full((count,), math.log(opacity / (1 - opacity))),
        "sh0": torch.from_numpy((scene.colors - 0.5) / SH_C0)[:, None, :].float(),
        "shn": torch.zeros((count, (degree + 1) ** 2 - 1, 3)),
    }
    return GaussianModel(tensors, degree).to(device)


def scene_scale(scene: SfmScene) -> float:
    centres = np.stack([c.camera_to_world[:3, 3] for c in scene.cameras])
    return 1.1 * float(np.linalg.norm(centres - centres.mean(0), axis=1).max())


def render(model, world_to_camera, intrinsics, width, height, degree, mode):
    from gsplat import rasterization

    params = model.params
    k = intrinsics.clone()
    # `Camera` uses integer pixel centres; gsplat samples at (u + 0.5, v + 0.5).
    k[0, 2] += 0.5
    k[1, 2] += 0.5
    colours, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat((params["sh0"], params["shn"]), dim=1),
        viewmats=world_to_camera[None],
        Ks=k[None],
        width=width,
        height=height,
        sh_degree=degree,
        packed=False,
        rasterize_mode=mode,
        render_mode="RGB+ED",
    )
    return colours[0, ..., :3], colours[0, ..., 3], alpha[0, ..., 0], info


class ImageCache:
    """uint8 RGB per (image, downscale), loaded in parallel and kept in pinned CPU memory."""

    def __init__(self, scene: SfmScene, levels: set[int], workers: int = 16):
        self.images: dict[tuple[int, int], torch.Tensor] = {}
        jobs = [(i, level) for level in sorted(levels) for i in range(len(scene.cameras))]
        with ThreadPoolExecutor(workers) as pool:
            for (i, level), image in zip(
                jobs, pool.map(lambda job: read_rgb(scene.image_paths[job[0]], job[1]), jobs)
            ):
                self.images[i, level] = torch.from_numpy(image).pin_memory()

    def get(self, index: int, level: int, device: str) -> torch.Tensor:
        return self.images[index, level].to(device, non_blocking=True).float().div_(255)


def _level(config: SfmSplatConfig, step: int) -> int:
    factor = 1
    for start, value in config.training.resolution_schedule:
        if step >= start:
            factor = value
    return factor * config.data.downscale


@torch.no_grad()
def evaluate(model, scene, corrections, images, indices, split, step, output, config, degree):
    """Metrics at training resolution and at the VGGT model grid (long side 518)."""
    device = model.device
    level = config.data.downscale
    directory = output / "evaluation" / f"{step:06d}" / split
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, index in enumerate(indices):
        camera = scene.cameras[index]
        target = images.get(index, level, device)
        height, width = target.shape[:2]
        resized = camera.resized(width, height)
        world_to_camera = torch.tensor(resized.world_to_camera, dtype=torch.float32, device=device)
        if split == "train" and config.training.pose_refinement:
            world_to_camera = torch.linalg.inv(
                corrections.camera_to_world(index, torch.linalg.inv(world_to_camera))
            )
        k = torch.tensor(resized.intrinsics, dtype=torch.float32, device=device)
        rgb, _, alpha, _ = render(
            model, world_to_camera, k, width, height, degree, config.rasterize_mode
        )
        rgb = rgb.clamp(0, 1)
        corrected = (
            corrections.apply_colour(index, rgb).clamp(0, 1)
            if split == "train" and config.training.appearance
            else fit_colour(rgb, target)
        )

        def psnr(a, b):
            return float(-10 * torch.log10((a - b).square().mean().clamp_min(1e-12)))

        scale = 518 / max(width, height)
        small = (round(width * scale), round(height * scale))

        def shrink(value, size=small):
            return torch.from_numpy(
                cv2.resize(value.cpu().numpy(), size, interpolation=cv2.INTER_AREA)
            )

        rows.append(
            {
                "frame_id": camera.frame_id,
                "timestamp_s": float(scene.timestamps[index]),
                "psnr_db": psnr(rgb, target),
                "psnr_cc_db": psnr(corrected, target),
                "ssim": float(ssim(rgb, target)),
                "ssim_cc": float(ssim(corrected, target)),
                "psnr_db_at_518": psnr(shrink(rgb), shrink(target)),
                "psnr_cc_db_at_518": psnr(shrink(corrected), shrink(target)),
                "coverage_alpha_gt_half": float((alpha > 0.5).float().mean()),
                "preview": f"{number:05d}.jpg",
            }
        )
        preview = torch.cat((target, rgb), dim=1).mul(255).round().byte().cpu().numpy()
        preview = cv2.resize(preview, (preview.shape[1] // 2, preview.shape[0] // 2))
        cv2.imwrite(str(directory / f"{number:05d}.jpg"), preview[..., ::-1])

    keys = [k for k in rows[0] if k not in ("frame_id", "timestamp_s", "preview")] if rows else []
    report = {
        "step": step,
        "split": split,
        "gaussians": model.count,
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "frames": rows,
    }
    write_json(directory / "metrics.json", report)
    return report


def _versions() -> dict:
    packages = {}
    for name in ("torch", "gsplat", "pycolmap", "numpy"):
        try:
            packages[name] = version(name)
        except Exception:  # noqa: BLE001 - missing optional package is just unrecorded
            packages[name] = None
    return packages


def train_sfm_splat(
    sfm_dir: str | Path,
    output_dir: str | Path,
    config: SfmSplatConfig | None = None,
    *,
    device: str = "cuda:0",
    resume: bool = False,
) -> dict:
    """Train one splat; returns the final metrics (also written to ``metrics.json``).

    ``resume`` continues from ``checkpoint.pt`` in ``output_dir``: Gaussians, camera and
    colour corrections and the step are restored; Adam moments restart from zero.
    """
    from gsplat.strategy import MCMCStrategy

    config = config or SfmSplatConfig()
    training, refinement = config.training, config.refinement
    output = Path(output_dir).resolve()
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(training.seed)
    rng = np.random.default_rng(training.seed)

    scene = load_sfm_scene(
        sfm_dir,
        config.data.holdout_geometry_every,
        config.data.holdout_exclusion_s,
        tuple(config.data.time_range_s) or None,
    )
    levels = {_level(config, step) for step, _ in training.resolution_schedule}
    levels.add(config.data.downscale)
    images = ImageCache(scene, levels)
    radius = scene_scale(scene)
    model = initial_model(scene, config, device)
    corrections = CameraCorrections(len(scene.cameras)).to(device)
    first_step = 0
    if resume:
        state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = GaussianModel(state["params"], state["sh_degree"]).to(device)
        corrections.load_state_dict(state["corrections"])
        first_step = int(state["step"])

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "trainer": "src.splatting.sfm_training",
        "sfm_dir": str(Path(sfm_dir).resolve()),
        "scene_units": "arbitrary (SfM gauge)",
        "scene_scale": radius,
        "camera_convention": "opencv_world_to_camera",
        "pixel_convention": "integer_centers",
        "sfm_splat_config": config.to_dict(),
        "versions": _versions(),
        "initial_points": len(scene.points),
        "train_frame_ids": [scene.frame_ids[i] for i in scene.train_indices],
        "val_frame_ids": [scene.frame_ids[i] for i in scene.val_indices],
        "initial_camera": scene.cameras[scene.train_indices[0]].to_dict(),
    }
    write_json(output / "run.json", metadata)

    rates = {
        "means": training.means_lr * radius,
        "scales": training.scales_lr,
        "quats": training.quats_lr,
        "opacities": training.opacities_lr,
        "sh0": training.sh0_lr,
        "shn": training.shn_lr,
    }
    optimizers = {
        key: torch.optim.Adam([model.params[key]], lr=rates[key], eps=1e-15) for key in rates
    }
    pose_optimizer = torch.optim.Adam(
        [corrections.pose], lr=training.pose_lr, weight_decay=training.pose_regularization
    )
    colour_optimizer = torch.optim.Adam([corrections.colour], lr=training.appearance_lr)
    decay = (training.means_lr_final / training.means_lr) ** (1 / training.steps)
    strategy = MCMCStrategy(
        cap_max=refinement.max_gaussians,
        noise_lr=refinement.noise_lr,
        refine_start_iter=refinement.start_step,
        refine_stop_iter=refinement.stop_step,
        refine_every=refinement.every,
        min_opacity=refinement.min_opacity,
    )
    strategy.check_sanity(model.params, optimizers)
    strategy_state = strategy.initialize_state()

    train_tensor = torch.tensor(scene.train_indices, device=device)
    log_path = output / "training.jsonl"
    order: list[int] = []
    started = time.perf_counter()
    for group in optimizers["means"].param_groups:
        group["lr"] *= decay**first_step
    for step in range(first_step, training.steps):
        if not order:
            order = list(rng.permutation(scene.train_indices))
        index = int(order.pop())
        degree = min(training.sh_degree, step // training.sh_interval)
        level = _level(config, step)

        target = images.get(index, level, device)
        height, width = target.shape[:2]
        camera = scene.cameras[index].resized(width, height)
        base = torch.tensor(camera.camera_to_world, dtype=torch.float32, device=device)
        refine_pose = training.pose_refinement and step >= training.pose_start_step
        camera_to_world = corrections.camera_to_world(index, base) if refine_pose else base
        k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
        rgb, _, _, info = render(
            model,
            torch.linalg.inv(camera_to_world),
            k,
            width,
            height,
            degree,
            config.rasterize_mode,
        )
        if training.appearance:
            rgb = corrections.apply_colour(index, rgb)

        l1 = (rgb - target).abs().mean()
        structure = ssim(rgb, target)
        loss = (1 - training.ssim_weight) * l1 + training.ssim_weight * (1 - structure)
        loss = loss + training.opacity_regularization * model.params["opacities"].sigmoid().mean()
        loss = loss + training.scale_regularization * model.params["scales"].exp().mean()
        if training.appearance:
            loss = (
                loss + training.appearance_regularization * corrections.colour[index].square().sum()
            )
        loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if refine_pose:
            pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)
        if training.appearance:
            colour_optimizer.step()
            # A shared offset in every image's colour transform is indistinguishable from
            # a change of the Gaussians' colours. Keeping the transforms zero-mean pins the
            # canonical colours (what viewers render) to the average training exposure.
            with torch.no_grad():
                colour = corrections.colour
                colour[train_tensor] -= colour[train_tensor].mean(0)
        colour_optimizer.zero_grad(set_to_none=True)
        strategy.step_post_backward(
            model.params,
            optimizers,
            strategy_state,
            step,
            info,
            lr=optimizers["means"].param_groups[0]["lr"],
        )
        for group in optimizers["means"].param_groups:
            group["lr"] *= decay

        if step % training.log_every == 0 or step == training.steps - 1:
            with torch.no_grad():
                psnr = float(-10 * torch.log10((rgb.clamp(0, 1) - target).square().mean()))
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "l1": float(l1.detach()),
                "ssim": float(structure.detach()),
                "psnr_db": psnr,
                "gaussians": model.count,
                "downscale": level,
                "elapsed_s": time.perf_counter() - started,
                "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                f"step={step} loss={row['loss']:.4f} psnr={psnr:.2f} n={model.count} "
                f"x{level} t={row['elapsed_s']:.0f}s mem={row['gpu_peak_gib']:.1f}G",
                flush=True,
            )
        if step > first_step and step % training.evaluate_every == 0 and scene.val_indices:
            evaluate(
                model,
                scene,
                corrections,
                images,
                scene.val_indices,
                "val",
                step,
                output,
                config,
                degree,
            )
        if step and step % training.checkpoint_every == 0:
            _save(output, model, corrections, step, metadata)

    step = training.steps
    degree = min(training.sh_degree, (step - 1) // training.sh_interval)
    _save(output, model, corrections, step, metadata)
    export_ply(output / "scene.ply", model, {**metadata, "active_sh_degree": degree, "step": step})
    _write_cameras(output, scene, corrections, config)
    every = max(1, len(scene.train_indices) // 60)
    train_report = evaluate(
        model,
        scene,
        corrections,
        images,
        scene.train_indices[::every],
        "train",
        step,
        output,
        config,
        degree,
    )
    val_report = (
        evaluate(
            model,
            scene,
            corrections,
            images,
            scene.val_indices,
            "val",
            step,
            output,
            config,
            degree,
        )
        if scene.val_indices
        else None
    )
    metrics = {
        "step": step,
        "gaussians": model.count,
        "train_seconds": time.perf_counter() - started,
        "train": train_report["mean"],
        "val": val_report["mean"] if val_report else None,
        "pose_correction": _pose_summary(scene, corrections, radius),
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def _save(output: Path, model, corrections, step: int, metadata: dict) -> None:
    payload = {
        "metadata": metadata,
        "step": step,
        "sh_degree": model.sh_degree,
        "params": {key: value.detach().cpu() for key, value in model.params.items()},
        "corrections": {
            key: value.detach().cpu() for key, value in corrections.state_dict().items()
        },
    }
    temporary = output / "checkpoint.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "checkpoint.pt")


@torch.no_grad()
def _refined_world_to_camera(scene, corrections, index, config) -> np.ndarray:
    base = torch.tensor(
        scene.cameras[index].camera_to_world, dtype=torch.float32, device=corrections.pose.device
    )
    if config.training.pose_refinement and index in set(scene.train_indices):
        base = corrections.camera_to_world(index, base)
    return np.linalg.inv(base.double().cpu().numpy())


def _write_cameras(output: Path, scene, corrections, config) -> None:
    cameras = []
    for index, camera in enumerate(scene.cameras):
        entry = camera.to_dict()
        entry["world_to_camera"] = _refined_world_to_camera(
            scene, corrections, index, config
        ).tolist()
        entry["split"] = "val" if index in set(scene.val_indices) else "train"
        cameras.append(entry)
    write_json(
        output / "cameras.json",
        {"schema_version": 1, "camera_convention": "opencv_world_to_camera", "cameras": cameras},
    )


@torch.no_grad()
def _pose_summary(scene, corrections, radius) -> dict:
    if not scene.train_indices:
        return {}
    pose = corrections.pose[scene.train_indices]
    rotation = rotation_6d(pose[:, :6] + corrections.identity)
    angle = torch.rad2deg(
        torch.arccos(((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    )
    shift = pose[:, 6:].norm(dim=-1) / radius
    return {
        "rotation_deg_median": float(angle.median()),
        "rotation_deg_max": float(angle.max()),
        "translation_over_scene_scale_median": float(shift.median()),
        "translation_over_scene_scale_max": float(shift.max()),
    }
