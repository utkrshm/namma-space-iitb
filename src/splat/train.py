"""Train one Gaussian splat on solved cameras.

Input: a `src.sfm.Solution` (undistorted cameras + images + SfM points) and,
optionally, monocular depth (``<out>/depth``).

Output (``<out>/splat/`` or a block directory):
    scene.ply (+ scene.json)   standard 3DGS PLY, SfM world coordinates
    cameras.json               cameras with refined poses (train) or as solved (held out)
    checkpoint.pt              parameters, per-image corrections, step
    training.jsonl             loss, PSNR, Gaussian count, speed, GPU memory
    metrics.json               held-out PSNR / SSIM / LPIPS (+ colour-corrected PSNR)

What the trainer does, and why:
- **MCMC densification** (Kheradmand et al. 2024) with a fixed Gaussian budget. Its noise
  is lr × covariance in world units, so it is divided by radius². Coorg's radius of about 11
  otherwise made it about 120× too strong, and the scene fogged.
- **Screen-size pruning:** a Gaussian covering more than 15% of the image is relocated. It
  stops near-camera floaters and out-of-memory errors in tile intersection.
- **Per-image pose and colour (affine) corrections.** Phones auto-expose and white-balance
  every frame; without this the splat bakes exposure changes into the scene as fog.
- **Depth and normal priors:** L1 against monocular inverse depth fitted to the SfM
  points (`src.depth.align_depth_maps`), and agreement of normals, weighted towards
  textureless pixels. A scale-free patch Pearson correlation, the earlier prior, cannot
  see a blob uniformly in front of a wall patch. Faded to zero at mid-training, it left
  blank walls bumpy and smeared off the filmed path (vyas-room). The weight now decays
  exponentially over the whole run.
- **Clipped pixels:** where the capture is saturated (blown-out windows), a brighter
  render is not penalized, so windows do not pull their surroundings into a grey haze.
- **Anti-aliased rasterization** (Mip-Splatting's 2D filter in gsplat): no shimmering when
  the tour zooms out or the viewer moves away.

Held-out views never contribute gradients. They are scored with identity colour
correction (standard PSNR) and with a fitted affine colour map (``psnr_cc``), because
exposure belongs to the capture, not the scene.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import SplatConfig
from ..contracts import Camera
from ..depth import depth_normals, pearson_loss, read_rgb
from .model import SH_C0, GaussianModel, load_ply, render, save_ply, write_json


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of [H,W,3] images, 11×11 Gaussian window (separable)."""
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


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10 * torch.log10((a - b).square().mean().clamp_min(1e-12)))


def rotation_6d(value: torch.Tensor) -> torch.Tensor:
    a, b = value[..., :3], value[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)


class Corrections(torch.nn.Module):
    """Per-image pose increment (camera frame) and affine colour map, both starting at identity."""

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
    gram = (x.T @ x).double() + 1e-6 * torch.eye(4, device=x.device, dtype=torch.float64)
    solution = torch.linalg.solve(gram, (x.T @ y).double()).float()
    return (x @ solution).reshape(rendered.shape).clamp(0, 1)


def patch_pearson(depth: torch.Tensor, prior: torch.Tensor, valid: torch.Tensor, grid: int = 4):
    """Mean of global and per-patch Pearson losses (local shape matters more than global)."""
    height, width = prior.shape
    losses = [pearson_loss(depth, prior, valid)]
    for row in range(grid):
        for column in range(grid):
            rows = slice(row * height // grid, (row + 1) * height // grid)
            columns = slice(column * width // grid, (column + 1) * width // grid)
            if valid[rows, columns].sum() > 64:
                losses.append(
                    pearson_loss(depth[rows, columns], prior[rows, columns], valid[rows, columns])
                )
    return torch.stack(losses).mean()


class Images:
    """uint8 RGB per (image, downscale) in pinned CPU memory, loaded in parallel."""

    def __init__(self, paths: list[Path], long_side: int, levels: set[int], workers: int = 16):
        self.images: dict[tuple[int, int], torch.Tensor] = {}
        jobs = [(i, level) for level in sorted(levels) for i in range(len(paths))]
        with ThreadPoolExecutor(workers) as pool:
            loaded = pool.map(lambda job: read_rgb(paths[job[0]], long_side, job[1]), jobs)
            for job, image in zip(jobs, loaded, strict=True):
                self.images[job] = torch.from_numpy(image).pin_memory()

    def get(self, index: int, level: int, device) -> torch.Tensor:
        return self.images[index, level].to(device, non_blocking=True).float().div_(255)


class Priors:
    """Monocular inverse depth per image, loaded lazily, with its metric fit if any.

    ``get`` returns ``(mono, metric)``: the relative inverse depth, and the inverse depth
    in SfM units (``a · mono + b`` from ``alignment.json``) or None without a fit.
    """

    def __init__(self, folder: Path, paths: list[Path]):
        self.paths = [Path(folder) / (p.stem + ".npy") for p in paths]
        fits_path = Path(folder) / "alignment.json"
        fits = json.loads(fits_path.read_text()) if fits_path.exists() else {}
        self.fits = [fits.get(p.stem) for p in paths]
        self.cache: dict[int, torch.Tensor] = {}

    def get(self, index: int, device) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        if index not in self.cache:
            if not self.paths[index].exists():
                return None
            self.cache[index] = torch.from_numpy(np.load(self.paths[index]).astype(np.float32))
        mono = self.cache[index].to(device, non_blocking=True)
        fit = self.fits[index]
        return mono, None if fit is None else fit["a"] * mono + fit["b"]


def textureless_weight(rgb: torch.Tensor, scale: float) -> torch.Tensor:
    """Per pixel, exp(−(gradient magnitude / scale)²) of the grey image [H,W,3] → [H,W]."""
    grey = rgb.mean(-1)[None, None]
    padded = F.pad(grey, (1, 1, 1, 1), mode="replicate")
    along_x = (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2]) / 2
    along_y = (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1]) / 2
    gradient = F.max_pool2d(torch.hypot(along_x, along_y), 3, stride=1, padding=1)[0, 0]
    return torch.exp(-((gradient / scale) ** 2))


def prior_losses(depth, alpha, target, intrinsics, prior, config: SplatConfig):
    """``(depth loss, normal loss)`` of one render against its monocular prior.

    ``depth`` and ``alpha`` are the render's, ``target`` the image, all at full size;
    ``prior`` is `Priors.get`'s pair. The comparison runs at the prior's resolution.
    """
    mono, metric = prior
    size = mono.shape
    small_depth = F.interpolate(depth[None, None], size=size, mode="area")[0, 0]
    small_alpha = F.interpolate(alpha[None, None], size=size, mode="area")[0, 0]
    opaque = small_alpha.detach() > 0.5
    if metric is None:
        return patch_pearson(small_depth, mono, opaque), small_depth.new_zeros(())

    # Monocular depth smears across object edges; compare only away from them.
    gy, gx = torch.gradient(metric)
    valid = (torch.hypot(gx, gy) / metric.clamp_min(1e-6) < 0.05) & (metric > 0)
    if valid.sum() < 64:
        zero = small_depth.new_zeros(())
        return zero, zero
    small_target = F.interpolate(target.permute(2, 0, 1)[None], size=size, mode="area")[0]
    textureless = textureless_weight(small_target.permute(1, 2, 0), config.textureless_gradient)

    # Alpha-weighted inverse depth, on every pixel. With the normalized expected depth (and
    # only opaque pixels compared), a floater in front of a wall can satisfy the prior by
    # turning transparent, and the black background shows through; dark smudges under the
    # ceiling appeared that way on vyas-room. Here a hole costs as much as a wrong depth.
    # Full weight on blank surfaces, a fifth on texture, where the photometric loss knows
    # better.
    inverse = small_alpha / small_depth.clamp_min(1e-6)
    residual = (inverse - metric)[valid].abs() / metric[valid].median()
    weight = (0.2 + 0.8 * textureless)[valid]
    depth_loss = (weight * residual).sum() / weight.sum()

    k = intrinsics.clone()
    k[:2] *= size[1] / depth.shape[1]
    rendered_normals = depth_normals(small_depth, k)
    prior_normals = depth_normals(1 / metric.clamp_min(1e-6), k)
    weight = textureless * valid * opaque
    disagreement = 1 - (rendered_normals * prior_normals).sum(-1)
    normal_loss = (weight * disagreement).sum() / weight.sum().clamp_min(1.0)
    return depth_loss, normal_loss


def photometric_l1(rgb: torch.Tensor, target: torch.Tensor, saturation_level: float):
    """Mean absolute error; where the capture is clipped, only a darker render counts."""
    error = rgb - target
    clipped = target >= saturation_level
    return torch.where(clipped, (-error).clamp_min(0), error.abs()).mean()


def split_indices(count: int, holdout_every: int) -> tuple[list[int], list[int]]:
    if holdout_every <= 0:
        return list(range(count)), []
    val = [i for i in range(count) if i % holdout_every == 0]
    return [i for i in range(count) if i % holdout_every], val


def scene_radius(cameras: list[Camera]) -> float:
    """1.1 × the 95th-percentile camera distance from the median camera centre.

    A percentile, not the maximum: one badly registered camera must not scale every
    learning rate and the MCMC noise.
    """
    centres = np.stack([c.centre for c in cameras])
    distance = np.linalg.norm(centres - np.median(centres, axis=0), axis=1)
    return max(1.1 * float(np.percentile(distance, 95)), 1e-3)


def scene_extent(cameras: list[Camera], points: np.ndarray) -> float:
    centres = np.stack([c.centre for c in cameras])
    distance = np.linalg.norm(points - np.median(centres, axis=0), axis=1) if len(points) else [0]
    return max(scene_radius(cameras), float(np.percentile(distance, 90)))


def initial_model(points: np.ndarray, colors: np.ndarray, config: SplatConfig, device):
    """Gaussians at SfM points; size = RMS distance to the 3 nearest neighbours."""
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k=4)
    spacing = np.sqrt((distances[:, 1:] ** 2).mean(1)).clip(1e-7)
    count = len(points)
    quats = np.zeros((count, 4), np.float32)
    quats[:, 0] = 1
    opacity = config.initial_opacity
    degree = config.sh_degree
    tensors = {
        "means": torch.from_numpy(points),
        "scales": torch.from_numpy(np.log(spacing * config.initial_scale).repeat(3).reshape(-1, 3)),
        "quats": torch.from_numpy(quats),
        "opacities": torch.full((count,), math.log(opacity / (1 - opacity))),
        "sh0": torch.from_numpy((colors - 0.5) / SH_C0)[:, None, :],
        "shn": torch.zeros((count, (degree + 1) ** 2 - 1, 3)),
    }
    return GaussianModel(tensors, degree).to(device)


def _level(config: SplatConfig, step: int) -> int:
    factor = 1
    for start, value in config.resolution_schedule:
        if step >= start:
            factor = value
    return factor


_LPIPS = {}


@torch.no_grad()
def lpips_vgg(a: torch.Tensor, b: torch.Tensor) -> float:
    """LPIPS (VGG) of [H,W,3] images in 0..1, as reported by 3DGS and Mip-NeRF 360."""
    if "vgg" not in _LPIPS:
        import lpips

        _LPIPS["vgg"] = lpips.LPIPS(net="vgg", verbose=False).eval().to(a.device)
    x = a.permute(2, 0, 1)[None] * 2 - 1
    y = b.permute(2, 0, 1)[None] * 2 - 1
    return float(_LPIPS["vgg"](x, y))


@torch.no_grad()
def evaluate(model, cameras, images, indices, corrections, config, output: Path, name: str, train):
    """Per-view metrics at full training resolution, plus side-by-side previews."""
    device = model.device
    directory = output / "evaluation" / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, index in enumerate(indices):
        target = images.get(index, 1, device)
        height, width = target.shape[:2]
        camera = cameras[index].resized(width, height)
        camera_to_world = torch.tensor(camera.camera_to_world, dtype=torch.float32, device=device)
        if train and config.pose_refinement:
            camera_to_world = corrections.camera_to_world(index, camera_to_world)
        k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
        rgb, _, alpha, _ = render(
            model, torch.linalg.inv(camera_to_world), k, width, height, mode=config.rasterize_mode
        )
        rgb = rgb.clamp(0, 1)
        if train and config.appearance:
            rgb = corrections.apply_colour(index, rgb).clamp(0, 1)
        corrected = fit_colour(rgb, target)
        rows.append(
            {
                "name": camera.name,
                "psnr": psnr(rgb, target),
                "psnr_cc": psnr(corrected, target),
                "ssim": float(ssim(rgb, target)),
                "lpips": lpips_vgg(rgb, target),
                "coverage": float((alpha > 0.5).float().mean()),
            }
        )
        if number % max(1, len(indices) // 12) == 0:
            import cv2

            preview = torch.cat((target, rgb), dim=1).mul(255).round().byte().cpu().numpy()
            scale = 960 / preview.shape[1]
            preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(directory / f"{number:04d}.jpg"), preview[..., ::-1])
    keys = [k for k in rows[0] if k != "name"] if rows else []
    report = {
        "views": len(rows),
        "gaussians": model.count,
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "frames": rows,
    }
    write_json(directory / "metrics.json", report)
    return report


def train_splat(
    cameras: list[Camera],
    image_paths: list[Path],
    points: np.ndarray,
    colors: np.ndarray,
    output_dir,
    config: SplatConfig | None = None,
    *,
    depth_dir: Path | None = None,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    initial: GaussianModel | str | Path | None = None,
    device: str = "cuda:0",
    log=print,
) -> dict:
    """Train and return ``metrics.json``; a finished run in ``output_dir`` is reused."""
    from gsplat.strategy import MCMCStrategy

    config = config or SplatConfig()
    output = Path(output_dir).resolve()
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and (output / "scene.ply").exists():
        return json.loads(metrics_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    if train_indices is None:
        train_indices, val_indices = split_indices(len(cameras), config.holdout_every)
    val_indices = val_indices or []
    levels = {_level(config, s) for s, _ in config.resolution_schedule} | {1}
    images = Images(image_paths, config.max_long_side, levels)
    radius = scene_radius([cameras[i] for i in train_indices])
    extent = scene_extent([cameras[i] for i in train_indices], points)
    if isinstance(initial, (str, Path)):
        model = load_ply(initial, device)
    elif isinstance(initial, GaussianModel):
        model = initial.to(device)
    else:
        model = initial_model(points, colors, config, device)
    if model.sh_degree != config.sh_degree:
        raise ValueError("Initial model SH degree differs from the configuration")
    corrections = Corrections(len(cameras)).to(device)
    priors = Priors(depth_dir, image_paths) if depth_dir and config.depth_prior_weight > 0 else None
    prior_decay = config.depth_prior_weight_final / max(config.depth_prior_weight, 1e-12)

    rates = {
        "means": config.means_lr * radius,
        "scales": config.scales_lr,
        "quats": config.quats_lr,
        "opacities": config.opacities_lr,
        "sh0": config.sh0_lr,
        "shn": config.shn_lr,
    }
    optimizers = {k: torch.optim.Adam([model.params[k]], lr=rates[k], eps=1e-15) for k in rates}
    pose_optimizer = torch.optim.Adam([corrections.pose], lr=config.pose_lr)
    colour_optimizer = torch.optim.Adam([corrections.colour], lr=config.appearance_lr)
    decay = (config.means_lr_final / config.means_lr) ** (1 / config.steps)
    strategy = MCMCStrategy(
        cap_max=max(config.max_gaussians, model.count),
        noise_lr=config.noise_lr,
        refine_start_iter=config.refine_start,
        refine_stop_iter=config.refine_stop,
        refine_every=config.refine_every,
        min_opacity=config.min_opacity,
    )
    strategy.check_sanity(model.params, optimizers)
    state = strategy.initialize_state()
    train_tensor = torch.tensor(train_indices, device=device)

    metadata = {
        "trainer": "src.splat.train",
        "config": asdict(config),
        "scene_radius": radius,
        "scene_extent": extent,
        "train": [cameras[i].name for i in train_indices],
        "val": [cameras[i].name for i in val_indices],
        "initial_gaussians": model.count,
    }
    write_json(output / "run.json", metadata)
    log_path = output / "training.jsonl"
    log_path.unlink(missing_ok=True)
    order: list[int] = []
    started = time.perf_counter()
    for step in range(config.steps):
        if not order:
            order = list(rng.permutation(train_indices))
        index = int(order.pop())
        level = _level(config, step)
        degree = min(config.sh_degree, step // config.sh_interval)

        target = images.get(index, level, device)
        height, width = target.shape[:2]
        camera = cameras[index].resized(width, height)
        base = torch.tensor(camera.camera_to_world, dtype=torch.float32, device=device)
        refine_pose = config.pose_refinement and step >= config.pose_start_step
        camera_to_world = corrections.camera_to_world(index, base) if refine_pose else base
        k = torch.tensor(camera.intrinsics, dtype=torch.float32, device=device)
        rgb, depth, alpha, info = render(
            model,
            torch.linalg.inv(camera_to_world),
            k,
            width,
            height,
            sh_degree=degree,
            mode=config.rasterize_mode,
        )
        if config.random_background:
            rgb = rgb + (1 - alpha)[..., None] * torch.rand(3, device=device)
        if config.appearance:
            rgb = corrections.apply_colour(index, rgb)

        l1 = photometric_l1(rgb, target, config.saturation_level)
        structure = ssim(rgb, target)
        loss = (1 - config.ssim_weight) * l1 + config.ssim_weight * (1 - structure)
        loss = loss + config.opacity_regularization * model.params["opacities"].sigmoid().mean()
        loss = loss + config.scale_regularization * model.params["scales"].exp().mean()
        if priors is not None:
            prior = priors.get(index, device)
            if prior is not None:
                depth_loss, normal_loss = prior_losses(depth, alpha, target, k, prior, config)
                fade = prior_decay ** (step / config.steps)
                loss = loss + fade * (
                    config.depth_prior_weight * depth_loss
                    + config.normal_prior_weight * normal_loss
                )
        if config.appearance:
            loss = (
                loss + config.appearance_regularization * corrections.colour[index].square().sum()
            )
        loss.backward()

        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if refine_pose:
            pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)
        if config.appearance:
            colour_optimizer.step()
            # A shared offset in every image's colour map is the same as a change of the
            # Gaussians' colours; zero-mean maps pin the canonical colours to the average
            # training exposure, which is what viewers and held-out views render.
            with torch.no_grad():
                corrections.colour[train_tensor] -= corrections.colour[train_tensor].mean(0)
        colour_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.params["scales"].clamp_(max=math.log(config.max_scale_fraction * extent))
        _prune_screen_size(model, config, info, step, max(width, height))
        strategy.step_post_backward(
            model.params,
            optimizers,
            state,
            step,
            info,
            lr=optimizers["means"].param_groups[0]["lr"] / radius**2,
        )
        for group in optimizers["means"].param_groups:
            group["lr"] *= decay

        if step % config.log_every == 0 or step == config.steps - 1:
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "psnr": psnr(rgb.detach().clamp(0, 1), target),
                "gaussians": model.count,
                "downscale": level,
                "elapsed_s": time.perf_counter() - started,
                "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                "means_lr": optimizers["means"].param_groups[0]["lr"],
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            log(
                f"step {step} loss {row['loss']:.4f} psnr {row['psnr']:.2f} n {model.count} "
                f"x{level} {row['elapsed_s']:.0f}s {row['gpu_peak_gib']:.1f}G"
            )
        if step and step % config.checkpoint_every == 0:
            _save(output, model, corrections, step)
        if step and step % config.evaluate_every == 0 and val_indices:
            report = evaluate(
                model,
                cameras,
                images,
                val_indices,
                corrections,
                config,
                output,
                f"{step:06d}",
                False,
            )
            log(f"held-out at {step}: {report['mean']}")

    _save(output, model, corrections, config.steps)
    save_ply(output / "scene.ply", model, metadata)
    _write_cameras(output, cameras, corrections, config, set(train_indices))
    every = max(1, len(train_indices) // 40)
    train_report = evaluate(
        model, cameras, images, train_indices[::every], corrections, config, output, "train", True
    )
    val_report = (
        evaluate(model, cameras, images, val_indices, corrections, config, output, "val", False)
        if val_indices
        else None
    )
    metrics = {
        "steps": config.steps,
        "gaussians": model.count,
        "train_seconds": time.perf_counter() - started,
        "train": train_report["mean"],
        "val": val_report["mean"] if val_report else None,
    }
    write_json(metrics_path, metrics)
    return metrics


@torch.no_grad()
def _prune_screen_size(model, config: SplatConfig, info: dict, step: int, side: int) -> None:
    """Hand Gaussians covering a large part of the image to MCMC relocation."""
    refining = config.refine_start <= step < config.refine_stop and step % config.refine_every == 0
    if config.max_screen_fraction <= 0 or not refining:
        return
    radii = info["radii"]
    if radii.dim() == 3:
        radii = radii.max(dim=-1).values
    oversize = radii.max(dim=0).values > max(config.max_screen_fraction * side, 64)
    if oversize.any():
        dead = math.log(config.min_opacity / 2 / (1 - config.min_opacity / 2))
        model.params["opacities"][oversize] = dead


def _save(output: Path, model, corrections, step: int) -> None:
    payload = {
        "step": step,
        "sh_degree": model.sh_degree,
        "params": {k: v.detach().cpu() for k, v in model.params.items()},
        "corrections": {k: v.detach().cpu() for k, v in corrections.state_dict().items()},
    }
    temporary = output / "checkpoint.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "checkpoint.pt")


@torch.no_grad()
def _write_cameras(output: Path, cameras, corrections, config, train: set[int]) -> None:
    entries = []
    for index, camera in enumerate(cameras):
        camera_to_world = torch.tensor(
            camera.camera_to_world, dtype=torch.float32, device=corrections.pose.device
        )
        if config.pose_refinement and index in train:
            camera_to_world = corrections.camera_to_world(index, camera_to_world)
        entry = camera.moved(np.linalg.inv(camera_to_world.double().cpu().numpy())).to_dict()
        entry["split"] = "train" if index in train else "val"
        entries.append(entry)
    write_json(
        output / "cameras.json", {"camera_convention": "opencv_world_to_camera", "cameras": entries}
    )


def load_cameras(path) -> tuple[list[Camera], list[str]]:
    listing = json.loads(Path(path).read_text())["cameras"]
    return [Camera.from_dict(c) for c in listing], [c.get("split", "train") for c in listing]
