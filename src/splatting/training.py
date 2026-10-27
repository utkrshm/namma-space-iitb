"""Single-GPU, fixed-camera Gaussian optimization from refined VGGT geometry."""

import json
import math
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import torch

from ..rendering.renderer import render_gaussians, require_cuda
from .config import SplattingConfig
from .contracts import SplattingResult
from .data import RefinedScene
from .geometry import initialize_points
from .losses import masked_ssim, rgb_depth_loss
from .model import GaussianModel
from .refinement import BudgetedStrategy
from .storage import cpu_tree, export_ply, load_checkpoint, save_checkpoint, write_json


def _frame_tensors(frame, device):
    return {
        "rgb": torch.as_tensor(frame.rgb, device=device),
        "depth": torch.as_tensor(frame.depth, device=device),
        "image_mask": torch.as_tensor(frame.image_mask, device=device),
        "geometry_mask": torch.as_tensor(frame.geometry_mask, device=device),
    }


def _optimizers(model, config, scene_scale):
    rates = {
        "means": config.means_lr * scene_scale,
        "scales": config.scales_lr,
        "quats": config.quats_lr,
        "opacities": config.opacities_lr,
        "sh0": config.sh0_lr,
        "shn": config.shn_lr,
    }
    return {
        key: torch.optim.Adam([parameter], lr=rates[key], eps=1e-15)
        for key, parameter in model.params.items()
    }


def _restore_strategy(state: dict, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


@torch.no_grad()
def evaluate(model, scene, indices, config, degree, output_dir: Path, step: int, split: str):
    """Evaluate all requested cameras, retaining provenance beside preview files."""
    directory = output_dir / "evaluation" / f"{step:06d}" / split
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, index in enumerate(indices):
        frame = scene.frame(index)
        target = _frame_tensors(frame, model.device)
        output = render_gaussians(model, frame.camera, config.render, sh_degree=degree)
        if not torch.isfinite(output.rgb).all() or not torch.isfinite(output.depth).all():
            raise RuntimeError(f"Nonfinite evaluation render: {frame.camera.frame_id}")
        # Clamp only for display/metrics. Optimization uses the raw SH RGB output.
        rgb = output.rgb.clamp(0, 1)
        mask = target["image_mask"]
        geometry = target["geometry_mask"]
        mse = (rgb - target["rgb"]).square()[mask].mean()
        relative_depth = (
            (output.depth[geometry] - target["depth"][geometry]).abs() / target["depth"][geometry]
        ).mean()
        coverage = (output.alpha[mask] > 0.5).float().mean()
        rows.append(
            {
                "frame_id": frame.camera.frame_id,
                "timestamp_s": scene.frames[index].timestamp_s,
                "psnr_db": float(-10 * torch.log10(mse.clamp_min(1e-12))),
                "ssim": float(masked_ssim(rgb, target["rgb"], mask)),
                "relative_depth_error": float(relative_depth),
                "coverage_alpha_gt_half": float(coverage),
                "preview": f"{number:05d}.png",
            }
        )
        # Side-by-side reference and prediction for visual server-side review.
        preview = torch.cat((target["rgb"], rgb), dim=1).mul(255).round().byte().cpu().numpy()
        if not cv2.imwrite(str(directory / f"{number:05d}.png"), preview[..., ::-1]):
            raise OSError("Failed to save evaluation preview")

    aggregate = (
        {
            key: sum(row[key] for row in rows) / len(rows)
            for key in (
                "psnr_db",
                "ssim",
                "relative_depth_error",
                "coverage_alpha_gt_half",
            )
        }
        if rows
        else {}
    )
    report = {
        "step": step,
        "split": split,
        "sh_degree": degree,
        "gaussians": model.count,
        "mean": aggregate,
        "frames": rows,
    }
    write_json(directory / "metrics.json", report)
    return report


def train_splat(
    manifest: str | Path,
    output_dir: str | Path,
    config: SplattingConfig | None = None,
    *,
    device: str = "cuda",
    resume: str | Path | None = None,
) -> SplattingResult:
    """Fit a scene on the development server; no implicit CPU/cloud fallback.

    Resume restores optimizer/refinement/RNG state. It requires byte-identical
    source inputs, the same configuration and the same runtime package versions.
    """
    config = config or SplattingConfig()
    target_device = require_cuda(device)
    scene = RefinedScene(manifest, config.data)
    fingerprint = scene.fingerprint()
    output_dir = Path(output_dir).resolve()
    resume_path = Path(resume).resolve() if resume is not None else None
    if output_dir == scene.root or output_dir in scene.path.parents:
        raise ValueError("Output directory must be separate from source input directories")
    if resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory or explicitly resume a checkpoint")
    if resume_path is not None and resume_path.parent != output_dir:
        raise ValueError("Resume into the checkpoint's existing output directory")
    packages = {
        name: version(name) for name in ("torch", "gsplat", "numpy", "opencv-python-headless")
    }
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    sampler = torch.Generator(device="cpu").manual_seed(config.training.seed)

    if resume_path is not None:
        model, saved = load_checkpoint(resume_path, str(target_device))
        same_config = json.dumps(saved["config"], sort_keys=True) == json.dumps(
            config.to_dict(), sort_keys=True
        )
        if saved["input_fingerprint"] != fingerprint or not same_config:
            raise ValueError("Resume rejected: source content or training configuration changed")
        if saved["versions"] != packages:
            raise ValueError("Resume rejected: runtime package versions changed")
        scene_scale = saved["scene_scale"]
        initialization = saved["initialization"]
        start_step = saved["step"]
    else:
        initial = initialize_points(scene, config.initialization, config.training.seed)
        model = GaussianModel.from_points(
            initial, config.training.sh_degree, config.initialization.opacity, str(target_device)
        )
        scene_scale = initial.scene_scale
        initialization = initial.diagnostics
        start_step = 0
    if not 0 <= start_step <= config.training.steps:
        raise ValueError("Checkpoint step is outside the configured training schedule")

    optimizers = _optimizers(model, config.training, scene_scale)
    strategy = BudgetedStrategy(config.refinement, len(scene.train_indices))
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    if resume_path is not None:
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][key])
        strategy_state = _restore_strategy(saved["strategy_state"], model.device)
        sampler.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device=model.device)
    strategy.check_sanity(model.params, optimizers)
    if resume_path is not None:
        del saved

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(scene.path),
        "input_fingerprint": fingerprint,
        "scene_units": scene.raw["scene_units"],
        "scene_scale": scene_scale,
        "camera_convention": "opencv_world_to_camera",
        "depth_convention": "camera_z",
        "point_convention": "world",
        "pixel_convention": "integer_centers",
        "config": config.to_dict(),
        "versions": packages,
        "initialization": initialization,
        "train_frame_ids": [scene.frames[i].camera.frame_id for i in scene.train_indices],
        "val_frame_ids": [scene.frames[i].camera.frame_id for i in scene.val_indices],
        "initial_camera": scene.frames[scene.train_indices[0]].camera.to_dict(),
    }
    write_json(output_dir / "run.json", metadata)
    write_json(output_dir / "input_manifest.json", scene.raw)
    checkpoint = output_dir / "checkpoint.pt"
    log_path = output_dir / "training.jsonl"

    def save(step):
        save_checkpoint(
            checkpoint,
            model,
            **metadata,
            step=step,
            active_sh_degree=min(config.training.sh_degree, step // config.training.sh_interval),
            optimizers={key: optimizer.state_dict() for key, optimizer in optimizers.items()},
            strategy_state=cpu_tree(strategy_state),
            sampler_rng=sampler.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(model.device),
        )

    # One frame is transferred at a time; the dataset cache remains on the CPU.
    for step in range(start_step + 1, config.training.steps + 1):
        position = int(torch.randint(len(scene.train_indices), (1,), generator=sampler))
        frame = scene.frame(scene.train_indices[position])
        target = _frame_tensors(frame, model.device)
        degree = min(config.training.sh_degree, step // config.training.sh_interval)
        fraction = (step - 1) / max(config.training.steps - 1, 1)
        mean_rate = config.training.means_lr * math.exp(
            fraction * math.log(config.training.means_lr_final / config.training.means_lr)
        )
        optimizers["means"].param_groups[0]["lr"] = mean_rate * scene_scale
        depth_weight = (
            1 - fraction
        ) * config.training.depth_weight + fraction * config.training.depth_weight_final
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        output = render_gaussians(
            model, frame.camera, config.render, sh_degree=degree, absgrad=config.refinement.absgrad
        )
        strategy.step_pre_backward(model.params, optimizers, strategy_state, step, output.info)
        loss, components = rgb_depth_loss(
            output,
            **target,
            ssim_weight=config.training.ssim_weight,
            depth_weight=depth_weight,
        )
        scale_loss = model.params["scales"].exp().mean() / scene_scale
        loss = loss + config.training.scale_regularization * scale_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}; last complete checkpoint retained")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0, error_if_nonfinite=True)
        for optimizer in optimizers.values():
            optimizer.step()
        # Mutate topology only AFTER Adam consumes the current parameters' gradients.
        strategy.step_post_backward(
            model.params, optimizers, strategy_state, step, output.info, packed=config.render.packed
        )
        model.constrain(scene_scale)

        if step == 1 or step % config.training.log_every == 0:
            model.validate()
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "gaussians": model.count,
                "depth_weight": depth_weight,
                **{key: float(value.detach()) for key, value in components.items()},
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"step={step} loss={row['loss']:.6f} gaussians={model.count}", flush=True)
        if step % config.training.checkpoint_every == 0:
            save(step)
        if scene.val_indices and step % config.training.evaluate_every == 0:
            evaluate(model, scene, scene.val_indices, config, degree, output_dir, step, "val")

    final_step = config.training.steps
    degree = min(config.training.sh_degree, final_step // config.training.sh_interval)
    save(final_step)
    # Do not advertise untrained higher-order SH bands for short runs.
    export_metadata = {**metadata, "active_sh_degree": degree, "step": final_step}
    export_ply(output_dir / "scene.ply", model, export_metadata)
    train_report = evaluate(
        model, scene, scene.train_indices, config, degree, output_dir, final_step, "train"
    )
    val_report = (
        evaluate(model, scene, scene.val_indices, config, degree, output_dir, final_step, "val")
        if scene.val_indices
        else None
    )
    write_json(output_dir / "geometry_checks.json", scene.diagnostics)
    write_json(
        output_dir / "metrics.json",
        {
            "step": final_step,
            "gaussians": model.count,
            "train": train_report,
            "val": val_report,
            "quality_status": "requires_visual_review",
            "note": "Metrics are measurements, not a guarantee of novel-view quality.",
        },
    )
    return SplattingResult(
        output_dir, checkpoint, output_dir / "scene.ply", output_dir / "metrics.json"
    )
