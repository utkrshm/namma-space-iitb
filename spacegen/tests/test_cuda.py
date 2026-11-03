"""GPU tests with real gsplat rendering and training (no mocks)."""

import json
import math

import cv2
import numpy as np
import pytest
import torch

from spacegen.config import SplatConfig
from spacegen.contracts import Camera, look_at
from spacegen.depth import align_depth_maps
from spacegen.sfm import Solution
from spacegen.splat.model import SH_C0, GaussianModel, load_ply, render_camera, save_ply
from spacegen.splat.train import train_splat
from spacegen.support import compute_support, support_map

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]


def textured_wall(device="cuda:0", count=60) -> GaussianModel:
    """A 2 × 2 wall in the x-z plane at y = 0, made of flat discs with a checker pattern."""
    u, v = np.meshgrid(np.linspace(-1, 1, count), np.linspace(-1, 1, count))
    means = np.stack((u.ravel(), np.zeros(u.size), v.ravel() + 1), axis=1)
    checker = ((np.floor(u * 4) + np.floor(v * 4)) % 2).ravel()
    colours = np.stack((0.2 + 0.6 * checker, 0.3 + 0.2 * (1 - checker), 0.5 * np.ones(u.size)), 1)
    spacing = 2 / (count - 1)
    scales = np.log(np.tile([spacing * 0.7, spacing * 0.05, spacing * 0.7], (u.size, 1)))
    quats = np.tile([1.0, 0, 0, 0], (u.size, 1))
    tensors = {
        "means": torch.tensor(means, dtype=torch.float32),
        "scales": torch.tensor(scales, dtype=torch.float32),
        "quats": torch.tensor(quats, dtype=torch.float32),
        "opacities": torch.full((u.size,), 4.0),
        "sh0": torch.tensor((colours - 0.5) / SH_C0, dtype=torch.float32)[:, None],
        "shn": torch.zeros((u.size, 15, 3)),
    }
    return GaussianModel(tensors, 3).to(device)


def camera_at(eye, target=(0, 0, 1), size=(160, 120), name="c") -> Camera:
    width, height = size
    focal = 0.9 * width
    k = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]])
    return Camera(
        name, width, height, k, look_at(np.asarray(eye, float), np.asarray(target, float))
    )


def front_cameras(count=24):
    cameras = []
    for number in range(count):
        angle = math.radians(-35 + 70 * number / (count - 1))
        eye = (2.5 * math.sin(angle), -2.5 * math.cos(angle), 1.0 + 0.3 * math.sin(3 * angle))
        cameras.append(camera_at(eye, name=f"{number:03d}"))
    return cameras


def test_support_marks_front_views_measured_and_back_views_not():
    wall = textured_wall()
    cameras = front_cameras()
    wall.extras["support"] = compute_support(wall, cameras, long_side=160)
    assert (wall.extras["support"] != 0).float().mean() > 0.95

    front, alpha, _ = support_map(wall, camera_at((0.3, -2.0, 1.1)), 35.0)
    covered = alpha > 0.5
    assert covered.float().mean() > 0.3
    assert front[covered].mean() > 0.95

    back, alpha, _ = support_map(wall, camera_at((0.0, 2.5, 1.0)), 35.0)
    assert back[alpha > 0.5].mean() < 0.05


def test_ply_round_trip_keeps_parameters_and_flags(tmp_path):
    wall = textured_wall()
    wall.extras["support"] = torch.randint(-(2**62), 2**62, (wall.count,), device="cuda:0")
    wall.extras["filled"] = (torch.rand(wall.count, device="cuda:0") > 0.5).long()
    save_ply(tmp_path / "wall.ply", wall)
    loaded = load_ply(tmp_path / "wall.ply")
    for key, value in wall.params.items():
        assert torch.allclose(loaded.params[key], value)
    assert torch.equal(loaded.extras["support"], wall.extras["support"])
    assert torch.equal(loaded.extras["filled"], wall.extras["filled"])


def test_training_recovers_a_rendered_wall(tmp_path):
    """Images rendered from a known wall; a splat trained from sparse points must match them."""
    truth = textured_wall()
    cameras = front_cameras(30)
    paths = []
    for camera in cameras:
        with torch.no_grad():
            rgb, _, _, _ = render_camera(truth, camera)
        path = tmp_path / f"{camera.name}.png"
        cv2.imwrite(str(path), (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()[..., ::-1])
        paths.append(path)
    rng = np.random.default_rng(0)
    points = np.c_[rng.uniform(-1, 1, 400), rng.normal(0, 0.02, 400), rng.uniform(0, 2, 400)]
    config = SplatConfig(
        steps=1500,
        max_gaussians=20_000,
        resolution_schedule=((0, 1),),
        holdout_every=0,
        pose_refinement=False,
        appearance=False,
        depth_prior_weight=0.0,
        refine_stop=1200,
        evaluate_every=10**9,
        checkpoint_every=10**9,
        log_every=500,
    )
    train = [i for i in range(30) if i % 5]
    val = [i for i in range(30) if i % 5 == 0]
    metrics = train_splat(
        cameras,
        paths,
        points.astype(np.float32),
        np.full((400, 3), 0.5, np.float32),
        tmp_path / "run",
        config,
        train_indices=train,
        val_indices=val,
        log=lambda _: None,
    )
    assert metrics["val"]["psnr"] > 24, metrics["val"]
    assert (tmp_path / "run" / "scene.ply").exists()

    # A run killed after its last step starts again from the checkpoint, without training.
    (tmp_path / "run" / "scene.ply").unlink()
    again = train_splat(
        cameras,
        paths,
        points.astype(np.float32),
        np.full((400, 3), 0.5, np.float32),
        tmp_path / "run",
        config,
        train_indices=train,
        val_indices=val,
        log=lambda _: None,
    )
    assert abs(again["val"]["psnr"] - metrics["val"]["psnr"]) < 1e-3
    assert again["train_seconds"] < metrics["train_seconds"] / 3
    assert (tmp_path / "run" / "scene.ply").exists()


def test_first_seen_reveals_each_part_of_the_wall_with_the_camera_that_sees_it():
    from spacegen.demo import first_seen

    wall = textured_wall()
    # Camera 0 looks at the left half from close by, camera 1 at the right half.
    left = camera_at((-0.5, -0.6, 1.0), target=(-0.5, 0, 1.0))
    right = camera_at((0.5, -0.6, 1.0), target=(0.5, 0, 1.0))
    seen = first_seen(wall.params["means"].detach(), [left, right], max_depth=5.0)
    x, z = wall.params["means"][:, 0].detach(), wall.params["means"][:, 2].detach()
    centre_band = (z - 1.0).abs() < 0.15
    assert (seen[centre_band & ((x + 0.5).abs() < 0.2)] == 0).all()
    assert (seen[centre_band & ((x - 0.5).abs() < 0.2)] == 1).all()
    assert (seen[(z - 1.0).abs() > 0.8] == 2).all()  # out of both views: never revealed


def test_metric_depth_prior_puts_a_blank_wall_in_place(tmp_path):
    """A wall textured only at its border, with stray points over its blank middle. The
    photometric loss cannot tell where a blank surface is, so without a prior the strays
    stay as blobs at wrong depths. With monocular depth fitted to the (border) points, the
    trained wall's middle must sit at the true depth."""
    truth = textured_wall()
    u = truth.params["means"][:, 0].detach().cpu().numpy()
    v = truth.params["means"][:, 2].detach().cpu().numpy() - 1
    blank = (np.abs(u) < 0.6) & (np.abs(v) < 0.6)
    with torch.no_grad():
        truth.params["sh0"][torch.from_numpy(blank).cuda()] = (0.85 - 0.5) / SH_C0
    cameras = front_cameras(30)
    paths, true_depths = [], []
    (tmp_path / "depth").mkdir()
    for camera in cameras:
        with torch.no_grad():
            rgb, depth, _, _ = render_camera(truth, camera)
        path = tmp_path / f"{camera.name}.png"
        cv2.imwrite(str(path), (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()[..., ::-1])
        paths.append(path)
        true_depths.append(depth.cpu().numpy())
        # A monocular network's output: inverse depth up to an unknown scale and shift.
        mono = 3.0 / np.maximum(depth.cpu().numpy(), 1e-3) + 0.4
        np.save(tmp_path / "depth" / f"{camera.name}.npy", mono.astype(np.float16))

    rng = np.random.default_rng(0)
    border = rng.uniform(-1, 1, (4000, 2))
    border = border[(np.abs(border) >= 0.65).any(axis=1)][:600]
    points = np.c_[border[:, 0], rng.normal(0, 0.01, len(border)), border[:, 1] + 1]
    # Stray points over the blank middle, in front of and behind the wall: what MCMC
    # relocation and wrong matches leave there in real rooms. Same colour from every
    # view, so the photometric loss has no reason to move them.
    middle = rng.uniform(-0.55, 0.55, (300, 2))
    strays = np.c_[middle[:, 0], rng.uniform(-0.8, 0.4, 300), middle[:, 1] + 1]
    points = np.r_[points, strays].astype(np.float32)
    solution = Solution(
        cameras,
        paths,
        points,
        np.full((len(points), 3), 0.5, np.float32),
        [np.arange(len(cameras))] * len(points),
        [{} for _ in cameras],
    )
    assert len(align_depth_maps(solution, tmp_path / "depth")) == len(cameras)

    def blank_error(depth_prior_weight: float, name: str) -> float:
        config = SplatConfig(
            steps=1500,
            max_gaussians=20_000,
            resolution_schedule=((0, 1),),
            holdout_every=0,
            pose_refinement=False,
            appearance=False,
            depth_prior_weight=depth_prior_weight,
            depth_prior_weight_final=depth_prior_weight / 10,
            refine_stop=1200,
            evaluate_every=10**9,
            checkpoint_every=10**9,
            log_every=500,
        )
        train = [i for i in range(30) if i % 5]
        val = [i for i in range(30) if i % 5 == 0]
        train_splat(
            cameras,
            paths,
            points,
            solution.colors,
            tmp_path / name,
            config,
            depth_dir=tmp_path / "depth",
            train_indices=train,
            val_indices=val,
            log=lambda _: None,
        )
        model = load_ply(tmp_path / name / "scene.ply")
        errors = []
        for index in val:
            with torch.no_grad():
                _, depth, _, _ = render_camera(model, cameras[index])
                _, _, middle, _ = render_camera(
                    truth.subset(torch.from_numpy(blank).cuda()), cameras[index]
                )
            inside = middle.cpu().numpy() > 0.5
            error = np.abs(depth.cpu().numpy() - true_depths[index]) / true_depths[index]
            errors.append(np.median(error[inside]))
        return float(np.mean(errors))

    with_prior = blank_error(0.3, "prior")
    # The position learning rate must still follow its own schedule with the priors on (a
    # shared variable name once multiplied it by the prior's fade, freezing the Gaussians).
    last = json.loads((tmp_path / "prior" / "training.jsonl").read_text().splitlines()[-1])
    run = json.loads((tmp_path / "prior" / "run.json").read_text())
    expected = SplatConfig().means_lr_final * run["scene_radius"]
    assert last["means_lr"] == pytest.approx(expected, rel=1e-3)
    without = blank_error(0.0, "none")
    assert with_prior < 0.02, (with_prior, without)
    assert with_prior < 0.5 * without, (with_prior, without)


def test_training_killed_mid_run_resumes_from_its_checkpoint(tmp_path):
    """A run stopped after its step-1000 checkpoint continues from step 1001, with its
    optimizer state, and ends as good as an uninterrupted run."""
    truth = textured_wall()
    cameras = front_cameras(30)
    paths = []
    for camera in cameras:
        with torch.no_grad():
            rgb, _, _, _ = render_camera(truth, camera)
        path = tmp_path / f"{camera.name}.png"
        cv2.imwrite(str(path), (rgb.clamp(0, 1) * 255).round().byte().cpu().numpy()[..., ::-1])
        paths.append(path)
    rng = np.random.default_rng(0)
    points = np.c_[rng.uniform(-1, 1, 400), rng.normal(0, 0.02, 400), rng.uniform(0, 2, 400)]
    config = SplatConfig(
        steps=1500,
        max_gaussians=20_000,
        resolution_schedule=((0, 1),),
        holdout_every=0,
        pose_refinement=False,
        appearance=False,
        depth_prior_weight=0.0,
        refine_stop=1200,
        evaluate_every=10**9,
        checkpoint_every=1000,
        log_every=100,
    )
    train = [i for i in range(30) if i % 5]
    val = [i for i in range(30) if i % 5 == 0]

    def run(name, log):
        return train_splat(
            cameras,
            paths,
            points.astype(np.float32),
            np.full((400, 3), 0.5, np.float32),
            tmp_path / name,
            config,
            train_indices=train,
            val_indices=val,
            log=log,
        )

    whole = run("whole", lambda _: None)

    class Killed(Exception):
        pass

    def kill(message):
        if message.startswith("step 1200 "):
            raise Killed

    with pytest.raises(Killed):
        run("killed", kill)
    messages = []
    again = run("killed", messages.append)
    assert "resumed from the checkpoint at step 1001" in messages
    assert not any(m.startswith("step 500 ") for m in messages)
    steps = [json.loads(r)["step"] for r in (tmp_path / "killed" / "training.jsonl").open()]
    assert steps == sorted(set(steps)) and steps[-1] == 1499
    assert again["val"]["psnr"] > whole["val"]["psnr"] - 0.5, (again["val"], whole["val"])
