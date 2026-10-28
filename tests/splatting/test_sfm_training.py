"""SfM-input splatting: correction modules, colour fit, scene loading and real CUDA training.

The scene is an analytic textured wall seen by cameras with exactly known poses, written
as a real COLMAP text model; the trainer, gsplat and pycolmap are not substituted.
"""

import json

import cv2
import numpy as np
import pytest
import torch

from src.splatting.sfm_config import SfmRefinementConfig, SfmSplatConfig, SfmTrainingConfig
from src.splatting.sfm_scene import load_sfm_scene
from src.splatting.sfm_training import CameraCorrections, fit_colour, rotation_6d, ssim

WIDTH, HEIGHT, FOCAL, WALL_Z = 64, 48, 60.0, 4.0


def _texture(size=256):
    rng = np.random.default_rng(3)
    noise = cv2.GaussianBlur(rng.uniform(0, 1, (size, size, 3)).astype(np.float32), (0, 0), 3)
    y, x = np.mgrid[:size, :size]
    checker = (((x // 32) + (y // 32)) % 2)[..., None] * 0.4
    return np.clip(0.3 + checker + 0.6 * (noise - noise.mean()), 0, 1).astype(np.float32)


def _wall_lookup(texture, x, y):
    """Texture colour at wall coordinates (x, y) in [-2, 2]."""
    size = texture.shape[0]
    map_x = ((x + 2) / 4 * (size - 1)).astype(np.float32)
    map_y = ((y + 2) / 4 * (size - 1)).astype(np.float32)
    return cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@pytest.fixture
def wall_sfm(tmp_path):
    """16 frames translating along x, COLMAP layout of `src.sfm` outputs."""
    texture = _texture()
    images = tmp_path / "undistorted" / "images"
    sparse = tmp_path / "undistorted" / "sparse"
    images.mkdir(parents=True)
    sparse.mkdir()
    v, u = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32) + 0.5  # COLMAP pixel centres
    frames, lines = [], []
    for index in range(16):
        centre_x = -0.6 + 0.08 * index
        x = centre_x + (u - WIDTH / 2) * WALL_Z / FOCAL
        y = (v - HEIGHT / 2) * WALL_Z / FOCAL
        rgb = _wall_lookup(texture, x, y)
        name = f"{index:05d}.jpg"
        cv2.imwrite(str(images / name), (rgb[..., ::-1] * 255).round().astype(np.uint8))
        lines.append(f"{index + 1} 1 0 0 0 {-centre_x} 0 0 1 {name}\n\n")
        frames.append(
            {
                "name": name,
                "frame_id": f"wall:{index}",
                "timestamp_s": index * 0.25,
                "geometry": True,
            }
        )
    (sparse / "cameras.txt").write_text(f"1 PINHOLE {WIDTH} {HEIGHT} {FOCAL} {FOCAL} 32 24\n")
    (sparse / "images.txt").write_text("".join(lines))
    rng = np.random.default_rng(4)
    wall = rng.uniform(-1.6, 1.6, (1500, 2)).astype(np.float32)
    colours = _wall_lookup(texture, wall[:, :1], wall[:, 1:])[:, 0]
    (sparse / "points3D.txt").write_text(
        "".join(
            f"{k + 1} {x} {y} {WALL_Z} {int(r * 255)} {int(g * 255)} {int(b * 255)} 0\n"
            for k, ((x, y), (r, g, b)) in enumerate(zip(wall, colours))
        )
    )
    (tmp_path / "frames.json").write_text(json.dumps(frames))
    return tmp_path


def test_corrections_start_at_identity():
    corrections = CameraCorrections(3)
    base = torch.eye(4)
    base[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    rgb = torch.rand(5, 5, 3)

    assert torch.allclose(rotation_6d(torch.tensor([1.0, 0, 0, 0, 1, 0])), torch.eye(3))
    assert torch.allclose(corrections.camera_to_world(1, base), base)
    assert torch.allclose(corrections.apply_colour(1, rgb), rgb)


def test_colour_fit_recovers_exposure_and_white_balance():
    rgb = torch.rand(20, 30, 3) * 0.8
    target = (rgb @ torch.diag(torch.tensor([1.1, 0.9, 1.05])).T + 0.03).clamp(0, 1)

    assert torch.allclose(fit_colour(rgb, target), target, atol=1e-5)
    assert float(ssim(target, target)) == pytest.approx(1.0, abs=1e-5)
    assert float(ssim(rgb, target)) < 0.999


def test_scene_loading_holds_out_every_eighth_geometry_frame(wall_sfm):
    scene = load_sfm_scene(wall_sfm, holdout_geometry_every=8, holdout_exclusion_s=0.3)

    assert scene.val_indices == [7, 15]
    # 0.3 s exclusion also removes the direct neighbours (0.25 s apart) from training.
    assert scene.train_indices == [i for i in range(16) if i not in (6, 7, 8, 14, 15)]
    assert np.allclose(scene.cameras[0].intrinsics[:2, 2], [31.5, 23.5])
    assert np.allclose(scene.cameras[3].camera_to_world[:3, 3], [-0.6 + 0.24, 0, 0])


@pytest.mark.cuda
def test_training_fits_held_out_views_and_exports_viewer_files(wall_sfm, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("gsplat")
    from src.rendering.cli import load_cameras
    from src.splatting import load_model, train_sfm_splat

    config = SfmSplatConfig(
        training=SfmTrainingConfig(
            steps=1500,
            sh_degree=1,
            sh_interval=200,
            resolution_schedule=((0, 1),),
            pose_start_step=200,
            log_every=100,
            evaluate_every=10_000,
            checkpoint_every=10_000,
        ),
        refinement=SfmRefinementConfig(max_gaussians=6000, start_step=100, stop_step=1200),
    )

    metrics = train_sfm_splat(wall_sfm, tmp_path / "splat", config)

    assert metrics["val"]["psnr_cc_db"] > 22  # one val view extrapolates past the path
    assert metrics["train"]["psnr_db"] > 24
    model, metadata = load_model(tmp_path / "splat" / "scene.ply", "cuda")
    extent = json.loads((tmp_path / "splat" / "run.json").read_text())["scene_extent"]
    assert float(model.params["scales"].detach().exp().max()) <= 0.1 * extent * 1.0001
    assert model.count == metrics["gaussians"] and metadata["active_sh_degree"] == 1
    assert len(load_cameras(tmp_path / "splat" / "cameras.json")) == 16

    # Resuming a finished run restores the saved Gaussians and corrections exactly.
    resumed = train_sfm_splat(wall_sfm, tmp_path / "splat", config, resume=True)
    assert resumed["gaussians"] == metrics["gaussians"]
    assert resumed["val"]["psnr_cc_db"] == pytest.approx(metrics["val"]["psnr_cc_db"], abs=0.05)


def test_block_ownership_follows_the_nearest_camera():
    from src.splatting.merge_blocks import block_owner

    centres = np.array([[0.0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]])
    labels = np.array([0, 0, 1, 1])
    points = np.array([[0.4, 2, 0], [5.4, 0, 0], [5.6, 0, 0], [12, -1, 0]])

    assert block_owner(centres, labels, points).tolist() == [0, 0, 1, 1]
