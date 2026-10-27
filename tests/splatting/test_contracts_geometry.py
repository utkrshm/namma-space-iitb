import json
from dataclasses import replace

import numpy as np
import pytest

from src.splatting import Camera, SplattingConfig, load_scene
from src.splatting.config import DataConfig, InitializationConfig
from src.splatting.contracts import load_array
from src.splatting.geometry import fuse_voxels, initialize_points, multiview_support


def test_camera_direction_and_half_pixel_resize():
    transform = np.eye(4)
    transform[:3, 3] = [-2, 0, 0]
    k = np.array([[100, 0, 49.5], [0, 100, 39.5], [0, 0, 1]])
    camera = Camera("clip:0", 100, 80, k, transform)
    np.testing.assert_allclose(camera.camera_to_world[:3, 3], [2, 0, 0])
    np.testing.assert_allclose(
        camera.resized(50, 40).intrinsics, [[50, 0, 24.5], [0, 50, 19.5], [0, 0, 1]]
    )
    np.testing.assert_allclose(camera.camera_to_world @ camera.world_to_camera, np.eye(4))
    with pytest.raises(ValueError, match="right-handed"):
        Camera("bad", 100, 80, k, np.diag([-1, 1, 1, 1]))


def test_named_and_indexed_arrays(tmp_path):
    stack = np.arange(24).reshape(2, 3, 4)
    np.savez(tmp_path / "geometry.npz", extrinsics=stack)
    np.testing.assert_array_equal(
        load_array({"path": "geometry.npz", "key": "extrinsics", "index": 1}, tmp_path), stack[1]
    )
    with pytest.raises(ValueError, match="key"):
        load_array("geometry.npz", tmp_path)
    np.save(tmp_path / "objects.npy", np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        load_array("objects.npy", tmp_path)


def test_alignment_guard_and_provenance(plane_scene):
    scene = load_scene(plane_scene)
    frame = scene.frame(0)
    assert frame.geometry_mask.all()
    assert scene.frames[0].timestamp_s == 0
    assert scene.frames[0].camera.frame_id == "clip:0"
    original = scene.fingerprint()
    manifest = json.loads(plane_scene.read_text())
    manifest["frames"][0]["world_to_camera"][0][3] = -1
    plane_scene.write_text(json.dumps(manifest))
    changed = load_scene(plane_scene)
    assert changed.fingerprint() != original
    with pytest.raises(ValueError, match="disagrees"):
        changed.frame(0)


def test_wrong_conventions_and_partial_splits_fail(plane_scene):
    manifest = json.loads(plane_scene.read_text())
    manifest["depth_convention"] = "ray_distance"
    plane_scene.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="depth_convention"):
        load_scene(plane_scene)
    manifest["depth_convention"] = "camera_z"
    del manifest["frames"][0]["split"]
    plane_scene.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every frame"):
        load_scene(plane_scene)


def test_multiview_support_rejects_off_surface_points(plane_scene):
    scene = load_scene(plane_scene)
    points = np.array([[0, 0, 2], [0, 0, 1], [0, 0, 3], [100, 100, 2]], dtype=np.float32)
    np.testing.assert_array_equal(multiview_support(scene, 0, points), [True, False, False, False])


def test_initialization_excludes_holdout_and_obeys_budget(plane_scene):
    pytest.importorskip("scipy", reason="Install the splatting extra")
    # Corrupt only holdout geometry. Training initialization must not read it.
    manifest = json.loads(plane_scene.read_text())
    np.save(plane_scene.parent / manifest["frames"][-1]["points"], np.zeros((2, 2, 3)))
    scene = load_scene(plane_scene)
    config = InitializationConfig(max_points=100, points_per_frame=200)
    points = initialize_points(scene, config, seed=5)
    repeat = initialize_points(scene, config, seed=5)
    assert 4 <= len(points.means) <= 100
    np.testing.assert_allclose(points.means[:, 2], 2)
    np.testing.assert_array_equal(points.means, repeat.means)
    assert np.isfinite(points.scales).all() and (points.scales > 0).all()
    assert "clip:400" not in points.diagnostics["per_frame"]


def test_masks_and_nonfinite_geometry(plane_scene):
    manifest = json.loads(plane_scene.read_text())
    path = plane_scene.parent / "0_depth.npy"
    depth = np.load(path)
    depth[0, 0] = np.nan
    depth[0, 1] = -1
    np.save(path, depth)
    mask = np.ones_like(depth, dtype=np.uint8)
    mask[1, 1] = 0
    np.save(plane_scene.parent / "mask.npy", mask)
    manifest["frames"][0]["mask"] = "mask.npy"
    plane_scene.write_text(json.dumps(manifest))
    frame = load_scene(plane_scene).frame(0)
    assert not frame.geometry_mask[0, :2].any()
    assert not frame.image_mask[1, 1] and frame.image_mask[0, 0]


def test_resize_cache_and_voxel_fusion(plane_scene):
    scene = load_scene(plane_scene, replace(DataConfig(), max_image_side=20, cache_frames=1))
    assert scene.frame(0).rgb.shape == (16, 20, 3)
    scene.frame(1)
    assert list(scene._cache) == [1]
    points = np.array([[0, 0, 0], [0.01, 0.01, 0.01], [2, 2, 2]])
    colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    fused, rgb = fuse_voxels(points, colors, 0.1)
    np.testing.assert_allclose(fused[0], [0.005, 0.005, 0.005])
    np.testing.assert_allclose(rgb[0], [0.5, 0, 0.5])


def test_config_rejects_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="Unknown"):
        SplattingConfig.from_dict({"training": {"stepz": 10}})
    with pytest.raises(ValueError, match="holdout"):
        SplattingConfig.from_dict({"data": {"holdout_every": 1}})
    with pytest.raises(ValueError, match="Gaussian budget"):
        SplattingConfig.from_dict({"refinement": {"max_gaussians": 1}})


def test_downsampling_does_not_mix_invalid_rgb_into_valid_pixels(plane_scene):
    manifest = json.loads(plane_scene.read_text())
    mask = np.ones((32, 40), dtype=np.uint8)
    mask[0, 0] = 0
    np.save(plane_scene.parent / "mask.npy", mask)
    manifest["frames"][0]["mask"] = "mask.npy"
    plane_scene.write_text(json.dumps(manifest))
    frame = load_scene(plane_scene, replace(DataConfig(), max_image_side=20)).frame(0)
    assert not frame.image_mask[0, 0]
    assert not frame.geometry_mask[0, 0]
    assert frame.image_mask[1:, 1:].all()
