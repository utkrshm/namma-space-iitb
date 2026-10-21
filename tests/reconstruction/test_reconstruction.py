"""Exercise real CPU geometry and handoff code; never instantiate or run VGGT."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import PreprocessConfig, load_result, preprocess_capture
from src.reconstruction import (
    FrameProvenance,
    ImageTransform,
    ReconstructionResult,
    VGGTInference,
    decode_cameras,
    load_images,
    load_preprocessed_frames,
    save_point_cloud,
    to_open3d,
    unproject_depth,
)
from src.reconstruction.cli import main


@pytest.fixture(scope="module")
def preprocessed(tmp_path_factory):
    fixture = Path(__file__).parents[1] / "fixtures" / "room_short.mp4"
    config = PreprocessConfig()
    config = replace(config, revisits=replace(config.revisits, enabled=False))
    root = tmp_path_factory.mktemp("reconstruction_handoff") / "preprocessing"
    return preprocess_capture([fixture], root, config=config)


@pytest.fixture
def plane_result():
    """Analytic plane and identity camera for testing processing, not model predictions."""
    transform = ImageTransform((3, 2), (3, 2), (3, 2), (0, 0))
    intrinsics = np.eye(3, dtype=np.float32)[None]
    extrinsics = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[None]
    cameras = decode_cameras(extrinsics, intrinsics, (transform,))
    depth = np.full((1, 2, 3), 2, dtype=np.float32)
    points = unproject_depth(depth, cameras)
    colors = np.array(
        [[[[255, 0, 0], [0, 255, 0], [0, 0, 255]], [[20, 40, 60], [90, 80, 70], [120, 130, 140]]]],
        dtype=np.uint8,
    )
    return ReconstructionResult(
        images=colors,
        valid_mask=np.ones((1, 2, 3), dtype=bool),
        transforms=(transform,),
        frames=(FrameProvenance("plane:0", None),),
        cameras=cameras,
        pose_encoding=np.zeros((1, 9), dtype=np.float32),
        pose_encoding_iterations=np.zeros((2, 1, 9), dtype=np.float32),
        depth=depth,
        depth_confidence=np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3),
        world_points=points.copy(),
        world_points_confidence=np.arange(6, 0, -1, dtype=np.float32).reshape(1, 2, 3),
        depth_world_points=points,
        model_id="analytic-plane-test-data",
        precision="float32",
    )


def test_pad_resize_preserves_rgb_and_pixel_centre_transform(tmp_path):
    source = np.full((80, 40, 3), [250, 10, 30], dtype=np.uint8)
    path = tmp_path / "portrait.png"
    Image.fromarray(source).save(path)
    batch = load_images(path, image_size=56)
    assert batch.images.shape == (1, 3, 56, 56)
    assert batch.valid_mask.sum() == 56 * 28
    transform = batch.transforms[0]
    assert transform.source_size == (40, 80)
    assert transform.resized_size == (28, 56)
    assert transform.offset == (14, 0)
    np.testing.assert_allclose(batch.images[0, :, 20, 20] * 255, [250, 10, 30], atol=1e-5)
    np.testing.assert_array_equal(batch.images[0, :, 0, 0], [1, 1, 1])
    xy = np.array([[0, 0], [39, 79], [15.5, 25.5]], dtype=np.float32)
    expected = xy * 0.7 + [14 - 0.15, -0.15]
    np.testing.assert_allclose(transform.transform_points(xy), expected, atol=1e-5)
    np.testing.assert_allclose(transform.transform_points(expected, inverse=True), xy, atol=1e-5)
    assert np.array_equal(load_images([path, source], image_size=56).images[0], batch.images[0])


def test_mixed_image_shapes_and_input_validation():
    batch = load_images([np.zeros((80, 40, 3), np.uint8), np.zeros((40, 80, 3), np.uint8)])
    assert batch.images.shape == (2, 3, 518, 518)
    assert batch.transforms[0].offset[0] > 0 and batch.transforms[1].offset[1] > 0
    with pytest.raises(ValueError, match="At least one"):
        load_images([])
    with pytest.raises(ValueError, match="uint8"):
        load_images(np.zeros((40, 40, 3), dtype=float))
    with pytest.raises(ValueError, match="multiple"):
        load_images(np.zeros((40, 40, 3), np.uint8), image_size=55)


def test_camera_inversion_depth_reprojection_and_original_intrinsics():
    transform = ImageTransform((80, 40), (56, 28), (56, 56), (0, 14))
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float32)
    extrinsics = np.column_stack((rotation, [1, 2, 3]))[None]
    k = np.array([[[40, 0, 28], [0, 42, 28], [0, 0, 1]]], np.float32)
    cameras = decode_cameras(extrinsics, k, (transform,))
    homogeneous = np.eye(4)
    homogeneous[:3] = extrinsics[0]
    np.testing.assert_allclose(homogeneous @ cameras.camera_to_world[0], np.eye(4), atol=1e-6)

    depth = np.full((1, 56, 56), 2.5, np.float32)
    world = unproject_depth(depth, cameras)
    camera_xyz = world[0] @ rotation.T + extrinsics[0, :, 3]
    pixels = camera_xyz @ k[0].T
    pixels = pixels[..., :2] / pixels[..., 2:]
    y, x = np.indices((56, 56))
    np.testing.assert_allclose(pixels, np.stack((x, y), -1), atol=1e-4)
    np.testing.assert_allclose(camera_xyz[..., 2], depth[0], atol=1e-6)
    source_xyz = np.array([15, 7, 1], np.float32)
    np.testing.assert_allclose(
        transform.source_to_model @ cameras.original_intrinsics[0] @ source_xyz,
        k[0] @ source_xyz,
        atol=1e-4,
    )


def test_point_cloud_filtering_uses_matching_head_and_provenance(plane_result):
    result = plane_result
    result.valid_mask[0, 0, 0] = False  # Padding must not influence percentile calculation.
    result.depth[0, 0, 1] = 0
    result.depth_world_points[0, 0, 2] = np.nan
    result.depth_confidence[0, 1, 0] = np.inf
    cloud = result.point_cloud(confidence_percentile=0, max_points=1)
    np.testing.assert_array_equal(cloud.pixel_xy, [[1, 1]])
    np.testing.assert_array_equal(cloud.colors, [[90, 80, 70]])
    np.testing.assert_array_equal(cloud.points, [[2, 2, 2]])
    assert cloud.frame_indices.tolist() == [0]
    assert cloud.confidence.tolist() == [5]

    direct = result.point_cloud(source="point_head", confidence_percentile=0, min_confidence=5)
    np.testing.assert_array_equal(direct.pixel_xy, [[1, 0]])
    assert direct.confidence.tolist() == [5]
    assert len(result.point_cloud(min_confidence=100).points) == 0
    with pytest.raises(ValueError, match="percentile"):
        result.point_cloud(confidence_percentile=float("nan"))


def test_all_outputs_save_as_numpy_with_json_provenance(plane_result, tmp_path):
    root = plane_result.save(tmp_path / "reconstruction")
    with np.load(root / "predictions.npz", allow_pickle=False) as saved:
        assert saved["depth"].shape == (1, 2, 3)
        for name, value in plane_result.to_dict().items():
            np.testing.assert_array_equal(saved[name], value)
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["frames"][0]["frame_id"] == "plane:0"
    assert metadata["units"] == "arbitrary scene units, not metres"
    assert metadata["arrays"]["camera_to_world"] == [1, 4, 4]


def test_preprocessed_handoff_is_ordered_relocatable_and_retains_provenance(preprocessed, tmp_path):
    relocated = tmp_path / "moved"
    shutil.copytree(preprocessed.output_dir, relocated)
    result = load_result(relocated)
    shuffled = replace(result, selected=tuple(reversed(result.selected)))
    batch = load_preprocessed_frames(shuffled, stride=2, max_frames=2, image_size=56)
    expected = sorted(result.geometry, key=lambda f: int(f.frame_id.rsplit(":", 1)[1]))[::2][:2]
    assert [frame.frame_id for frame in batch.frames] == [frame.frame_id for frame in expected]
    assert batch.images.shape == (len(expected), 3, 56, 56)
    scores = {score.frame_id: score for score in result.scores}
    for frame in batch.frames:
        assert Path(frame.image_path).is_relative_to(relocated)
        assert frame.timestamp_s == scores[frame.frame_id].t_s
        assert frame.time_base == result.clips[0].time_base
        assert frame.selected_frame["rotation"]["rotation_deg"] == -90

    ids = [frame.frame_id for frame in expected]
    filtered = load_preprocessed_frames(relocated, frame_ids=list(reversed(ids)), image_size=56)
    assert [frame.frame_id for frame in filtered.frames] == ids
    appearance = load_preprocessed_frames(relocated, pool="appearance", max_frames=2, image_size=56)
    assert all(frame.selected_frame["pool"] == "appearance" for frame in appearance.frames)
    with pytest.raises(ValueError, match="absent"):
        load_preprocessed_frames(result, frame_ids=["missing:1"])
    with pytest.raises(ValueError, match="stride"):
        load_preprocessed_frames(result, stride=0)


def test_lazy_wrapper_and_query_validation_without_model_loading():
    wrapper = VGGTInference()
    assert not wrapper.is_loaded
    wrapper.unload()
    with pytest.raises(ValueError, match="CUDA"):
        VGGTInference(device="cpu")
    batch = load_images(np.zeros((80, 40, 3), np.uint8), image_size=56)
    wrapper._validate_batch(batch)
    queries = batch.transforms[0].transform_points(np.array([[10, 20]], np.float32))
    np.testing.assert_allclose(wrapper._validate_queries(queries, batch), queries)
    with pytest.raises(ValueError, match="padding"):
        wrapper._validate_queries([[0, 0]], batch)
    with pytest.raises(ValueError, match="inside"):
        wrapper._validate_queries([[56, 20]], batch)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from src import VGGTInference; "
                "assert 'torch' not in sys.modules; assert 'open3d' not in sys.modules; "
                "assert not VGGTInference().is_loaded"
            ),
        ],
        check=True,
    )


def test_open3d_rgb_ply_round_trip_without_gui_or_inference(plane_result, tmp_path):
    o3d = pytest.importorskip("open3d", reason="Open3D viewer extra is not installed")
    cloud = plane_result.point_cloud(confidence_percentile=0)
    geometry = to_open3d(cloud)
    np.testing.assert_allclose(np.asarray(geometry.colors), cloud.colors / 255)
    path = save_point_cloud(cloud, tmp_path / "scene.ply")
    restored = o3d.io.read_point_cloud(str(path))
    np.testing.assert_allclose(np.asarray(restored.points), cloud.points)
    np.testing.assert_allclose(np.asarray(restored.colors) * 255, cloud.colors)


def test_cli_help_and_rejects_invalid_requests_before_model_loading(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "--view" in capsys.readouterr().out
    for arguments in ([], ["--view", "scene.ply", "--no-view"], ["unused", "--max-frames", "0"]):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2
