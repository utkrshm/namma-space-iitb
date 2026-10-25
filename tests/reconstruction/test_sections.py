import json

import numpy as np
import pytest

from src.reconstruction.section_registration import Sim3, robust_sim3
from src.reconstruction.sections import (
    _cached_section,
    _sample_frame_correspondences,
    _transform_camera,
    create_sections,
    run_section_alignment,
)


def test_section_windows_cover_frames_with_expected_overlap():
    sections = create_sections(161)
    assert [section.frame_indices[0] for section in sections] == [0, 25, 50, 75, 100, 125]
    assert all(len(section.frame_indices) == 36 for section in sections)
    assert sections[-1].frame_indices[-1] == 160
    assert set().union(*(set(section.frame_indices) for section in sections)) == set(range(161))
    assert [sections[i + 1].frame_indices[0] for i in range(5)] == [25, 50, 75, 100, 125]


def test_section_windows_short_clip_and_tail_deduplication():
    assert create_sections(12, 36, 11)[0].frame_indices == tuple(range(12))
    sections = create_sections(50, 36, 11)
    assert [section.frame_indices[0] for section in sections] == [0, 14]
    assert len(sections) == 2


@pytest.mark.parametrize("args", [(0, 36, 11), (3, 0, 0), (10, 3, 3), (10, 3, -1)])
def test_invalid_section_configuration(args):
    with pytest.raises(ValueError):
        create_sections(*args)


def test_overlap_sampling_pairs_exact_model_pixels_and_keeps_frame_labels():
    height = width = 16
    points = np.zeros((1, height, width, 3), dtype=np.float32)
    points[0, ..., 0] = np.arange(width)[None, :]
    points[0, ..., 1] = np.arange(height)[:, None]
    confidence = np.ones((1, height, width), dtype=np.float32)
    prediction = {
        "depth_world_points": points,
        "depth_confidence": confidence,
        "depth": np.ones((1, height, width), dtype=np.float32),
        "valid_mask": np.ones((1, height, width), dtype=bool),
    }
    shifted = {
        **prediction,
        "depth_world_points": points + np.array([2, 3, 4], dtype=np.float32),
    }
    source_later, target_earlier, weights, pixels, frame_labels = _sample_frame_correspondences(
        prediction, shifted, 0, 0, 17, 3, 500
    )
    np.testing.assert_allclose(
        target_earlier - source_later,
        np.broadcast_to([-2, -3, -4], source_later.shape),
    )
    np.testing.assert_allclose(weights.sum(), 1)
    np.testing.assert_array_equal(target_earlier[:, 0], pixels[:, 0])
    np.testing.assert_array_equal(target_earlier[:, 1], pixels[:, 1])
    np.testing.assert_array_equal(frame_labels, 17)
    assert len(source_later) == height * width
    transform, _ = robust_sim3(source_later, target_earlier, weights=weights, seed=2)
    np.testing.assert_allclose(transform.apply(source_later), target_earlier, atol=1e-5)
    invalid_depth = {**shifted, "depth": np.zeros((1, height, width), dtype=np.float32)}
    assert _sample_frame_correspondences(prediction, invalid_depth, 0, 0, 17, 3, 500) is None


def test_camera_transform_scales_center_but_not_orientation():
    camera = np.eye(4)
    camera[:3, 3] = [1, 2, 3]
    angle = np.pi / 2
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    transform = Sim3(2, rotation, np.array([4, 5, 6]))
    camera_to_world, world_to_camera = _transform_camera(camera, transform)
    np.testing.assert_allclose(camera_to_world[:3, 3], [0, 7, 12])
    np.testing.assert_allclose(camera_to_world[:3, :3], rotation)
    np.testing.assert_allclose(world_to_camera @ camera_to_world, np.eye(4)[:3])


def test_alignment_runs_from_saved_sections_through_point_cloud_and_cameras(tmp_path):
    root = tmp_path / "section_alignment"
    root.mkdir()
    frame_ids = [f"frame-{index}" for index in range(7)]
    image_size = 16
    yy, xx = np.mgrid[:image_size, :image_size]
    base = np.stack((xx / 8, yy / 8, 3 + xx * yy / 1000), axis=-1).astype(np.float32)
    translation = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    digest = "identical-model-pixels"
    transform = {
        "source_size": [16, 16],
        "resized_size": [16, 16],
        "model_size": [16, 16],
        "offset": [0, 0],
    }

    for section_index, ids in enumerate((frame_ids[:6], frame_ids[1:])):
        section_dir = root / f"section_{section_index:02d}"
        section_dir.mkdir()
        is_earlier = section_index == 0
        points = np.stack([base + (translation if is_earlier else 0) for _ in ids])
        shape = (len(ids), image_size, image_size)
        cameras = np.broadcast_to(np.eye(4, dtype=np.float32), (len(ids), 4, 4)).copy()
        metadata = {
            "frames": [{"frame_id": fid, "timestamp_s": frame_ids.index(fid) / 30} for fid in ids],
            "transforms": [transform for _ in ids],
        }
        (section_dir / "metadata.json").write_text(json.dumps(metadata))
        np.savez_compressed(
            section_dir / "predictions.npz",
            depth_world_points=points,
            depth_confidence=np.full(shape, 10, dtype=np.float32),
            depth=np.full(shape, 3, dtype=np.float32),
            valid_mask=np.ones(shape, dtype=bool),
            images=np.full((len(ids), image_size, image_size, 3), 127, dtype=np.uint8),
            camera_to_world=cameras,
        )
    manifest_sections = []
    for index, ids in enumerate((frame_ids[:6], frame_ids[1:])):
        manifest_sections.append(
            {
                "section_index": index,
                "frame_ids": ids,
                "image_hashes": [digest] * len(ids),
                "status": "cached",
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "frame_ids": frame_ids,
                "image_size": image_size,
                "sections": manifest_sections,
            }
        )
    )

    report = run_section_alignment(root, try_icp=False, max_points=700)
    assert report["frame_count"] == len(frame_ids)
    assert (root / "point_cloud.ply").is_file()
    assert (root / "graph.json").is_file()
    with np.load(root / "point_cloud.npz") as cloud:
        np.testing.assert_array_equal(np.unique(cloud["frame_indices"]), np.arange(7))
        assert cloud["camera_centers"].shape == (7, 3)
        assert cloud["camera_rotations"].shape == (7, 3, 3)
        assert cloud["camera_translations"].shape == (7, 3, 4)
        assert np.all(cloud["depths"] > 0)


def test_section_cache_rejects_changed_timestamp_or_transform(tmp_path):
    section = tmp_path / "cached-section"
    section.mkdir()
    expected = {"section_index": 0}
    frames = [{"frame_id": "f0", "timestamp_s": 1.0}]
    transforms = [{"source_to_model": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]
    mask = np.ones((1, 4, 4), dtype=bool)
    metadata = {"section_alignment": expected, "frames": frames, "transforms": transforms}
    (section / "metadata.json").write_text(json.dumps(metadata))
    np.savez_compressed(section / "predictions.npz", valid_mask=mask)
    from src.reconstruction.sections import _array_hash

    mask_hash = [_array_hash(mask[0])]
    assert _cached_section(section, expected, frames, transforms, mask_hash)
    changed_frame = [{**frames[0], "timestamp_s": 2.0}]
    assert not _cached_section(section, expected, changed_frame, transforms, mask_hash)
    changed_transform = [{**transforms[0], "source_to_model": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}]
    assert not _cached_section(section, expected, frames, changed_transform, mask_hash)
