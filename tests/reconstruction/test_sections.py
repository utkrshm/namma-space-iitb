import json

import numpy as np
import pytest

from src.reconstruction.section_registration import Sim3, robust_sim3
from src.reconstruction.sections import (
    _absolute_quality_gate,
    _archive_incompatible_section,
    _cached_section,
    _registration_threshold,
    _sample_frame_correspondences,
    _split_overlap_frames,
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


def test_tail_sections_preserve_first_five_windows_and_split_last_window():
    sections = create_sections(161, tail_section_size=24)
    assert [section.frame_indices[0] for section in sections] == [0, 25, 50, 75, 100, 125, 137]
    assert [len(section.frame_indices) for section in sections] == [36, 36, 36, 36, 36, 24, 24]
    assert sections[5].frame_indices[-1] == 148
    assert sections[6].frame_indices[-1] == 160
    assert set().union(*(set(section.frame_indices) for section in sections)) == set(range(161))
    assert set(sections[4].frame_indices) & set(sections[5].frame_indices) == set(range(125, 136))
    assert set(sections[5].frame_indices) & set(sections[6].frame_indices) == set(range(137, 149))


def test_tail_context_extends_bridge_without_changing_first_five_sections():
    sections = create_sections(161, tail_section_size=24, tail_context_frames=11)
    assert [section.frame_indices[0] for section in sections] == [0, 25, 50, 75, 100, 114, 127, 137]
    assert [len(section.frame_indices) for section in sections] == [36, 36, 36, 36, 36, 24, 24, 24]
    assert [section.frame_indices for section in sections[:5]] == [
        section.frame_indices for section in create_sections(161)[:5]
    ]
    assert set(sections[4].frame_indices) & set(sections[5].frame_indices) == set(range(114, 136))
    assert set().union(*(set(section.frame_indices) for section in sections)) == set(range(161))


def test_tail_start_replaces_selected_normal_window_and_all_later_windows():
    sections = create_sections(161, tail_section_size=24, tail_start_section=4)
    assert [section.frame_indices[0] for section in sections] == [0, 25, 50, 75, 100, 113, 126, 137]
    assert [len(section.frame_indices) for section in sections] == [36, 36, 36, 36, 24, 24, 24, 24]
    assert [section.frame_indices for section in sections[:4]] == [
        section.frame_indices for section in create_sections(161)[:4]
    ]
    assert set(sections[3].frame_indices) & set(sections[4].frame_indices) == set(range(100, 111))
    assert set().union(*(set(section.frame_indices) for section in sections)) == set(range(161))


def test_smaller_tail_size_replaces_same_start_index_with_full_coverage():
    sections = create_sections(161, tail_section_size=18, tail_start_section=4)
    assert [section.frame_indices[0] for section in sections[4:]] == [
        100,
        107,
        114,
        121,
        128,
        135,
        142,
        143,
    ]
    assert all(len(section.frame_indices) == 18 for section in sections[4:])
    assert set().union(*(set(section.frame_indices) for section in sections)) == set(range(161))


@pytest.mark.parametrize("tail_size", [0, 11, 37])
def test_invalid_tail_section_size(tail_size):
    with pytest.raises(ValueError, match="tail_section_size"):
        create_sections(161, tail_section_size=tail_size)


def test_tail_context_requires_tail_sections():
    with pytest.raises(ValueError, match="requires tail_section_size"):
        create_sections(161, tail_context_frames=1)
    with pytest.raises(ValueError, match="tail_start_section requires"):
        create_sections(161, tail_start_section=2)


def test_overlap_splits_keep_endpoints_stress_only_and_whole_frame_validation():
    common = [f"frame-{index}" for index in range(11)]
    training, validation, stress = _split_overlap_frames(common)
    assert validation == [common[2], common[5], common[8]]
    assert stress == [common[0], common[-1]]
    assert len(training) == 6
    assert not set(training) & set(validation)
    assert not set(training) & set(stress)
    assert not set(validation) & set(stress)
    short_training, short_validation, short_stress = _split_overlap_frames(common[:5])
    assert len(short_training) == 3
    assert len(short_validation) == 2
    assert short_stress == []
    seven_training, seven_validation, seven_stress = _split_overlap_frames(common[:7])
    assert len(seven_training) == 3
    assert len(seven_validation) == 2
    assert seven_stress == [common[0], common[6]]


def test_registration_threshold_uses_target_camera_distance_units():
    assert _registration_threshold(0.51375) == pytest.approx(0.0154125)
    assert _registration_threshold(2.0) == pytest.approx(0.06)
    with pytest.raises(ValueError, match="finite and positive"):
        _registration_threshold(0)
    with pytest.raises(ValueError, match="existing normal window"):
        create_sections(161, tail_section_size=24, tail_start_section=6)


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
    frame_ids = [f"frame-{index}" for index in range(9)]
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

    for section_index, ids in enumerate((frame_ids[:8], frame_ids[1:])):
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
    for index, ids in enumerate((frame_ids[:8], frame_ids[1:])):
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

    report = run_section_alignment(root, try_icp=False, max_points=900)
    assert report["frame_count"] == len(frame_ids)
    assert (root / "point_cloud.ply").is_file()
    assert (root / "graph.json").is_file()
    with np.load(root / "point_cloud.npz") as cloud:
        np.testing.assert_array_equal(np.unique(cloud["frame_indices"]), np.arange(9))
        assert cloud["camera_centers"].shape == (9, 3)
        assert cloud["camera_rotations"].shape == (9, 3, 3)
        assert cloud["camera_translations"].shape == (9, 3, 4)
        assert np.all(cloud["depths"] > 0)
    with np.load(root / "overlap_01_00.npz") as overlap:
        assert set(overlap["stress_frame_indices"]) == {1, 7}
        assert not set(overlap["stress_frame_indices"]) & set(overlap["train_frame_indices"])
        assert not set(overlap["stress_frame_indices"]) & set(overlap["validation_frame_indices"])
    graph = json.loads((root / "graph.json").read_text())
    stress = graph["edges"][0]["endpoint_stress"]
    assert set(stress["per_frame"]) == {"frame-1", "frame-7"}
    assert "post_graph" in stress


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
    nested_frames = [{"frame_id": "f0", "selected_frame": {"source_size": (8, 6)}}]
    normalized_frames = json.loads(json.dumps(nested_frames))
    metadata["frames"] = normalized_frames
    (section / "metadata.json").write_text(json.dumps(metadata))
    assert _cached_section(section, expected, normalized_frames, transforms, mask_hash)


def test_archive_only_moves_incompatible_window_and_uses_unique_destination(tmp_path):
    output = tmp_path / "section_alignment"
    section = output / "section_05"
    section.mkdir(parents=True)
    existing_archive = output / "experiments" / "initial_36" / "section_05"
    existing_archive.mkdir(parents=True)
    (existing_archive / "preserve.txt").write_text("do not overwrite")
    (section / "metadata.json").write_text(
        json.dumps(
            {
                "section_alignment": {
                    "frame_ids": [str(i) for i in range(125, 161)],
                    "section_size": 36,
                }
            }
        )
    )
    (section / "predictions.npz").write_bytes(b"saved inference")
    expected = {"frame_ids": [str(i) for i in range(125, 149)], "section_size": 24}
    archived = _archive_incompatible_section(section, expected, output)
    assert archived == output / "experiments" / "initial_36" / "section_05_1"
    assert (archived / "predictions.npz").read_bytes() == b"saved inference"
    assert (existing_archive / "preserve.txt").read_text() == "do not overwrite"

    replacement = output / "section_05"
    replacement.mkdir()
    (replacement / "metadata.json").write_text(
        json.dumps({"section_alignment": {"frame_ids": expected["frame_ids"], "section_size": 24}})
    )
    assert _archive_incompatible_section(replacement, expected, output) is None


def test_replaced_24_frame_tail_uses_superseded_archive(tmp_path):
    output = tmp_path / "section_alignment"
    section = output / "section_06"
    section.mkdir(parents=True)
    (section / "metadata.json").write_text(
        json.dumps(
            {
                "section_alignment": {
                    "frame_ids": [str(i) for i in range(137, 161)],
                    "section_size": 24,
                }
            }
        )
    )
    (section / "predictions.npz").write_bytes(b"older tail inference")
    expected = {"frame_ids": [str(i) for i in range(127, 151)], "section_size": 24}
    archived = _archive_incompatible_section(section, expected, output)
    assert archived == output / "experiments" / "superseded_sections" / "section_06"
    assert (archived / "predictions.npz").read_bytes() == b"older tail inference"


def test_absolute_quality_gate_tolerates_one_bad_heldout_frame_not_bad_edge():
    per_frame = {
        f"frame-{i}": {"normalized_median": value} for i, value in enumerate((0.03, 0.05, 0.24))
    }
    assert _absolute_quality_gate(0.04, per_frame)
    assert not _absolute_quality_gate(0.11, per_frame)
    assert not _absolute_quality_gate(
        0.04,
        {f"frame-{i}": {"normalized_median": 0.2} for i in range(3)},
    )
