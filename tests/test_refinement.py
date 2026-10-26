"""Analytic multi-camera recovery and source-pixel contracts using the real optimizer."""

from dataclasses import replace

import numpy as np
import pytest

from src.refinement import (
    ObservationTracks,
    RefinementConfig,
    RefinementResult,
    RefinementScene,
    evaluate_refinement,
    optimize_multiview,
    world_observations,
)
from src.refinement.export import multiview_support, refined_depth
from src.refinement.inputs import field_values, sample_image
from src.refinement.matching import link_tracks, split_tracks


def analytic_scene():
    count, height, width = 4, 64, 64
    intrinsics = np.tile([[50.0, 0, 31.5], [0, 50.0, 31.5], [0, 0, 1]], (count, 1, 1))
    true_centers = np.array([[0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0]])
    centers = true_centers + np.array(
        [[0.0, 0, 0], [0.08, 0.02, 0.04], [-0.05, 0.04, 0], [0.1, -0.04, 0.05]]
    )
    wrong_scales = np.array([1.0, 1.15, 0.85, 1.1])
    depth = np.ones((count, height, width), np.float32) * 4 * wrong_scales[:, None, None]
    scene = RefinementScene(
        np.zeros((count, height, width, 3), np.uint8),
        depth,
        np.ones_like(depth),
        np.ones_like(depth, bool),
        intrinsics,
        np.tile(np.eye(3), (count, 1, 1)),
        centers,
        np.tile(np.eye(3), (count, 1, 1)),
        np.tile([0, 0, 63, 63], (count, 1)),
        np.zeros(count, int),
        tuple({"frame_id": str(i)} for i in range(count)),
        "analytic",
    )
    x, y = np.meshgrid(np.linspace(-1.2, 1.2, 9), np.linspace(-1.2, 1.2, 8))
    landmarks = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 4)))
    frames, uv, track_ids = [], [], []
    for track, point in enumerate(landmarks):
        for frame in range(count):
            camera = point - true_centers[frame]
            pixel = intrinsics[frame] @ camera
            frames.append(frame)
            uv.append(pixel[:2] / pixel[2])
            track_ids.append(track)
    track_ids = np.asarray(track_ids)
    tracks = ObservationTracks(
        np.asarray(frames),
        np.asarray(uv),
        track_ids,
        split_tracks(track_ids),
        np.array([[0, 1], [1, 2], [2, 3]]),
        track_ids.copy(),
    )
    return scene, tracks


def test_track_conflicts_and_whole_track_split():
    groups, conflicts = link_tracks(
        [((0, 1), (1, 2)), ((1, 2), (2, 3)), ((0, 9), (2, 3)), ((4, 1), (5, 1))]
    )
    assert conflicts == 1
    assert [(0, 1), (1, 2), (2, 3)] in groups
    tids = np.repeat(np.arange(100), 3)
    split = split_tracks(tids)
    assert set(split) == {0, 1, 2}
    assert all(len(set(split[tids == track])) == 1 for track in range(100))


def test_border_sampling_and_content_field_interpolation():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    values = sample_image(image, [[0, 0], [3, 3], [-0.01, 0], [1.5, 1.5]])
    np.testing.assert_allclose(values[[0, 1, 3]], [0, 15, 7.5])
    assert np.isnan(values[2])
    assert sample_image(image, np.empty((0, 2))).shape == (0,)
    scene, _ = analytic_scene()
    fields = np.tile([[-0.1, 0.1], [-0.1, 0.1]], (4, 1, 1))
    np.testing.assert_allclose(
        field_values(scene, 0, np.array([[0, 0], [63, 63], [31.5, 31.5]]), fields), [-0.1, 0.1, 0]
    )


def test_sampling_large_point_batch_matches_small_batch():
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    xy = np.tile([[0, 0], [7, 7], [-1, 0], [2.25, 4.5]], (20000, 1))
    expected = np.tile(sample_image(image, xy[:4]), 20000)
    np.testing.assert_allclose(sample_image(image, xy), expected, equal_nan=True)


def test_real_optimizer_recovers_heldout_multicamera_agreement():
    scene, tracks = analytic_scene()
    before = evaluate_refinement(scene, tracks)
    result = optimize_multiview(
        scene,
        tracks,
        RefinementConfig(
            device="cpu", pose_steps=250, field_steps=0, learning_rate=0.005, retrieval_top_k=0
        ),
    )
    after = evaluate_refinement(scene, tracks, result)
    assert (
        after["disagreement_normalized"]["median"]
        < before["disagreement_normalized"]["median"] * 0.3
    )
    assert after["reprojection_pixels"]["median"] < before["reprojection_pixels"]["median"] * 0.3
    np.testing.assert_array_equal(result.centers[0], scene.centers[0])
    assert result.log_scales[0] == 0
    np.testing.assert_allclose(
        result.rotations @ result.rotations.transpose(0, 2, 1),
        np.eye(3)[None].repeat(4, axis=0),
        atol=1e-10,
    )


def test_scale_invariant_metrics_and_depth_camera_contract():
    scene, tracks = analytic_scene()
    scaled = replace(scene, depth=scene.depth * 5, centers=scene.centers * 5)
    a, b = evaluate_refinement(scene, tracks), evaluate_refinement(scaled, tracks)
    assert a["disagreement_normalized"]["median"] == pytest.approx(
        b["disagreement_normalized"]["median"], rel=1e-6
    )
    fields = np.tile([[-0.1, 0.1], [-0.1, 0.1]], (4, 1, 1))
    result = RefinementResult(scene.rotations, scene.centers, np.full(4, np.log(1.1)), fields, {})
    pixels = np.array([[10, 10], [30, 40], [63, 63]])
    points = world_observations(
        scene,
        np.zeros(3, int),
        pixels,
        result.rotations,
        result.centers,
        result.log_scales,
        result.log_depth_fields,
    )
    camera_z = ((points - scene.centers[0]) @ scene.rotations[0])[:, 2]
    depth = refined_depth(scene, result, 0)
    np.testing.assert_allclose(camera_z, depth[pixels[:, 1], pixels[:, 0]], rtol=1e-6)


def test_occluded_points_are_not_free_space_contradictions():
    scene, _ = analytic_scene()
    scene = replace(scene, centers=np.zeros_like(scene.centers), depth=np.full_like(scene.depth, 4))
    result = RefinementResult(scene.rotations, scene.centers, np.zeros(4), np.zeros((4, 2, 2)), {})
    maps = scene.depth.copy()
    maps[1] = 2  # Other camera sees an occluder in front of the queried surface.
    maps[2] = 8  # Other camera predicts free space where the queried surface sits.
    _, support, contradiction = multiview_support(
        scene, result, maps, 0, np.array([[30, 30]]), [1, 2, 3], 0.04
    )
    assert support[0] == 1
    assert contradiction[0] == 1
