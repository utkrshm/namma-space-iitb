import json
import shutil
import subprocess
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import pairwise

import av
import cv2
import numpy as np
import pytest

from src import load_result, preprocess_capture
from src.preprocessing.config import PreprocessConfig
from src.preprocessing.contracts import ClipInfo, PreprocessResult, RotationTransform, read_json
from src.preprocessing.decode import decode_frames, rotation_transform
from src.preprocessing.graph import build_graph
from src.preprocessing.probe import probe_clip
from src.preprocessing.revisits import revisits_available, spatial_spread, verify_matches
from src.preprocessing.scoring import read_scores, score_clip, tiled_sharpness, write_scores
from src.preprocessing.selection import PoolSelector, select_clip
from src.preprocessing.tracking import KLTTracker


def test_probe_rotation_vfr_and_privacy(clip, fixture_path):
    assert clip.rotation_deg == -90
    assert clip.variable_frame_rate
    assert Fraction(clip.time_base) > 0
    assert not clip.is_hdr
    assert (clip.width, clip.height) == (640, 360)
    with av.open(str(fixture_path)) as container:
        assert not any("location" in key.lower() for key in container.metadata)
        assert not any("location" in key.lower() for key in container.streams.video[0].metadata)
    assert "location" not in clip.to_json().lower()


def test_decode_upright_monotonic_and_start_time(frames, clip):
    assert len(frames) > 250
    assert frames[0].image.shape == (640, 360)
    assert all(b.pts > a.pts for a, b in pairwise(frames))
    assert frames[0].t_s == pytest.approx(0.0)
    assert frames[0].frame_id == f"{clip.clip_id}:{frames[0].pts}"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_decode_matches_ffmpeg_color_and_rotation(clip, config, tmp_path):
    reference = tmp_path / "reference.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip.path, "-frames:v", "1", str(reference)], check=True
    )
    actual = next(decode_frames(clip, config.decode, full_resolution=True, rgb=True)).image
    expected = cv2.cvtColor(cv2.imread(str(reference)), cv2.COLOR_BGR2RGB)
    assert actual.shape == expected.shape
    error = np.abs(actual.astype(float) - expected.astype(float))
    assert error.mean() < 1.0
    assert np.percentile(error, 99) <= 2


def test_rotation_transform_maps_pixels(clip, config):
    transform = rotation_transform(clip, config.decode.long_side)
    assert RotationTransform.from_json(transform.to_json()) == transform
    assert np.array_equal(np.array(transform.matrix) @ [0, 0, 1], [359, 0, 1])
    assert np.array_equal(np.array(transform.matrix) @ [639, 359, 1], [0, 639, 1])


def test_real_blur_reduces_sharpness(frames):
    image = frames[80].image
    blurred = cv2.GaussianBlur(image, (15, 15), 4)
    assert tiled_sharpness(blurred) < tiled_sharpness(image) * 0.4


def test_sustained_real_blur_is_flagged_without_reliable_flow(frames, config, tmp_path):
    path = tmp_path / "blurred.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = 360, 640
        stream.pix_fmt = "yuv420p"
        for frame in frames[:90]:
            blur = cv2.GaussianBlur(frame.image, (61, 61), 15)
            rgb = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    rows = score_clip(probe_clip(path), config)
    # The entire clip is blurred: a local relative score near one must not hide it.
    assert all("low_detail_or_blur" in row.flags for row in rows)
    assert rows[0].flow_px is None
    assert "low_detail_or_blur" in rows[0].flags


def test_tracking_known_real_frame_and_unknown_motion(frames, config):
    measurement = KLTTracker(frames[0].image, config.tracking).step(frames[0].image)
    assert measurement.reliable
    assert measurement.overlap == pytest.approx(1)
    assert measurement.displacement_px < 0.1
    # Real frame with all texture removed by a measured crop is not a fake pipeline/model.
    blank = np.full_like(frames[0].image, int(frames[0].image.mean()))
    unknown = KLTTracker(blank, config.tracking).step(blank)
    assert not unknown.reliable
    assert unknown.displacement_px is None
    assert unknown.overlap is None


def test_csv_json_round_trips(clip, config, tmp_path):
    assert ClipInfo.from_json(clip.to_json()) == clip
    scores = score_clip(clip, config)
    path = tmp_path / "scores.csv"
    write_scores(path, scores)
    assert read_scores(path) == scores
    assert all(r.flow_px is None for r in scores if not r.tracking_reliable)
    assert not any("timing_gap_info" in r.flags for r in scores if r.dt_s < 0.066)
    with pytest.raises(ValueError):
        ClipInfo.from_json({**asdict(clip), "schema_version": 999})


def test_raising_overlap_threshold_does_not_reduce_count(clip, config):
    scores = score_clip(clip, config)
    counts = []
    for threshold in (0.45, 0.65, 0.85):
        changed = replace(
            config,
            selection=replace(
                config.selection,
                geometry_overlap=threshold,
                appearance_overlap=0.9,
                appearance=False,
            ),
        )
        selected, edges, measurements, _links = select_clip(clip, scores, changed)
        counts.append(len(selected))
        assert len({f.frame_id for f in selected}) == len(selected)
        assert all(e.inliers >= config.tracking.min_tracks for e in edges)
        assert all(e.inlier_coverage >= config.tracking.min_coverage for e in edges)
        assert all(m["reference_id"] != m["frame_id"] for m in measurements)
        assert all(edge.a != edge.b for edge in edges)
    assert counts == sorted(counts)


def test_buffered_reference_replay_and_cut_support(clip, frames, config):
    scores = score_clip(clip, config)
    lookup = {s.frame_id: s for s in scores}
    selector = PoolSelector(clip, lookup, config, "geometry", config.selection.geometry_overlap)
    replayed = False
    for frame in frames:
        old_count = len(selector.selected)
        selector.accept(frame)
        if len(selector.selected) > old_count and selector.anchor.t_s < frame.t_s:
            replayed = True
            reference = KLTTracker(selector.anchor.image, config.tracking)
            for buffered, _ in selector.buffer:
                reference.step(buffered.image)
            np.testing.assert_allclose(selector.tracker.current, reference.current, atol=0.01)
    assert replayed
    cut = frames[3]
    lookup[cut.frame_id] = replace(lookup[cut.frame_id], flags=("possible_cut",))
    selector = PoolSelector(clip, lookup, config, "geometry", config.selection.geometry_overlap)
    for frame in frames:
        selector.accept(frame)
    times = {f.frame_id: f.t_s for f in frames}
    assert any(not link["supported"] for link in selector.links)
    assert not any(times[e.a] < cut.t_s <= times[e.b] for e in selector.edges)


def test_empty_graph_does_not_invent_edges():
    assert build_graph([], [], []) == {"nodes": [], "edges": [], "components": []}


def test_spatial_spread_rejects_clustered_matches(frames, config):
    # Correspondences from the actual image, with realistic duplicate-view geometry.
    gray = frames[0].image
    points = cv2.goodFeaturesToTrack(gray, 200, 0.01, 3).reshape(-1, 2)
    coverage, area = spatial_spread(points, gray.shape)
    assert coverage > 0 and area > 0
    cluster = points[:5]
    assert verify_matches(cluster, cluster, gray.shape, gray.shape, config.revisits) is None


def test_config_round_trip_and_validation(tmp_path):
    config = PreprocessConfig.from_toml("configs/preprocessing.toml")
    assert config == PreprocessConfig()
    with pytest.raises(ValueError):
        replace(config, selection=replace(config.selection, geometry_overlap=0.95))


def test_end_to_end_and_resume(fixture_path, config, tmp_path):
    output = tmp_path / "preprocessing"
    result = preprocess_capture([fixture_path], output, config)
    assert len(result.scores) > 250
    assert result.geometry and len(result.appearance) >= len(result.geometry)
    assert PreprocessResult.from_json(result.to_json()) == result
    assert load_result(output) == result
    for record in result.selected:
        image = cv2.imread(str(output / record.image_path))
        assert image.shape[:2] == (record.height, record.width) == (640, 360)
    for path in output.glob("*.json"):
        assert json.loads(path.read_text())["schema_version"] == 1
    report = read_json(output / "report.json")
    assert report["revisits"]["status"] == "not run"
    preprocess_capture([fixture_path], output, config)
    assert all(p["resumed"] for p in read_json(output / "run.json")["passes"].values())
    # Export corruption must invalidate only export, without rerunning tracking.
    (output / result.geometry[0].image_path).write_bytes(b"interrupted export")
    preprocess_capture([fixture_path], output, config)
    run = read_json(output / "run.json")["passes"]
    assert not run["export"]["resumed"] and run["select"]["resumed"]
    changed = replace(config, export=replace(config.export, jpeg_quality=90))
    preprocess_capture([fixture_path], output, changed)
    run = read_json(output / "run.json")["passes"]
    assert not run["export"]["resumed"] and run["score"]["resumed"]


@pytest.mark.skipif(not revisits_available(), reason="optional revisits extra not installed")
def test_real_revisit_pipeline(fixture_path, tmp_path):
    config = PreprocessConfig()
    config = replace(config, revisits=replace(config.revisits, temporal_exclusion_s=1.0, top_k=1))
    result = preprocess_capture([fixture_path], tmp_path / "revisits", config)
    report = read_json(tmp_path / "revisits/report.json")
    assert report["revisits"]["status"] == "run"
    assert report["revisits"]["pairs_tested"] > 0
    assert all(
        e.inliers >= config.revisits.min_inliers for e in result.edges if e.kind == "revisit"
    )
