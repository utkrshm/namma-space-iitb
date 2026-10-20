from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from src.preprocessing.decode import decode_frames
from src.preprocessing.graph import build_graph
from src.preprocessing.probe import probe_clip
from src.preprocessing.report import build_report
from src.preprocessing.revisits import match_image_pair, revisits_available
from src.preprocessing.scoring import score_clip
from src.preprocessing.selection import select_clip

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.skipif(not revisits_available(), reason="optional revisits extra not installed")
def test_real_door_positive_control(config):
    root = FIXTURES / "revisits"
    diagnostics = {}
    result = match_image_pair(
        root / "coorg_701592.png",
        root / "coorg_7302316.png",
        config.revisits,
        diagnostics=diagnostics,
    )
    assert result is not None, diagnostics
    assert result[0] >= config.revisits.min_inliers
    assert diagnostics["unambiguous_matches"] >= result[0]
    assert diagnostics["raw_matches"] == (
        diagnostics["unambiguous_matches"] + diagnostics["ambiguous_matches"]
    )


@pytest.mark.skipif(not revisits_available(), reason="optional revisits extra not installed")
def test_repeated_windows_do_not_both_connect_to_same_target(config):
    root = FIXTURES / "revisits"
    accepted = []
    for pts in (1421182, 2521548):
        diagnostics = {}
        result = match_image_pair(
            root / f"coorg_{pts}.png",
            root / "coorg_5011622.png",
            config.revisits,
            diagnostics=diagnostics,
        )
        accepted.append(result is not None)
        assert diagnostics["ambiguous_matches"] > 0
        # Both were geometric positives without the guard: reject repetition, not low count.
        assert diagnostics["inliers_before"] >= config.revisits.min_inliers
        if result is None:
            assert diagnostics["rejected_repetitive"]
    assert sum(accepted) <= 1, accepted


def test_real_splice_has_one_tracking_break_without_cross_boundary_edge(config, tmp_path):
    clips = [probe_clip(FIXTURES / "room_short.mp4"), probe_clip(FIXTURES / "coorg_ceiling.mp4")]
    sequences = [list(decode_frames(clip, config.decode, rgb=True)) for clip in clips]
    spacing = float(np.median(np.diff([f.t_s for f in sequences[0]])))
    boundary = sequences[0][-1].t_s + spacing
    path = tmp_path / "real_splice.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=60)
        stream.width, stream.height = 360, 640
        stream.pix_fmt = "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 90000)
        for offset, frames in zip((0.0, boundary), sequences):
            for frame in frames:
                image = av.VideoFrame.from_ndarray(frame.image, format="rgb24")
                image.pts = round((offset + frame.t_s) * 90000)
                image.time_base = Fraction(1, 90000)
                for packet in stream.encode(image):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    clip = probe_clip(path)
    scores = score_clip(clip, config)
    assert len(scores) == sum(map(len, sequences))
    selected, edges, _, links = select_clip(
        clip, scores, replace(config, selection=replace(config.selection, appearance=False))
    )
    times = {row.frame_id: row.t_s for row in scores}
    breaks = [link for link in links if not link["supported"]]
    assert len(breaks) == 1, breaks
    assert breaks[0]["t_start_s"] < boundary <= breaks[0]["t_end_s"]
    assert not any(times[e.a] < boundary <= times[e.b] for e in edges)
    assert any(boundary <= times[f.frame_id] for f in selected)
    report = build_report(
        [clip],
        scores,
        links,
        edges,
        build_graph(selected, edges, []),
        {"status": "not run"},
        config,
    )
    tracking_breaks = [item for item in report["gaps"] if item["kind"] == "tracking_break"]
    cuts = [row for row in scores if "possible_cut" in row.flags]
    assert len(tracking_breaks) == 1 or len(cuts) == 1
