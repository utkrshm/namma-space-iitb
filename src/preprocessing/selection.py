"""Buffered keyframe selection. A new reference always replays the buffered frames."""

from collections import deque
from dataclasses import asdict, replace

from .contracts import Edge, SelectedFrame
from .decode import decode_frames, rotation_transform
from .tracking import KLTTracker

QUALITY_FLAGS = {"suspected_blur", "low_detail_or_blur", "exposure", "low_contrast"}


class PoolSelector:
    def __init__(self, clip, scores, config, pool, threshold):
        self.clip, self.scores, self.config = clip, scores, config
        self.pool, self.threshold = pool, threshold
        self.anchor, self.tracker = None, None
        self.buffer = deque()
        self.cut_times = []
        self.selected, self.edges, self.measurements, self.links = [], [], [], []

    def record(self, frame, measurement, reason):
        transform = rotation_transform(self.clip, self.config.decode.long_side)
        w, h = self.clip.width, self.clip.height
        if self.clip.rotation_deg % 180:
            w, h = h, w
        return SelectedFrame(
            frame.frame_id,
            self.pool,
            frame.frame_id,
            tuple(reason),
            measurement.overlap if measurement else None,
            measurement.initial_coverage if measurement else self.tracker.initial_coverage,
            measurement.displacement_px if measurement else None,
            "",
            w,
            h,
            transform,
        )

    def accept(self, frame):
        cfg = self.config.selection
        if self.anchor is None:
            self.anchor = frame
            self.tracker = KLTTracker(frame.image, self.config.tracking)
            self.selected.append(self.record(frame, None, ["clip_start"]))
            return
        measurement = self.tracker.step(frame.image)
        self.measurements.append(
            {
                "frame_id": frame.frame_id,
                "t_s": frame.t_s,
                "reference_id": self.anchor.frame_id,
                **asdict(measurement),
            }
        )
        self.buffer.append((frame, measurement))
        while self.buffer and self.buffer[0][0].t_s < frame.t_s - cfg.buffer_s:
            self.buffer.popleft()
        elapsed = frame.t_s - self.anchor.t_s
        reason = []
        if measurement.overlap is None or measurement.overlap < self.threshold:
            reason.append("overlap")
        if not measurement.reliable:
            reason.append("tracking_weak")
        if elapsed >= cfg.max_interval_s:
            reason.append("max_interval")
        if "possible_cut" in self.scores[frame.frame_id].flags:
            self.cut_times.append(frame.t_s)
        if self.cut_times:
            reason.append("possible_cut")
        if not reason or elapsed < cfg.min_interval_s:
            return
        candidates = [
            (f, m) for f, m in self.buffer if f.t_s - self.anchor.t_s >= cfg.min_interval_s
        ]
        if not candidates:
            return
        good = [
            (f, m)
            for f, m in candidates
            if not QUALITY_FLAGS.intersection(self.scores[f.frame_id].flags)
        ]
        supported = [
            (f, m) for f, m in good if m.reliable and m.overlap >= cfg.minimum_edge_overlap
        ]
        chosen, support = max(
            supported or good or candidates,
            key=lambda item: self.scores[item[0].frame_id].sharpness,
        )
        if not good:
            reason.append("flagged_fallback")
        # Never connect across a possible cut, even when a tracker happens to survive it.
        crossed_cut = any(t <= chosen.t_s for t in self.cut_times)
        linked = (
            support.reliable
            and support.overlap is not None
            and support.overlap >= cfg.minimum_edge_overlap
            and not crossed_cut
        )
        evidence = {
            "a": self.anchor.frame_id,
            "b": chosen.frame_id,
            "clip_id": self.clip.clip_id,
            "t_start_s": self.anchor.t_s,
            "t_end_s": chosen.t_s,
            "supported": linked,
            **asdict(support),
        }
        self.links.append(evidence)
        if linked:
            self.edges.append(
                Edge(
                    self.anchor.frame_id,
                    chosen.frame_id,
                    "sequential",
                    support.overlap,
                    support.tracks,
                    support.current_coverage,
                )
            )
        self.selected.append(self.record(chosen, support, reason))
        self.anchor = chosen
        self.cut_times = [t for t in self.cut_times if t > chosen.t_s]
        self.tracker = KLTTracker(chosen.image, self.config.tracking)
        # Do not reuse measurements from the old reference after choosing an earlier frame.
        replay = [f for f, _ in self.buffer if f.t_s > chosen.t_s]
        self.buffer.clear()
        for buffered in replay:
            self.buffer.append((buffered, self.tracker.step(buffered.image)))


def select_clip(clip, scores, config):
    lookup = {s.frame_id: s for s in scores}
    geometry = PoolSelector(clip, lookup, config, "geometry", config.selection.geometry_overlap)
    appearance = (
        PoolSelector(clip, lookup, config, "appearance", config.selection.appearance_overlap)
        if config.selection.appearance
        else None
    )
    for frame in decode_frames(clip, config.decode):
        if frame.frame_id not in lookup:
            raise ValueError("Score and selection decoding disagree")
        geometry.accept(frame)
        if appearance:
            appearance.accept(frame)
    selected = list(geometry.selected)
    if appearance:
        # Include every geometry observation in the appearance pool, without duplicate IDs/pool.
        dense = {f.frame_id: f for f in appearance.selected}
        dense.update({f.frame_id: replace(f, pool="appearance") for f in geometry.selected})
        keyframes = sorted(geometry.selected, key=lambda f: lookup[f.frame_id].t_s)
        for frame in sorted(dense.values(), key=lambda f: lookup[f.frame_id].t_s):
            key = min(
                keyframes, key=lambda k: abs(lookup[k.frame_id].t_s - lookup[frame.frame_id].t_s)
            )
            selected.append(replace(frame, keyframe_id=key.frame_id))
    return selected, geometry.edges, geometry.measurements, geometry.links
