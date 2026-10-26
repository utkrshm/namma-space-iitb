"""Method A: VGGT-Long-style loop chunks turned into validated section-graph loop edges.

For a verified loop pair (i, j), VGGT jointly infers the frames around i and around j
(one "loop chunk"). The chunk is registered to the section owning each side using exact
shared model pixels, which is the same contract as adjacent section overlaps. Composing the
two registrations gives a direct Sim(3) between those sections. The verified image matches
between i and j independently check that the chunk co-registered both sides.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..reconstruction.section_registration import SectionEdge, Sim3, robust_sim3
from ..reconstruction.sections import (
    _absolute_quality_gate,
    _registration_threshold,
    _sample_frame_correspondences,
    _scene_scale,
    _sim3_json,
    _split_overlap_frames,
)
from .contracts import FrameMatches, LoopConfig, LoopPair
from .store import SectionCache

_CHUNK_KEYS = (
    "depth_world_points",
    "depth_confidence",
    "depth",
    "valid_mask",
    "camera_to_world",
    "intrinsics",
)


def chunk_frames(pair: LoopPair, frame_count: int, half_window: int) -> tuple[list[int], list[int]]:
    """Frames around each loop end, clipped to the sequence and kept disjoint."""
    side_a = list(
        range(max(0, pair.frame_a - half_window), min(frame_count, pair.frame_a + half_window + 1))
    )
    side_b = list(
        range(max(0, pair.frame_b - half_window), min(frame_count, pair.frame_b + half_window + 1))
    )
    side_b = [frame for frame in side_b if frame not in side_a]
    return side_a, side_b


def infer_loop_chunk(
    cache: SectionCache,
    frames: list[int],
    chunk_dir: Path,
    model,
) -> dict:
    """Run (or resume) joint VGGT inference on the listed frames; returns chunk arrays."""
    from ..reconstruction.preprocessed import reconstruct_preprocessed

    frame_ids = [cache.frame_ids[f] for f in sorted(frames)]
    settings = {
        "frame_ids": frame_ids,
        "image_size": cache.image_size,
        "model_id": model.model_id,
        "precision": model.precision,
    }
    metadata_path = chunk_dir / "metadata.json"
    if metadata_path.is_file() and (chunk_dir / "predictions.npz").is_file():
        saved = json.loads(metadata_path.read_text()).get("loop_chunk", {})
        if all(saved.get(k) == v for k, v in settings.items()):
            with np.load(chunk_dir / "predictions.npz") as archive:
                return {key: archive[key] for key in _CHUNK_KEYS} | {"frame_ids": frame_ids}
    started = time.perf_counter()
    result = reconstruct_preprocessed(cache.preprocessing_dir, model, frame_ids=frame_ids)
    if [frame.frame_id for frame in result.frames] != frame_ids:
        raise ValueError("Loop chunk inference reordered or dropped frames")
    result.save(chunk_dir)
    metadata = json.loads(metadata_path.read_text())
    metadata["loop_chunk"] = {**settings, "wall_seconds": time.perf_counter() - started}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    arrays = result.to_dict()
    return {key: arrays[key] for key in _CHUNK_KEYS} | {"frame_ids": frame_ids}


def _register_side(cache: SectionCache, chunk: dict, section: int, frames: list[int], seed: int):
    """Chunk-local -> section-local Sim(3) from identical model pixels, with held-out frames."""
    section_arrays = cache.arrays(section)
    chunk_local = {cache.index[fid]: i for i, fid in enumerate(chunk["frame_ids"])}
    common = [cache.frame_ids[f] for f in frames]
    train, validation, _ = _split_overlap_frames(common)
    parts = {"train": [], "validation": [], "stress": []}
    for fid in common:
        frame = cache.index[fid]
        split = "train" if fid in train else "validation" if fid in validation else "stress"
        sample = _sample_frame_correspondences(
            section_arrays,
            chunk,
            cache.local(section, frame),
            chunk_local[frame],
            frame,
            seed + frame,
            500,
        )
        if sample is not None:
            parts[split].append(sample)
    if len(parts["train"]) < 3 or len(parts["validation"]) < 2:
        raise ValueError("loop side has fewer than 3 usable train or 2 held-out frames")

    def combine(items):
        return tuple(np.concatenate([item[i] for item in items]) for i in range(5))

    src, dst, weights, _, labels = combine(parts["train"])
    centers = np.asarray(
        [section_arrays["camera_to_world"][cache.local(section, int(i)), :3, 3] for i in labels]
    )
    scale = _scene_scale(dst, centers)
    estimate, fit = robust_sim3(src, dst, weights, _registration_threshold(scale), seed)
    val_src, val_dst, _, _, val_labels = combine(parts["validation"])
    residual = np.linalg.norm(estimate.apply(val_src) - val_dst, axis=1) / scale
    per_frame = {
        cache.frame_ids[int(i)]: {"normalized_median": float(np.median(residual[val_labels == i]))}
        for i in np.unique(val_labels)
    }
    validation_report = {
        "normalized_median": float(np.median(residual)),
        "normalized_p90": float(np.percentile(residual, 90)),
        "per_frame": per_frame,
        "absolute_gate_passed": _absolute_quality_gate(float(np.median(residual)), per_frame),
    }
    fit.pop("residuals", None)
    return estimate, {
        "section": section,
        "frames": common,
        "training": fit,
        "validation": validation_report,
        "scene_scale": scale,
        "train_count": len(src),
    }


def cross_side_agreement(chunk: dict, cache: SectionCache, matches: FrameMatches) -> dict:
    """Do the loop-pair image matches land on the same 3D point inside the chunk?"""
    local = {cache.index[fid]: i for i, fid in enumerate(chunk["frame_ids"])}
    a, b = local[matches.frame_a], local[matches.frame_b]
    xa = np.rint(matches.xy_a).astype(int)
    xb = np.rint(matches.xy_b).astype(int)
    pa = chunk["depth_world_points"][a][xa[:, 1], xa[:, 0]]
    pb = chunk["depth_world_points"][b][xb[:, 1], xb[:, 0]]
    valid = (
        chunk["valid_mask"][a][xa[:, 1], xa[:, 0]]
        & chunk["valid_mask"][b][xb[:, 1], xb[:, 0]]
        & (chunk["depth"][a][xa[:, 1], xa[:, 0]] > 0)
        & (chunk["depth"][b][xb[:, 1], xb[:, 0]] > 0)
    )
    if valid.sum() < 8:
        return {"count": int(valid.sum()), "normalized_median": None, "passed": False}
    centers = (chunk["camera_to_world"][a][:3, 3] + chunk["camera_to_world"][b][:3, 3]) / 2
    scale = float(np.median(np.linalg.norm(np.r_[pa[valid], pb[valid]] - centers, axis=1)))
    distance = np.linalg.norm(pa[valid] - pb[valid], axis=1) / scale
    return {
        "count": int(valid.sum()),
        "normalized_median": float(np.median(distance)),
        "normalized_p90": float(np.percentile(distance, 90)),
        "scene_scale": scale,
    }


def build_loop_edges(
    cache: SectionCache,
    loops: list[LoopPair],
    matches_for,
    config: LoopConfig,
    output_dir: Path,
    *,
    device: str = "cuda:0",
    precision: str = "bfloat16",
    local_files_only: bool = True,
) -> tuple[list[SectionEdge], list[dict]]:
    """Infer loop chunks for every `role == "chunk"` pair and return accepted Sim(3) edges."""
    from ..reconstruction.vggt import VGGTInference

    model_id = cache.manifest["model_id"]
    model = VGGTInference(
        model_id,
        device=device,
        precision=precision,
        image_size=cache.image_size,
        local_files_only=local_files_only,
    )
    edges, reports = [], []
    try:
        for number, pair in enumerate(p for p in loops if p.role == "chunk"):
            side_a, side_b = chunk_frames(pair, len(cache.frame_ids), config.chunk_half_window)
            record = {
                "loop": number,
                "frame_a": pair.frame_a,
                "frame_b": pair.frame_b,
                "time_a": float(cache.timestamps[pair.frame_a]),
                "time_b": float(cache.timestamps[pair.frame_b]),
                "side_a_frames": side_a,
                "side_b_frames": side_b,
            }
            try:
                chunk = infer_loop_chunk(
                    cache, side_a + side_b, output_dir / "loop_chunks" / f"loop_{number:02d}", model
                )
                agreement = cross_side_agreement(chunk, cache, matches_for(pair))
                agreement["passed"] = (
                    agreement.get("normalized_median") is not None
                    and agreement["normalized_median"] <= config.loop_gate_normalized_median
                )
                record["cross_side_agreement"] = agreement
                transforms, sides = [], []
                for side, frames in (("a", side_a), ("b", side_b)):
                    sections = cache.sections_with([pair.frame_a if side == "a" else pair.frame_b])
                    best = None
                    for section in sections:
                        usable = [f for f in frames if f in cache.sections[section]]
                        if len(usable) < 5:
                            continue
                        try:
                            estimate, side_report = _register_side(
                                cache, chunk, section, usable, 7000 + number
                            )
                        except ValueError as error:
                            side_report, estimate = {"section": section, "error": str(error)}, None
                        sides.append({"side": side, **side_report})
                        if (
                            estimate is not None
                            and side_report["validation"]["absolute_gate_passed"]
                        ):
                            best = (section, estimate, side_report)
                            break
                    transforms.append(best)
                record["registrations"] = sides
                if any(item is None for item in transforms):
                    record["accepted"] = False
                    record["reason"] = "a loop side failed held-out registration to its section"
                elif not agreement["passed"]:
                    record["accepted"] = False
                    record["reason"] = "loop chunk did not co-register the matched loop views"
                elif transforms[0][0] == transforms[1][0]:
                    record["accepted"] = False
                    record["reason"] = "both loop sides registered to the same section"
                else:
                    (section_a, to_a, report_a), (section_b, to_b, report_b) = transforms
                    # Points in section B -> chunk -> section A.
                    relative: Sim3 = to_a.compose(to_b.inverse())
                    weight = float(
                        min(
                            report_a["training"]["inlier_ratio"] * report_a["train_count"],
                            report_b["training"]["inlier_ratio"] * report_b["train_count"],
                        )
                    )
                    edges.append(
                        SectionEdge(
                            section_b,
                            section_a,
                            relative,
                            max(weight, 1.0),
                            "loop_chunk",
                            report_a["scene_scale"],
                        )
                    )
                    record.update(
                        accepted=True,
                        source_section=section_b,
                        target_section=section_a,
                        weight=weight,
                        transform=_sim3_json(relative),
                    )
            except (ValueError, RuntimeError) as error:
                record.update(accepted=False, reason=f"{type(error).__name__}: {error}")
            reports.append(record)
            print(
                f"Loop chunk {number}: frames {pair.frame_a}<->{pair.frame_b} "
                f"accepted={record.get('accepted')} {record.get('reason', '')}",
                flush=True,
            )
    finally:
        model.unload()
    return edges, reports
