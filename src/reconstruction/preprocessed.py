"""Read the frozen preprocessing handoff, preserving frame order and provenance."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

from ..preprocessing import PreprocessResult, load_result
from ..preprocessing.storage import artifact_path
from .contracts import FrameProvenance, ImageBatch, ReconstructionResult
from .images import load_images
from .vggt import VGGTInference


def load_preprocessed_frames(
    output_dir: str | Path | PreprocessResult,
    *,
    pool: Literal["geometry", "appearance"] = "geometry",
    stride: int = 1,
    max_frames: int | None = None,
    frame_ids: Sequence[str] | None = None,
    image_size: int = 518,
) -> ImageBatch:
    """Load selected images in clip-manifest order, then source PTS order.

    frame_ids filters identities without reordering them. stride then max_frames
    select an explicit subset; max_frames takes the first N, not a full-scene sample.
    The geometry/appearance pools are never concatenated (appearance includes geometry).
    """
    result = output_dir if isinstance(output_dir, PreprocessResult) else load_result(output_dir)
    if pool not in ("geometry", "appearance"):
        raise ValueError("pool must be 'geometry' or 'appearance'")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")

    clips = {clip.clip_id: clip for clip in result.clips}
    clip_order = {clip_id: index for index, clip_id in enumerate(clips)}
    scores = {score.frame_id: score for score in result.scores}
    selected = [frame for frame in result.selected if frame.pool == pool]
    if frame_ids is not None:
        wanted = set(frame_ids)
        missing = wanted - {frame.frame_id for frame in selected}
        if missing:
            raise ValueError(f"Frame IDs absent from {pool} pool: {sorted(missing)}")
        selected = [frame for frame in selected if frame.frame_id in wanted]
    if len({frame.frame_id for frame in selected}) != len(selected):
        raise ValueError("Duplicate frame IDs in the selected preprocessing pool")
    if any(frame.frame_id not in scores for frame in selected):
        raise ValueError("Selected frames are missing score/timestamp records")
    selected.sort(
        key=lambda frame: (clip_order[scores[frame.frame_id].clip_id], scores[frame.frame_id].pts)
    )
    selected = selected[::stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError(f"No {pool} frames selected")

    paths = [artifact_path(result.output_dir, frame.image_path) for frame in selected]
    batch = load_images(paths, image_size=image_size)
    provenance = []
    for frame, transform, path in zip(selected, batch.transforms, paths):
        if transform.source_size != (frame.width, frame.height):
            raise ValueError(f"Exported image dimensions disagree with manifest: {frame.frame_id}")
        score = scores[frame.frame_id]
        clip = clips[score.clip_id]
        provenance.append(
            FrameProvenance(
                frame.frame_id,
                str(path),
                clip.clip_id,
                score.pts,
                score.t_s,
                clip.time_base,
                frame.to_dict(),
            )
        )
    return replace(batch, frames=tuple(provenance))


def reconstruct_preprocessed(
    output_dir: str | Path,
    model: VGGTInference | None = None,
    *,
    pool: Literal["geometry", "appearance"] = "geometry",
    stride: int = 1,
    max_frames: int | None = None,
    frame_ids: Sequence[str] | None = None,
    query_points=None,
) -> ReconstructionResult:
    """Load one frozen preprocessing directory and jointly infer the selected views."""
    model = VGGTInference() if model is None else model
    batch = load_preprocessed_frames(
        output_dir,
        pool=pool,
        stride=stride,
        max_frames=max_frames,
        frame_ids=frame_ids,
        image_size=model.image_size,
    )
    result = model.predict(batch, query_points=query_points)
    return replace(result, preprocessing_dir=str(Path(output_dir).resolve()))
