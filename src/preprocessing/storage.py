"""Source retrieval and relocatable artifact paths; no generated files beside inputs."""

import re
from pathlib import Path

from .decode import decode_frames


def safe_clip_id(stem):
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    if not value:
        raise ValueError("Empty clip ID")
    return value


def image_name(frame_id, extension):
    clip_id, pts = frame_id.rsplit(":", 1)
    return f"{clip_id}_{pts}.{extension}"


def retrieve_frames(clips, frame_ids, decode_config, *, full_resolution=False, rgb=True):
    """One sequential decode per needed clip, with exact PTS identity (no approximate seeks)."""
    wanted = set(frame_ids)
    for clip in clips:
        remaining = {fid for fid in wanted if fid.startswith(clip.clip_id + ":")}
        if not remaining:
            continue
        for frame in decode_frames(clip, decode_config, full_resolution=full_resolution, rgb=rgb):
            if frame.frame_id in remaining:
                remaining.remove(frame.frame_id)
                yield frame
                if not remaining:
                    break
        if remaining:
            raise ValueError(f"Missing source frames: {sorted(remaining)[:5]}")


def artifact_path(output_dir, relative):
    root = Path(output_dir).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes output directory")
    return path
