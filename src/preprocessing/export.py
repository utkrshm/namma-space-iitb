"""Export upright original-resolution images with a relocatable provenance manifest."""

from dataclasses import replace

import cv2

from .contracts import write_json
from .storage import image_name, retrieve_frames


def export_frames(clips, selected, config, output_dir):
    records = []
    by_id = {}
    for record in selected:
        folder, ext = (
            ("geometry_frames", "png")
            if record.pool == "geometry"
            else ("appearance_frames", "jpg")
        )
        path = f"{folder}/{image_name(record.frame_id, ext)}"
        updated = replace(record, image_path=path)
        records.append(updated)
        by_id.setdefault(record.frame_id, []).append(updated)
    for frame in retrieve_frames(clips, by_id, config.decode, full_resolution=True):
        bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        for record in by_id[frame.frame_id]:
            path = output_dir / record.image_path
            path.parent.mkdir(parents=True, exist_ok=True)
            parameters = (
                [cv2.IMWRITE_PNG_COMPRESSION, config.export.png_compression]
                if record.pool == "geometry"
                else [cv2.IMWRITE_JPEG_QUALITY, config.export.jpeg_quality]
            )
            if not cv2.imwrite(str(path), bgr, parameters):
                raise OSError(f"Could not export {path}")
    return records


def write_manifest(output_dir, selected, clips, candidates):
    write_json(
        output_dir / "manifest.json",
        {
            "clips": [c.to_dict() for c in clips],
            "frames": [f.to_dict() for f in selected],
            "candidate_pool": candidates,
            "coordinates": {
                "origin": "top-left",
                "x": "right",
                "y": "down",
                "units": "pixels",
                "time": "seconds relative to stream start_pts",
                "analysis_resize": "x'=sx*(x+0.5)-0.5; y'=sy*(y+0.5)-0.5",
                "poses": "not estimated",
                "spatial_scale": "not estimated",
            },
        },
    )
