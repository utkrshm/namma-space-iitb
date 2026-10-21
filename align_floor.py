"""Export a separate Z-up, floor-aligned cloud from a saved reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.reconstruction import (
    PointCloud,
    estimate_floor_alignment,
    save_point_cloud,
    segment_floor_images,
    transform_observations,
)
from src.reconstruction.floor import FLOOR_MODEL


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reconstruction_dir", type=Path)
    parser.add_argument("--mode", choices=("rigid", "per-view"), default="rigid")
    parser.add_argument("--output-dir", type=Path, help="Default: sibling floor_aligned directory")
    parser.add_argument("--floor-masks", type=Path, help="Optional existing semantic_masks.npz")
    parser.add_argument("--device", default="cuda", help="Device for semantic segmentation only")
    args = parser.parse_args(argv)

    source = args.reconstruction_dir.resolve()
    destination = (args.output_dir or source.parent / "floor_aligned").resolve()
    if destination == source or destination.is_relative_to(source):
        parser.error("Save aligned outputs outside the original reconstruction directory")
    metadata = json.loads((source / "metadata.json").read_text())
    preview = json.loads((source / "preview.json").read_text())
    point_source = preview["source"]
    if point_source not in ("depth", "point_head"):
        parser.error("Unknown point-cloud source in preview.json")
    frame_ids = np.asarray([frame["frame_id"] for frame in metadata["frames"]])
    timestamps = np.asarray([frame["timestamp_s"] for frame in metadata["frames"]])
    clip_ids = {frame["clip_id"] for frame in metadata["frames"]}
    if len(clip_ids) != 1:
        parser.error("Floor interpolation currently requires one continuous source clip")

    with np.load(source / "predictions.npz", allow_pickle=False) as data:
        images, valid = data["images"], data["valid_mask"]
        if point_source == "depth":
            points, confidence = data["depth_world_points"], data["depth_confidence"]
        else:
            points, confidence = data["world_points"], data["world_points_confidence"]
        poses = data["camera_to_world"]
    image_hash = hashlib.sha256(images.tobytes()).hexdigest()
    if args.floor_masks is None:
        masks = segment_floor_images(images, valid, device=args.device)
    else:
        with np.load(args.floor_masks, allow_pickle=False) as data:
            masks = data["floor"]
            if "model_id" not in data or str(data["model_id"]) != FLOOR_MODEL:
                parser.error("Cached floor masks must identify the supported semantic model")
            if "frame_ids" in data and not np.array_equal(data["frame_ids"], frame_ids):
                parser.error("Cached floor masks have different source frame IDs")
            if "image_sha256" in data and str(data["image_sha256"]) != image_hash:
                parser.error("Cached floor masks have different model-grid RGB")
    if masks.shape != valid.shape or masks.dtype != bool:
        parser.error("Floor masks must be boolean and match the saved prediction grid")
    masks = masks & valid

    alignment = estimate_floor_alignment(
        points,
        masks,
        np.where(valid, confidence, np.nan),
        poses,
        timestamps,
        correct_per_view=args.mode == "per-view",
    )
    with np.load(source / "point_cloud.npz", allow_pickle=False) as data:
        cloud = PointCloud(
            transform_observations(
                data["points"], data["frame_indices"], alignment.frame_to_aligned
            ),
            data["colors"],
            data["confidence"],
            data["frame_indices"],
            data["pixel_xy"],
        )

    destination.mkdir(parents=True, exist_ok=True)
    path = save_point_cloud(cloud, destination / "point_cloud.ply")
    np.savez_compressed(destination / "point_cloud.npz", **vars(cloud))
    np.savez_compressed(
        destination / "semantic_masks.npz",
        floor=masks,
        model_id=FLOOR_MODEL,
        frame_ids=frame_ids,
        image_sha256=image_hash,
    )
    aligned_poses = alignment.frame_to_aligned @ poses
    np.savez_compressed(
        destination / "alignment.npz",
        scene_to_aligned=alignment.scene_to_aligned,
        frame_to_aligned=alignment.frame_to_aligned,
        floor_planes=alignment.floor_planes,
        reliable_frames=alignment.reliable_frames,
        camera_to_world=aligned_poses,
        world_to_camera=np.linalg.inv(aligned_poses)[:, :3],
        frame_ids=frame_ids,
    )
    report = dict(
        alignment.report,
        source_reconstruction=str(source),
        semantic_model=FLOOR_MODEL,
        source_cloud="point_cloud.npz",
        point_source=point_source,
        point_count=len(cloud.points),
        source_metadata_sha256=hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest(),
        source_cloud_sha256=hashlib.sha256((source / "point_cloud.npz").read_bytes()).hexdigest(),
        model_rgb_sha256=image_hash,
        input_floor_masks=str(args.floor_masks.resolve()) if args.floor_masks else None,
    )
    (destination / "alignment.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )

    print(f"Saved {len(cloud.points):,} points to {path}")
    print(f"Reliable floor planes: {alignment.report['reliable_floor_views']}/{len(poses)}")
    for label in ("before", "after"):
        print(
            f"Floor |height| p50/p90/p95 {label}: "
            f"{alignment.report[f'absolute_floor_residual_{label}_p50_p90_p95']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
