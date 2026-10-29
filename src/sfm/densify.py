"""Add more frames to a solved SfM model by PnP registration against its 3D points.

Re-solving a capture at a higher frame rate means repeating matching, verification and
the loop checks. Instead, extra frames (e.g. 8 per second instead of 4) are matched with
LightGlue to the nearest already-registered frames. Matched keypoints that observe a 3D
point give 2D–3D correspondences, and the pose is solved with robust PnP and refined
(pycolmap). The existing cameras and points are not changed. Extra frames are undistorted
with the model's own camera (any COLMAP model), and the result is written as a new SfM
directory in the `src.sfm` hand-off layout:

    frames.json                    base frames + extra frames (``geometry`` false)
    undistorted/images/            symlinks to the base images + undistorted extra frames
    undistorted/sparse/*.txt       COLMAP text model: base and extra poses, base points
    densify.json                   per-frame registration statistics
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..preprocessing import load_result
from .contracts import SfmConfig
from .frames import decode_selected, select_frames
from .matching import extract_features, load_features


def undistortion_maps(distorted, undistorted):
    """cv2.remap maps from the undistorted PINHOLE grid to the distorted image.

    Works for any COLMAP camera model (`img_from_cam`). COLMAP places pixel centres at
    +0.5; array indices are integer-centred.
    """
    fx, fy, ux, uy = undistorted.params
    v, u = np.mgrid[: undistorted.height, : undistorted.width].astype(np.float64) + 0.5
    rays = np.stack(((u - ux) / fx, (v - uy) / fy, np.ones_like(u)), axis=-1).reshape(-1, 3)
    pixels = distorted.img_from_cam(rays, check_cheirality=False).reshape(*u.shape, 2) - 0.5
    return pixels[..., 0].astype(np.float32), pixels[..., 1].astype(np.float32)


def _pose_line(image_id: int, world_to_camera: np.ndarray, camera_id: int, name: str) -> str:
    x, y, z, w = Rotation.from_matrix(world_to_camera[:3, :3]).as_quat()
    t = world_to_camera[:3, 3]
    return f"{image_id} {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} {camera_id} {name}\n\n"


def densify(
    base_dir: str | Path,
    preprocess_dir: str | Path,
    output_dir: str | Path,
    *,
    frames_per_second: float = 8.0,
    neighbours: int = 4,
    min_inliers: int = 100,
    holdout_geometry_every: int = 0,
    holdout_exclusion_s: float = 0.0,
    config: SfmConfig | None = None,
    device: str = "cuda:0",
) -> dict:
    import pycolmap
    import torch
    from lightglue import LightGlue

    config = config or SfmConfig()
    base, output = Path(base_dir).resolve(), Path(output_dir).resolve()
    base_frames = json.loads((base / "frames.json").read_text())
    model = pycolmap.Reconstruction(base / "sparse" / "0")
    undistorted_model = pycolmap.Reconstruction(base / "undistorted" / "sparse")
    distorted_camera = next(iter(model.cameras.values()))
    pinhole = next(iter(undistorted_model.cameras.values()))

    # Extra frames: the selection at the higher rate minus the frames already solved.
    result = load_result(preprocess_dir)
    geometry = {f.frame_id for f in result.selected if f.pool == "geometry"}
    known = {f["frame_id"] for f in base_frames}
    candidates = select_frames(
        result.scores, geometry, replace(config, frames_per_second=frames_per_second)
    )
    # Held-out views of splat training (every Nth geometry frame, as in `load_sfm_scene`)
    # keep no extra frames within `holdout_exclusion_s`, so they stay unseen.
    geometry_frames = sorted(
        (f for f in base_frames if f["geometry"]), key=lambda f: f["timestamp_s"]
    )
    held_out = np.array([
        f["timestamp_s"] for k, f in enumerate(geometry_frames)
        if holdout_geometry_every and (k + 1) % holdout_geometry_every == 0
    ])  # fmt: skip

    def near_held_out(frame):
        return (
            len(held_out) and np.abs(held_out - frame["timestamp_s"]).min() <= holdout_exclusion_s
        )

    extra = [dict(f, name=f"d{k:05d}.jpg") for k, f in enumerate(
        f for f in candidates if f["frame_id"] not in known and not near_held_out(f)
    )]  # fmt: skip

    # Decode and extract features for the extra frames only.
    (output / "images").mkdir(parents=True, exist_ok=True)
    by_id = {f["frame_id"]: f for f in extra}
    for clip in result.clips:
        wanted = {f["frame_id"] for f in extra if f["clip_id"] == clip.clip_id}
        for frame_id, image in decode_selected(clip, wanted, config.decode_threads):
            path = output / "images" / by_id[frame_id]["name"]
            if not path.exists():
                cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            by_id[frame_id]["width"], by_id[frame_id]["height"] = image.shape[1], image.shape[0]
    extract_features(output, extra, config, device)
    extra_features = load_features(output, extra)
    base_features = load_features(base, base_frames)

    # Keypoint → 3D point lookup for every registered base frame.
    registered = {}
    for image in model.images.values():
        if image.has_pose:
            ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D])
            registered[image.name] = ids
    points3D = {pid: p.xyz for pid, p in model.points3D.items()}
    base_index = [i for i, f in enumerate(base_frames) if f["name"] in registered]
    base_times = np.array([base_frames[i]["timestamp_s"] for i in base_index])

    matcher = LightGlue(features="aliked").eval().to(device)

    def tensors(feature):
        return {
            "keypoints": torch.from_numpy(feature["keypoints"])[None].to(device),
            "descriptors": torch.from_numpy(feature["descriptors"].astype(np.float32))[None].to(
                device
            ),
            "image_size": torch.from_numpy(feature["image_size"])[None].to(device),
        }

    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = 2 * config.ransac_px
    stats, poses = [], {}
    with torch.inference_mode():
        for frame, feature in zip(extra, extra_features):
            nearest = np.argsort(np.abs(base_times - frame["timestamp_s"]))[:neighbours]
            correspondences = {}
            query = tensors(feature)
            for k in nearest:
                old = base_frames[base_index[k]]
                matches = matcher(
                    {"image0": query, "image1": tensors(base_features[base_index[k]])}
                )
                matches = matches["matches"][0].cpu().numpy()
                ids = registered[old["name"]][matches[:, 1]]
                for keypoint, point in zip(matches[:, 0], ids):
                    if point >= 0:
                        correspondences.setdefault(int(keypoint), int(point))
            record = {"name": frame["name"], "correspondences": len(correspondences)}
            if len(correspondences) >= min_inliers:
                keypoints = np.array(list(correspondences))
                xy = feature["keypoints"][keypoints].astype(np.float64) + 0.5
                xyz = np.array([points3D[correspondences[k]] for k in keypoints])
                solved = pycolmap.estimate_and_refine_absolute_pose(
                    xy, xyz, distorted_camera, options
                )
                if solved is not None:
                    inliers = int(np.sum(solved["inlier_mask"]))
                    record["inliers"] = inliers
                    if inliers >= min_inliers:
                        world_to_camera = np.eye(4)
                        world_to_camera[:3] = solved["cam_from_world"].matrix()
                        poses[frame["name"]] = world_to_camera
            record["registered"] = frame["name"] in poses
            stats.append(record)

    # Undistort registered extra frames; link the base images.
    images_dir = output / "undistorted" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    map_x, map_y = undistortion_maps(distorted_camera, pinhole)
    for frame in extra:
        if frame["name"] in poses:
            image = cv2.imread(str(output / "images" / frame["name"]))
            undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imwrite(
                str(images_dir / frame["name"]), undistorted, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
    for frame in base_frames:
        link = images_dir / frame["name"]
        if not link.exists():
            link.symlink_to(base / "undistorted" / "images" / frame["name"])

    # COLMAP text model: one PINHOLE camera, base + extra poses, base points (no tracks).
    sparse = output / "undistorted" / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    camera_id = pinhole.camera_id
    fx, fy, ux, uy = pinhole.params
    (sparse / "cameras.txt").write_text(
        f"{camera_id} PINHOLE {pinhole.width} {pinhole.height} {fx} {fy} {ux} {uy}\n"
    )
    lines, next_id = [], 1
    for image in undistorted_model.images.values():
        if image.has_pose:
            world_to_camera = np.eye(4)
            world_to_camera[:3] = image.cam_from_world().matrix()
            lines.append(_pose_line(next_id, world_to_camera, camera_id, image.name))
            next_id += 1
    for name, world_to_camera in poses.items():
        lines.append(_pose_line(next_id, world_to_camera, camera_id, name))
        next_id += 1
    (sparse / "images.txt").write_text("".join(lines))
    (sparse / "points3D.txt").write_text(
        "".join(
            f"{pid} {p.xyz[0]} {p.xyz[1]} {p.xyz[2]} {p.color[0]} {p.color[1]} {p.color[2]} 0\n"
            for pid, p in undistorted_model.points3D.items()
        )
    )
    frames = base_frames + [f for f in extra if f["name"] in poses]
    (output / "frames.json").write_text(json.dumps(frames, indent=1) + "\n")
    report = {
        "base_dir": str(base),
        "frames_per_second": frames_per_second,
        "holdout_geometry_every": holdout_geometry_every,
        "holdout_exclusion_s": holdout_exclusion_s,
        "extra_candidates": len(extra),
        "extra_registered": len(poses),
        "median_inliers": float(np.median([s.get("inliers", 0) for s in stats])) if stats else 0,
        "frames": stats,
    }
    (output / "densify.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Register extra video frames into an SfM model")
    parser.add_argument("base_dir", type=Path, help="Solved src.sfm directory (with sparse/0)")
    parser.add_argument("preprocess_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--holdout-every", type=int, default=0, help="Splat held-out rule")
    parser.add_argument("--holdout-exclusion-s", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    report = densify(
        args.base_dir, args.preprocess_dir, args.output_dir,
        frames_per_second=args.fps, holdout_geometry_every=args.holdout_every,
        holdout_exclusion_s=args.holdout_exclusion_s, device=args.device,
    )  # fmt: skip
    print({k: report[k] for k in ("extra_candidates", "extra_registered", "median_inliers")})
    return 0
