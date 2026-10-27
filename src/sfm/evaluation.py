"""SfM quality report, independent comparison with VGGT cameras, and inspection plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def camera_poses(model, frames: list[dict]) -> dict[str, np.ndarray]:
    """``frame_id`` → camera-to-world 4×4 for every registered image."""
    by_name = {frame["name"]: frame["frame_id"] for frame in frames}
    poses = {}
    for image in model.images.values():
        if image.has_pose:
            world_to_camera = np.eye(4)
            world_to_camera[:3] = image.cam_from_world().matrix()
            poses[by_name[image.name]] = np.linalg.inv(world_to_camera)
    return poses


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity ``target ≈ s·R·source + t``."""
    mean_s, mean_t = source.mean(0), target.mean(0)
    a, b = source - mean_s, target - mean_t
    u, d, vt = np.linalg.svd(b.T @ a / len(source))
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(d) @ sign) / (a**2).sum(1).mean())
    return scale, rotation, mean_t - scale * rotation @ mean_s


def compare_with_vggt(poses: dict[str, np.ndarray], vggt_cameras: Path) -> dict:
    """Align VGGT camera centres to SfM with one similarity and report the residuals.

    Centre errors are divided by the SfM trajectory's RMS radius, so they are scale-free.
    Rotation errors compare camera orientations after the same alignment.
    """
    with np.load(vggt_cameras) as data:
        ids, vggt = list(data["frame_ids"]), data["camera_to_world"]
    shared = [k for k, fid in enumerate(ids) if fid in poses]
    if len(shared) < 3:
        return {"shared_frames": len(shared)}
    sfm = np.stack([poses[ids[k]] for k in shared])
    source, target = vggt[shared, :3, 3], sfm[:, :3, 3]
    scale, rotation, translation = umeyama(source, target)
    aligned = scale * source @ rotation.T + translation
    radius = float(np.sqrt(((target - target.mean(0)) ** 2).sum(1).mean()))
    centre_error = np.linalg.norm(aligned - target, axis=1) / radius
    relative = np.einsum("nji,njk->nik", sfm[:, :3, :3], rotation @ vggt[shared, :3, :3])
    angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return {
        "shared_frames": len(shared),
        "centre_error_over_radius": {
            "median": float(np.median(centre_error)),
            "p90": float(np.percentile(centre_error, 90)),
            "max": float(centre_error.max()),
        },
        "rotation_error_deg": {
            "median": float(np.median(angle)),
            "p90": float(np.percentile(angle, 90)),
            "max": float(angle.max()),
        },
        "per_frame": [
            {"frame_id": ids[k], "centre_error": float(e), "rotation_error_deg": float(a)}
            for k, e, a in zip(shared, centre_error, angle)
        ],
        "alignment": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
        },
    }


def trajectory_continuity(poses: dict[str, np.ndarray], frames: list[dict]) -> dict:
    """Breaks between time-consecutive registered frames.

    Consecutive selected frames are about 0.25 s apart, so a camera cannot move many
    typical steps or turn tens of degrees between them. Such a break means a mapper placed
    a segment wrongly, even when reprojection errors look fine.
    """
    ordered = [f for f in frames if f["frame_id"] in poses]
    centres = np.stack([poses[f["frame_id"]][:3, 3] for f in ordered])
    rotations = np.stack([poses[f["frame_id"]][:3, :3] for f in ordered])
    step = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    cosine = (np.einsum("nij,nij->n", rotations[:-1], rotations[1:]) - 1) / 2
    turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    typical = float(np.median(step))
    breaks = np.flatnonzero((step > 8 * typical) | (turn > 25))
    return {
        "median_step": typical,
        "breaks": [
            {
                "from_s": ordered[i]["timestamp_s"],
                "to_s": ordered[i + 1]["timestamp_s"],
                "step_over_median": float(step[i] / typical),
                "turn_deg": float(turn[i]),
            }
            for i in breaks
        ],
    }


def model_report(model, frames: list[dict]) -> dict:
    registered = {image.name for image in model.images.values() if image.has_pose}
    times = [f["timestamp_s"] for f in frames if f["name"] not in registered]
    camera = next(iter(model.cameras.values()))
    return {
        "frames": len(frames),
        "registered": len(registered),
        "unregistered_timestamps_s": [round(t, 2) for t in times],
        "points3D": model.num_points3D(),
        "mean_reprojection_error_px": float(model.compute_mean_reprojection_error()),
        "mean_track_length": float(model.compute_mean_track_length()),
        "mean_observations_per_image": float(model.compute_mean_observations_per_reg_image()),
        "camera": {"model": camera.model.name, "params": camera.params.tolist()},
    }


def world_up(poses: dict[str, np.ndarray]) -> np.ndarray:
    """Phones are held upright: minus the mean image-down axis approximates world up."""
    up = -np.mean([pose[:3, 1] for pose in poses.values()], axis=0)
    return up / np.linalg.norm(up)


def plan_axes(up: np.ndarray) -> np.ndarray:
    first = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(first) < 1e-3:
        first = np.cross(up, [0.0, 0.0, 1.0])
    first /= np.linalg.norm(first)
    return np.stack((first, np.cross(up, first)))


def render_inspection(model, frames, poses, vggt: dict | None, vggt_cameras, path: Path):
    """Top-down plot: sparse points in RGB, SfM trajectory by time, aligned VGGT centres."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    up = world_up(poses)
    axes = plan_axes(up)
    times = {f["frame_id"]: f["timestamp_s"] for f in frames}
    ids = sorted(poses, key=times.get)
    centres = np.stack([poses[i][:3, 3] for i in ids])
    xyz = np.array([p.xyz for p in model.points3D.values()])
    rgb = np.array([p.color for p in model.points3D.values()]) / 255.0
    heights = (xyz - centres.mean(0)) @ up
    band = np.abs(heights - np.median(heights)) < 2 * np.std(centres @ up) + 1e-9

    figure, panels = plt.subplots(1, 2, figsize=(20, 10))
    plane = xyz @ axes.T
    panels[0].scatter(plane[:, 0], plane[:, 1], s=0.2, c=rgb)
    panels[0].set_title(f"Sparse points ({len(xyz)}) and cameras")
    panels[1].scatter(plane[band, 0], plane[band, 1], s=0.1, c="0.75")
    track = centres @ axes.T
    for panel in panels:
        points = panel.scatter(
            track[:, 0], track[:, 1], s=4, c=[times[i] for i in ids], cmap="viridis"
        )
        panel.set_aspect("equal")
    figure.colorbar(points, ax=panels[1], label="source time (s)")
    if vggt and "alignment" in vggt:
        with np.load(vggt_cameras) as data:
            source = data["camera_to_world"][:, :3, 3]
        alignment = vggt["alignment"]
        aligned = alignment["scale"] * source @ np.array(alignment["rotation"]).T + np.array(
            alignment["translation"]
        )
        flat = aligned @ axes.T
        panels[1].plot(flat[:, 0], flat[:, 1], "r-", lw=0.8, label="VGGT cameras (aligned)")
        panels[1].legend()
    panels[1].set_title("Trajectory: SfM (dots, by time) vs VGGT")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)
