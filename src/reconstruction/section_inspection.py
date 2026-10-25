"""Headless visual inspection for section-alignment exports and their baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_transform(record):
    from .section_registration import Sim3

    return Sim3(record["scale"], np.asarray(record["rotation"]), np.asarray(record["translation"]))


def _read_cloud(
    path: Path, expected: np.ndarray | None = None, expected_colors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"PLY is empty or contains nonfinite points: {path}")
    if expected is not None and not np.array_equal(points, expected):
        raise ValueError(f"PLY points do not exactly match NPZ points: {path}")
    if expected_colors is not None and not np.array_equal(
        np.rint(colors * 255).astype(np.uint8), expected_colors
    ):
        raise ValueError(f"PLY RGB does not exactly match NPZ colors: {path}")
    if not len(colors):
        colors = np.full_like(points, 0.7)
    return points, colors


def _bounded(points: np.ndarray, colors: np.ndarray, limit: int = 150_000):
    if len(points) > limit:
        indices = np.linspace(0, len(points) - 1, limit, dtype=int)
        return points[indices], colors[indices]
    return points, colors


def _plot_comparison(baseline, current, output: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = ((0, 1), (0, 2), (1, 2))
    names = ("XY", "XZ", "YZ")
    common_limits = np.percentile(np.concatenate([baseline[0], current[0]]), [0.2, 99.8], axis=0)
    figure, panels = plt.subplots(2, 3, figsize=(18, 11), facecolor="#111111")
    for row, (points, colors, label) in enumerate(
        ((*baseline, "baseline"), (*current, "sections"))
    ):
        for column, (xy, name) in enumerate(zip(axes, names, strict=True)):
            axis = panels[row, column]
            axis.set_facecolor("#111111")
            axis.scatter(points[:, xy[0]], points[:, xy[1]], c=colors, s=0.25, linewidths=0)
            axis.set_title(f"{label} {name}", color="white")
            axis.tick_params(colors="#aaaaaa")
            axis.set_xlim(common_limits[:, xy[0]])
            axis.set_ylim(common_limits[:, xy[1]])
            axis.set_aspect("equal", adjustable="box")
            for spine in axis.spines.values():
                spine.set_color("#555555")
    figure.suptitle(title, color="white")
    figure.tight_layout()
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def _render_views(points, colors, destination):
    """Orthographic software z-buffer: inspect opaque surfaces without requiring EGL."""
    import cv2

    size = 1000
    panels = []
    for azimuth, elevation in ((30, 25), (150, 25), (270, 25), (0, 90)):
        az, el = np.radians([azimuth, elevation])
        right = np.array([-np.sin(az), np.cos(az), 0.0])
        forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        up = np.cross(forward, right)
        projected = np.column_stack([points @ right, points @ up, points @ forward])
        lo, hi = np.percentile(projected[:, :2], [0.1, 99.9], axis=0)
        scale = (size - 30) / max(hi - lo)
        pixels = np.rint((projected[:, :2] - (hi + lo) / 2) * scale + size / 2).astype(int)
        valid = ((pixels >= 0) & (pixels < size)).all(1)
        indices = np.flatnonzero(valid)
        keys = pixels[valid, 1] * size + pixels[valid, 0]
        # Sort each pixel from far to near; keep the last (nearest) observation.
        order = np.lexsort((projected[valid, 2], keys))
        sorted_keys = keys[order]
        selected = indices[order[np.r_[sorted_keys[1:] != sorted_keys[:-1], True]]]
        image = np.full((size, size, 3), 22, dtype=np.uint8)
        image[size - 1 - pixels[selected, 1], pixels[selected, 0]] = np.rint(
            colors[selected] * 255
        ).astype(np.uint8)
        cv2.putText(
            image,
            f"azimuth {azimuth}, elevation {elevation}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
        )
        panels.append(image)
    sheet = np.concatenate(
        [np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0
    )
    cv2.imwrite(str(destination), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _plot_sections(points, colors, sections, destination: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = int(sections.max()) + 1
    figure, axes = plt.subplots(count, 3, figsize=(15, max(4, count * 3.5)), squeeze=False)
    figure.patch.set_facecolor("#111111")
    for section in range(count):
        selected = sections == section
        p, c = _bounded(points[selected], colors[selected], 50_000)
        for column, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            axis = axes[section, column]
            axis.set_facecolor("#111111")
            axis.scatter(p[:, a], p[:, b], c=c, s=0.4, linewidths=0)
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_title(f"section {section}: {('XY', 'XZ', 'YZ')[column]}", color="white")
            axis.tick_params(colors="#aaaaaa")
    figure.tight_layout()
    figure.savefig(destination, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)


def _plot_cameras(cloud, baseline_centers, gauge, floor_matrix, destination: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = gauge.apply(cloud["camera_centers"])
    centers = centers @ floor_matrix[:3, :3].T + floor_matrix[:3, 3]
    owners = np.asarray(cloud["owner_section_by_frame"])
    figure = plt.figure(figsize=(10, 8), facecolor="#111111")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#111111")
    palette = plt.get_cmap("tab10")
    axis.plot(*baseline_centers.T, color="#bbbbbb", linewidth=1, label="baseline")
    for section in np.unique(owners):
        selected = owners == section
        axis.plot(
            *centers[selected].T, color=palette(int(section) % 10), label=f"section {section}"
        )
        axis.scatter(*centers[selected].T, color=palette(int(section) % 10), s=12)
    axis.set_box_aspect(np.ptp(centers, axis=0) + 1e-12)
    axis.legend()
    figure.savefig(destination, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def _plot_boundaries(root: Path, graph: dict, destination: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transforms = [_load_transform(item) for item in graph["optimized_transforms"]]
    boundary_count = len(transforms) - 1
    figure, axes = plt.subplots(
        max(1, boundary_count), 2, figsize=(12, 5 * max(1, boundary_count)), squeeze=False
    )
    palette = plt.get_cmap("tab10")
    for boundary in range(1, len(transforms)):
        path = root / f"overlap_{boundary:02d}_{boundary - 1:02d}.npz"
        if not path.is_file():
            continue
        with np.load(path) as pair:
            groups = (
                ("train", "train_source", "train_target"),
                ("heldout", "validation_source", "validation_target"),
            )
            for column, (label, source_key, target_key) in enumerate(groups):
                source = transforms[boundary].apply(pair[source_key])
                target = transforms[boundary - 1].apply(pair[target_key])
                if not len(source):
                    continue
                indices = np.linspace(0, len(source) - 1, min(len(source), 300), dtype=int)
                axis = axes[boundary - 1, column]
                axis.set_facecolor("#111111")
                axis.scatter(
                    source[indices, 0],
                    source[indices, 2],
                    c=[palette(boundary % 10)],
                    s=8,
                    label="later section",
                )
                axis.scatter(
                    target[indices, 0],
                    target[indices, 2],
                    c=["#eeeeee"],
                    s=8,
                    marker="x",
                    label="earlier section",
                )
                axis.set_aspect("equal", adjustable="datalim")
                axis.set_title(f"boundary {boundary - 1}/{boundary}: {label}")
                axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _gauge_from_shared_pixels(root: Path, baseline_dir: Path, max_samples: int = 1800):
    """Fit a visualization-only Sim3 using identical source pixels in section 0 and baseline."""
    import cv2

    from .section_registration import robust_sim3

    manifest = json.loads((root / "manifest.json").read_text())
    section_meta = json.loads((root / "section_00" / "metadata.json").read_text())
    baseline_meta = json.loads((baseline_dir / "metadata.json").read_text())
    with np.load(root / "section_00" / "predictions.npz") as archive:
        new = {
            key: archive[key]
            for key in (
                "depth_world_points",
                "depth_confidence",
                "depth",
                "valid_mask",
                "source_to_model",
            )
        }
    with np.load(baseline_dir / "predictions.npz") as archive:
        old = {
            key: archive[key]
            for key in (
                "depth_world_points",
                "depth_confidence",
                "depth",
                "valid_mask",
                "source_to_model",
            )
        }
    frame_lookup = {frame["frame_id"]: index for index, frame in enumerate(baseline_meta["frames"])}
    new_lookup = {frame["frame_id"]: index for index, frame in enumerate(section_meta["frames"])}
    source_size = section_meta["transforms"][0]["source_size"]
    width, height = source_size
    stride = max(12, int(np.sqrt(width * height / 3000)))
    sx, sy = np.meshgrid(np.arange(8, width - 8, stride), np.arange(8, height - 8, stride))
    source_xy = np.column_stack((sx.ravel(), sy.ravel())).astype(np.float32)
    src_parts, dst_parts = [], []
    for frame_id in manifest["frame_ids"][:20]:
        if frame_id not in frame_lookup or frame_id not in new_lookup:
            continue
        oi, ni = frame_lookup[frame_id], new_lookup[frame_id]
        sampled = []
        for data, index in ((new, ni), (old, oi)):
            matrix = data["source_to_model"][index]
            hom = np.column_stack((source_xy, np.ones(len(source_xy)))) @ matrix.T
            xy = (hom[:, :2] / hom[:, 2:]).astype(np.float32)
            xyz = data["depth_world_points"][index]
            valid = (
                data["valid_mask"][index]
                & np.isfinite(xyz).all(-1)
                & np.isfinite(data["depth"][index])
                & (data["depth"][index] > 0)
                & np.isfinite(data["depth_confidence"][index])
            )
            if valid.any():
                valid &= data["depth_confidence"][index] >= np.percentile(
                    data["depth_confidence"][index][valid], 25
                )
            inside = (
                (xy[:, 0] >= 1)
                & (xy[:, 0] < valid.shape[1] - 2)
                & (xy[:, 1] >= 1)
                & (xy[:, 1] < valid.shape[0] - 2)
            )
            values = np.full((len(xy), 3), np.nan, dtype=np.float32)
            if inside.any():
                selected = xy[inside]
                maps_x, maps_y = selected[:, 0, None], selected[:, 1, None]
                remapped = cv2.remap(xyz.astype(np.float32), maps_x, maps_y, cv2.INTER_LINEAR)[:, 0]
                ix, iy = np.floor(selected[:, 0]).astype(int), np.floor(selected[:, 1]).astype(int)
                good = valid[iy, ix] & valid[iy + 1, ix] & valid[iy, ix + 1] & valid[iy + 1, ix + 1]
                values[np.flatnonzero(inside)[good]] = remapped[good]
            sampled.append(values)
        good = np.isfinite(sampled[0]).all(1) & np.isfinite(sampled[1]).all(1)
        if good.any():
            src_parts.append(sampled[0][good])
            dst_parts.append(sampled[1][good])
    if not src_parts:
        raise ValueError("No shared source-pixel depth correspondences for visualization gauge")
    source, target = np.concatenate(src_parts), np.concatenate(dst_parts)
    if len(source) > max_samples:
        selected = np.linspace(0, len(source) - 1, max_samples, dtype=int)
        source, target = source[selected], target[selected]
    scale = float(np.median(np.linalg.norm(target - np.median(target, axis=0), axis=1)))
    gauge, fit = robust_sim3(source, target, threshold=max(scale * 0.08, 1e-6), seed=17)
    return gauge, {
        "correspondence_count": len(source),
        "fit": fit,
        "purpose": "visualization gauge only; not ground truth",
    }


def _contact_sheet(baseline_dir: Path, output: Path):
    from PIL import Image, ImageDraw

    metadata = json.loads((baseline_dir / "metadata.json").read_text())
    frames = metadata["frames"]
    selected = sorted({0, len(frames) // 2, len(frames) - 1})
    tiles = []
    for index in selected:
        frame = frames[index]
        image = Image.open(frame["image_path"]).convert("RGB")
        image.thumbnail((480, 270))
        tiles.append((index, frame, image.copy()))
    sheet = Image.new("RGB", (500, len(tiles) * 320), "#202020")
    draw = ImageDraw.Draw(sheet)
    for row, (index, frame, image) in enumerate(tiles):
        sheet.paste(image, (10, row * 320 + 35))
        draw.text(
            (10, row * 320 + 8),
            f"frame {index}: {frame['frame_id']}  timestamp={frame.get('timestamp_s')}",
            fill="white",
        )
    sheet.save(output)


def inspect_section_cloud(output_dir: str | Path, baseline_dir: str | Path) -> dict:
    """Create actual-PLY backed headless comparison plots under ``output_dir/inspection``."""
    root, baseline_dir = Path(output_dir).resolve(), Path(baseline_dir).resolve()
    inspection = root / "inspection"
    inspection.mkdir(parents=True, exist_ok=True)
    graph = json.loads((root / "graph.json").read_text())
    with np.load(root / "point_cloud.npz") as archive:
        current = {key: archive[key] for key in archive.files}
    with np.load(baseline_dir / "point_cloud.npz") as archive:
        baseline_npz = {key: archive[key] for key in archive.files}
    with np.load(baseline_dir / "predictions.npz") as archive:
        baseline_camera_poses = archive["camera_to_world"]
    ply_new, _ = _read_cloud(root / "point_cloud.ply", current["points"], current["colors"])
    ply_old, _ = _read_cloud(
        baseline_dir / "point_cloud.ply", baseline_npz["points"], baseline_npz["colors"]
    )
    gauge, gauge_report = _gauge_from_shared_pixels(root, baseline_dir)
    floor_dir = baseline_dir.parent / "floor_aligned"
    floor_matrix = np.eye(4)
    if (floor_dir / "alignment.npz").is_file():
        with np.load(floor_dir / "alignment.npz") as archive:
            floor_matrix = archive["scene_to_aligned"]
    new_points = (gauge.apply(current["points"]) @ floor_matrix[:3, :3].T) + floor_matrix[:3, 3]
    old_points = baseline_npz["points"] @ floor_matrix[:3, :3].T + floor_matrix[:3, 3]
    new_colors = (
        current["colors"] / 255 if current["colors"].max(initial=0) > 1 else current["colors"]
    )
    old_colors = (
        baseline_npz["colors"] / 255
        if baseline_npz["colors"].max(initial=0) > 1
        else baseline_npz["colors"]
    )
    old_sample, old_rgb = _bounded(old_points, old_colors)
    new_sample, new_rgb = _bounded(new_points, new_colors)
    _plot_comparison(
        (old_sample, old_rgb),
        (new_sample, new_rgb),
        inspection / "baseline_vs_sections.png",
        "Source-pixel visualization gauge; floor-aligned display",
    )
    _render_views(old_points, old_colors, inspection / "baseline_views.jpg")
    _render_views(new_points, new_colors, inspection / "section_views.jpg")
    import matplotlib.pyplot as plt

    section_colors = plt.get_cmap("tab10")(current["section_indices"] % 10)[:, :3]
    _render_views(new_points, section_colors, inspection / "section_colored_views.jpg")
    _plot_sections(new_points, new_colors, current["section_indices"], inspection / "sections.png")
    cameras = {key: current[key] for key in ("camera_centers",)}
    cameras["owner_section_by_frame"] = json.loads((root / "alignment_report.json").read_text())[
        "owner_section_by_frame"
    ]
    baseline_centers = (
        baseline_camera_poses[:, :3, 3] @ floor_matrix[:3, :3].T + floor_matrix[:3, 3]
    )
    _plot_cameras(
        cameras,
        baseline_centers,
        gauge,
        floor_matrix,
        inspection / "camera_trajectory.png",
    )
    _plot_boundaries(root, graph, inspection / "boundary_correspondences.png")
    _contact_sheet(baseline_dir, inspection / "source_frames.png")
    bounds = {
        "baseline": np.percentile(old_points, [0.5, 99.5], axis=0).tolist(),
        "sections": np.percentile(new_points, [0.5, 99.5], axis=0).tolist(),
    }
    full_bounds = {
        "baseline": np.stack([old_points.min(0), old_points.max(0)]).tolist(),
        "sections": np.stack([new_points.min(0), new_points.max(0)]).tolist(),
    }
    report = {
        "baseline_point_count": len(ply_old),
        "section_point_count": len(ply_new),
        "ply_npz_exact": True,
        "visualization_gauge": gauge_report,
        "visualization_gauge_transform": {
            "scale": gauge.scale,
            "rotation": gauge.rotation.tolist(),
            "translation": gauge.translation.tolist(),
        },
        "floor_alignment_applied_for_display": bool((floor_dir / "alignment.npz").is_file()),
        "display_bounds_robust_0_5_99_5_percentile": bounds,
        "full_bounds": full_bounds,
        "plots": [
            "baseline_vs_sections.png",
            "sections.png",
            "camera_trajectory.png",
            "boundary_correspondences.png",
            "source_frames.png",
            "baseline_views.jpg",
            "section_views.jpg",
            "section_colored_views.jpg",
        ],
    }
    (inspection / "inspection.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
