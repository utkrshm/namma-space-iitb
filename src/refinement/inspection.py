"""Actual-PLY, common-framing and source-camera visual diagnostics without a GUI."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .inputs import load_refinement_scene, world_observations
from .matching import load_observation_tracks
from .optimization import observation_pairs, scene_distance


def _tracks_path(root, manifest):
    effective = root / "effective_tracks.npz"
    if effective.is_file():
        return effective
    source = manifest.get("tracks_source")
    if source:
        candidate = Path(source)
        candidates = (candidate, root / candidate, root.parent / candidate)
        for path in candidates:
            if path.is_file():
                return path
    caches = sorted((root / "dense_matches").glob("tracks_*.npz"))
    if len(caches) == 1:
        return caches[0]
    fallback = root / "tracks.npz"
    return fallback if fallback.is_file() else None


def _components(tracks, frame_count):
    parent = list(range(frame_count))

    def root(frame):
        while parent[frame] != frame:
            parent[frame] = parent[parent[frame]]
            frame = parent[frame]
        return frame

    first, second = observation_pairs(tracks, 0)
    for a, b in zip(tracks.frame_indices[first], tracks.frame_indices[second], strict=True):
        parent[max(root(int(a)), root(int(b)))] = min(root(int(a)), root(int(b)))
    labels = [root(frame) for frame in range(frame_count)]
    return {
        label: [i for i, value in enumerate(labels) if value == label]
        for label in sorted(set(labels))
    }


def _track_diagnostics(
    root,
    manifest,
    scene,
    destination,
    before,
    after,
    frame_indices,
    rotations,
    centers,
    scales,
    fields,
):
    path = _tracks_path(root, manifest)
    if path is None:
        return {"tracks_source": None, "components": []}
    tracks = load_observation_tracks(path, scene)
    groups = _components(tracks, scene.frame_count)
    baseline = tracks.feature_indices >= 0
    denominator = (
        scene_distance(
            scene,
            tracks.__class__(
                tracks.frame_indices[baseline],
                tracks.pixel_xy[baseline],
                tracks.track_indices[baseline],
                tracks.split[baseline],
                tracks.pairs,
                tracks.feature_indices[baseline],
            ),
        )
        if baseline.any()
        else None
    )
    test_first, test_second = observation_pairs(tracks, 2)
    test_frames_a = tracks.frame_indices[test_first]
    test_frames_b = tracks.frame_indices[test_second]
    test_points_before = world_observations(scene, tracks.frame_indices, tracks.pixel_xy)
    test_points_after = world_observations(
        scene, tracks.frame_indices, tracks.pixel_xy, rotations, centers, scales, fields
    )
    component_data = []
    times = np.array([frame.get("timestamp_s", i) for i, frame in enumerate(scene.frames)])
    displacement = (
        np.linalg.norm(centers - scene.centers, axis=1) / denominator
        if denominator
        else np.full(scene.frame_count, np.nan)
    )
    for label, frames in groups.items():
        selected = np.isin(test_frames_a, frames) & np.isin(test_frames_b, frames)
        variants = {}
        for name, points in (("before", test_points_before), ("after", test_points_after)):
            residuals = np.linalg.norm(
                points[test_first[selected]] - points[test_second[selected]], axis=1
            )
            normalized = residuals / denominator if denominator else np.empty(0)
            variants[name] = _residual_stats(normalized)
        frame_stats = [
            {
                "frame": int(frame),
                "timestamp_s": float(times[frame]),
                "camera_displacement_normalized": float(displacement[frame]),
                "field_min": float(fields[frame].min()),
                "field_max": float(fields[frame].max()),
            }
            for frame in frames
        ]
        component_data.append(
            {
                "component": int(label),
                "frames": frames,
                "test_residuals": variants,
                "per_frame": frame_stats,
            }
        )
        point_mask = _component_point_mask(frame_indices, frames)
        _overview(
            (before[0][point_mask], before[1][point_mask]),
            (after[0][point_mask], after[1][point_mask]),
            destination / f"component_{label:03d}_geometry.jpg",
            f"component {label} source frames",
        )
    (destination / "components.json").write_text(
        json.dumps({"components": component_data}, indent=2) + "\n"
    )
    dense_pairs = sorted(
        {
            tuple(sorted((int(tracks.frame_indices[i]), int(tracks.frame_indices[j]))))
            for i, j in zip(*observation_pairs(tracks, 0), strict=True)
            if tracks.feature_indices[i] == -2 and tracks.feature_indices[j] == -2
        }
    )
    if dense_pairs:
        draw_pairs = dense_pairs
        if len(draw_pairs) > 40:
            indices = np.linspace(0, len(draw_pairs) - 1, 40, dtype=int)
            draw_pairs = [draw_pairs[i] for i in indices]
        for start in range(0, len(draw_pairs), 4):
            batch = draw_pairs[start : start + 4]
            width, height = scene.images.shape[2], scene.images.shape[1]
            sheet = Image.new("RGB", (2 * width, len(batch) * (height + 30)), "#161616")
            draw = ImageDraw.Draw(sheet)
            for row, (a, b) in enumerate(batch):
                pair_mask = (tracks.feature_indices == -2) & np.isin(tracks.frame_indices, (a, b))
                ids, counts = np.unique(tracks.track_indices[pair_mask], return_counts=True)
                ids = ids[counts == 2]
                if len(ids) > 40:
                    ids = ids[np.linspace(0, len(ids) - 1, 40, dtype=int)]
                for column, frame in enumerate((a, b)):
                    sheet.paste(
                        Image.fromarray(scene.images[frame]),
                        (column * width, row * (height + 30) + 30),
                    )
                    timestamp = scene.frames[frame].get("timestamp_s", frame)
                    draw.text(
                        (column * width + 5, row * (height + 30) + 7),
                        f"frame {frame} | {timestamp}s",
                        fill="white",
                    )
                draw.text(
                    (width + 150, row * (height + 30) + 7),
                    f"accepted pair: {len(ids)} shown / {int((counts == 2).sum())} tracks",
                    fill="white",
                )
                for track_id in ids:
                    points = []
                    for frame in (a, b):
                        match = np.flatnonzero(
                            (tracks.track_indices == track_id) & (tracks.frame_indices == frame)
                        )
                        if len(match):
                            points.append(tracks.pixel_xy[match[0]])
                    if len(points) == 2:
                        x0, y0 = points[0] + np.array([0, row * (height + 30) + 30])
                        x1, y1 = points[1] + np.array([width, row * (height + 30) + 30])
                        draw.line((x0, y0, x1, y1), fill="#00ff66", width=1)
                        draw.ellipse((x0 - 3, y0 - 3, x0 + 3, y0 + 3), fill="#ff3030")
                        draw.ellipse((x1 - 3, y1 - 3, x1 + 3, y1 + 3), fill="#ff3030")
            sheet.save(destination / f"dense_matches_{start:02d}.jpg")
    dense_report = {
        "tracks_source": str(path),
        "fixed_original_feature_scene_distance": denominator,
        "components": component_data,
        "dense_accepted_pairs": dense_pairs,
    }
    return dense_report


def _residual_stats(values):
    if not len(values):
        return {"count": 0, "median": None, "p90": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _component_point_mask(frame_indices, frames):
    return np.isin(frame_indices, frames)


def _cloud(root, name, *, include_frame_indices=False):
    import open3d as o3d

    with np.load(root / f"{name}.npz", allow_pickle=False) as archive:
        points, colors = archive["points"], archive["colors"]
        frame_indices = archive["frame_indices"] if include_frame_indices else None
    ply = o3d.io.read_point_cloud(str(root / f"{name}.ply"))
    if not np.array_equal(np.asarray(ply.points), points):
        raise ValueError(f"Actual PLY XYZ differs from its provenance archive: {name}")
    if not np.array_equal(np.rint(np.asarray(ply.colors) * 255).astype(np.uint8), colors):
        raise ValueError(f"Actual PLY RGB differs from its provenance archive: {name}")
    cloud = (points.astype(np.float64), colors)
    return (*cloud, frame_indices) if include_frame_indices else cloud


def _paint(xy, z, colors, width, height):
    image = np.full((height, width, 3), 22, np.uint8)
    valid = np.isfinite(xy).all(1) & np.isfinite(z)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    xy, z, colors = np.rint(xy[valid]).astype(int), z[valid], colors[valid]
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    if not len(xy):
        return image
    keys = xy[:, 1] * width + xy[:, 0]
    order = np.lexsort((-z, keys))
    selected = order[np.r_[keys[order][1:] != keys[order][:-1], True]]
    image[xy[selected, 1], xy[selected, 0]] = colors[selected]
    return image


def _overview(before, after, destination, title):
    # First source camera's upright display convention, not estimated gravity.
    display = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    clouds = [(p @ display.T, c) for p, c in (before, after)]
    width, height = 800, 800
    sheet = Image.new("RGB", (3 * width, 2 * (height + 36)), "#161616")
    draw = ImageDraw.Draw(sheet)
    clipped = []
    for column, (azimuth, elevation) in enumerate(((30, 20), (150, 20), (0, 90))):
        az, el = np.radians([azimuth, elevation])
        right = np.array([-np.sin(az), np.cos(az), 0])
        forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        up = np.cross(forward, right)
        projected = [np.column_stack((p @ right, p @ up, -p @ forward)) for p, _ in clouds]
        bounds = [np.percentile(p[:, :2], [0.05, 99.95], axis=0) for p in projected]
        lo = np.minimum(bounds[0][0], bounds[1][0])
        hi = np.maximum(bounds[0][1], bounds[1][1])
        scale = (width - 30) / max(hi - lo)
        for row, (p, (_, colors)) in enumerate(zip(projected, clouds, strict=True)):
            xy = (p[:, :2] - (lo + hi) / 2) * scale + width / 2
            xy[:, 1] = height - 1 - xy[:, 1]
            image = _paint(xy, p[:, 2], colors, width, height)
            sheet.paste(Image.fromarray(image), (column * width, row * (height + 36) + 36))
            draw.text(
                (column * width + 8, row * (height + 36) + 8),
                f"{title}: {('before', 'after')[row]} | az={azimuth}, el={elevation}",
                fill="white",
            )
            clipped.append(int(((xy < 0) | (xy >= width)).any(1).sum()))
    sheet.save(destination)
    return clipped


def _camera(points, colors, rotation, center, intrinsics, shape):
    camera = (points - center) @ rotation
    positive = camera[:, 2] > 1e-5
    camera, colors = camera[positive], colors[positive]
    uvz = camera @ intrinsics.T
    return _paint(uvz[:, :2] / camera[:, 2, None], camera[:, 2], colors, shape[1], shape[0])


def inspect_refinement(output_dir, *, scene=None):
    root = Path(output_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    scene = scene or load_refinement_scene(manifest["input_dir"])
    if scene.fingerprint != manifest["scene_fingerprint"]:
        raise ValueError("Inspection input has changed since refinement")
    destination = root / "inspection"
    destination.mkdir(exist_ok=True)
    report = json.loads((root / "refinement_report.json").read_text())
    with np.load(root / "corrections.npz", allow_pickle=False) as data:
        rotations, centers = data["rotations"], data["centers"]
        scales, fields = data["log_scales"], data["log_depth_fields"]
    before_points, before_colors, frame_indices = _cloud(
        root, "geometry_before", include_frame_indices=True
    )
    before = (before_points, before_colors)
    after = _cloud(root, "geometry_after")
    clips = {
        "same_observations": _overview(
            before, after, destination / "geometry_comparison.jpg", "same observations"
        )
    }
    raw_before = _cloud(root, "unfiltered_before")
    raw_after = _cloud(root, "unfiltered_after")
    clips["spatial_unfiltered"] = _overview(
        raw_before,
        raw_after,
        destination / "spatial_geometry_comparison.jpg",
        "unfiltered spatial samples",
    )
    supported_before = _cloud(root, "supported_before")
    supported = _cloud(root, "point_cloud")
    clips["supported"] = _overview(
        supported_before,
        supported,
        destination / "supported_comparison.jpg",
        "multi-view supported",
    )
    diagnostics = _track_diagnostics(
        root,
        manifest,
        scene,
        destination,
        before,
        after,
        frame_indices,
        rotations,
        centers,
        scales,
        fields,
    )

    times = np.array(
        [
            f.get("timestamp_s") if f.get("timestamp_s") is not None else i
            for i, f in enumerate(scene.frames)
        ]
    )
    selected = {int(np.argmin(abs(times - t))) for t in np.linspace(times.min(), times.max(), 16)}
    worst_pairs = []
    for variant in ("before", "after"):
        supported_pairs = [
            p for p in report["metrics"]["test"][variant]["pairs"] if p["count"] >= 20
        ]
        worst_pairs.extend(sorted(supported_pairs, key=lambda p: p["p90"], reverse=True)[:3])
    for pair in worst_pairs:
        selected.update((pair["first"], pair["second"]))
    selected = sorted(selected)
    height, width = scene.images.shape[1:3]
    for start in range(0, len(selected), 4):
        sheet = Image.new("RGB", (4 * width, 4 * (height + 36)), "#161616")
        draw = ImageDraw.Draw(sheet)
        for row, index in enumerate(selected[start : start + 4]):
            pictures = [
                scene.images[index],
                _camera(
                    *before,
                    scene.rotations[index],
                    scene.centers[index],
                    scene.intrinsics[index],
                    (height, width),
                ),
                _camera(
                    *after,
                    rotations[index],
                    centers[index],
                    scene.intrinsics[index],
                    (height, width),
                ),
                _camera(
                    *supported,
                    rotations[index],
                    centers[index],
                    scene.intrinsics[index],
                    (height, width),
                ),
            ]
            labels = ("input", "original geometry", "refined SAME pixels", "refined supported")
            for column, (picture, label) in enumerate(zip(pictures, labels, strict=True)):
                sheet.paste(Image.fromarray(picture), (column * width, row * (height + 36) + 36))
                draw.text(
                    (column * width + 6, row * (height + 36) + 8),
                    f"{times[index]:.2f}s frame {index}: {label}",
                    fill="white",
                )
        sheet.save(destination / f"camera_comparison_{start:02d}.jpg")

    # Orthogonal plots keep the same coordinate frame and limits for both variants.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="#161616")
    sample = np.random.default_rng(17).choice(
        len(before[0]), min(150_000, len(before[0])), replace=False
    )
    for column, (a, b) in enumerate(((0, 2), (0, 1), (2, 1))):
        bounds = np.percentile(
            np.concatenate((before[0][:, [a, b]], after[0][:, [a, b]])), [0.05, 99.95], axis=0
        )
        for row, (p, c) in enumerate((before, after)):
            axis = axes[row, column]
            axis.set_facecolor("#161616")
            axis.scatter(p[sample, a], p[sample, b], c=c[sample] / 255, s=0.3, linewidths=0)
            axis.set_xlim(bounds[:, 0])
            axis.set_ylim(bounds[:, 1])
            axis.set_aspect("equal")
            axis.set_title(f"{('before', 'after')[row]} axes {a}/{b}", color="white")
            axis.tick_params(colors="white")
    figure.tight_layout()
    figure.savefig(
        destination / "orthogonal_comparison.png", dpi=140, facecolor=figure.get_facecolor()
    )
    plt.close(figure)
    inspection = {
        "actual_ply_npz_exact": True,
        "camera_frames": selected,
        "well_supported_worst_pairs": worst_pairs,
        "display_convention": "first-source-camera upright; no gravity or floor constraint",
        "overview_clipped_point_counts": clips,
        "comparison": "same-observation geometry and support selection shown separately",
        "track_diagnostics": diagnostics,
    }
    (destination / "inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")
    return inspection
