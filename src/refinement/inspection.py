"""Actual-PLY, common-framing and source-camera visual diagnostics without a GUI."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .inputs import load_refinement_scene


def _cloud(root, name):
    import open3d as o3d

    with np.load(root / f"{name}.npz", allow_pickle=False) as archive:
        points, colors = archive["points"], archive["colors"]
    ply = o3d.io.read_point_cloud(str(root / f"{name}.ply"))
    if not np.array_equal(np.asarray(ply.points), points):
        raise ValueError(f"Actual PLY XYZ differs from its provenance archive: {name}")
    if not np.array_equal(np.rint(np.asarray(ply.colors) * 255).astype(np.uint8), colors):
        raise ValueError(f"Actual PLY RGB differs from its provenance archive: {name}")
    return points.astype(np.float64), colors


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
    before = _cloud(root, "geometry_before")
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

    times = np.array(
        [
            f.get("timestamp_s") if f.get("timestamp_s") is not None else i
            for i, f in enumerate(scene.frames)
        ]
    )
    selected = {int(np.argmin(abs(times - t))) for t in np.linspace(times.min(), times.max(), 16)}
    for pair in sorted(
        report["metrics"]["test"]["before"]["pairs"], key=lambda p: p["median"], reverse=True
    )[:3]:
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
        "display_convention": "first-source-camera upright; no gravity or floor constraint",
        "overview_clipped_point_counts": clips,
        "comparison": "same-observation geometry and support selection shown separately",
    }
    (destination / "inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")
    return inspection
