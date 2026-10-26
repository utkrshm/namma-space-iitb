"""Full-export consistency audit for multi-view refinement artifacts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..reconstruction.images import load_images
from .export import multiview_support
from .inputs import field_values, load_refinement_scene, world_observations
from .matching import _source_path

EXPORTS = (
    "geometry_before",
    "geometry_after",
    "unfiltered_before",
    "unfiltered_after",
    "supported_before",
    "point_cloud",
)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _check_provenance(actual_frames, actual_pixels, expected_frames, expected_pixels, name):
    _check(
        np.array_equal(actual_frames, expected_frames)
        and np.array_equal(actual_pixels, expected_pixels),
        f"{name} provenance mismatch",
    )


def _check_support(support, contradictions, minimum, name):
    _check(np.all(support >= minimum), f"Minimum support violated in {name}")
    _check(np.all(contradictions <= support), f"Contradictions exceed support in {name}")


def _read_ply(path):
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.rint(np.asarray(cloud.colors) * 255).astype(np.uint8)
    _check(
        len(points) > 0
        and points.ndim == 2
        and points.shape[1] == 3
        and colors.shape == points.shape,
        f"Invalid PLY: {path}",
    )
    return points, colors


def audit_refinement(output_dir, input_dir=None):
    """Check exported PLY/NPZ, source evidence, cameras, depths and support labels.

    Raises ValueError on any inconsistency and writes audit_report.json on success.
    """
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(input_dir or manifest["input_dir"]).resolve()
    scene = load_refinement_scene(source)
    _check(manifest["scene_fingerprint"] == scene.fingerprint, "Scene fingerprint mismatch")
    _check(manifest["frames"] == list(scene.frames), "Manifest frame provenance mismatch")
    source_images = [_source_path(scene, i, source) for i in range(scene.frame_count)]
    image_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_images]
    image_batch = load_images(source_images, image_size=scene.images.shape[1])
    recreated_rgb = np.rint(image_batch.images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    _check(np.allclose(recreated_rgb, scene.images, atol=1), "Source model-grid RGB mismatch")
    _check(
        np.allclose(
            np.stack([transform.source_to_model for transform in image_batch.transforms]),
            scene.source_to_model,
        ),
        "Recreated source image transforms mismatch",
    )

    with np.load(output / "corrections.npz", allow_pickle=False) as correction_file:
        corrections = {key: correction_file[key] for key in correction_file.files}
    rotations, centers = corrections["rotations"], corrections["centers"]
    _check(
        np.array_equal(corrections["intrinsics"], scene.intrinsics),
        "Correction intrinsics differ from source",
    )
    _check(
        np.array_equal(corrections["source_to_model"], scene.source_to_model),
        "Correction image transforms differ from source",
    )
    _check(
        np.array_equal(corrections["content_bounds"], scene.content_bounds),
        "Correction content bounds differ from source",
    )
    expected_frame_ids = np.asarray([frame["frame_id"] for frame in scene.frames])
    _check(
        np.array_equal(corrections["frame_ids"].astype(str), expected_frame_ids),
        "Correction frame IDs differ from source",
    )
    _check(
        np.array_equal(corrections["initial_rotations"], scene.rotations),
        "Initial rotations differ from source",
    )
    _check(
        np.array_equal(corrections["initial_centers"], scene.centers),
        "Initial centers differ from source",
    )
    _check(np.isfinite(rotations).all() and np.isfinite(centers).all(), "Nonfinite corrections")
    _check(
        np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-5),
        "Invalid rotations",
    )
    _check(np.allclose(np.linalg.det(rotations), 1, atol=2e-5), "Rotations are not proper")
    with np.load(output / "cameras.npz", allow_pickle=False) as cameras:
        _check(np.allclose(cameras["rotations"], rotations), "Camera rotations mismatch")
        _check(np.allclose(cameras["centers"], centers), "Camera centers mismatch")
        extrinsics = cameras["world_to_camera"]
        expected_ext = np.concatenate(
            (
                rotations.transpose(0, 2, 1),
                -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), centers)[:, :, None],
            ),
            axis=2,
        )
        _check(np.allclose(extrinsics, expected_ext, atol=2e-5), "Camera extrinsics mismatch")
        _check(
            np.allclose(
                cameras["original_intrinsics"],
                np.linalg.solve(cameras["source_to_model"], cameras["intrinsics"]),
            ),
            "Original intrinsics mismatch",
        )
        _check(
            np.allclose(cameras["source_to_model"], scene.source_to_model),
            "Image transform mismatch",
        )
        _check(
            np.array_equal(cameras["intrinsics"], scene.intrinsics),
            "Camera intrinsics differ from source",
        )
        _check(
            np.array_equal(cameras["source_to_model"], scene.source_to_model),
            "Source transform differs from source",
        )
        _check(
            np.array_equal(
                cameras["original_intrinsics"],
                np.linalg.solve(scene.source_to_model, scene.intrinsics),
            ),
            "Original intrinsics differ from source",
        )

    with np.load(output / "depths.npz", allow_pickle=False) as depths:
        depth = depths["depth"]
        y_grid, x_grid = np.indices(scene.depth.shape[1:])
        grid_pixels = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        expected_depth = np.stack(
            [
                scene.depth[i]
                * np.exp(
                    corrections["log_scales"][i]
                    + field_values(scene, i, grid_pixels, corrections["log_depth_fields"]).reshape(
                        scene.depth.shape[1:]
                    )
                )
                for i in range(scene.frame_count)
            ]
        ).astype(np.float32)
        _check(
            depth.shape == scene.depth.shape
            and np.isfinite(depth[scene.valid_mask]).all()
            and (depth[scene.valid_mask] > 0).all(),
            "Invalid depth export",
        )
        _check(
            np.allclose(
                depth[scene.valid_mask], expected_depth[scene.valid_mask], rtol=1e-6, atol=1e-6
            ),
            "Depth export differs from corrections",
        )
        _check(np.array_equal(depths["valid_mask"], scene.valid_mask), "Depth valid mask mismatch")
        _check(
            np.array_equal(
                depths["frame_ids"].astype(str), np.asarray([f["frame_id"] for f in scene.frames])
            ),
            "Depth frame IDs mismatch",
        )
    report = json.loads((output / "export_report.json").read_text())
    support_records = {int(row["frame"]): row for row in report["frames"]}
    config = manifest.get("export_config", manifest["config"])
    min_support, seed = int(report["min_support"]), int(config["seed"])
    budget = max(1, int(config["max_points"]) // scene.frame_count)
    rng = np.random.default_rng(seed)
    expected_unfiltered = []
    expected_supported_before = []
    expected_supported_after = []
    identity = SimpleNamespace(
        rotations=scene.rotations,
        centers=scene.centers,
        log_scales=np.zeros(scene.frame_count),
        log_depth_fields=np.zeros_like(corrections["log_depth_fields"]),
    )
    corrected = SimpleNamespace(
        rotations=rotations,
        centers=centers,
        log_scales=corrections["log_scales"],
        log_depth_fields=corrections["log_depth_fields"],
    )
    for frame in range(scene.frame_count):
        eligible = np.flatnonzero(scene.valid_mask[frame])
        chosen = rng.choice(eligible, min(len(eligible), 2 * budget), replace=False)
        xy = np.column_stack(
            (chosen % scene.depth.shape[2], chosen // scene.depth.shape[2])
        ).astype(np.int32)
        expected_unfiltered.append((np.full(min(len(chosen), budget), frame), xy[:budget]))
        record = support_records[frame]
        _, before_support, before_conflict = multiview_support(
            scene,
            identity,
            scene.depth,
            frame,
            xy,
            record["neighbors"],
            report["relative_z_tolerance"],
        )
        _, after_support, after_conflict = multiview_support(
            scene, corrected, depth, frame, xy, record["neighbors"], report["relative_z_tolerance"]
        )
        expected_supported_before.append(
            (
                np.full(
                    min(
                        budget,
                        int(
                            (
                                (before_support >= min_support)
                                & (before_conflict <= before_support)
                            ).sum()
                        ),
                    ),
                    frame,
                ),
                xy[
                    np.flatnonzero(
                        (before_support >= min_support) & (before_conflict <= before_support)
                    )[:budget]
                ],
            )
        )
        expected_supported_after.append(
            (
                np.full(
                    min(
                        budget,
                        int(
                            (
                                (after_support >= min_support) & (after_conflict <= after_support)
                            ).sum()
                        ),
                    ),
                    frame,
                ),
                xy[
                    np.flatnonzero(
                        (after_support >= min_support) & (after_conflict <= after_support)
                    )[:budget]
                ],
            )
        )
    support_summary = {}
    digest = hashlib.sha256()
    for name in EXPORTS:
        with np.load(output / f"{name}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        points = arrays["points"]
        frames, pixels = arrays["frame_indices"], arrays["pixel_xy"]
        _check(points.ndim == 2 and points.shape[1] == 3, f"Invalid points in {name}")
        _check(
            len(points) == len(frames) == len(pixels) and np.isfinite(points).all(),
            f"Invalid observation arrays in {name}",
        )
        _check(
            np.all((frames >= 0) & (frames < scene.frame_count)), f"Invalid frame index in {name}"
        )
        x, y = pixels.T
        _check(
            np.all((x >= 0) & (x < scene.depth.shape[2]) & (y >= 0) & (y < scene.depth.shape[1])),
            f"Invalid pixel in {name}",
        )
        _check(scene.valid_mask[frames, y, x].all(), f"Invalid source pixel in {name}")
        _check(
            np.array_equal(arrays["colors"], scene.images[frames, y, x]), f"RGB mismatch in {name}"
        )
        _check(
            np.array_equal(arrays["confidence"], scene.confidence[frames, y, x]),
            f"Confidence mismatch in {name}",
        )
        _check(
            np.array_equal(arrays["section_indices"], scene.owners[frames]),
            f"Owner mismatch in {name}",
        )
        before = name.endswith("before")
        _check(
            np.array_equal(
                arrays["frame_ids"].astype(str), np.asarray([f["frame_id"] for f in scene.frames])
            ),
            f"Frame identity metadata mismatch in {name}",
        )
        variant_rot = scene.rotations if before else rotations
        variant_centers = scene.centers if before else centers
        expected_translation = -np.einsum(
            "nij,nj->ni", variant_rot.transpose(0, 2, 1), variant_centers
        )
        _check(
            np.array_equal(arrays["camera_rotations"], variant_rot),
            f"Camera rotations mismatch in {name}",
        )
        _check(
            np.array_equal(arrays["camera_centers"], variant_centers),
            f"Camera centers mismatch in {name}",
        )
        _check(
            np.allclose(
                arrays["camera_translations"],
                np.concatenate(
                    (variant_rot.transpose(0, 2, 1), expected_translation[:, :, None]), axis=2
                ),
                atol=2e-5,
            ),
            f"Camera translations mismatch in {name}",
        )
        scales = np.zeros(scene.frame_count) if before else corrections["log_scales"]
        fields = (
            np.zeros_like(corrections["log_depth_fields"])
            if before
            else corrections["log_depth_fields"]
        )
        expected = world_observations(
            scene, frames, pixels, variant_rot, variant_centers, scales, fields
        )
        _check(
            np.allclose(points, expected, rtol=1e-6, atol=2e-5), f"Unprojection mismatch in {name}"
        )
        camera_z = np.einsum(
            "nij,nj->ni", variant_rot[frames].transpose(0, 2, 1), points - variant_centers[frames]
        )[:, 2]
        _check(
            np.all(camera_z > 0) and np.allclose(arrays["depths"], camera_z, atol=2e-5),
            f"Camera depth mismatch in {name}",
        )
        ply_points, ply_colors = _read_ply(output / f"{name}.ply")
        _check(
            np.array_equal(ply_points, points.astype(np.float64))
            and np.array_equal(ply_colors, arrays["colors"]),
            f"PLY/NPZ mismatch in {name}",
        )
        digest.update((output / f"{name}.npz").read_bytes())
        if "support_count" in arrays:
            variant_depth = scene.depth if before else depth
            counts, contradictions = [], []
            for frame in range(scene.frame_count):
                mask = frames == frame
                if mask.any():
                    record = support_records[frame]
                    _, support, conflict = multiview_support(
                        scene,
                        type(
                            "Result",
                            (),
                            {
                                "rotations": variant_rot,
                                "centers": variant_centers,
                                "log_scales": scales,
                                "log_depth_fields": fields,
                            },
                        )(),
                        variant_depth,
                        frame,
                        pixels[mask],
                        record["neighbors"],
                        report["relative_z_tolerance"],
                    )
                    counts.extend(support.tolist())
                    contradictions.extend(conflict.tolist())
            _check(
                np.array_equal(arrays["support_count"], counts),
                f"Support counts mismatch in {name}",
            )
            _check(
                np.array_equal(arrays["contradiction_count"], contradictions),
                f"Contradiction counts mismatch in {name}",
            )
            if name in {"supported_before", "point_cloud"}:
                _check_support(
                    arrays["support_count"], arrays["contradiction_count"], min_support, name
                )
            support_summary[name] = {
                "points": len(points),
                "support_min": int(min(counts, default=0)),
                "support_max": int(max(counts, default=0)),
            }
        else:
            support_summary[name] = {"points": len(points)}
    with (
        np.load(source / "point_cloud.npz", allow_pickle=False) as original,
        np.load(output / "geometry_before.npz", allow_pickle=False) as before_archive,
    ):
        _check_provenance(
            before_archive["frame_indices"],
            before_archive["pixel_xy"],
            original["frame_indices"],
            original["pixel_xy"],
            "geometry_before",
        )
    with (
        np.load(source / "point_cloud.npz", allow_pickle=False) as original,
        np.load(output / "geometry_after.npz", allow_pickle=False) as after_archive,
    ):
        _check_provenance(
            after_archive["frame_indices"],
            after_archive["pixel_xy"],
            original["frame_indices"],
            original["pixel_xy"],
            "geometry_after",
        )
    for left, right in (("unfiltered_before", "unfiltered_after"),):
        with np.load(output / f"{left}.npz") as a, np.load(output / f"{right}.npz") as b:
            _check_provenance(
                a["frame_indices"],
                a["pixel_xy"],
                b["frame_indices"],
                b["pixel_xy"],
                "Unfiltered before/after",
            )
    for name, expected in (
        ("unfiltered_before", expected_unfiltered),
        ("unfiltered_after", expected_unfiltered),
        ("supported_before", expected_supported_before),
        ("point_cloud", expected_supported_after),
    ):
        with np.load(output / f"{name}.npz", allow_pickle=False) as archive:
            expected_frames = np.concatenate([item[0] for item in expected])
            expected_pixels = np.concatenate([item[1] for item in expected])
            _check_provenance(
                archive["frame_indices"],
                archive["pixel_xy"],
                expected_frames,
                expected_pixels,
                f"Seeded sampling {name}",
            )
        _check(
            report["counts"][name] == len(expected_frames),
            f"Export report count mismatch for {name}",
        )
    _check(len(support_records) == scene.frame_count, "Export report frame records incomplete")
    audit = {
        "status": "passed",
        "scene_fingerprint": scene.fingerprint,
        "source_image_sha256": image_hashes,
        "exports": support_summary,
        "exports_sha256": digest.hexdigest(),
    }
    (output / "audit_report.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
