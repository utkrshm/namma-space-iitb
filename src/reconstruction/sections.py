"""Overlapping VGGT section inference and reconstruction exports."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessed import load_preprocessed_frames, reconstruct_preprocessed
from .vggt import VGGTInference


@dataclass(frozen=True)
class SectionSpec:
    """A section index and its frame indices in the ordered source pool."""

    index: int
    frame_indices: tuple[int, ...]


def create_sections(
    frame_count: int, section_size: int = 36, overlap: int = 11
) -> list[SectionSpec]:
    """Create full windows, anchoring the last window at the end and deduplicating."""
    if frame_count < 1 or section_size < 1 or not 0 <= overlap < section_size:
        raise ValueError(
            "Require positive frame_count/section_size and 0 <= overlap < section_size"
        )
    if frame_count <= section_size:
        starts = [0]
    else:
        step = section_size - overlap
        starts = list(range(0, frame_count - section_size + 1, step))
        starts.append(frame_count - section_size)
        starts = list(dict.fromkeys(starts))
    return [
        SectionSpec(i, tuple(range(start, min(start + section_size, frame_count))))
        for i, start in enumerate(starts)
    ]


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _cached_section(
    path: Path, expected: dict, frames: list, transforms: list, valid_hashes: list[str]
) -> bool:
    metadata = path / "metadata.json"
    predictions = path / "predictions.npz"
    if not metadata.is_file() or not predictions.is_file():
        return False
    try:
        metadata_value = json.loads(metadata.read_text())
        actual = metadata_value.get("section_alignment")
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(actual, dict) or not all(
        actual.get(key) == value for key, value in expected.items()
    ):
        return False
    if metadata_value.get("frames") != frames or metadata_value.get("transforms") != transforms:
        return False
    try:
        with np.load(predictions) as archive:
            masks = archive["valid_mask"]
            return len(masks) == len(valid_hashes) and all(
                _array_hash(masks[i]) == digest for i, digest in enumerate(valid_hashes)
            )
    except (OSError, KeyError, ValueError):
        return False


def run_section_inference(
    preprocessing_dir: str | Path,
    output_dir: str | Path,
    *,
    section_size: int = 36,
    overlap: int = 11,
    image_size: int = 518,
    device: str = "cuda:0",
    precision: str = "bfloat16",
    resume: bool = True,
    local_files_only: bool = False,
    model_id: str = "facebook/VGGT-1B",
    save_ply: bool = True,
) -> dict:
    """Infer ordered overlapping windows, saving a standard result per section.

    Resume validates source pixel hashes and all model/window settings before reuse.
    CUDA OOM is allowed to propagate without changing section size or frame coverage.
    """
    from .visualization import save_point_cloud

    source = Path(preprocessing_dir).resolve()
    output = Path(output_dir).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Section outputs must be outside frozen preprocessing")
    # Also prevent writing into the original reconstruction output tree.
    if output.name != "section_alignment":
        raise ValueError("Section outputs must use a dedicated section_alignment directory")

    batch = load_preprocessed_frames(source, image_size=image_size)
    frame_ids = [frame.frame_id for frame in batch.frames]
    provenance = [frame.__dict__ for frame in batch.frames]
    transforms = json.loads(json.dumps([transform.__dict__ for transform in batch.transforms]))
    sections = create_sections(len(frame_ids), section_size, overlap)
    image_hashes = [
        hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in batch.images
    ]
    valid_mask_hashes = [_array_hash(mask) for mask in batch.valid_mask]
    del batch
    output.mkdir(parents=True, exist_ok=True)
    model = VGGTInference(
        model_id,
        device=device,
        precision=precision,
        image_size=image_size,
        local_files_only=local_files_only,
    )
    manifest = {
        "schema_version": 1,
        "cache_contract_version": 2,
        "preprocessing_dir": str(source),
        "frame_ids": frame_ids,
        "image_hashes": image_hashes,
        "frame_provenance": provenance,
        "transforms": transforms,
        "section_size": section_size,
        "overlap": overlap,
        "image_size": image_size,
        "model_id": model_id,
        "device": device,
        "precision": precision,
        "local_files_only": local_files_only,
        "sections": [],
    }
    try:
        for section in sections:
            frame_indices = section.frame_indices
            section_dir = output / f"section_{section.index:02d}"
            settings = {
                "frame_ids": [frame_ids[i] for i in frame_indices],
                "image_hashes": [image_hashes[i] for i in frame_indices],
                "section_index": section.index,
                "section_size": section_size,
                "overlap": overlap,
                "image_size": image_size,
                "model_id": model_id,
                "device": device,
                "precision": precision,
            }
            selected_frames = [provenance[i] for i in frame_indices]
            selected_transforms = [transforms[i] for i in frame_indices]
            selected_valid_hashes = [valid_mask_hashes[i] for i in frame_indices]
            if resume and _cached_section(
                section_dir, settings, selected_frames, selected_transforms, selected_valid_hashes
            ):
                manifest["sections"].append({**settings, "status": "cached"})
                continue
            import torch

            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            ids = [frame_ids[i] for i in frame_indices]
            reconstruction = reconstruct_preprocessed(source, model, frame_ids=ids)
            with torch.cuda.device(device):
                torch.cuda.synchronize(device)
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
            elapsed = time.perf_counter() - started
            reconstruction.save(section_dir)
            metadata_path = section_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["section_alignment"] = settings
            metadata["section_alignment"].update(
                {
                    "wall_seconds": elapsed,
                    "cuda_peak_allocated_bytes": allocated,
                    "cuda_peak_reserved_bytes": reserved,
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            if save_ply:
                cloud = reconstruction.point_cloud(max_points=1_000_000)
                save_point_cloud(cloud, section_dir / "point_cloud.ply")
            manifest["sections"].append(
                {
                    **settings,
                    "status": "inferred",
                    "wall_seconds": elapsed,
                    "cuda_peak_allocated_bytes": allocated,
                    "cuda_peak_reserved_bytes": reserved,
                }
            )
            del reconstruction
    finally:
        model.unload()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def _load_metadata(section_dir: Path) -> dict:
    return json.loads((section_dir / "metadata.json").read_text())


def _sample_frame_correspondences(first, second, frame_a, frame_b, frame_id, seed, cap):
    """Match predictions on identical model pixels, balanced over a 4x4 grid."""
    a = first["depth_world_points"][frame_a]
    b = second["depth_world_points"][frame_b]
    ca = first["depth_confidence"][frame_a]
    cb = second["depth_confidence"][frame_b]
    da, db = first["depth"][frame_a], second["depth"][frame_b]
    va = (
        first["valid_mask"][frame_a]
        & np.isfinite(a).all(axis=-1)
        & np.isfinite(ca)
        & np.isfinite(da)
        & (da > 0)
    )
    vb = (
        second["valid_mask"][frame_b]
        & np.isfinite(b).all(axis=-1)
        & np.isfinite(cb)
        & np.isfinite(db)
        & (db > 0)
    )
    if va.any():
        va &= ca >= np.percentile(ca[va], 25)
    if vb.any():
        vb &= cb >= np.percentile(cb[vb], 25)
    valid = va & vb
    height, width = valid.shape
    rng = np.random.default_rng(seed)
    chosen = []
    for row in range(4):
        for column in range(4):
            y0, y1 = row * height // 4, (row + 1) * height // 4
            x0, x1 = column * width // 4, (column + 1) * width // 4
            ys, xs = np.where(valid[y0:y1, x0:x1])
            pixels = np.column_stack((xs + x0, ys + y0))
            if len(pixels) > cap // 16:
                pixels = pixels[rng.choice(len(pixels), cap // 16, replace=False)]
            chosen.extend(map(tuple, pixels.tolist()))
    pixels = np.asarray(chosen, dtype=np.int32).reshape(-1, 2)
    if len(pixels) > cap:
        pixels = pixels[rng.choice(len(pixels), cap, replace=False)]
    if not len(pixels):
        return None
    x, y = pixels.T
    # Confidence heads are scores (often exp+1), not calibrated probabilities.
    score = np.log1p(np.maximum(ca[y, x], 0)) + np.log1p(np.maximum(cb[y, x], 0))
    cells = (y * 4 // height) * 4 + (x * 4 // width)
    weights = np.zeros(len(pixels), dtype=np.float32)
    for cell in np.unique(cells):
        mask = cells == cell
        cell_score = score[mask]
        positive = np.maximum(cell_score - np.min(cell_score) + 1e-3, 1e-3)
        weights[mask] = positive / positive.sum() / 16
    weights /= weights.sum()
    return (
        b[y, x],
        a[y, x],
        weights,
        pixels,
        np.full(len(pixels), frame_id, dtype=np.int32),
    )


def _scene_scale(points: np.ndarray, camera_centers: np.ndarray) -> float:
    distances = np.linalg.norm(points - camera_centers, axis=-1)
    scale = float(np.median(distances[np.isfinite(distances) & (distances > 0)]))
    return max(scale, 1e-8)


def _sim3_json(transform) -> dict:
    return {
        "scale": transform.scale,
        "rotation": transform.rotation.tolist(),
        "translation": transform.translation.tolist(),
        "matrix": transform.as_matrix().tolist(),
    }


def _transform_camera(camera_to_world: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Apply Sim3 to camera center and orientation, returning c2w and inverse w2c."""
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    aligned = np.eye(4)
    aligned[:3, :3] = transform.rotation @ camera_to_world[:3, :3]
    aligned[:3, 3] = transform.apply(camera_to_world[:3, 3])
    return aligned, np.linalg.inv(aligned)[:3]


def run_section_alignment(
    output_dir: str | Path,
    *,
    try_icp: bool = True,
    extra_edges=None,
    max_points: int = 1_000_000,
) -> dict:
    """Align cached sections, export one owner prediction per source frame, and report diagnostics."""
    from .section_registration import SectionEdge, optimize_section_graph, robust_sim3
    from .visualization import save_point_cloud

    root = Path(output_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    frame_ids = manifest["frame_ids"]
    sections = manifest["sections"]
    section_indices = [entry["section_index"] for entry in sections]
    if section_indices != list(range(len(sections))):
        raise ValueError("Manifest sections must be complete and ordered")
    frame_to_sections = {i: [] for i in range(len(frame_ids))}
    for section in sections:
        for local, frame_id in enumerate(section["frame_ids"]):
            index = frame_ids.index(frame_id)
            frame_to_sections[index].append(
                (section["section_index"], local, len(section["frame_ids"]))
            )
    for items in frame_to_sections.values():
        if not items:
            raise ValueError("Section manifest does not cover every source frame")

    edges, edge_reports = [], []
    for earlier in range(len(sections) - 1):
        later = earlier + 1
        common = [
            frame_ids.index(fid)
            for fid in sections[earlier]["frame_ids"]
            if fid in set(sections[later]["frame_ids"])
        ]
        validation_count = min(3, len(common) - 3)
        if validation_count < 2:
            raise ValueError(
                f"Overlap {earlier}-{later} needs at least five common frames for 3 train/2 validation"
            )
        validation_positions = np.linspace(0, len(common) - 1, validation_count, dtype=int)
        validation = {common[index] for index in validation_positions}
        train = [item for item in common if item not in validation]
        if len(train) < 3 or not validation:
            raise ValueError(f"Insufficient overlap for sections {earlier}, {later}")
        a_dir, b_dir = root / f"section_{earlier:02d}", root / f"section_{later:02d}"
        a_meta, b_meta = _load_metadata(a_dir), _load_metadata(b_dir)
        a_ids, b_ids = a_meta["frames"], b_meta["frames"]
        a_local = {f["frame_id"]: i for i, f in enumerate(a_ids)}
        b_local = {f["frame_id"]: i for i, f in enumerate(b_ids)}
        a_transforms = {f["frame_id"]: a_meta["transforms"][i] for i, f in enumerate(a_ids)}
        b_transforms = {f["frame_id"]: b_meta["transforms"][i] for i, f in enumerate(b_ids)}
        section_hashes = {
            fid: digest
            for fid, digest in zip(
                sections[earlier]["frame_ids"], sections[earlier]["image_hashes"], strict=True
            )
        }
        later_hashes = dict(
            zip(sections[later]["frame_ids"], sections[later]["image_hashes"], strict=True)
        )
        if any(
            section_hashes[fid] != later_hashes[fid] or a_transforms[fid] != b_transforms[fid]
            for fid in (frame_ids[i] for i in common)
        ):
            raise ValueError(
                f"Overlap image pixels/transforms differ for sections {earlier}, {later}"
            )
        keys = ("depth_world_points", "depth_confidence", "depth", "valid_mask", "camera_to_world")
        with np.load(a_dir / "predictions.npz") as archive:
            first = {key: archive[key] for key in keys}
        with np.load(b_dir / "predictions.npz") as archive:
            second = {key: archive[key] for key in keys}
        train_parts, valid_parts = [], []
        for frame_index in common:
            fid = frame_ids[frame_index]
            split = valid_parts if frame_index in validation else train_parts
            correspondence = _sample_frame_correspondences(
                first, second, a_local[fid], b_local[fid], frame_index, 1000 + frame_index, 500
            )
            if correspondence is not None:
                split.append(correspondence)
        valid_train_frames = {int(index) for part in train_parts for index in part[4]}
        valid_validation_frames = {int(index) for part in valid_parts for index in part[4]}
        if len(valid_train_frames) < 3 or len(valid_validation_frames) < 2:
            raise ValueError(
                f"Overlap {earlier}-{later} has fewer than 3 usable train or 2 held-out frames"
            )

        def combine(parts):
            return tuple(np.concatenate([part[i] for part in parts]) for i in range(5))

        src, dst, weights, pixels, labels = combine(train_parts)
        val_src, val_dst, val_weights, val_pixels, val_labels = combine(valid_parts)
        edge_path = root / f"overlap_{later:02d}_{earlier:02d}.npz"
        np.savez_compressed(
            edge_path,
            train_source=src,
            train_target=dst,
            train_weights=weights,
            train_pixel_xy=pixels,
            train_frame_indices=labels,
            validation_source=val_src,
            validation_target=val_dst,
            validation_weights=val_weights,
            validation_pixel_xy=val_pixels,
            validation_frame_indices=val_labels,
        )
        centers = np.asarray(
            [first["camera_to_world"][a_local[frame_ids[int(i)]], :3, 3] for i in labels]
        )
        scale = _scene_scale(dst, centers)
        estimate, fit = robust_sim3(src, dst, weights=weights, seed=earlier)
        fit_inliers = np.asarray(fit["residuals"]) <= fit["threshold"]
        inlier_frames = np.unique(labels[fit_inliers])
        inlier_cells = np.unique(
            (pixels[fit_inliers, 1] * 4 // manifest["image_size"]) * 4
            + pixels[fit_inliers, 0] * 4 // manifest["image_size"]
        )
        if len(inlier_frames) < 3 or len(inlier_cells) < 2:
            raise ValueError(
                f"Overlap {earlier}-{later} Sim3 consensus must span at least 3 frames and 2 cells"
            )
        residual = np.linalg.norm(estimate.apply(val_src) - val_dst, axis=1)
        validation_report = {
            "count": len(residual),
            "median": float(np.median(residual)),
            "p90": float(np.percentile(residual, 90)),
            "normalized_median": float(np.median(residual) / scale),
            "normalized_p90": float(np.percentile(residual, 90) / scale),
            "scene_scale": scale,
        }
        if (
            validation_report["normalized_median"] > 0.10
            or validation_report["normalized_p90"] > 0.25
        ):
            raise ValueError(
                f"Overlap {earlier}-{later} held-out quality gate failed: "
                f"normalized median={validation_report['normalized_median']:.4f} "
                f"(limit 0.10), p90={validation_report['normalized_p90']:.4f} (limit 0.25)"
            )
        if try_icp:
            try:
                from .section_evaluation import evaluate_icp

                train_cloud_a, train_cloud_b = [], []
                for frame_index in train:
                    fid = frame_ids[frame_index]
                    ai, bi = a_local[fid], b_local[fid]
                    for data, local, clouds in (
                        (first, ai, train_cloud_a),
                        (second, bi, train_cloud_b),
                    ):
                        point = data["depth_world_points"][local]
                        conf = data["depth_confidence"][local]
                        depth = data["depth"][local]
                        valid = (
                            data["valid_mask"][local]
                            & np.isfinite(point).all(axis=-1)
                            & np.isfinite(conf)
                            & np.isfinite(depth)
                            & (depth > 0)
                        )
                        if valid.any():
                            valid &= conf >= np.percentile(conf[valid], 25)
                        clouds.append(point[valid])
                train_cloud_a = np.concatenate(train_cloud_a)
                train_cloud_b = np.concatenate(train_cloud_b)
                estimate, icp_report = evaluate_icp(
                    train_cloud_b, train_cloud_a, estimate, val_src, val_dst, scene_scale=scale
                )
            except (ImportError, RuntimeError, ValueError) as error:
                icp_report = {"accepted": False, "error": str(error)}
        else:
            icp_report = {"accepted": False, "skipped": True}
        weight = max(float(fit["inlier_ratio"] * len(src)), 1.0)
        edges.append(SectionEdge(later, earlier, estimate, weight, "overlap", scale))
        edge_reports.append(
            {
                "source": later,
                "target": earlier,
                "kind": "overlap",
                "correspondences": str(edge_path.name),
                "weight": weight,
                "translation_scale": scale,
                "training": fit,
                "validation": validation_report,
                "icp": icp_report,
                "transform": _sim3_json(estimate),
            }
        )
        del first, second
    if extra_edges:
        for edge in extra_edges:
            edges.append(edge)
            edge_reports.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind,
                    "weight": edge.weight,
                    "translation_scale": edge.translation_scale,
                    "transform": _sim3_json(edge.transform),
                    "injected": True,
                }
            )
    from .section_registration import Sim3

    baseline = [None] * len(sections)
    baseline[0] = Sim3.identity()
    for edge in edges:
        if edge.kind == "overlap" and edge.target == edge.source - 1:
            baseline[edge.source] = baseline[edge.target].compose(edge.transform)
    if any(pose is None for pose in baseline):
        raise ValueError("Adjacent overlap edges do not provide a connected baseline")
    transforms, initial_report = optimize_section_graph(len(sections), edges)
    graph_path = root / "graph.json"
    graph = {
        "schema_version": 1,
        "convention": "section-local -> global; world points map as s*R*p+t",
        "initial_transforms": [_sim3_json(t) for t in baseline],
        "optimized_transforms": [_sim3_json(t) for t in transforms],
        "edges": edge_reports,
        "optimizer": initial_report,
    }
    graph_path.write_text(json.dumps(graph, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        root / "graph.npz",
        initial_matrices=np.stack([t.as_matrix() for t in baseline]),
        optimized_matrices=np.stack([t.as_matrix() for t in transforms]),
    )

    # Select a single section for each source frame, preferring maximum boundary distance.
    owners = [
        max(items, key=lambda item: (min(item[1], item[2] - 1 - item[1]), -item[0]))
        for items in frame_to_sections.values()
    ]
    if max_points < len(frame_ids):
        raise ValueError("max_points must allow at least one observation per source frame")
    budget = max_points // len(frame_ids)
    cloud_by_frame = [None] * len(frame_ids)
    camera_centers = np.empty((len(frame_ids), 3), dtype=np.float64)
    camera_rotations = np.empty((len(frame_ids), 3, 3), dtype=np.float64)
    camera_translations = np.empty((len(frame_ids), 3, 4), dtype=np.float64)
    for owner in range(len(sections)):
        assigned = [i for i, item in enumerate(owners) if item[0] == owner]
        if not assigned:
            continue
        section_dir = root / f"section_{owner:02d}"
        keys = (
            "depth_world_points",
            "depth_confidence",
            "depth",
            "valid_mask",
            "images",
            "camera_to_world",
        )
        with np.load(section_dir / "predictions.npz") as archive:
            arrays = {key: archive[key] for key in keys}
        for frame_index in assigned:
            _, local, _ = owners[frame_index]
            points = arrays["depth_world_points"][local]
            confidence = arrays["depth_confidence"][local]
            depth = arrays["depth"][local]
            valid = (
                arrays["valid_mask"][local]
                & np.isfinite(points).all(axis=-1)
                & np.isfinite(confidence)
                & np.isfinite(depth)
                & (depth > 0)
            )
            if valid.any():
                valid &= confidence >= np.percentile(confidence[valid], 25)
            indices = np.flatnonzero(valid)
            if not len(indices):
                raise ValueError(
                    f"Owner section {owner} has no valid observations for frame {frame_index}"
                )
            if len(indices) > budget:
                indices = indices[np.linspace(0, len(indices) - 1, budget, dtype=int)]
            width = valid.shape[1]
            xy = np.column_stack((indices % width, indices // width)).astype(np.int32)
            cloud_by_frame[frame_index] = (
                transforms[owner].apply(points.reshape(-1, 3)[indices]).astype(np.float32),
                arrays["images"][local].reshape(-1, 3)[indices],
                confidence.reshape(-1)[indices].astype(np.float32),
                xy,
                (depth.reshape(-1)[indices] * transforms[owner].scale).astype(np.float32),
            )
            global_camera, world_to_camera = _transform_camera(
                arrays["camera_to_world"][local], transforms[owner]
            )
            camera_centers[frame_index] = global_camera[:3, 3]
            camera_rotations[frame_index] = global_camera[:3, :3]
            camera_translations[frame_index] = world_to_camera
        del arrays
    if any(item is None for item in cloud_by_frame):
        raise ValueError("Every source frame must contribute a cloud observation")
    points = np.concatenate([item[0] for item in cloud_by_frame])
    colors = np.concatenate([item[1] for item in cloud_by_frame])
    confidence = np.concatenate([item[2] for item in cloud_by_frame])
    pixel_xy = np.concatenate([item[3] for item in cloud_by_frame])
    cloud_depths = np.concatenate([item[4] for item in cloud_by_frame])
    frame_indices = np.concatenate(
        [np.full(len(item[0]), i, np.int32) for i, item in enumerate(cloud_by_frame)]
    )
    owner_ids = np.concatenate(
        [np.full(len(item[0]), owners[i][0], np.int16) for i, item in enumerate(cloud_by_frame)]
    )
    arrays = {
        "points": points,
        "colors": colors,
        "confidence": confidence,
        "frame_indices": frame_indices,
        "pixel_xy": pixel_xy,
        "section_indices": owner_ids,
        "camera_centers": np.asarray(camera_centers),
        "camera_rotations": np.asarray(camera_rotations),
        "camera_translations": np.asarray(camera_translations),
        "depths": cloud_depths,
        "frame_ids": np.asarray(frame_ids),
    }
    np.savez_compressed(root / "point_cloud.npz", **arrays)
    from .contracts import PointCloud

    save_point_cloud(
        PointCloud(
            arrays["points"],
            arrays["colors"],
            arrays["confidence"],
            arrays["frame_indices"],
            arrays["pixel_xy"],
        ),
        root / "point_cloud.ply",
    )
    report = {
        "frame_count": len(frame_ids),
        "point_count": len(arrays["points"]),
        "owner_section_by_frame": [int(item[0]) for item in owners],
        "graph": "graph.json",
        "cloud": "point_cloud.npz",
        "edges": edge_reports,
    }
    (root / "alignment_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def align_sections(
    preprocessing_dir: str | Path,
    output_dir: str | Path,
    *,
    inference_only: bool = False,
    skip_inference: bool = False,
    try_icp: bool = True,
    extra_edges=None,
    max_points: int = 1_000_000,
    **kwargs,
) -> dict:
    """Run inference unless skipped, then align and export without geometry fusion."""
    if not skip_inference:
        run_section_inference(preprocessing_dir, output_dir, **kwargs)
    if inference_only:
        return {"inference_only": True}
    return run_section_alignment(
        output_dir, try_icp=try_icp, extra_edges=extra_edges, max_points=max_points
    )
