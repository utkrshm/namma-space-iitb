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
    frame_count: int,
    section_size: int = 36,
    overlap: int = 11,
    tail_section_size: int | None = None,
    tail_context_frames: int = 0,
    tail_start_section: int | None = None,
) -> list[SectionSpec]:
    """Create overlapping windows, optionally replacing a normal-window suffix with tails.

    ``tail_start_section`` indexes the original full-size windows before any tails are
    added. A context extension moves the first replacement window earlier, but never
    before the previous normal window's start plus one.
    """
    if frame_count < 1 or section_size < 1 or not 0 <= overlap < section_size:
        raise ValueError(
            "Require positive frame_count/section_size and 0 <= overlap < section_size"
        )
    if tail_section_size is not None and not overlap < tail_section_size <= section_size:
        raise ValueError("tail_section_size must be greater than overlap and at most section_size")
    if not isinstance(tail_context_frames, int) or tail_context_frames < 0:
        raise ValueError("tail_context_frames must be a nonnegative integer")
    if tail_context_frames and tail_section_size is None:
        raise ValueError("tail_context_frames requires tail_section_size")
    if tail_start_section is not None and tail_section_size is None:
        raise ValueError("tail_start_section requires tail_section_size")
    if frame_count <= section_size:
        starts = [0]
    else:
        step = section_size - overlap
        starts = list(range(0, frame_count - section_size + 1, step))
        starts.append(frame_count - section_size)
        starts = list(dict.fromkeys(starts))
    windows = [(start, min(start + section_size, frame_count)) for start in starts]
    if tail_section_size is not None:
        tail_index = len(windows) - 1 if tail_start_section is None else tail_start_section
        if not isinstance(tail_index, int) or not 0 <= tail_index < len(windows):
            raise ValueError("tail_start_section must identify an existing normal window")
        tail_start = windows[tail_index][0]
        minimum_start = windows[tail_index - 1][0] + 1 if tail_index > 0 else 0
        tail_start = max(minimum_start, tail_start - tail_context_frames)
        last_tail_start = max(0, frame_count - tail_section_size)
        tail_starts = list(range(tail_start, last_tail_start + 1, tail_section_size - overlap))
        tail_starts.append(last_tail_start)
        windows = windows[:tail_index] + [
            (start, min(start + tail_section_size, frame_count))
            for start in dict.fromkeys(tail_starts)
        ]
    return [SectionSpec(i, tuple(range(start, end))) for i, (start, end) in enumerate(windows)]


def _absolute_quality_gate(pooled_normalized_median: float, per_frame: dict) -> bool:
    """Accept a useful edge despite an isolated held-out outlier frame."""
    return (
        pooled_normalized_median <= 0.10
        and sum(frame["normalized_median"] <= 0.10 for frame in per_frame.values()) >= 2
    )


def _split_overlap_frames(common: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Return disjoint training, primary-validation, and endpoint-stress frame IDs."""
    if len(common) >= 8:
        stress = [common[0], common[-1]]
        positions = np.rint(np.quantile(np.arange(len(common)), [0.25, 0.5, 0.75])).astype(int)
        validation = [common[index] for index in positions]
        training = [frame for frame in common if frame not in {*stress, *validation}]
    elif len(common) == 7:
        stress = [common[0], common[-1]]
        interior = common[1:-1]
        positions = np.rint(np.quantile(np.arange(len(interior)), [1 / 3, 2 / 3])).astype(int)
        validation = [interior[index] for index in positions]
        training = [frame for frame in interior if frame not in validation]
    else:
        stress = []
        validation_count = min(3, len(common) - 3)
        if validation_count < 2:
            raise ValueError("Overlap needs at least five frames for 3 train/2 validation")
        positions = np.linspace(0, len(common) - 1, validation_count, dtype=int)
        validation = [common[index] for index in positions]
        training = [frame for frame in common if frame not in validation]
    if len(training) < 3 or len(validation) < 2:
        raise ValueError("Overlap split requires at least three train and two validation frames")
    if (
        set(training) & set(validation)
        or set(training) & set(stress)
        or set(validation) & set(stress)
    ):
        raise ValueError("Training, validation, and endpoint-stress frames must be disjoint")
    return training, validation, stress


def _registration_threshold(target_camera_distance: float) -> float:
    """RANSAC tolerance in target reconstruction units, scaled by camera distance."""
    if not np.isfinite(target_camera_distance) or target_camera_distance <= 0:
        raise ValueError("target_camera_distance must be finite and positive")
    return 0.03 * target_camera_distance


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
    tail_section_size: int | None = None,
    tail_context_frames: int = 0,
    tail_start_section: int | None = None,
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
    Tail sizing, start index, and context change boundaries only; image size/model pixels stay fixed.
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
    provenance = json.loads(json.dumps([frame.__dict__ for frame in batch.frames]))
    transforms = json.loads(json.dumps([transform.__dict__ for transform in batch.transforms]))
    sections = create_sections(
        len(frame_ids),
        section_size,
        overlap,
        tail_section_size,
        tail_context_frames,
        tail_start_section,
    )
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
        "tail_section_size": tail_section_size,
        "tail_context_frames": tail_context_frames,
        "tail_start_section": tail_start_section,
        "overlap": overlap,
        "image_size": image_size,
        "model_id": model_id,
        "device": device,
        "precision": precision,
        "local_files_only": local_files_only,
        "sections": [{"section_index": section.index, "status": "pending"} for section in sections],
        "archived_sections": [],
    }
    _write_manifest(output, manifest)
    try:
        for section in sections:
            frame_indices = section.frame_indices
            section_dir = output / f"section_{section.index:02d}"
            actual_size = len(frame_indices)
            is_tail = tail_section_size is not None and actual_size != section_size
            settings = {
                "frame_ids": [frame_ids[i] for i in frame_indices],
                "image_hashes": [image_hashes[i] for i in frame_indices],
                "section_index": section.index,
                "section_size": actual_size,
                "overlap": overlap,
                "image_size": image_size,
                "model_id": model_id,
                "device": device,
                "precision": precision,
            }
            if is_tail:
                settings["tail_section_size"] = tail_section_size
            manifest["sections"][section.index] = {**settings, "status": "pending"}
            selected_frames = [provenance[i] for i in frame_indices]
            selected_transforms = [transforms[i] for i in frame_indices]
            selected_valid_hashes = [valid_mask_hashes[i] for i in frame_indices]
            if resume and _cached_section(
                section_dir, settings, selected_frames, selected_transforms, selected_valid_hashes
            ):
                metadata = _load_metadata(section_dir)
                cached_timing = metadata["section_alignment"]
                entry = {**settings, "status": "cached"}
                for key in (
                    "wall_seconds",
                    "cuda_peak_allocated_bytes",
                    "cuda_peak_reserved_bytes",
                ):
                    if key in cached_timing:
                        entry[key] = cached_timing[key]
                manifest["sections"][section.index] = entry
                _write_manifest(output, manifest)
                seconds = cached_timing.get("wall_seconds")
                memory = cached_timing.get("cuda_peak_allocated_bytes")
                print(
                    f"Section {section.index + 1}/{len(sections)} cached: "
                    f"wall={seconds if seconds is not None else 'unknown'}s, "
                    f"peak_allocated={memory / 2**30:.2f} GiB"
                    if memory is not None
                    else f"Section {section.index + 1}/{len(sections)} cached: wall={seconds}s",
                    flush=True,
                )
                continue
            archived = _archive_incompatible_section(section_dir, settings, output)
            if archived is not None:
                manifest["archived_sections"].append(
                    {"section_index": section.index, "path": str(archived.relative_to(output))}
                )
                _write_manifest(output, manifest)
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
            manifest["sections"][section.index] = {
                **settings,
                "status": "inferred",
                "wall_seconds": elapsed,
                "cuda_peak_allocated_bytes": allocated,
                "cuda_peak_reserved_bytes": reserved,
            }
            _write_manifest(output, manifest)
            print(
                f"Section {section.index + 1}/{len(sections)} complete in {elapsed:.1f}s; "
                f"CUDA peak allocated/reserved {allocated / 2**30:.2f}/{reserved / 2**30:.2f} GiB",
                flush=True,
            )
            del reconstruction
    finally:
        model.unload()
    _write_manifest(output, manifest)
    return manifest


def _write_manifest(output: Path, manifest: dict) -> None:
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def _archive_incompatible_section(section_dir: Path, expected: dict, output: Path) -> Path | None:
    """Archive predictions only when the frame window itself changed; never overwrite archives."""
    metadata_path = section_dir / "metadata.json"
    predictions_path = section_dir / "predictions.npz"
    if not metadata_path.is_file() or not predictions_path.is_file():
        return None
    try:
        saved = json.loads(metadata_path.read_text()).get("section_alignment", {})
    except (OSError, json.JSONDecodeError):
        return None
    window_changed = (
        saved.get("frame_ids") != expected["frame_ids"]
        or saved.get("section_size") != expected["section_size"]
    )
    if not window_changed:
        return None
    archive_name = "initial_36" if saved.get("section_size") == 36 else "superseded_sections"
    archive_root = output / "experiments" / archive_name
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / section_dir.name
    suffix = 1
    while destination.exists():
        destination = archive_root / f"{section_dir.name}_{suffix}"
        suffix += 1
    section_dir.rename(destination)
    return destination


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
            fid
            for fid in sections[earlier]["frame_ids"]
            if fid in set(sections[later]["frame_ids"])
        ]
        train, validation, stress_frames = _split_overlap_frames(common)
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
            for fid in common
        ):
            raise ValueError(
                f"Overlap image pixels/transforms differ for sections {earlier}, {later}"
            )
        keys = ("depth_world_points", "depth_confidence", "depth", "valid_mask", "camera_to_world")
        with np.load(a_dir / "predictions.npz") as archive:
            first = {key: archive[key] for key in keys}
        with np.load(b_dir / "predictions.npz") as archive:
            second = {key: archive[key] for key in keys}
        train_parts, valid_parts, stress_parts = [], [], []
        for fid in common:
            frame_index = frame_ids.index(fid)
            if fid in train:
                split = train_parts
            elif fid in validation:
                split = valid_parts
            else:
                split = stress_parts
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
        if stress_parts:
            stress_src, stress_dst, stress_weights, stress_pixels, stress_labels = combine(
                stress_parts
            )
        else:
            stress_src = np.empty((0, 3), dtype=np.float32)
            stress_dst = np.empty((0, 3), dtype=np.float32)
            stress_weights = np.empty((0,), dtype=np.float32)
            stress_pixels = np.empty((0, 2), dtype=np.int32)
            stress_labels = np.empty((0,), dtype=np.int32)
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
            stress_source=stress_src,
            stress_target=stress_dst,
            stress_weights=stress_weights,
            stress_pixel_xy=stress_pixels,
            stress_frame_indices=stress_labels,
        )
        centers = np.asarray(
            [first["camera_to_world"][a_local[frame_ids[int(i)]], :3, 3] for i in labels]
        )
        scale = _scene_scale(dst, centers)
        threshold = _registration_threshold(scale)
        estimate, fit = robust_sim3(src, dst, weights=weights, threshold=threshold, seed=earlier)
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

        def validation_metrics(
            transform,
            validation_source=val_src,
            validation_target=val_dst,
            validation_labels=val_labels,
            scene_scale=scale,
        ):
            residuals = np.linalg.norm(
                transform.apply(validation_source) - validation_target, axis=1
            )
            per_frame = {}
            for global_index in sorted(set(map(int, validation_labels))):
                selected = validation_labels == global_index
                frame_residual = residuals[selected]
                per_frame[frame_ids[global_index]] = {
                    "count": len(frame_residual),
                    "median": float(np.median(frame_residual)),
                    "p90": float(np.percentile(frame_residual, 90)),
                    "normalized_median": float(np.median(frame_residual) / scene_scale),
                    "normalized_p90": float(np.percentile(frame_residual, 90) / scene_scale),
                }
            report = {
                "count": len(residuals),
                "median": float(np.median(residuals)),
                "p90": float(np.percentile(residuals, 90)),
                "normalized_median": float(np.median(residuals) / scene_scale),
                "normalized_p90": float(np.percentile(residuals, 90) / scene_scale),
                "scene_scale": scene_scale,
                "per_frame": per_frame,
                "frames_with_normalized_median_at_most_0_10": sum(
                    value["normalized_median"] <= 0.10 for value in per_frame.values()
                ),
                "p90_warning": float(np.percentile(residuals, 90) / scene_scale) > 0.35,
            }
            report["absolute_gate_passed"] = _absolute_quality_gate(
                report["normalized_median"], per_frame
            )
            return report

        validation_before_icp = validation_metrics(estimate)
        validation_report = validation_before_icp
        if try_icp:
            try:
                from .section_evaluation import evaluate_icp

                train_cloud_a, train_cloud_b = [], []
                for fid in train:
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
                initial_estimate = estimate
                candidate, icp_report = evaluate_icp(
                    train_cloud_b, train_cloud_a, estimate, val_src, val_dst, scene_scale=scale
                )
                if icp_report.get("accepted"):
                    candidate_report = validation_metrics(candidate)
                    if candidate_report["absolute_gate_passed"]:
                        estimate = candidate
                    else:
                        icp_report["accepted"] = False
                        icp_report["reason"] = (
                            "ICP candidate failed the absolute held-out quality gate"
                        )
                        icp_report["candidate_validation"] = candidate_report
                        estimate = initial_estimate
            except (ImportError, RuntimeError, ValueError) as error:
                icp_report = {"accepted": False, "error": str(error)}
        else:
            icp_report = {"accepted": False, "skipped": True}
        validation_report = validation_metrics(estimate)
        if not validation_report["absolute_gate_passed"]:
            raise ValueError(
                f"Overlap {earlier}-{later} failed absolute held-out alignment gate: "
                f"pooled normalized median={validation_report['normalized_median']:.4f} "
                f"(limit 0.10), only "
                f"{validation_report['frames_with_normalized_median_at_most_0_10']} of "
                f"{len(validation_report['per_frame'])} held-out frames have median <= 0.10"
            )
        camera_diagnostics = []
        for fid in common:
            earlier_camera = first["camera_to_world"][a_local[fid]]
            later_camera = second["camera_to_world"][b_local[fid]]
            center_error = np.linalg.norm(
                estimate.apply(later_camera[:3, 3]) - earlier_camera[:3, 3]
            )
            rotation_error = earlier_camera[:3, :3].T @ estimate.rotation @ later_camera[:3, :3]
            angle = np.degrees(np.arccos(np.clip((np.trace(rotation_error) - 1) / 2, -1.0, 1.0)))
            camera_diagnostics.append(
                {
                    "frame_id": fid,
                    "role": (
                        "endpoint_stress"
                        if fid in stress_frames
                        else "primary_validation"
                        if fid in validation
                        else "training"
                    ),
                    "center_distance": float(center_error),
                    "normalized_center_distance": float(center_error / scale),
                    "orientation_disagreement_degrees": float(angle),
                }
            )
        stress_report = {"count": len(stress_src), "per_frame": {}}
        if len(stress_src):
            stress_errors = np.linalg.norm(estimate.apply(stress_src) - stress_dst, axis=1)
            stress_report.update(
                median=float(np.median(stress_errors)),
                p90=float(np.percentile(stress_errors, 90)),
                normalized_median=float(np.median(stress_errors) / scale),
                normalized_p90=float(np.percentile(stress_errors, 90) / scale),
            )
            for global_index in sorted(set(map(int, stress_labels))):
                selected = stress_labels == global_index
                frame_errors = stress_errors[selected]
                stress_report["per_frame"][frame_ids[global_index]] = {
                    "count": len(frame_errors),
                    "median": float(np.median(frame_errors)),
                    "p90": float(np.percentile(frame_errors, 90)),
                    "normalized_median": float(np.median(frame_errors) / scale),
                    "normalized_p90": float(np.percentile(frame_errors, 90) / scale),
                }
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
                "training_frame_ids": train,
                "primary_validation_frame_ids": validation,
                "endpoint_stress_frame_ids": stress_frames,
                "validation": validation_report,
                "validation_before_icp": validation_before_icp,
                "endpoint_stress": stress_report,
                "duplicate_camera_pose_diagnostics": camera_diagnostics,
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
    # Graph optimization must preserve independently held-out overlap agreement.
    post_graph = []
    for edge in edge_reports:
        if edge["kind"] != "overlap":
            continue
        with np.load(root / edge["correspondences"]) as paired:
            relative = transforms[edge["target"]].inverse().compose(transforms[edge["source"]])
            distances = (
                np.linalg.norm(
                    relative.apply(paired["validation_source"]) - paired["validation_target"],
                    axis=1,
                )
                / edge["translation_scale"]
            )
            labels = paired["validation_frame_indices"]
            stress_source = paired["stress_source"]
            stress_target = paired["stress_target"]
            stress_labels = paired["stress_frame_indices"]
        per_frame = {
            frame_ids[int(index)]: {
                "normalized_median": float(np.median(distances[labels == index]))
            }
            for index in np.unique(labels)
        }
        median = float(np.median(distances))
        passed = _absolute_quality_gate(median, per_frame)
        post_graph.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "normalized_median": median,
                "normalized_p90": float(np.percentile(distances, 90)),
                "per_frame": per_frame,
                "absolute_gate_passed": passed,
            }
        )
        if not passed:
            raise ValueError(
                f"Optimized graph degraded held-out overlap {edge['source']}-{edge['target']}"
            )
        stress_post_graph = {"count": len(stress_source), "per_frame": {}}
        if len(stress_source):
            stress_distances = (
                np.linalg.norm(relative.apply(stress_source) - stress_target, axis=1)
                / edge["translation_scale"]
            )
            stress_post_graph.update(
                normalized_median=float(np.median(stress_distances)),
                normalized_p90=float(np.percentile(stress_distances, 90)),
            )
            for index in np.unique(stress_labels):
                frame_distance = stress_distances[stress_labels == index]
                stress_post_graph["per_frame"][frame_ids[int(index)]] = {
                    "normalized_median": float(np.median(frame_distance)),
                    "normalized_p90": float(np.percentile(frame_distance, 90)),
                }
        edge["endpoint_stress"]["post_graph"] = stress_post_graph
    graph_path = root / "graph.json"
    graph = {
        "schema_version": 1,
        "convention": "section-local -> global; world points map as s*R*p+t",
        "initial_transforms": [_sim3_json(t) for t in baseline],
        "optimized_transforms": [_sim3_json(t) for t in transforms],
        "edges": edge_reports,
        "optimizer": initial_report,
        "post_graph_validation": post_graph,
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
    use_revisits: bool = True,
    max_points: int = 1_000_000,
    **kwargs,
) -> dict:
    """Run inference unless skipped, then align and export without geometry fusion."""
    if not skip_inference:
        run_section_inference(preprocessing_dir, output_dir, **kwargs)
    if inference_only:
        return {"inference_only": True}
    if extra_edges is None and use_revisits:
        from .section_benchmark import build_revisit_edges

        extra_edges, _ = build_revisit_edges(output_dir, preprocessing_dir)
    return run_section_alignment(
        output_dir, try_icp=try_icp, extra_edges=extra_edges, max_points=max_points
    )
