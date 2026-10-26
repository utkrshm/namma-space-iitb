"""Execute refinement separately from source reconstruction and preprocessing."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import RefinementConfig, RefinementResult
from .inputs import load_refinement_scene
from .matching import build_observation_tracks, load_observation_tracks, validate_track_cache
from .optimization import optimize_multiview


def _same_tracks(first, second):
    return all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in first.__dataclass_fields__
    )


def _export_tracks(output, previous, scene, explicit_path=None):
    """Load the exact saved export tracks without invoking augmentation stages."""
    if explicit_path is not None:
        explicit = load_observation_tracks(explicit_path, scene)
        effective_path = output / "effective_tracks.npz"
        if effective_path.is_file():
            effective = load_observation_tracks(effective_path, scene)
            if not _same_tracks(explicit, effective):
                raise ValueError("--tracks must match the saved effective tracks for export-only")
        else:
            raise ValueError(
                "--tracks cannot override export-only tracks; no saved effective tracks exist"
            )
        return effective

    effective_path = output / "effective_tracks.npz"
    if effective_path.is_file():
        return load_observation_tracks(effective_path, scene)

    configured = previous.get("config", {})
    if configured.get("flow_anchors"):
        raise ValueError(
            "Export-only cannot reconstruct the saved flow tracks: "
            "effective_tracks.npz is missing. Restore it from the original refinement run."
        )

    source = previous.get("tracks_source")
    candidates = [Path(source)] if source else []
    candidates.append(output / "tracks.npz")
    dense_cache = output / "dense_matches"
    if configured.get("dense_anchors"):
        dense_candidates = sorted(dense_cache.glob("tracks_*.npz")) if dense_cache.is_dir() else []
        if source and Path(source).is_file() and Path(source).parent == dense_cache:
            return load_observation_tracks(source, scene)
        if len(dense_candidates) == 1:
            return load_observation_tracks(dense_candidates[0], scene)
        raise ValueError(
            "Export-only cannot identify the saved dense tracks unambiguously: "
            "effective_tracks.npz is missing. Restore it from the original refinement run."
        )
    existing = list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise ValueError(
            "Export-only needs saved effective tracks or one unambiguous base tracks cache"
        )
    return load_observation_tracks(existing[0], scene)


def refine_multiview(
    input_dir,
    output_dir=None,
    *,
    config=None,
    tracks_path=None,
    matching_only=False,
    export_only=False,
    inspect=True,
):
    config = config or RefinementConfig()
    if matching_only and export_only:
        raise ValueError("matching_only and export_only cannot be used together")
    source = Path(input_dir).resolve()
    output = Path(output_dir or source.parent / "multiview_refinement").resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Refinement output must be separate from the source artifact directory")
    output.mkdir(parents=True, exist_ok=True)

    scene = load_refinement_scene(source)
    print(
        f"Loaded {scene.frame_count} views; scene fingerprint {scene.fingerprint[:16]}", flush=True
    )
    if export_only:
        previous = json.loads((output / "manifest.json").read_text())
        if previous["scene_fingerprint"] != scene.fingerprint:
            raise ValueError("Saved refinement does not match current source predictions")
    manifest = {
        "schema_version": 1,
        "input_dir": str(source),
        "scene_fingerprint": scene.fingerprint,
        "config": asdict(config),
        "frames": scene.frames,
        "conventions": {
            "coordinates": "input world, arbitrary scene units, OpenCV cameras",
            "depth": "camera z; refined = initial * exp(log_scale + field)",
            "field": "bilinear zero-mean grid over content bounds, constant edge extension",
        },
        "source_to_model": scene.source_to_model.tolist(),
        "content_bounds": scene.content_bounds.tolist(),
    }
    cached = Path(tracks_path) if tracks_path else output / "tracks.npz"
    if tracks_path and not cached.is_file():
        raise FileNotFoundError(cached)
    if export_only:
        tracks = _export_tracks(output, previous, scene, tracks_path)
    elif cached.is_file():
        tracks = (
            load_observation_tracks(cached, scene)
            if tracks_path
            else validate_track_cache(cached, scene, source, config)
        )
    else:
        tracks = build_observation_tracks(scene, source, output, config)
    if not export_only and config.flow_anchors:
        from .flow import augment_texture_tracks

        tracks = augment_texture_tracks(scene, tracks, output)
    if not export_only and config.dense_anchors:
        from .dense_matching import augment_dense_tracks

        tracks = augment_dense_tracks(scene, tracks, output, config, cached.with_suffix(".json"))
    manifest["tracks_source"] = str(cached.resolve())
    manifest["explicit_tracks_override"] = tracks_path is not None
    if not export_only:
        np.savez_compressed(
            output / "effective_tracks.npz", **asdict(tracks), scene_fingerprint=scene.fingerprint
        )
    if not export_only:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if matching_only:
        return {
            "status": "matched",
            "frames": scene.frame_count,
            "tracks": int(tracks.track_indices.max()) + 1,
            "observations": len(tracks.frame_indices),
        }

    if export_only:
        with np.load(output / "corrections.npz", allow_pickle=False) as data:
            result = RefinementResult(
                *(data[k] for k in ("rotations", "centers", "log_scales", "log_depth_fields")),
                json.loads((output / "refinement_report.json").read_text()),
            )
        manifest["config"] = previous["config"]
        manifest["tracks_source"] = previous.get("tracks_source", str(cached.resolve()))
        manifest["explicit_tracks_override"] = previous.get("explicit_tracks_override", False)
        manifest["export_config"] = asdict(config)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        result = optimize_multiview(scene, tracks, config)
    np.savez_compressed(
        output / "corrections.npz",
        rotations=result.rotations,
        centers=result.centers,
        log_scales=result.log_scales,
        log_depth_fields=result.log_depth_fields,
        intrinsics=scene.intrinsics,
        source_to_model=scene.source_to_model,
        content_bounds=scene.content_bounds,
        frame_ids=[f["frame_id"] for f in scene.frames],
        initial_rotations=scene.rotations,
        initial_centers=scene.centers,
    )
    (output / "refinement_report.json").write_text(
        json.dumps(result.report, indent=2, allow_nan=False) + "\n"
    )

    from .export import export_refinement

    export_report = export_refinement(scene, result, tracks, source, output, config)
    (output / "export_report.json").write_text(
        json.dumps(export_report, indent=2, allow_nan=False) + "\n"
    )
    if inspect:
        from .inspection import inspect_refinement

        inspect_refinement(output, scene=scene)
    return {
        "status": "complete",
        "selected_candidate": result.report["selected_candidate"],
        "test": result.report["metrics"]["test"],
        "export": export_report,
    }
