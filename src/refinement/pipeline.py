"""Execute refinement separately from source reconstruction and preprocessing."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .contracts import RefinementConfig, RefinementResult
from .inputs import load_refinement_scene
from .matching import build_observation_tracks, load_observation_tracks
from .optimization import optimize_multiview


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
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cached = Path(tracks_path) if tracks_path else output / "tracks.npz"
    if tracks_path and not cached.is_file():
        raise FileNotFoundError(cached)
    tracks = (
        load_observation_tracks(cached, scene)
        if cached.is_file()
        else build_observation_tracks(scene, source, output, config)
    )
    if config.flow_anchors:
        from .flow import augment_texture_tracks

        tracks = augment_texture_tracks(scene, tracks, output)
    if config.dense_anchors:
        from .dense_matching import augment_dense_tracks

        tracks = augment_dense_tracks(scene, tracks, output, config, cached.with_suffix(".json"))
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
