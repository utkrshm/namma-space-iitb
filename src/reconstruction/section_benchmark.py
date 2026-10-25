"""Independent section revisit constraints and resolution comparison helpers."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from .section_registration import SectionEdge, Sim3, robust_sim3


def map_pixel_centres(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map xy integer-centred pixels with a 3x3 source-to-destination affine map."""
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("Expected N x 2 points and a 3 x 3 pixel transform")
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def normalized_disagreement(
    first: np.ndarray, second: np.ndarray, camera_centers: np.ndarray
) -> dict:
    """Compare paired 3D points in each model's own arbitrary units."""
    first, second, centers = (
        np.asarray(value, dtype=float) for value in (first, second, camera_centers)
    )
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("Point arrays must have matching shape N x 3")
    if centers.shape != first.shape:
        raise ValueError("Camera centers must match point arrays")
    if not len(first) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Point arrays must be nonempty and finite")
    scale_a = float(np.median(np.linalg.norm(first - centers, axis=1)))
    scale_b = float(np.median(np.linalg.norm(second - centers, axis=1)))
    if min(scale_a, scale_b) <= 0:
        raise ValueError("Scene scale must be positive")
    distance = np.linalg.norm(first - second, axis=1)
    return {
        "count": len(distance),
        "median": float(np.median(distance)),
        "p90": float(np.percentile(distance, 90)),
        "first_scene_scale": scale_a,
        "second_scene_scale": scale_b,
        "first_normalized_median": float(np.median(distance / scale_a)),
        "second_normalized_median": float(np.median(distance / scale_b)),
    }


def _bilinear(points: np.ndarray, image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height, width = valid.shape
    xy = np.asarray(points, dtype=np.float32)
    inside = (xy[:, 0] >= 1) & (xy[:, 0] < width - 2) & (xy[:, 1] >= 1) & (xy[:, 1] < height - 2)
    result = np.full((len(xy), image.shape[-1]), np.nan, dtype=np.float32)
    if inside.any():
        values = cv2.remap(
            image.astype(np.float32), xy[inside, 0, None], xy[inside, 1, None], cv2.INTER_LINEAR
        )
        sample_xy = xy[inside]
        x0, y0 = np.floor(sample_xy[:, 0]).astype(int), np.floor(sample_xy[:, 1]).astype(int)
        good = valid[y0, x0] & valid[y0, x0 + 1] & valid[y0 + 1, x0] & valid[y0 + 1, x0 + 1]
        values = np.squeeze(values, axis=1)
        result[np.flatnonzero(inside)[good]] = values[good]
    return result


def _verifier(config):
    """Create one real CPU matcher and bounded feature cache for an entire run."""
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image

    extractor = ALIKED(max_num_keypoints=config.keypoints).eval().to("cpu")
    matcher = LightGlue(features="aliked").eval().to("cpu")
    cache: OrderedDict[str, tuple[dict, tuple[int, int]]] = OrderedDict()

    def features(path):
        key = str(path)
        if key not in cache:
            image = load_image(key, resize=config.image_long_side).to("cpu")
            with torch.inference_mode():
                feature = extractor.extract(image)
            cache[key] = (feature, tuple(image.shape[-2:]))
            if len(cache) > config.feature_cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def match(path_a, path_b):
        from ..preprocessing.revisits import verify_feature_pair

        first, shape_a = features(path_a)
        second, shape_b = features(path_b)
        with torch.inference_mode():
            verified, report = verify_feature_pair(first, second, matcher, shape_a, shape_b, config)
        report["feature_shapes"] = {str(path_a): shape_a, str(path_b): shape_b}
        return verified, report

    return match


def _source_pixels(points, feature_shape, source_size):
    height, width = feature_shape
    source_width, source_height = source_size
    # LightGlue coordinates refer to resized pixel centres.
    scale = np.array([source_width / width, source_height / height])
    return (np.asarray(points, dtype=float) + 0.5) * scale - 0.5


def _config(preprocessing_dir):
    from ..preprocessing.config import RevisitConfig

    settings = json.loads((Path(preprocessing_dir) / "run.json").read_text())["config"]["revisits"]
    return RevisitConfig(**{**settings, "device": "cpu"})


class _SectionStore:
    """Bounded CPU cache: decompress each required section array once, not per pixel/frame."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.metadata = {}
        self.membership = {}
        self.frames = {}
        self.cache = OrderedDict()
        for section in self.manifest["sections"]:
            sid = section["section_index"]
            meta = json.loads((self.root / f"section_{sid:02d}" / "metadata.json").read_text())
            self.metadata[sid] = meta
            for local, frame in enumerate(meta["frames"]):
                self.membership.setdefault(frame["frame_id"], []).append((sid, local))
                self.frames[frame["frame_id"]] = frame

    def arrays(self, sid):
        if sid not in self.cache:
            keys = (
                "depth_world_points",
                "depth_confidence",
                "depth",
                "valid_mask",
                "source_to_model",
                "camera_to_world",
            )
            with np.load(self.root / f"section_{sid:02d}" / "predictions.npz") as archive:
                self.cache[sid] = {key: archive[key] for key in keys}
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(sid)
        return self.cache[sid]

    def sample(self, sid, fid, source_xy):
        local = next(i for i, f in enumerate(self.metadata[sid]["frames"]) if f["frame_id"] == fid)
        data = self.arrays(sid)
        return _sample(data, local, source_xy), data["camera_to_world"][local, :3, 3]


def _sample(data, local, source_xy):
    xyz = data["depth_world_points"][local]
    conf, depth = data["depth_confidence"][local], data["depth"][local]
    valid = (
        data["valid_mask"][local]
        & np.isfinite(xyz).all(-1)
        & np.isfinite(conf)
        & np.isfinite(depth)
        & (depth > 0)
    )
    if valid.any():
        valid &= conf >= np.percentile(conf[valid], 25)
    return _bilinear(map_pixel_centres(source_xy, data["source_to_model"][local]), xyz, valid)


def _matched_source_pixels(store, a, b, config, matcher, purpose):
    """Persist genuine image matches; cache identity includes source identities and settings."""
    root = store.root / "feature_matches"
    root.mkdir(exist_ok=True)
    frame_a, frame_b = store.frames[a], store.frames[b]
    paths = [Path(frame["image_path"]) for frame in (frame_a, frame_b)]
    signature = {
        "config": asdict(config),
        "purpose": purpose,
        "frames": [a, b],
        "files": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in paths],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = root / f"{digest}.npz"
    if cache.is_file():
        with np.load(cache) as arrays:
            return arrays["a"], arrays["b"], json.loads(str(arrays["details"]))
    verified, details = matcher(*paths)
    if verified is None:
        points = (np.empty((0, 2)), np.empty((0, 2)))
    else:
        points = []
        for fid, path, xy in zip((a, b), paths, verified[2:], strict=True):
            sid, local = store.membership[fid][0]
            size = store.metadata[sid]["transforms"][local]["source_size"]
            points.append(_source_pixels(xy, details["feature_shapes"][str(path)], size))
    np.savez_compressed(cache, a=points[0], b=points[1], details=json.dumps(details))
    return *points, details


def _summary(distances, scale=1.0):
    values = np.asarray(distances, dtype=float)
    if not len(values):
        return {"count": 0, "median": None, "p90": None, "rmse": None}
    values = values / scale
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def _require_spatial_support(xy, image_size, points, scene_scale):
    """Depth filtering must not turn broad image matches into a tiny scale anchor."""
    normalized = np.asarray(xy, dtype=np.float64) / np.asarray(image_size)
    cells = np.minimum((normalized * 4).astype(int), 3)
    coverage = len(np.unique(cells[:, 1] * 4 + cells[:, 0])) / 16
    area = cv2.contourArea(cv2.convexHull(normalized.astype(np.float32)))
    spread = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False) / np.sqrt(
        len(points)
    )
    if coverage < 0.125 or area < 0.015 or spread[1] < 0.01 * scene_scale:
        raise ValueError(
            "depth-valid revisit support is spatially concentrated or geometrically tiny"
        )


def build_revisit_edges(output_dir, preprocessing_dir) -> tuple[list[SectionEdge], dict]:
    """Reverify conservative image revisits, then validate independent 3D similarity fits."""
    store = _SectionStore(output_dir)
    config = _config(preprocessing_dir)
    matcher = _verifier(config)
    candidates = json.loads((Path(preprocessing_dir) / "covisibility.json").read_text())[
        "debug_matches"
    ]
    edges, records = [], []
    directory = store.root / "revisit_correspondences"
    directory.mkdir(exist_ok=True)
    for candidate in candidates:
        a, b = candidate["a"], candidate["b"]
        xy_a, xy_b, verification = _matched_source_pixels(store, a, b, config, matcher, "revisit")
        pairs = [
            (sa, sb) for sa, _ in store.membership[a] for sb, _ in store.membership[b] if sa != sb
        ]
        for source, target in pairs:
            record = {
                "a": a,
                "b": b,
                "source": source,
                "target": target,
                "accepted": False,
                "image_verification": verification,
            }
            try:
                if not len(xy_a):
                    raise ValueError("conservative image verification rejected")
                src, _ = store.sample(source, a, xy_a)
                dst, camera = store.sample(target, b, xy_b)
                good = np.flatnonzero(np.isfinite(src).all(1) & np.isfinite(dst).all(1))
                held = good[::3]
                train = np.setdiff1d(good, held)
                if len(train) < 12 or len(held) < 6:
                    raise ValueError("insufficient confident 3D train/held-out support")
                scene_scale = float(np.median(np.linalg.norm(dst[good] - camera, axis=1)))
                source_center = store.arrays(source)["camera_to_world"][
                    next(local for sid, local in store.membership[a] if sid == source), :3, 3
                ]
                source_scale = float(np.median(np.linalg.norm(src[good] - source_center, axis=1)))
                size_a = store.metadata[source]["transforms"][
                    next(local for sid, local in store.membership[a] if sid == source)
                ]["source_size"]
                size_b = store.metadata[target]["transforms"][
                    next(local for sid, local in store.membership[b] if sid == target)
                ]["source_size"]
                _require_spatial_support(xy_a[good], size_a, src[good], source_scale)
                _require_spatial_support(xy_b[good], size_b, dst[good], scene_scale)
                estimate, fit = robust_sim3(
                    src[train],
                    dst[train],
                    threshold=scene_scale * 0.03,
                    seed=source * 1009 + target,
                )
                residual = np.linalg.norm(estimate.apply(src[held]) - dst[held], axis=1)
                validation = _summary(residual, scene_scale)
                record.update(training=fit, validation=validation)
                if (
                    fit["inlier_ratio"] < 0.4
                    or validation["median"] > 0.05
                    or validation["p90"] > 0.15
                ):
                    raise ValueError("3D similarity lacks robust held-out support")
                consensus = train[np.asarray(fit["residuals"]) <= fit["threshold"]]
                for selected in (consensus, held):
                    _require_spatial_support(xy_a[selected], size_a, src[selected], source_scale)
                    _require_spatial_support(xy_b[selected], size_b, dst[selected], scene_scale)
                edge = SectionEdge(
                    source, target, estimate, min(fit["inlier_count"], 1000), "revisit", scene_scale
                )
                edges.append(edge)
                record.update(
                    accepted=True,
                    scale=estimate.scale,
                    rotation=estimate.rotation.tolist(),
                    translation=estimate.translation.tolist(),
                    weight=edge.weight,
                    translation_scale=scene_scale,
                )
                np.savez_compressed(
                    directory / f"edge_{len(edges):02d}.npz",
                    source=src,
                    target=dst,
                    train_indices=train,
                    held_indices=held,
                    source_pixel_xy=xy_a,
                    target_pixel_xy=xy_b,
                )
            except ValueError as error:
                record["reason"] = str(error)
            records.append(record)
            print(
                f"Revisit {a} -> {b}, sections {source}->{target}: {record['accepted']}", flush=True
            )
    report = {
        "candidate_count": len(candidates),
        "accepted_count": len(edges),
        "candidates": records,
    }
    (store.root / "revisit_graph.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return edges, report


def _validate_artifacts(root, manifest, cloud):
    import open3d as o3d

    frame_count = len(manifest["frame_ids"])
    if cloud["frame_ids"].tolist() != manifest["frame_ids"]:
        raise ValueError("Exported source identities differ from manifest")
    if not np.array_equal(np.unique(cloud["frame_indices"]), np.arange(frame_count)):
        raise ValueError("Every source frame must contribute observations")
    if not np.isfinite(cloud["points"]).all() or not np.isfinite(cloud["camera_centers"]).all():
        raise ValueError("Nonfinite exported geometry")
    rotation = cloud["camera_rotations"]
    if not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Camera orientations are not orthonormal")
    w2c = cloud["camera_translations"]
    if not np.allclose(w2c[:, :3, :3] @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("Inverse camera orientations disagree")
    if not np.allclose(
        np.einsum("nij,nj->ni", w2c[:, :, :3], cloud["camera_centers"]) + w2c[:, :, 3], 0, atol=1e-6
    ):
        raise ValueError("Inverse camera centers disagree")
    frame_indices = cloud["frame_indices"]
    z_depth = (
        np.einsum("ni,ni->n", cloud["points"], w2c[frame_indices, 2, :3]) + w2c[frame_indices, 2, 3]
    )
    if not np.allclose(z_depth, cloud["depths"], atol=1e-5, rtol=1e-5):
        raise ValueError("Aligned point depths disagree with scaled camera z-depth")
    ply = o3d.io.read_point_cloud(str(root / "point_cloud.ply"))
    if not np.array_equal(np.asarray(ply.points), cloud["points"]):
        raise ValueError("PLY XYZ differs from the saved observations")
    if not np.array_equal(np.rint(np.asarray(ply.colors) * 255).astype(np.uint8), cloud["colors"]):
        raise ValueError("PLY colors differ from the saved observations")
    graph = json.loads((root / "graph.json").read_text())
    for transform in graph["optimized_transforms"]:
        Sim3(
            transform["scale"], np.array(transform["rotation"]), np.array(transform["translation"])
        )
    return {
        "frame_count": frame_count,
        "point_count": len(cloud["points"]),
        "ply_npz_exact": True,
        "camera_inverse_consistent": True,
        "scaled_depth_consistent": True,
        "graph_connected": graph["optimizer"]["connected"],
        "section_count": len(graph["optimized_transforms"]),
        "edge_count": len(graph["edges"]),
        "cycle_rank": len(graph["edges"]) - len(graph["optimized_transforms"]) + 1,
    }


def _loop_graph_metrics(root, graph):
    """Audit held-out loop observations before/after joint graph optimization."""
    path = root / "revisit_graph.json"
    if not path.is_file():
        return []
    candidates = json.loads(path.read_text())["candidates"]
    accepted = [record for record in candidates if record["accepted"]]
    poses = {
        name: [
            Sim3(t["scale"], np.array(t["rotation"]), np.array(t["translation"]))
            for t in graph[key]
        ]
        for name, key in (("before", "initial_transforms"), ("after", "optimized_transforms"))
    }
    reports = []
    for index, edge in enumerate(accepted, 1):
        with np.load(root / "revisit_correspondences" / f"edge_{index:02d}.npz") as paired:
            selected = paired["held_indices"]
            source, target = paired["source"][selected], paired["target"][selected]
        report = {
            "source": edge["source"],
            "target": edge["target"],
            "a": edge["a"],
            "b": edge["b"],
        }
        for stage, transforms in poses.items():
            relative = transforms[edge["target"]].inverse().compose(transforms[edge["source"]])
            errors = np.linalg.norm(relative.apply(source) - target, axis=1)
            report[stage] = _summary(errors, edge["translation_scale"])
        reports.append(report)
    return reports


def build_benchmark_section_alignment(output_dir, baseline_dir, preprocessing_dir) -> dict:
    """Compare identical real feature observations with one characteristic scale per method."""
    store = _SectionStore(output_dir)
    root = store.root
    with np.load(root / "point_cloud.npz") as archive:
        cloud = {key: archive[key] for key in archive.files}
    artifact_checks = _validate_artifacts(root, store.manifest, cloud)
    alignment = json.loads((root / "alignment_report.json").read_text())
    owners = alignment["owner_section_by_frame"]
    graph = json.loads((root / "graph.json").read_text())
    transforms = [
        Sim3(t["scale"], np.array(t["rotation"]), np.array(t["translation"]))
        for t in graph["optimized_transforms"]
    ]
    baseline_dir = Path(baseline_dir)
    baseline_meta = json.loads((baseline_dir / "metadata.json").read_text())
    if [f["frame_id"] for f in baseline_meta["frames"]] != store.manifest["frame_ids"]:
        raise ValueError("Baseline source pool differs; cannot compare identical observations")
    with np.load(baseline_dir / "predictions.npz") as archive:
        old = {
            key: archive[key]
            for key in (
                "depth_world_points",
                "depth_confidence",
                "depth",
                "valid_mask",
                "source_to_model",
                "camera_to_world",
            )
        }

    # Evaluation accepts fewer inliers than loop construction, without altering loop constraints.
    config = replace(
        _config(preprocessing_dir), min_inliers=20, min_coverage=0.125, min_hull_area=0.025
    )
    matcher = _verifier(config)
    fids = store.manifest["frame_ids"]
    boundaries = {i for i in range(len(fids) - 1) if owners[i] != owners[i + 1]}
    indexes = sorted(set(range(0, len(fids) - 1, 4)) | boundaries)
    pairs, observations = [], []
    for i in indexes:
        a, b = fids[i : i + 2]
        xy_a, xy_b, verification = _matched_source_pixels(store, a, b, config, matcher, "benchmark")
        record = {
            "first": i,
            "second": i + 1,
            "boundary": i in boundaries,
            "image_matches": len(xy_a),
            "verification": verification,
        }
        if not len(xy_a):
            record["eligible_matches"] = 0
            pairs.append(record)
            continue
        old_a, old_b = _sample(old, i, xy_a), _sample(old, i + 1, xy_b)
        new_a, ca = store.sample(owners[i], a, xy_a)
        new_b, cb = store.sample(owners[i + 1], b, xy_b)
        new_a, new_b = transforms[owners[i]].apply(new_a), transforms[owners[i + 1]].apply(new_b)
        ca, cb = transforms[owners[i]].apply(ca), transforms[owners[i + 1]].apply(cb)
        good = np.logical_and.reduce([np.isfinite(p).all(1) for p in (old_a, old_b, new_a, new_b)])
        old_a, old_b, new_a, new_b = [p[good] for p in (old_a, old_b, new_a, new_b)]
        record["eligible_matches"] = int(good.sum())
        pairs.append(record)
        if not good.any():
            continue
        observations.append(
            {
                "pair": len(pairs) - 1,
                "old": np.linalg.norm(old_a - old_b, axis=1),
                "new": np.linalg.norm(new_a - new_b, axis=1),
                "old_distance": np.r_[
                    np.linalg.norm(old_a - old["camera_to_world"][i, :3, 3], axis=1),
                    np.linalg.norm(old_b - old["camera_to_world"][i + 1, :3, 3], axis=1),
                ],
                "new_distance": np.r_[
                    np.linalg.norm(new_a - ca, axis=1), np.linalg.norm(new_b - cb, axis=1)
                ],
            }
        )
        print(
            f"Benchmark frames {i}-{i + 1}: {int(good.sum())} identical eligible matches",
            flush=True,
        )
    if not observations:
        raise ValueError("No independently verified observations eligible in both reconstructions")
    old_scale = float(np.median(np.concatenate([o["old_distance"] for o in observations])))
    new_scale = float(np.median(np.concatenate([o["new_distance"] for o in observations])))
    old_all, new_all, pair_labels = [], [], []
    for observation in observations:
        record = pairs[observation["pair"]]
        record["old_normalized"] = _summary(observation["old"], old_scale)
        record["new_normalized"] = _summary(observation["new"], new_scale)
        old_all.append(observation["old"])
        new_all.append(observation["new"])
        pair_labels.append(np.full(len(observation["old"]), observation["pair"], np.int32))
    old_all, new_all, labels = map(np.concatenate, (old_all, new_all, pair_labels))
    boundary_mask = np.array([pairs[index]["boundary"] for index in labels])
    report = {
        "artifact_checks": artifact_checks,
        "protocol": "CPU ALIKED/LightGlue, index-paired observations; identical confidence-eligible support at both resolutions; every fourth adjacent pair plus owner boundaries",
        "evaluation_config": asdict(config),
        "old_scene_scale": old_scale,
        "new_scene_scale": new_scale,
        "all": {"old": _summary(old_all, old_scale), "new": _summary(new_all, new_scale)},
        "boundaries": {
            "old": _summary(old_all[boundary_mask], old_scale),
            "new": _summary(new_all[boundary_mask], new_scale),
        },
        "pairs": pairs,
        "held_out_loop_graph": _loop_graph_metrics(root, graph),
    }
    report["median_improvement_fraction"] = (
        1 - report["all"]["new"]["median"] / report["all"]["old"]["median"]
    )
    np.savez_compressed(
        root / "benchmark_observations.npz",
        old_disagreement=old_all,
        new_disagreement=new_all,
        pair_indices=labels,
        boundary=boundary_mask,
        old_scene_scale=old_scale,
        new_scene_scale=new_scale,
    )
    (root / "section_benchmark.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
