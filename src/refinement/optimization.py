"""Robust pose/depth adjustment with whole-track validation and untouched test tracks."""

import copy
from itertools import pairwise

import numpy as np

from .contracts import RefinementResult
from .inputs import sample_image, world_observations


def _stats(values):
    values = np.asarray(values)
    if not len(values):
        return {"count": 0, "median": None, "p90": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def observation_pairs(tracks, split=None):
    """Consecutive observations of each landmark, preserving exact evaluation support."""
    first, second = [], []
    order = np.lexsort((tracks.frame_indices, tracks.track_indices))
    boundaries = np.flatnonzero(np.r_[True, np.diff(tracks.track_indices[order]) != 0, True])
    for begin, end in pairwise(boundaries):
        indices = order[begin:end]
        if split is not None and tracks.split[indices[0]] != split:
            continue
        indices = indices[np.argsort(tracks.frame_indices[indices])]
        first.extend(indices[:-1])
        second.extend(indices[1:])
    return np.asarray(first, int), np.asarray(second, int)


def scene_distance(scene, tracks):
    depths = []
    for frame in np.unique(tracks.frame_indices):
        uv = tracks.pixel_xy[tracks.frame_indices == frame]
        rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(scene.intrinsics[frame]).T
        depths.extend(np.linalg.norm(rays, axis=1) * sample_image(scene.depth[frame], uv))
    return float(np.median(depths))


def evaluate_refinement(scene, tracks, result=None, split=2, *, per_pair=True):
    scale = scene_distance(scene, tracks)
    rotations = scene.rotations if result is None else result.rotations
    centers = scene.centers if result is None else result.centers
    points = world_observations(
        scene,
        tracks.frame_indices,
        tracks.pixel_xy,
        rotations,
        centers,
        None if result is None else result.log_scales,
        None if result is None else result.log_depth_fields,
    )
    a, b = observation_pairs(tracks, split)
    distance = np.linalg.norm(points[a] - points[b], axis=1)
    reprojections = []
    behind = 0
    for source, target in ((a, b), (b, a)):
        frames = tracks.frame_indices[target]
        camera = np.einsum(
            "nij,nj->ni", rotations[frames].transpose(0, 2, 1), points[source] - centers[frames]
        )
        uvz = np.einsum("nij,nj->ni", scene.intrinsics[frames], camera)
        behind += int((camera[:, 2] <= 0).sum())
        projected = uvz[:, :2] / np.maximum(camera[:, 2, None], 1e-8)
        error = np.linalg.norm(projected - tracks.pixel_xy[target], axis=1)
        reprojections.append(error)
    current_distance = float(
        np.median(np.linalg.norm(points - centers[tracks.frame_indices], axis=1))
    )
    report = {
        "fixed_input_scene_distance": scale,
        "current_scene_distance": current_distance,
        "disagreement_normalized": _stats(distance / scale),
        "disagreement_current_normalized": _stats(distance / current_distance),
        "reprojection_pixels": _stats(np.concatenate(reprojections)),
        "behind_camera_projections": behind,
    }
    if per_pair:
        pairs = np.column_stack((tracks.frame_indices[a], tracks.frame_indices[b]))
        report["pairs"] = []
        for pair in np.unique(pairs, axis=0):
            selected = (pairs == pair).all(1)
            report["pairs"].append(
                {
                    "first": int(pair[0]),
                    "second": int(pair[1]),
                    **_stats(distance[selected] / scale),
                }
            )
    return report


def _gauge_anchors(tracks, count):
    parent = list(range(count))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    a, b = observation_pairs(tracks, 0)
    for i, j in zip(tracks.frame_indices[a], tracks.frame_indices[b], strict=True):
        ri, rj = root(int(i)), root(int(j))
        parent[max(ri, rj)] = min(ri, rj)
    anchors = sorted({root(i) for i in range(count)})
    support = np.bincount(tracks.frame_indices[tracks.split == 0], minlength=count)
    return anchors, support


def optimize_multiview(scene, tracks, config):
    """Fit training tracks; choose pose/field candidate using validation tracks only."""
    import torch
    from torch.nn import functional

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    dtype = torch.float64
    scale = scene_distance(scene, tracks)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot establish a positive observation-distance scale")
    train = tracks.split == 0
    frames = tracks.frame_indices[train]
    uv = tracks.pixel_xy[train]
    tids = tracks.track_indices[train]
    unique, tids = np.unique(tids, return_inverse=True)
    if len(unique) < 8:
        raise ValueError("At least eight independent training tracks are required")
    anchors, support = _gauge_anchors(tracks, scene.frame_count)

    def tensor(array, kind=dtype):
        return torch.as_tensor(array, dtype=kind, device=device)

    fids = tensor(frames, torch.long)
    track_ids = tensor(tids, torch.long)
    pixels = tensor(uv)
    depths, scores, rays = np.empty(len(frames)), np.empty(len(frames)), np.empty((len(frames), 3))
    for frame in np.unique(frames):
        selected = frames == frame
        depths[selected] = sample_image(scene.depth[frame], uv[selected])
        scores[selected] = sample_image(scene.confidence[frame], uv[selected])
        scores[selected] /= max(float(np.median(scores[selected])), 1e-8)
        rays[selected] = (
            np.column_stack((uv[selected], np.ones(selected.sum())))
            @ np.linalg.inv(scene.intrinsics[frame]).T
        )
    weights = np.clip(scores, 0.25, 2) / np.sqrt(np.maximum(support[frames], 1))
    weights[tracks.feature_indices[train] < 0] *= 0.5
    weights /= weights.mean()
    weights = tensor(weights)
    ray_depth = tensor(rays * depths[:, None])
    initial_r, initial_c, intrinsics = (
        tensor(scene.rotations),
        tensor(scene.centers),
        tensor(scene.intrinsics),
    )
    mutable = np.ones(scene.frame_count)
    mutable[anchors] = 0
    mutable = tensor(mutable)
    observed = tensor((support > 0).astype(float))

    bounds = scene.content_bounds[frames]
    normalized = np.clip((uv - bounds[:, :2]) / np.maximum(bounds[:, 2:] - bounds[:, :2], 1), 0, 1)
    grid_xy = normalized * (config.grid_size - 1)
    lo = np.minimum(grid_xy.astype(int), config.grid_size - 2)
    fraction = grid_xy - lo
    grid_indices = tensor(
        np.stack(
            [
                lo[:, 1] * config.grid_size + lo[:, 0],
                lo[:, 1] * config.grid_size + lo[:, 0] + 1,
                (lo[:, 1] + 1) * config.grid_size + lo[:, 0],
                (lo[:, 1] + 1) * config.grid_size + lo[:, 0] + 1,
            ],
            axis=1,
        ),
        torch.long,
    )
    x, y = fraction.T
    grid_weights = tensor(np.stack([(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y], axis=1))

    # Dense initialization prior: restrain unobserved surfaces, not just pose parameters.
    prior_frames, prior_pixels = [], []
    for frame, (x0, y0, x1, y1) in enumerate(scene.content_bounds):
        px, py = np.meshgrid(np.linspace(x0 + 1, x1 - 1, 12), np.linspace(y0 + 1, y1 - 1, 12))
        pixels_grid = np.column_stack((px.ravel(), py.ravel()))
        keep = sample_image(scene.valid_mask[frame], pixels_grid) > 0.999
        prior_frames.extend([frame] * int(keep.sum()))
        prior_pixels.extend(pixels_grid[keep])
    prior_frames = np.asarray(prior_frames, int)
    prior_pixels = np.asarray(prior_pixels)
    prior_world = world_observations(scene, prior_frames, prior_pixels)
    prior_ray_depth = np.einsum(
        "nij,nj->ni",
        scene.rotations[prior_frames].transpose(0, 2, 1),
        prior_world - scene.centers[prior_frames],
    )
    pb = scene.content_bounds[prior_frames]
    pg = np.clip((prior_pixels - pb[:, :2]) / np.maximum(pb[:, 2:] - pb[:, :2], 1), 0, 1) * (
        config.grid_size - 1
    )
    pl = np.minimum(pg.astype(int), config.grid_size - 2)
    px, py = (pg - pl).T
    prior_ids = tensor(
        np.stack(
            (
                pl[:, 1] * config.grid_size + pl[:, 0],
                pl[:, 1] * config.grid_size + pl[:, 0] + 1,
                (pl[:, 1] + 1) * config.grid_size + pl[:, 0],
                (pl[:, 1] + 1) * config.grid_size + pl[:, 0] + 1,
            ),
            axis=1,
        ),
        torch.long,
    )
    prior_weights = tensor(
        np.stack(((1 - px) * (1 - py), px * (1 - py), (1 - px) * py, px * py), axis=1)
    )
    prior_fids, prior_world, prior_ray_depth = (
        tensor(prior_frames, torch.long),
        tensor(prior_world),
        tensor(prior_ray_depth),
    )

    rotation = torch.nn.Parameter(torch.zeros((scene.frame_count, 3), dtype=dtype, device=device))
    translation = torch.nn.Parameter(torch.zeros_like(rotation))
    log_scale = torch.nn.Parameter(torch.zeros(scene.frame_count, dtype=dtype, device=device))
    field = torch.nn.Parameter(
        torch.zeros(
            (scene.frame_count, config.grid_size, config.grid_size), dtype=dtype, device=device
        )
    )

    def parameters():
        rv = rotation * mutable[:, None]
        zero = torch.zeros_like(rv[:, 0])
        rx, ry, rz = rv.unbind(1)
        skew = torch.stack([zero, -rz, ry, rz, zero, -rx, -ry, rx, zero], dim=1).reshape(-1, 3, 3)
        rotations = torch.matrix_exp(skew) @ initial_r
        centers = initial_c + scale * translation * mutable[:, None]
        scales = log_scale * mutable
        fields = (field - field.mean(dim=(1, 2), keepdim=True)) * observed[:, None, None]
        return rotations, centers, scales, fields

    def snapshot():
        with torch.no_grad():
            r, c, s, g = parameters()
            return RefinementResult(*(v.detach().cpu().numpy().copy() for v in (r, c, s, g)), {})

    def loss():
        r, c, s, g = parameters()
        correction = s[fids] + (
            g.reshape(scene.frame_count, -1)[fids[:, None], grid_indices] * grid_weights
        ).sum(1)
        world = torch.einsum("nij,nj->ni", r[fids], ray_depth * correction.exp()[:, None]) + c[fids]
        total = torch.zeros((len(unique), 3), dtype=dtype, device=device).index_add_(
            0, track_ids, world
        )
        count = torch.bincount(track_ids, minlength=len(unique)).to(dtype)
        mean = total / count[:, None]
        delta = (world - mean[track_ids]) / (0.02 * scale)
        geometric = (
            functional.huber_loss(delta, torch.zeros_like(delta), reduction="none").mean(1)
            * weights
        ).mean()

        camera = torch.einsum("nji,nj->ni", r[fids], mean[track_ids] - c[fids])
        projected = torch.einsum("nij,nj->ni", intrinsics[fids], camera)
        pixel_error = (projected[:, :2] / camera[:, 2, None].clamp_min(1e-5) - pixels) / 3
        reprojection = (
            functional.huber_loss(
                pixel_error, torch.zeros_like(pixel_error), reduction="none"
            ).mean(1)
            * weights
        ).mean()
        prior = 0.01 * ((rotation / 0.15).square().mean() + (translation / 0.15).square().mean())
        prior += 0.02 * (s / 0.25).square().mean() + 0.04 * (g / 0.15).square().mean()
        prior += 0.02 * (
            ((g[:, 1:] - g[:, :-1]) / 0.07).square().mean()
            + ((g[:, :, 1:] - g[:, :, :-1]) / 0.07).square().mean()
        )
        pc = s[prior_fids] + (
            g.reshape(scene.frame_count, -1)[prior_fids[:, None], prior_ids] * prior_weights
        ).sum(1)
        pw = (
            torch.einsum("nij,nj->ni", r[prior_fids], prior_ray_depth * pc.exp()[:, None])
            + c[prior_fids]
        )
        displacement = (pw - prior_world) / (0.05 * scale)
        prior += config.geometry_prior_weight * functional.huber_loss(
            displacement, torch.zeros_like(displacement), reduction="mean"
        )
        return geometric + 0.2 * reprojection + prior

    initial = snapshot()
    before_validation = evaluate_refinement(scene, tracks, initial, split=1, per_pair=False)

    def score(metrics):
        d = metrics["disagreement_normalized"]
        return d["median"] + 0.25 * d["p90"] + 0.0005 * metrics["reprojection_pixels"]["median"]

    best_score, best = score(before_validation), initial
    best_state = [p.detach().clone() for p in (rotation, translation, log_scale, field)]
    history = [{"stage": "initial", "step": 0, "validation": before_validation}]
    selected = "initial"
    for stage, steps in (("pose_scale", config.pose_steps), ("smooth_depth", config.field_steps)):
        if not steps:
            continue
        field.requires_grad_(stage == "smooth_depth")
        optimized = [rotation, translation, log_scale] + ([field] if field.requires_grad else [])
        optimizer = torch.optim.Adam(optimized, lr=config.learning_rate)
        for step in range(steps):
            optimizer.zero_grad()
            value = loss()
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite refinement objective")
            value.backward()
            torch.nn.utils.clip_grad_norm_(optimized, 20)
            optimizer.step()
            with torch.no_grad():
                rotation.clamp_(-0.35, 0.35)
                translation.clamp_(-0.5, 0.5)
                log_scale.clamp_(-0.69, 0.69)
                field.clamp_(-0.4, 0.4)
            if (step + 1) % 100 == 0 or step == steps - 1:
                candidate = snapshot()
                metrics = evaluate_refinement(scene, tracks, candidate, split=1, per_pair=False)
                history.append(
                    {
                        "stage": stage,
                        "step": step + 1,
                        "training_loss": float(value.detach()),
                        "validation": metrics,
                    }
                )
                candidate_score = score(metrics)
                if candidate_score < best_score and metrics["behind_camera_projections"] == 0:
                    best_score, best, selected = candidate_score, candidate, stage
                    best_state = [
                        p.detach().clone() for p in (rotation, translation, log_scale, field)
                    ]
                print(
                    f"{stage} {step + 1}/{steps}: loss={value.item():.4f}, validation median/p90="
                    f"{metrics['disagreement_normalized']['median']:.5f}/{metrics['disagreement_normalized']['p90']:.5f}",
                    flush=True,
                )
        with torch.no_grad():
            for parameter, value in zip(
                (rotation, translation, log_scale, field), best_state, strict=True
            ):
                parameter.copy_(value)

    report = {
        "selected_candidate": selected,
        "gauge_anchor_frames": anchors,
        "training_observations_by_frame": support.tolist(),
        "unsupported_frames": np.flatnonzero(support == 0).tolist(),
        "checkpoint_selection": "validation tracks only; test tracks never select parameters",
        "history": history,
        "metrics": {},
    }
    for name, split in (("train", 0), ("validation", 1), ("test", 2)):
        report["metrics"][name] = {
            "before": evaluate_refinement(scene, tracks, split=split),
            "after": evaluate_refinement(scene, tracks, best, split=split),
        }
    report["corrections"] = {
        "camera_displacement_normalized": _stats(
            np.linalg.norm(best.centers - scene.centers, axis=1) / scale
        ),
        "depth_scale_range": np.exp(best.log_scales).tolist(),
        "spatial_depth_multiplier_range": [
            float(np.exp(best.log_depth_fields).min()),
            float(np.exp(best.log_depth_fields).max()),
        ],
    }
    return RefinementResult(
        best.rotations, best.centers, best.log_scales, best.log_depth_fields, copy.deepcopy(report)
    )
