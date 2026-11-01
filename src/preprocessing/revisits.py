"""Optional real DINOv2 retrieval and ALIKED/LightGlue geometric verification."""

from collections import OrderedDict
from dataclasses import replace
from importlib.util import find_spec

import cv2
import numpy as np

from .contracts import Edge
from .storage import image_name, retrieve_frames
from .tracking import cell_ids


def revisits_available():
    return all(
        find_spec(name) is not None
        for name in ("torch", "torchvision", "transformers", "lightglue")
    )


def spatial_spread(points, shape, grid=4):
    coverage = len(np.unique(cell_ids(points, shape, grid))) / grid**2
    area = (
        float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32))))
        / (shape[0] * shape[1])
        if len(points) >= 3
        else 0.0
    )
    return coverage, area


def verify_matches(points_a, points_b, shape_a, shape_b, config):
    """Both image distributions must pass; many clustered inliers are insufficient."""
    if len(points_a) < max(8, config.min_inliers):
        return None
    matrix, mask = cv2.findFundamentalMat(
        np.asarray(points_a, np.float32),
        np.asarray(points_b, np.float32),
        cv2.USAC_MAGSAC,
        config.ransac_px,
        config.confidence,
        config.max_iterations,
    )
    if matrix is None or mask is None:
        return None
    valid = mask.ravel().astype(bool)
    count = int(valid.sum())
    if count < config.min_inliers or count / len(valid) < config.min_inlier_ratio:
        return None
    a, b = np.asarray(points_a)[valid], np.asarray(points_b)[valid]
    ca, aa = spatial_spread(a, shape_a)
    cb, ab = spatial_spread(b, shape_b)
    coverage = min(ca, cb)
    if coverage < config.min_coverage or min(aa, ab) < config.min_hull_area:
        return None
    return count, coverage, a, b


def unambiguous_matches(first, second, matches, config):
    """Lowe-style distance ratio against spatially distant SAME-image alternatives.

    ALIKED descriptors are explicitly L2-normalized. Compare squared Euclidean
    distances with ratio squared, on both sides. Equal-distance ties are ambiguous.
    Keypoints within radius pixels are excluded to avoid treating adjacent detections
    of the same physical feature as repeated structure.
    """
    import torch
    from torch.nn.functional import normalize

    if not len(matches):
        return torch.empty(0, dtype=torch.bool, device=matches.device)
    descriptors_a = normalize(first["descriptors"], dim=-1)
    descriptors_b = normalize(second["descriptors"], dim=-1)

    def best_distant_distance(descriptors, keypoints):
        distance = (2 - 2 * (descriptors @ descriptors.T)).clamp_min(0)
        spatial = torch.cdist(keypoints.float(), keypoints.float())
        distance.masked_fill_(spatial <= config.self_match_radius_px, float("inf"))
        return distance.min(dim=1).values

    own_a = best_distant_distance(descriptors_a, first["keypoints"])
    own_b = best_distant_distance(descriptors_b, second["keypoints"])
    cross = ((descriptors_a[matches[:, 0]] - descriptors_b[matches[:, 1]]) ** 2).sum(dim=-1)
    threshold = config.self_match_ratio**2
    return (cross < threshold * own_a[matches[:, 0]]) & (cross < threshold * own_b[matches[:, 1]])


def verify_feature_pair(first, second, matcher, shape_a, shape_b, config):
    """Shared real verification for the pipeline and standalone pair controls."""
    from lightglue.utils import rbd

    matches = rbd(matcher({"image0": first, "image1": second}))["matches"]
    first, second = rbd(first), rbd(second)
    a = first["keypoints"][matches[:, 0]].cpu().numpy()
    b = second["keypoints"][matches[:, 1]].cpu().numpy()
    keep = unambiguous_matches(first, second, matches, config).cpu().numpy()
    # The unfiltered result is diagnostic only: it never supplies accepted inliers.
    before = verify_matches(a, b, shape_a, shape_b, config)
    verified = verify_matches(a[keep], b[keep], shape_a, shape_b, config)
    ambiguous = int((~keep).sum())
    diagnostics = {
        "raw_matches": len(matches),
        "ambiguous_matches": ambiguous,
        "unambiguous_matches": int(keep.sum()),
        "inliers_before": before[0] if before is not None else 0,
        "inliers_after": verified[0] if verified is not None else 0,
        "accepted": verified is not None,
        "rejected_repetitive": before is not None and verified is None and ambiguous > 0,
    }
    return verified, diagnostics


def match_image_pair(path_a, path_b, config, *, diagnostics=None):
    """Run the real local verifier directly, also useful for independent negative checks."""
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )
    extractor = ALIKED(max_num_keypoints=config.keypoints).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    with torch.inference_mode():
        a_image = load_image(path_a, resize=config.image_long_side).to(device)
        b_image = load_image(path_b, resize=config.image_long_side).to(device)
        first, second = extractor.extract(a_image), extractor.extract(b_image)
        verified, details = verify_feature_pair(
            first, second, matcher, a_image.shape[-2:], b_image.shape[-2:], config
        )
    if diagnostics is not None:
        diagnostics.update(details)
    return verified


def find_revisits(clips, selected, scores, config, output_dir):
    cfg = config.revisits
    if not cfg.enabled:
        return [], {"status": "not run", "reason": "disabled in config"}, []
    if not revisits_available():
        return [], {"status": "not run", "reason": "revisits extra not installed"}, []
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image
    from transformers import AutoImageProcessor, AutoModel

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    geometry = [f for f in selected if f.pool == "geometry"]
    if len(geometry) < 2:
        return [], {"status": "run", "pairs_tested": 0, "accepted": 0}, []
    cache_dir = output_dir / ".cache" / "revisits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for frame in retrieve_frames(
        clips, [f.frame_id for f in geometry], replace(config.decode, long_side=cfg.image_long_side)
    ):
        path = cache_dir / image_name(frame.frame_id, "png")
        if not cv2.imwrite(str(path), cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write {path}")
        paths[frame.frame_id] = path
    processor = AutoImageProcessor.from_pretrained(cfg.model)
    model = AutoModel.from_pretrained(cfg.model).eval().to(device)
    descriptors = []
    with torch.inference_mode():
        for frame in geometry:
            rgb = cv2.cvtColor(cv2.imread(str(paths[frame.frame_id])), cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt").to(device)
            descriptor = model(**inputs).last_hidden_state[:, 0]
            descriptors.append(torch.nn.functional.normalize(descriptor, dim=-1).cpu().numpy()[0])
    del model, processor
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    similarity = np.stack(descriptors) @ np.stack(descriptors).T
    lookup = {r.frame_id: r for r in scores}
    pairs = set()
    for i, first in enumerate(geometry):
        eligible = []
        for j, second in enumerate(geometry):
            a, b = lookup[first.frame_id], lookup[second.frame_id]
            if i == j or (a.clip_id == b.clip_id and abs(a.t_s - b.t_s) < cfg.temporal_exclusion_s):
                continue
            if similarity[i, j] >= cfg.min_similarity:
                eligible.append(j)
        for j in sorted(eligible, key=lambda j: float(similarity[i, j]), reverse=True)[: cfg.top_k]:
            pairs.add(tuple(sorted((i, j))))
    extractor = ALIKED(max_num_keypoints=cfg.keypoints).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    cache = OrderedDict()

    def features(index):
        if index not in cache:
            image = load_image(paths[geometry[index].frame_id]).to(device)
            feature = extractor.extract(image)
            cache[index] = {k: v.cpu() if torch.is_tensor(v) else v for k, v in feature.items()}
            if len(cache) > cfg.feature_cache_size:
                cache.popitem(last=False)
        cache.move_to_end(index)
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in cache[index].items()}

    edges, debug, pair_diagnostics = [], [], []
    with torch.inference_mode():
        for i, j in sorted(pairs):
            first, second = features(i), features(j)
            shape_a = cv2.imread(str(paths[geometry[i].frame_id])).shape[:2]
            shape_b = cv2.imread(str(paths[geometry[j].frame_id])).shape[:2]
            verified, details = verify_feature_pair(first, second, matcher, shape_a, shape_b, cfg)
            details.update(
                a=geometry[i].frame_id,
                b=geometry[j].frame_id,
                t_a_s=lookup[geometry[i].frame_id].t_s,
                t_b_s=lookup[geometry[j].frame_id].t_s,
            )
            pair_diagnostics.append(details)
            if verified is None:
                continue
            count, coverage, a, b = verified
            edges.append(
                Edge(
                    geometry[i].frame_id,
                    geometry[j].frame_id,
                    "revisit",
                    count / max(1, details["unambiguous_matches"]) * coverage,
                    count,
                    coverage,
                    details["ambiguous_matches"],
                )
            )
            debug.append(
                {
                    "a": geometry[i].frame_id,
                    "b": geometry[j].frame_id,
                    "points_a": a[:200].tolist(),
                    "points_b": b[:200].tolist(),
                    "t_a_s": lookup[geometry[i].frame_id].t_s,
                    "t_b_s": lookup[geometry[j].frame_id].t_s,
                    "ambiguous_matches": details["ambiguous_matches"],
                    "image_path": f"debug/revisit_{image_name(geometry[i].frame_id, 'jpg')[:-4]}__{image_name(geometry[j].frame_id, 'jpg')}",
                }
            )
    return (
        edges,
        {
            "status": "run",
            "device": device,
            "model": cfg.model,
            "pairs_tested": len(pairs),
            "accepted": len(edges),
            "rejected_repetitive": [d for d in pair_diagnostics if d["rejected_repetitive"]],
            "pair_diagnostics": pair_diagnostics,
        },
        debug,
    )
