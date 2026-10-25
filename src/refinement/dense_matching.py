"""Detector-free image anchors across weakly connected feature-observation groups."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .contracts import ObservationTracks
from .inputs import sample_image
from .optimization import observation_pairs, scene_distance


def _components(tracks, count):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a, b = observation_pairs(tracks, 0)
    graph = coo_matrix(
        (np.ones(len(a)), (tracks.frame_indices[a], tracks.frame_indices[b])), shape=(count, count)
    )
    return connected_components(graph, directed=False)[1]


def augment_dense_tracks(scene, tracks, output_dir, config, candidate_report=None):
    """Use real indoor LoFTR; original validation/test feature locations remain excluded."""
    import torch
    from kornia.feature import LoFTR
    from kornia.feature.loftr.loftr import default_cfg

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(scene.fingerprint.encode())
    for value in asdict(tracks).values():
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(b"loftr-indoor-new-v2-confidence0.5-reciprocal-protect10")
    token = digest.hexdigest()[:16]
    cache = root / "dense_matches"
    cache.mkdir(exist_ok=True)
    final = cache / f"tracks_{token}.npz"
    if final.is_file():
        from .matching import load_observation_tracks

        return load_observation_tracks(final, scene)

    labels = _components(tracks, scene.frame_count)
    count = np.bincount(tracks.frame_indices, minlength=scene.frame_count)
    candidates = {(i, i + d) for d in (1, 2, 3, 4, 8) for i in range(scene.frame_count - d)}
    if candidate_report and Path(candidate_report).is_file():
        previous = json.loads(Path(candidate_report).read_text())
        candidates |= {(p["first"], p["second"]) for p in previous["pairs"]}
    distance = scene_distance(scene, tracks)
    for a in range(scene.frame_count):
        for b in range(a + 1, scene.frame_count):
            if (
                labels[a] != labels[b]
                and scene.rotations[a, :, 2] @ scene.rotations[b, :, 2] > 0.5
                and np.linalg.norm(scene.centers[a] - scene.centers[b]) < 0.75 * distance
            ):
                candidates.add((a, b))
    candidates = {p for p in candidates if labels[p[0]] != labels[p[1]]}
    selected_pairs = set()
    label_pairs = {tuple(sorted(labels[list(p)])) for p in candidates}
    for label_pair in sorted(label_pairs):
        relevant = [p for p in candidates if tuple(sorted(labels[list(p)])) == label_pair]

        def rank(pair):
            a, b = pair
            angle = 1 - float(scene.rotations[a, :, 2] @ scene.rotations[b, :, 2])
            separation = np.linalg.norm(scene.centers[a] - scene.centers[b]) / distance
            return angle + 0.3 * separation + 0.002 * min(abs(a - b), 30)

        selected_pairs.update(sorted(relevant, key=rank)[:6])
        selected_pairs.update(sorted(relevant, key=lambda p: abs(p[1] - p[0]))[:4])

    cfg = copy.deepcopy(default_cfg)
    cfg["coarse"]["temp_bug_fix"] = True
    model = LoFTR(pretrained=None, config=cfg).eval().to(config.device)
    url = "https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    images, gray, sizes, allowed = {}, {}, {}, {}
    for frame in {i for pair in selected_pairs for i in pair}:
        x0, y0, x1, y1 = scene.content_bounds[frame]
        crop = scene.images[frame, y0 : y1 + 1, x0 : x1 + 1]
        h, w = crop.shape[:2]
        height, width = max(8, round(h / 8) * 8), max(8, round(w / 8) * 8)
        image = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (width, height))
        images[frame] = torch.from_numpy(image.astype(np.float32) / 255)[None, None].to(
            config.device
        )
        gray[frame] = cv2.cvtColor(scene.images[frame], cv2.COLOR_RGB2GRAY)
        sizes[frame] = np.array([w / width, h / height])
        mask = cv2.erode(scene.valid_mask[frame].astype(np.uint8), np.ones((5, 5), np.uint8))
        for point in tracks.pixel_xy[(tracks.frame_indices == frame) & (tracks.split != 0)]:
            cv2.circle(mask, tuple(np.rint(point).astype(int)), 10, 0, -1)
        allowed[frame] = mask

    records, extra_frames, extra_xy, extra_ids, pair_list = [], [], [], [], []
    offset = int(tracks.track_indices.max()) + 1
    rng = np.random.default_rng(config.seed)
    for number, (a, b) in enumerate(sorted(selected_pairs)):
        file = cache / f"{token}_{a:05d}_{b:05d}.npz"
        if file.is_file():
            with np.load(file, allow_pickle=False) as data:
                xa, xb = data["a"], data["b"]
        else:
            with torch.inference_mode():
                prediction = model({"image0": images[a], "image1": images[b]})
            with torch.inference_mode():
                reverse = model({"image0": images[b], "image1": images[a]})
            keep = prediction["confidence"].cpu().numpy() >= 0.5
            xa, xb = [prediction[key].cpu().numpy()[keep] for key in ("keypoints0", "keypoints1")]
            from scipy.spatial import cKDTree

            if len(xa) and len(reverse["keypoints1"]):
                ra, rb = [reverse[key].cpu().numpy() for key in ("keypoints1", "keypoints0")]
                delta, nearest = cKDTree(ra).query(xa)
                reciprocal = (delta < 1.5) & (np.linalg.norm(rb[nearest] - xb, axis=1) < 1.5)
                xa, xb = xa[reciprocal], xb[reciprocal]
            else:
                xa, xb = np.empty((0, 2)), np.empty((0, 2))
            xa = (xa + 0.5) * sizes[a] - 0.5 + scene.content_bounds[a, :2]
            xb = (xb + 0.5) * sizes[b] - 0.5 + scene.content_bounds[b, :2]
            np.savez_compressed(file, a=xa, b=xb)
        valid = (sample_image(allowed[a], xa) > 0.999) & (sample_image(allowed[b], xb) > 0.999)
        xa, xb = xa[valid], xb[valid]
        record = {"first": a, "second": b, "eligible": len(xa), "accepted": 0}
        if len(xa) >= 30:
            matrix, inliers = cv2.findFundamentalMat(xa, xb, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
            if matrix is not None and inliers is not None:
                xa, xb = xa[inliers.ravel().astype(bool)], xb[inliers.ravel().astype(bool)]
                from ..preprocessing.revisits import spatial_spread

                spread = [
                    spatial_spread(
                        xy - scene.content_bounds[f, :2],
                        tuple(
                            (scene.content_bounds[f, 2:] - scene.content_bounds[f, :2] + 1)[::-1]
                        ),
                    )
                    for f, xy in ((a, xa), (b, xb))
                ]
                if (
                    len(xa) >= 30
                    and min(s[0] for s in spread) >= 0.1875
                    and min(s[1] for s in spread) >= 0.03
                ):
                    chosen = rng.choice(len(xa), min(len(xa), 400), replace=False)
                    xa, xb = xa[chosen], xb[chosen]
                    for pa, pb in zip(xa, xb, strict=True):
                        extra_frames.extend((a, b))
                        extra_xy.extend((pa, pb))
                        extra_ids.extend((offset, offset))
                        offset += 1
                    pair_list.append((a, b))
                    record["accepted"] = len(xa)
        records.append(record)
        print(
            f"Dense image anchors {number + 1}/{len(selected_pairs)}: {a}/{b}, {record['accepted']} accepted",
            flush=True,
        )

    if not extra_frames:
        result = tracks
    else:
        count = len(extra_frames)
        result = ObservationTracks(
            np.r_[tracks.frame_indices, extra_frames],
            np.concatenate((tracks.pixel_xy, np.asarray(extra_xy, np.float32))),
            np.r_[tracks.track_indices, extra_ids],
            np.r_[tracks.split, np.zeros(count, np.int8)],
            np.unique(np.concatenate((tracks.pairs, np.asarray(pair_list, np.int32))), axis=0),
            np.r_[tracks.feature_indices, np.full(count, -2, np.int32)],
        )
    np.savez_compressed(final, **asdict(result), scene_fingerprint=scene.fingerprint)
    report = {
        "model": url,
        "min_confidence": 0.5,
        "reciprocal_tolerance_matcher_pixels": 1.5,
        "protected_radius": 10,
        "components_before": int(labels.max()) + 1,
        "components_after": int(_components(result, scene.frame_count).max()) + 1,
        "added_observations": len(extra_frames),
        "pairs": records,
    }
    (root / "dense_tracks.json").write_text(json.dumps(report, indent=2) + "\n")
    return result
