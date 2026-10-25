"""Real feature matching, nonlocal retrieval, conflict-free tracks and disjoint splits."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .contracts import ObservationTracks
from .inputs import sample_image


def split_tracks(track_indices, seed=17):
    """Reserve whole landmarks; a feature observation never occurs in two splits."""
    tracks = np.unique(track_indices)
    shuffled = np.random.default_rng(seed).permutation(tracks)
    assignment = np.zeros(int(tracks.max()) + 1, np.int8)
    count = max(1, int(len(tracks) * 0.15))
    assignment[shuffled[:count]] = 2
    assignment[shuffled[count : 2 * count]] = 1
    return assignment[track_indices]


def link_tracks(matches):
    """Union exact (frame, feature) identities, rejecting conflicting same-frame merges."""
    parent, members = {}, {}

    def root(key):
        if key not in parent:
            parent[key] = key
            members[key] = {key[0]: key[1]}
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    conflicts = 0
    for first, second in matches:
        a, b = root(first), root(second)
        if a == b:
            continue
        if any(
            frame in members[b] and members[b][frame] != feature
            for frame, feature in members[a].items()
        ):
            conflicts += 1
            continue
        if len(members[a]) < len(members[b]):
            a, b = b, a
        parent[b] = a
        members[a].update(members.pop(b))
    groups = sorted(
        (sorted(v.items()) for v in members.values() if len(v) >= 2), key=lambda x: x[0]
    )
    return groups, conflicts


def _source_path(scene, index, input_dir):
    frame = scene.frames[index]
    direct = Path(frame.get("image_path") or "")
    if direct.is_file():
        return direct.resolve()
    relative = (frame.get("selected_frame") or {}).get("image_path")
    if relative:
        alternative = Path(input_dir).parent / "preprocessing" / relative
        if alternative.is_file():
            return alternative.resolve()
    raise FileNotFoundError(f"Cannot locate source image for {frame['frame_id']}")


def _retrieve(scene, paths, config):
    if not config.retrieval_top_k:
        return set()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(config.device)
    descriptors = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                batch = processor(images=image.convert("RGB"), return_tensors="pt").to(
                    config.device
                )
            descriptor = model(**batch).last_hidden_state[:, 0]
            descriptors.append(torch.nn.functional.normalize(descriptor, dim=-1)[0].cpu().numpy())
    del model, processor
    similarity = np.stack(descriptors) @ np.stack(descriptors).T
    pairs = set()
    for i, first in enumerate(scene.frames):
        eligible = []
        for j, second in enumerate(scene.frames):
            same_clip = first.get("clip_id") == second.get("clip_id")
            ta = first.get("timestamp_s") if first.get("timestamp_s") is not None else i
            tb = second.get("timestamp_s") if second.get("timestamp_s") is not None else j
            if i == j or (same_clip and abs(ta - tb) < config.retrieval_exclusion_s):
                continue
            if similarity[i, j] >= 0.75:
                eligible.append(j)
        for j in sorted(eligible, key=lambda j: float(similarity[i, j]), reverse=True)[
            : config.retrieval_top_k
        ]:
            pairs.add(tuple(sorted((i, j))))
    return pairs


def build_observation_tracks(scene, input_dir, output_dir, config):
    """Persist genuine image evidence and independently split complete multi-frame tracks."""
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image
    from scipy.spatial import cKDTree

    from ..preprocessing.config import RevisitConfig
    from ..preprocessing.revisits import verify_feature_pair

    root = Path(output_dir)
    cache = root / "matches"
    cache.mkdir(parents=True, exist_ok=True)
    paths = [_source_path(scene, i, input_dir) for i in range(scene.frame_count)]
    signature = {
        "version": 1,
        "scene": scene.fingerprint,
        "keypoints": config.keypoints,
        "feature_size": config.feature_size,
        "retrieval_top_k": config.retrieval_top_k,
        "retrieval_exclusion_s": config.retrieval_exclusion_s,
        "neighbor_offsets": config.neighbor_offsets,
        "images": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
    }
    token = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    verifier = replace(
        RevisitConfig(),
        min_inliers=20,
        keypoints=config.keypoints,
        image_long_side=config.feature_size,
    )

    candidate_file = cache / f"{token}_candidates.json"
    if candidate_file.is_file():
        pairs = {tuple(p) for p in json.loads(candidate_file.read_text())}
    else:
        pairs = _retrieve(scene, paths, config)
    pairs |= {
        (i, i + offset)
        for offset in config.neighbor_offsets
        for i in range(scene.frame_count - offset)
    }
    candidate_file.write_text(json.dumps(sorted(pairs)))

    extractor = ALIKED(max_num_keypoints=config.keypoints).eval().to(config.device)
    matcher = LightGlue(features="aliked").eval().to(config.device)
    features, shapes, source_xy, model_xy = {}, {}, {}, {}
    with torch.inference_mode():
        for index, path in enumerate(paths):
            file = cache / f"{token}_features_{index:05d}.npz"
            if file.is_file():
                with np.load(file, allow_pickle=False) as data:
                    shape = tuple(data["shape"])
                    feature = {
                        k: torch.from_numpy(data[k]).to(config.device)
                        for k in data.files
                        if k != "shape"
                    }
            else:
                image = load_image(str(path), resize=config.feature_size).to(config.device)
                shape = tuple(image.shape[-2:])
                feature = extractor.extract(image)
                np.savez_compressed(
                    file, shape=shape, **{k: v.cpu().numpy() for k, v in feature.items()}
                )
            features[index], shapes[index] = feature, shape
            from PIL import Image

            with Image.open(path) as image:
                size = np.asarray(image.size)
            xy = feature["keypoints"][0].cpu().numpy()
            source_xy[index] = (xy + 0.5) * (size / np.asarray(shape[::-1])) - 0.5
            matrix = scene.source_to_model[index]
            model_xy[index] = source_xy[index] @ matrix[:2, :2].T + matrix[:2, 2]

    links, records, accepted_pairs = [], [], []
    for position, (a, b) in enumerate(sorted(pairs, key=lambda p: (p[1] - p[0], p[0]))):
        file = cache / f"{token}_pair_{a:05d}_{b:05d}.npz"
        if file.is_file():
            with np.load(file, allow_pickle=False) as data:
                ia, ib = data["a"], data["b"]
                details = json.loads(str(data["details"]))
        else:
            with torch.inference_mode():
                verified, details = verify_feature_pair(
                    features[a], features[b], matcher, shapes[a], shapes[b], verifier
                )
            ia, ib = np.empty(0, int), np.empty(0, int)
            if verified is not None:
                ia = cKDTree(features[a]["keypoints"][0].cpu().numpy()).query(verified[2])[1]
                ib = cKDTree(features[b]["keypoints"][0].cpu().numpy()).query(verified[3])[1]
            np.savez_compressed(file, a=ia, b=ib, details=json.dumps(details))

        good = np.ones(len(ia), bool)
        for frame, feature_ids in ((a, ia), (b, ib)):
            xy = model_xy[frame][feature_ids]
            valid = sample_image(scene.valid_mask[frame], xy) > 0.999
            valid &= np.isfinite(sample_image(scene.depth[frame], xy))
            good &= valid
        ia, ib = ia[good], ib[good]
        if len(ia) >= 8:
            accepted_pairs.append((a, b))
            links.extend(((a, int(i)), (b, int(j))) for i, j in zip(ia, ib, strict=True))
        records.append({"first": a, "second": b, "depth_valid_matches": len(ia), **details})
        if position % 40 == 0:
            print(
                f"Feature graph {position + 1}/{len(pairs)} pairs; {len(accepted_pairs)} supported",
                flush=True,
            )

    groups, conflicts = link_tracks(links)
    if len(groups) < 20:
        raise ValueError("Too few verified multi-frame tracks to refine this scene")
    fids, pixels, tids, kids = [], [], [], []
    for track, group in enumerate(groups):
        for frame, feature in group:
            fids.append(frame)
            pixels.append(model_xy[frame][feature])
            tids.append(track)
            kids.append(feature)
    tids = np.asarray(tids, np.int32)
    tracks = ObservationTracks(
        np.asarray(fids, np.int32),
        np.asarray(pixels, np.float32),
        tids,
        split_tracks(tids, config.seed),
        np.asarray(accepted_pairs, np.int32).reshape(-1, 2),
        np.asarray(kids, np.int32),
    )
    np.savez_compressed(root / "tracks.npz", **asdict(tracks), scene_fingerprint=scene.fingerprint)
    summary = {
        "scene_fingerprint": scene.fingerprint,
        "matcher_signature": signature,
        "candidate_pairs": len(pairs),
        "supported_pairs": len(accepted_pairs),
        "track_count": len(groups),
        "observation_count": len(fids),
        "conflicting_merges_rejected": conflicts,
        "pairs": records,
        "observations_by_frame": np.bincount(
            tracks.frame_indices, minlength=scene.frame_count
        ).tolist(),
    }
    (root / "tracks.json").write_text(json.dumps(summary, indent=2) + "\n")
    return tracks


def load_observation_tracks(path, scene):
    with np.load(path, allow_pickle=False) as data:
        if str(data["scene_fingerprint"]) != scene.fingerprint:
            raise ValueError("Cached tracks do not belong to these source predictions")
        tracks = ObservationTracks(**{k: data[k] for k in ObservationTracks.__dataclass_fields__})
    for track in np.unique(tracks.track_indices):
        mask = tracks.track_indices == track
        if len(np.unique(tracks.split[mask])) != 1:
            raise ValueError("Feature track leaks across evaluation splits")
        if len(np.unique(tracks.frame_indices[mask])) != mask.sum():
            raise ValueError("Feature track contains conflicting observations in one frame")
    return tracks
