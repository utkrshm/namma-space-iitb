"""ALIKED keypoints, DINOv2 global descriptors and LightGlue matches, all cached on disk.

Keypoints are in ``frames/images`` pixel coordinates with integer pixel centres.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..config import SfmConfig


def extract_features(images: Path, names: list[str], cache: Path, config: SfmConfig, device: str):
    """Write ``<cache>/<stem>.npz`` (keypoints, scores, float16 descriptors, image size)."""
    import torch
    from lightglue import ALIKED

    cache.mkdir(parents=True, exist_ok=True)
    extractor = None
    for name in names:
        path = cache / (Path(name).stem + ".npz")
        if path.exists():
            continue
        if extractor is None:
            extractor = ALIKED(
                max_num_keypoints=config.max_keypoints,
                detection_threshold=config.detection_threshold,
            )
            extractor = extractor.eval().to(device)
        rgb = cv2.cvtColor(cv2.imread(str(images / name)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(device)
        with torch.inference_mode():
            feature = extractor.extract(tensor, resize=config.feature_long_side)
        np.savez(
            path,
            keypoints=feature["keypoints"][0].cpu().numpy().astype(np.float32),
            scores=feature["keypoint_scores"][0].cpu().numpy().astype(np.float32),
            descriptors=feature["descriptors"][0].cpu().numpy().astype(np.float16),
            image_size=np.array(rgb.shape[1::-1], np.float32),
        )
    del extractor


def load_features(cache: Path, names: list[str]) -> list[dict]:
    features = []
    for name in names:
        with np.load(cache / (Path(name).stem + ".npz")) as data:
            features.append({key: data[key] for key in data.files})
    return features


def global_descriptors(images: Path, names: list[str], cache: Path, config: SfmConfig, device):
    """L2-normalized DINOv2 CLS descriptors, cached in ``<cache>/descriptors.npy``."""
    path = cache / "descriptors.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(names):
            return cached

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(config.descriptor_model)
    model = AutoModel.from_pretrained(config.descriptor_model).eval().to(device)
    descriptors = []
    with torch.inference_mode():
        for start in range(0, len(names), 16):
            batch = [Image.open(images / n).convert("RGB") for n in names[start : start + 16]]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            output = model(**inputs).last_hidden_state[:, 0]
            descriptors.append(torch.nn.functional.normalize(output, dim=-1).cpu().numpy())
    del model
    descriptors = np.concatenate(descriptors)
    np.save(path, descriptors)
    return descriptors


def build_pairs(descriptors: np.ndarray, config: SfmConfig) -> list[tuple[int, int, str]]:
    """Sequential neighbours (capture order), plus top-k retrieval pairs farther apart."""
    count = len(descriptors)
    pairs = {
        (i, j): "sequential"
        for i in range(count)
        for j in range(i + 1, min(count, i + 1 + config.sequential_window))
    }
    similarity = descriptors @ descriptors.T
    index = np.arange(count)
    far = np.abs(index[:, None] - index[None, :]) >= max(
        config.retrieval_min_gap, config.sequential_window + 1
    )
    similarity = np.where(far, similarity, -np.inf)
    for i in range(count):
        for j in np.argsort(-similarity[i])[: config.retrieval_top_k]:
            if similarity[i, j] >= config.retrieval_min_similarity:
                pairs.setdefault((min(i, int(j)), max(i, int(j))), "retrieval")
    return [(i, j, kind) for (i, j), kind in sorted(pairs.items())]


def planar_fraction(xy_a: np.ndarray, xy_b: np.ndarray, ransac_px: float) -> tuple[int, float]:
    """Fundamental-matrix inliers and the fraction of them one homography explains."""
    if len(xy_a) < 16:
        return 0, 1.0
    _, mask = cv2.findFundamentalMat(xy_a, xy_b, cv2.USAC_MAGSAC, ransac_px, 0.999, 10000)
    if mask is None:
        return 0, 1.0
    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    if count < 16:
        return count, 1.0
    _, plane = cv2.findHomography(
        xy_a[inliers], xy_b[inliers], cv2.USAC_MAGSAC, 2 * ransac_px, confidence=0.999
    )
    on_plane = int(plane.sum()) if plane is not None else count
    return count, on_plane / count


def match_pairs(cache: Path, features: list[dict], pairs, config: SfmConfig, device: str):
    """LightGlue matches; retrieval pairs must pass the non-planar epipolar gate.

    Returns ``(matches, listing)``: accepted matches per pair and one record per pair.
    Both are cached in ``<cache>/matches.npz`` and ``<cache>/pairs.json``.
    """
    matches_path, listing_path = cache / "matches.npz", cache / "pairs.json"
    if matches_path.exists() and listing_path.exists():
        listing = json.loads(listing_path.read_text())
        if [(p["i"], p["j"], p["kind"]) for p in listing] == [tuple(p) for p in pairs]:
            with np.load(matches_path) as data:
                matches = {tuple(map(int, key.split("-"))): data[key] for key in data.files}
            return matches, listing

    import torch
    from lightglue import LightGlue

    matcher = LightGlue(features="aliked").eval().to(device)
    on_device: dict[int, dict] = {}

    def tensors(index):
        if index not in on_device:
            if len(on_device) > 64:  # pairs are sorted by i, so old entries are done
                on_device.pop(next(iter(on_device)))
            feature = features[index]
            on_device[index] = {
                "keypoints": torch.from_numpy(feature["keypoints"])[None].to(device),
                "descriptors": torch.from_numpy(feature["descriptors"].astype(np.float32))[None].to(
                    device
                ),
                "image_size": torch.from_numpy(feature["image_size"])[None].to(device),
            }
        return on_device[index]

    matches, listing = {}, []
    with torch.inference_mode():
        for number, (i, j, kind) in enumerate(pairs):
            if number % 2000 == 0:
                print(f"matching pair {number}/{len(pairs)}", flush=True)
            result = matcher({"image0": tensors(i), "image1": tensors(j)})
            pair_matches = result["matches"][0].cpu().numpy().astype(np.uint32)
            record = {"i": i, "j": j, "kind": kind, "raw_matches": len(pair_matches)}
            if kind == "retrieval":
                xy_a = features[i]["keypoints"][pair_matches[:, 0]]
                xy_b = features[j]["keypoints"][pair_matches[:, 1]]
                inliers, planar = planar_fraction(xy_a, xy_b, config.ransac_px)
                record.update(fundamental_inliers=inliers, planar_fraction=planar)
                record["accepted"] = bool(
                    inliers >= config.retrieval_min_inliers
                    and planar <= config.retrieval_max_planar_fraction
                )
            else:
                record["accepted"] = bool(len(pair_matches) >= config.min_inliers)
            if record["accepted"]:
                matches[(i, j)] = pair_matches
            listing.append(record)

    np.savez(matches_path, **{f"{i}-{j}": value for (i, j), value in matches.items()})
    listing_path.write_text(json.dumps(listing) + "\n")
    return matches, listing
