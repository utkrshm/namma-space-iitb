"""ALIKED features, sequential and retrieval pairs, LightGlue matching on full-resolution frames."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .contracts import SfmConfig


def extract_features(output_dir: Path, frames: list[dict], config: SfmConfig, device: str):
    """ALIKED keypoints in ``images/`` pixel coordinates (integer-centre), cached per image."""
    import torch
    from lightglue import ALIKED

    folder = output_dir / "features"
    folder.mkdir(exist_ok=True)
    extractor = None
    for frame in frames:
        path = folder / (Path(frame["name"]).stem + ".npz")
        if path.exists():
            continue
        if extractor is None:
            extractor = ALIKED(
                max_num_keypoints=config.max_keypoints,
                detection_threshold=config.detection_threshold,
            )
            extractor = extractor.eval().to(device)
        rgb = cv2.cvtColor(
            cv2.imread(str(output_dir / "images" / frame["name"])), cv2.COLOR_BGR2RGB
        )
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


def load_features(output_dir: Path, frames: list[dict]) -> list[dict]:
    features = []
    for frame in frames:
        with np.load(output_dir / "features" / (Path(frame["name"]).stem + ".npz")) as data:
            features.append({key: data[key] for key in data.files})
    return features


def global_descriptors(output_dir: Path, frames: list[dict], config: SfmConfig, device: str):
    """L2-normalized DINOv2 CLS descriptors; cached in ``descriptors.npy``."""
    path = output_dir / "descriptors.npy"
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(frames):
            return cached

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(config.descriptor_model)
    model = AutoModel.from_pretrained(config.descriptor_model).eval().to(device)
    descriptors = []
    with torch.inference_mode():
        for start in range(0, len(frames), 16):
            images = [
                Image.open(output_dir / "images" / f["name"]).convert("RGB")
                for f in frames[start : start + 16]
            ]
            inputs = processor(images=images, return_tensors="pt").to(device)
            output = model(**inputs).last_hidden_state[:, 0]
            descriptors.append(torch.nn.functional.normalize(output, dim=-1).cpu().numpy())
    del model
    descriptors = np.concatenate(descriptors)
    np.save(path, descriptors)
    return descriptors


def build_pairs(
    timestamps: np.ndarray, descriptors: np.ndarray, config: SfmConfig
) -> list[tuple[int, int, str]]:
    """Sequential neighbours within the window, plus top-k retrieval pairs far apart in time."""
    count = len(timestamps)
    pairs = {
        (i, j): "sequential"
        for i in range(count)
        for j in range(i + 1, min(count, i + 1 + config.sequential_window))
    }
    similarity = descriptors @ descriptors.T
    far = np.abs(timestamps[:, None] - timestamps[None, :]) > config.retrieval_exclusion_s
    similarity = np.where(far, similarity, -np.inf)
    for i in range(count):
        for j in np.argsort(-similarity[i])[: config.retrieval_top_k]:
            if similarity[i, j] >= config.retrieval_min_similarity:
                pairs.setdefault((min(i, int(j)), max(i, int(j))), "retrieval")
    return [(i, j, kind) for (i, j), kind in sorted(pairs.items())]


def planar_fraction(xy_a: np.ndarray, xy_b: np.ndarray, ransac_px: float) -> tuple[int, float]:
    """Fundamental-matrix inliers and the fraction of them that one homography explains."""
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


def match_pairs(
    output_dir: Path,
    features: list[dict],
    pairs: list[tuple[int, int, str]],
    config: SfmConfig,
    device: str,
) -> tuple[dict[tuple[int, int], np.ndarray], list[dict]]:
    """LightGlue matches for every pair; retrieval pairs must pass the non-planar epipolar gate.

    Results are cached in ``matches.npz`` / ``pairs.json`` and reused when the pair list is
    unchanged.
    """
    matches_path, listing_path = output_dir / "matches.npz", output_dir / "pairs.json"
    wanted = [[i, j, kind] for i, j, kind in pairs]
    if matches_path.exists() and listing_path.exists():
        listing = json.loads(listing_path.read_text())
        if [[p["i"], p["j"], p["kind"]] for p in listing] == wanted:
            with np.load(matches_path) as data:
                matches = {tuple(map(int, key.split("-"))): data[key] for key in data.files}
            return matches, listing

    import torch
    from lightglue import LightGlue

    matcher = LightGlue(features="aliked").eval().to(device)
    on_device = {}

    def tensors(index):
        if index not in on_device:
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
            if number % 1000 == 0:
                print(f"matching pair {number}/{len(pairs)}", flush=True)
            result = matcher({"image0": tensors(i), "image1": tensors(j)})
            pair_matches = result["matches"][0].cpu().numpy().astype(np.uint32)
            record = {"i": i, "j": j, "kind": kind, "raw_matches": len(pair_matches)}
            if kind == "retrieval":
                xy_a = features[i]["keypoints"][pair_matches[:, 0]]
                xy_b = features[j]["keypoints"][pair_matches[:, 1]]
                inliers, planar = planar_fraction(xy_a, xy_b, config.ransac_px)
                record.update(fundamental_inliers=inliers, planar_fraction=planar)
                record["accepted"] = (
                    inliers >= config.retrieval_min_inliers
                    and planar <= config.retrieval_max_planar_fraction
                )
            else:
                record["accepted"] = len(pair_matches) >= config.min_inliers
            if record["accepted"]:
                matches[(i, j)] = pair_matches
            listing.append(record)

    np.savez(matches_path, **{f"{i}-{j}": value for (i, j), value in matches.items()})
    listing_path.write_text(json.dumps(listing) + "\n")
    return matches, listing
