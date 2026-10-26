"""Loop retrieval (DINOv2) and ALIKED/LightGlue verification on the shared model grid."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .contracts import FrameMatches, LoopConfig, LoopPair
from .store import SectionCache


def global_descriptors(cache: SectionCache, config: LoopConfig, device: str) -> np.ndarray:
    """L2-normalized DINOv2 CLS descriptors of the upright source frames, in frame order."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(config.descriptor_model)
    model = AutoModel.from_pretrained(config.descriptor_model).eval().to(device)
    descriptors = []
    with torch.inference_mode():
        for frame in range(len(cache.frame_ids)):
            image = Image.open(cache.image_paths[frame]).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            output = model(**inputs).last_hidden_state[:, 0]
            descriptors.append(torch.nn.functional.normalize(output, dim=-1)[0].cpu().numpy())
    del model
    return np.stack(descriptors)


def retrieve_candidates(
    descriptors: np.ndarray, timestamps: np.ndarray, config: LoopConfig
) -> list[tuple[int, int, float]]:
    """Mutual top-k pairs above a cosine similarity, outside the temporal exclusion window."""
    similarity = descriptors @ descriptors.T
    far = np.abs(timestamps[:, None] - timestamps[None, :]) > config.temporal_exclusion_s
    similarity = np.where(far, similarity, -np.inf)
    pairs = {}
    for frame in range(len(similarity)):
        order = np.argsort(-similarity[frame])[: config.top_k]
        for other in order:
            value = float(similarity[frame, other])
            if value >= config.min_similarity:
                key = (min(frame, int(other)), max(frame, int(other)))
                pairs[key] = value
    return sorted(((a, b, v) for (a, b), v in pairs.items()), key=lambda item: -item[2])


class FeatureBank:
    """ALIKED features of each frame's model-grid image (owner-section pixels, valid crop)."""

    def __init__(self, cache: SectionCache, owners: list[int], config: LoopConfig, device: str):
        import torch
        from lightglue import ALIKED, LightGlue

        self.cache, self.owners, self.config, self.device = cache, owners, config, device
        self.extractor = ALIKED(max_num_keypoints=config.keypoints).eval().to(device)
        self.matcher = LightGlue(features="aliked").eval().to(device)
        self.features: dict[int, tuple[dict, tuple[int, int], np.ndarray]] = {}
        self._torch = torch

    def _extract_all(self) -> None:
        torch = self._torch
        for section in sorted(set(self.owners)):
            arrays = self.cache.arrays(section)
            for frame in [f for f, o in enumerate(self.owners) if o == section]:
                local = self.cache.local(section, frame)
                mask = arrays["valid_mask"][local]
                ys, xs = np.where(mask)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                crop = arrays["images"][local][y0:y1, x0:x1]
                tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div(255).to(self.device)
                with torch.inference_mode():
                    feature = self.extractor.extract(tensor)
                self.features[frame] = (feature, crop.shape[:2], np.array([x0, y0], float))

    def match(self, frame_a: int, frame_b: int, kind: str) -> FrameMatches | None:
        from ..preprocessing.revisits import verify_feature_pair

        if not self.features:
            self._extract_all()
        config = self.config
        if kind == "sequential":
            config = replace(
                config,
                min_inliers=config.sequential_min_inliers,
                min_coverage=config.sequential_min_coverage,
                min_hull_area=config.sequential_min_hull_area,
            )
        first, shape_a, offset_a = self.features[frame_a]
        second, shape_b, offset_b = self.features[frame_b]
        with self._torch.inference_mode():
            verified, details = verify_feature_pair(
                first, second, self.matcher, shape_a, shape_b, config
            )
            if verified is None and kind == "loop" and details.get("rejected_repetitive"):
                verified = self._nonplanar_override(first, second, shape_a, shape_b, details)
        if verified is None:
            return FrameMatches(frame_a, frame_b, np.empty((0, 2)), np.empty((0, 2)), kind, details)
        _, coverage, xy_a, xy_b = verified
        details["coverage"] = float(coverage)
        return FrameMatches(frame_a, frame_b, xy_a + offset_a, xy_b + offset_b, kind, details)

    def _nonplanar_override(self, first, second, shape_a, shape_b, details):
        """Accept a repetition-flagged pair only with clearly non-planar epipolar support.

        Repeated grilles and floor inlays are planar patterns: one homography (or two)
        explains almost all of their fundamental-matrix inliers. On Coorg the true
        wide-baseline living-room loops had planar fractions of 0.30-0.38, while grille and
        inlay false positives had 0.79-0.98, so the guard keeps a wide margin.
        """
        import cv2
        from lightglue.utils import rbd

        from ..preprocessing.revisits import verify_matches

        config = self.config
        matches = rbd(self.matcher({"image0": first, "image1": second}))["matches"]
        a = rbd(first)["keypoints"][matches[:, 0]].cpu().numpy()
        b = rbd(second)["keypoints"][matches[:, 1]].cpu().numpy()
        raw = verify_matches(a, b, shape_a, shape_b, config)
        if raw is None:
            return None
        count, coverage, inliers_a, inliers_b = raw
        _, mask = cv2.findHomography(
            inliers_a, inliers_b, cv2.USAC_MAGSAC, config.planar_ransac_px, confidence=0.999
        )
        on_plane = mask.ravel().astype(bool) if mask is not None else np.zeros(count, bool)
        off_a, off_b = inliers_a[~on_plane], inliers_b[~on_plane]
        second_plane = 0
        if len(off_a) >= 8:
            _, mask2 = cv2.findHomography(
                off_a, off_b, cv2.USAC_MAGSAC, config.planar_ransac_px, confidence=0.999
            )
            second_plane = int(mask2.sum()) if mask2 is not None else 0
        planar_fraction = float(on_plane.mean())
        two_plane_fraction = float((on_plane.sum() + second_plane) / count)
        details["nonplanar_override"] = {
            "fundamental_inliers": int(count),
            "planar_fraction": planar_fraction,
            "two_plane_fraction": two_plane_fraction,
            "off_plane_inliers": len(off_a),
        }
        accepted = (
            planar_fraction <= config.max_planar_fraction
            and two_plane_fraction <= config.max_two_plane_fraction
            and len(off_a) >= config.min_off_plane_inliers
        )
        details["nonplanar_override"]["accepted"] = accepted
        if not accepted:
            return None
        details["accepted"] = True
        details["inliers_after"] = int(count)
        return count, coverage, inliers_a, inliers_b


_MATCHING_FIELDS = (
    "keypoints",
    "min_inliers",
    "min_inlier_ratio",
    "min_coverage",
    "min_hull_area",
    "self_match_radius_px",
    "self_match_ratio",
    "ransac_px",
    "confidence",
    "max_iterations",
    "planar_ransac_px",
    "max_planar_fraction",
    "max_two_plane_fraction",
    "min_off_plane_inliers",
    "sequential_min_inliers",
    "sequential_min_coverage",
    "sequential_min_hull_area",
)


def matching_settings(config: LoopConfig) -> dict:
    """Only the settings that change verified matches; solver settings do not invalidate them."""
    return {name: getattr(config, name) for name in _MATCHING_FIELDS}


class MatchStore:
    """Disk cache of verified matches, keyed by frame identity, owners and settings."""

    def __init__(
        self, root: Path, cache: SectionCache, owners: list[int], config: LoopConfig, bank_factory
    ):
        self.root = root / "matches"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache, self.owners, self.config = cache, owners, config
        self._bank_factory = bank_factory
        self._bank: FeatureBank | None = None

    def get(self, frame_a: int, frame_b: int, kind: str) -> FrameMatches:
        signature = {
            "frames": [self.cache.frame_ids[frame_a], self.cache.frame_ids[frame_b]],
            "owners": [self.owners[frame_a], self.owners[frame_b]],
            "sections_dir": str(self.cache.root),
            "kind": kind,
            "config": matching_settings(self.config),
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        path = self.root / f"{frame_a:03d}_{frame_b:03d}_{kind}_{digest}.npz"
        if path.is_file():
            with np.load(path) as data:
                return FrameMatches(
                    frame_a,
                    frame_b,
                    data["xy_a"],
                    data["xy_b"],
                    kind,
                    json.loads(str(data["details"])),
                )
        if self._bank is None:
            self._bank = self._bank_factory()
        result = self._bank.match(frame_a, frame_b, kind)
        np.savez_compressed(
            path,
            xy_a=result.xy_a,
            xy_b=result.xy_b,
            details=json.dumps(result.details, default=float),
        )
        return result


def select_loops(
    verified: list[LoopPair], cache: SectionCache, owners: list[int], config: LoopConfig
) -> list[LoopPair]:
    """Assign roles: chunk centres (method A), training pairs and held-out pairs.

    Held-out pairs are every `holdout_every`-th verified pair (by inliers) that is not a
    chunk centre; they enter neither method and are used only by the comparison.
    """
    times = cache.timestamps
    ranked = sorted(verified, key=lambda pair: -pair.inliers)
    chunks: list[LoopPair] = []
    per_section_pair: dict[tuple[int, int], int] = {}
    for pair in ranked:
        key = tuple(sorted((owners[pair.frame_a], owners[pair.frame_b])))
        if key[0] == key[1] or per_section_pair.get(key, 0) >= config.max_loops_per_section_pair:
            continue
        close = any(
            abs(times[pair.frame_a] - times[c.frame_a]) < config.min_loop_separation_s
            and abs(times[pair.frame_b] - times[c.frame_b]) < config.min_loop_separation_s
            for c in chunks
        )
        if close or len(chunks) >= config.max_loop_chunks:
            continue
        chunks.append(replace(pair, role="chunk"))
        per_section_pair[key] = per_section_pair.get(key, 0) + 1
    chunk_keys = {(c.frame_a, c.frame_b) for c in chunks}
    rest = [p for p in ranked if (p.frame_a, p.frame_b) not in chunk_keys]
    roles = [
        replace(
            p,
            role="holdout" if index % config.holdout_every == config.holdout_every - 1 else "train",
        )
        for index, p in enumerate(rest)
    ]
    return sorted(chunks + roles, key=lambda p: (p.frame_a, p.frame_b))


def detect_loops(
    cache: SectionCache,
    owners: list[int],
    matches: MatchStore,
    config: LoopConfig,
    output_dir: Path,
    device: str,
) -> tuple[list[LoopPair], dict]:
    """Retrieve, verify and assign roles; writes `loops.json`. Cached descriptors are reused."""
    descriptor_path = output_dir / "descriptors.npz"
    signature = f"{config.descriptor_model}|{cache.root}|{len(cache.frame_ids)}"
    descriptors = None
    if descriptor_path.is_file():
        with np.load(descriptor_path) as data:
            if str(data["signature"]) == signature:
                descriptors = data["descriptors"]
    if descriptors is None:
        descriptors = global_descriptors(cache, config, device)
        np.savez_compressed(descriptor_path, descriptors=descriptors, signature=signature)

    candidates = retrieve_candidates(descriptors, cache.timestamps, config)
    verified, records = [], []
    for frame_a, frame_b, similarity in candidates:
        result = matches.get(frame_a, frame_b, "loop")
        record = {
            "frame_a": frame_a,
            "frame_b": frame_b,
            "time_a": float(cache.timestamps[frame_a]),
            "time_b": float(cache.timestamps[frame_b]),
            "similarity": similarity,
            **{
                k: result.details.get(k)
                for k in ("raw_matches", "ambiguous_matches", "inliers_after", "accepted")
            },
        }
        records.append(record)
        if len(result.xy_a):
            verified.append(
                LoopPair(
                    frame_a,
                    frame_b,
                    similarity,
                    len(result.xy_a),
                    float(result.details["coverage"]),
                )
            )
    selected = select_loops(verified, cache, owners, config)
    report = {
        "config": asdict(config),
        "candidate_count": len(candidates),
        "verified_count": len(verified),
        "candidates": records,
        "loops": [
            {
                **asdict(p),
                "time_a": float(cache.timestamps[p.frame_a]),
                "time_b": float(cache.timestamps[p.frame_b]),
                "owner_a": owners[p.frame_a],
                "owner_b": owners[p.frame_b],
            }
            for p in selected
        ],
    }
    (output_dir / "loops.json").write_text(json.dumps(report, indent=2) + "\n")
    return selected, report
