"""Strict refined-geometry handoff and a bounded CPU image/map cache."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DataConfig
from .contracts import Camera, FrameRecord, load_array


@dataclass
class FrameData:
    camera: Camera
    rgb: np.ndarray
    depth: np.ndarray
    points: np.ndarray
    image_mask: np.ndarray
    geometry_mask: np.ndarray


class RefinedScene:
    def __init__(self, manifest: str | Path, config: DataConfig):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.config = config
        self.raw = json.loads(self.path.read_text())
        conventions = {
            "schema_version": 1,
            "camera_convention": "opencv_world_to_camera",
            "depth_convention": "camera_z",
            "point_convention": "world",
            "camera_model": "pinhole",
            "pixel_convention": "integer_centers",
        }
        for key, expected in conventions.items():
            if self.raw.get(key) != expected:
                raise ValueError(f"Manifest requires {key}={expected!r}")
        if not isinstance(self.raw.get("scene_units"), str) or not self.raw["scene_units"]:
            raise ValueError("Declare scene_units, e.g. 'arbitrary' or 'metres'")
        if not isinstance(self.raw.get("frames"), list) or len(self.raw["frames"]) < 2:
            raise ValueError("At least two posed frames are required")

        self.frames = [self._record(value) for value in self.raw["frames"]]
        ids = [frame.camera.frame_id for frame in self.frames]
        if len(set(ids)) != len(ids):
            raise ValueError("Frame IDs must be unique")
        explicit = [frame.split is not None for frame in self.frames]
        if any(explicit) and not all(explicit):
            raise ValueError("Specify split for every frame or for none")
        if all(explicit):
            self.train_indices = [i for i, f in enumerate(self.frames) if f.split == "train"]
            self.val_indices = [i for i, f in enumerate(self.frames) if f.split == "val"]
        else:
            every = config.holdout_every
            self.val_indices = [
                i for i in range(len(self.frames)) if every and (i + 1) % every == 0
            ]
            self.train_indices = [i for i in range(len(self.frames)) if i not in self.val_indices]
        if len(self.train_indices) < max(2, config.min_consistent_views + 1):
            raise ValueError("Insufficient training cameras for the requested multiview support")
        self._cache: OrderedDict[int, FrameData] = OrderedDict()
        self.diagnostics: dict[str, dict] = {}

    def _record(self, value: dict) -> FrameRecord:
        camera = Camera.from_dict(value, self.root)
        image = (self.root / value["image"]).resolve()
        if not image.is_file():
            raise ValueError(f"Missing RGB image: {image}")
        timestamp = float(value["timestamp_s"])
        transform = np.asarray(value["source_to_image"], dtype=np.float64)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("source_to_image must be a finite 3×3 pixel transform")
        if abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("source_to_image must be invertible")
        split = value.get("split")
        if split not in (None, "train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        for name in ("depth", "points"):
            if name not in value:
                raise ValueError(f"Missing {name} reference for {camera.frame_id}")
        return FrameRecord(
            camera,
            image,
            value["depth"],
            value["points"],
            timestamp,
            transform,
            split,
            value.get("confidence"),
            value.get("mask"),
            value.get("metadata", {}),
        )

    def fingerprint(self) -> str:
        """Content hash inputs, including images/maps; used to reject stale resumes."""
        paths = {self.path}
        for value in self.raw["frames"]:
            paths.add((self.root / value["image"]).resolve())
            for key in ("depth", "points", "confidence", "mask", "intrinsics", "world_to_camera"):
                ref = value.get(key)
                if isinstance(ref, str):
                    paths.add((self.root / ref).resolve())
                elif isinstance(ref, dict):
                    paths.add((self.root / ref["path"]).resolve())
        digest = hashlib.sha256()
        for path in sorted(paths):
            digest.update(str(path).encode())
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    def _map(self, source, shape: tuple, name: str) -> np.ndarray:
        array = np.asarray(load_array(source, self.root))
        if array.shape == (*shape, 1):
            array = array[..., 0]
        if array.shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {array.shape}; maps must align to RGB")
        if np.iscomplexobj(array):
            raise ValueError(f"{name} cannot contain complex numbers")
        return array

    def frame(self, index: int) -> FrameData:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        record = self.frames[index]
        camera = record.camera
        bgr = cv2.imread(str(record.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None or bgr.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"Image cannot be read or dimensions disagree: {record.image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        shape = (camera.height, camera.width)
        depth = self._map(record.depth, shape, "depth").astype(np.float32)
        points = self._map(record.points, (*shape, 3), "points").astype(np.float32)
        image_mask = np.ones(shape, dtype=bool)
        if record.mask is not None:
            mask = self._map(record.mask, shape, "mask")
            if not np.isin(mask, [0, 1]).all():
                raise ValueError("Validity mask must be boolean or contain only 0/1")
            image_mask &= mask.astype(bool)
        valid = image_mask & np.isfinite(depth) & (depth > self.config.depth_min)
        valid &= (depth < self.config.depth_max) & np.isfinite(points).all(axis=-1)
        if record.confidence is not None:
            confidence = self._map(record.confidence, shape, "confidence")
            valid &= np.isfinite(confidence) & (confidence >= self.config.confidence_min)

        # Check before resizing: refined maps must actually match the supplied poses/K.
        ys, xs = np.nonzero(valid)
        xyz = points[ys, xs].astype(np.float64)
        xyz_cam = xyz @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
        projected = xyz_cam @ camera.intrinsics.T
        z = xyz_cam[:, 2]
        uv = projected[:, :2] / np.maximum(z[:, None], 1e-12)
        reprojection = np.linalg.norm(uv - np.column_stack((xs, ys)), axis=1)
        relative = np.abs(z - depth[ys, xs]) / np.maximum(depth[ys, xs], 1e-12)
        consistent = (
            (z > 0)
            & (reprojection <= self.config.reprojection_px)
            & (relative <= self.config.depth_relative_tolerance)
        )
        rejection = 1 - float(consistent.mean()) if len(consistent) else 1.0
        if rejection > self.config.max_geometry_rejection:
            raise ValueError(
                f"{camera.frame_id}: {rejection:.1%} of valid geometry disagrees with camera/depth; "
                "check pose direction, scene scale, pixel grid and image transforms"
            )
        valid[ys[~consistent], xs[~consistent]] = False
        self.diagnostics[camera.frame_id] = {
            "candidate_pixels": len(xs),
            "valid_pixels": int(valid.sum()),
            "geometry_rejection_fraction": rejection,
        }
        if not image_mask.any() or not valid.any():
            raise ValueError(f"{camera.frame_id}: no valid image/geometry pixels")

        ratio = min(1.0, self.config.max_image_side / max(shape))
        if ratio < 1:
            width, height = (
                max(1, round(camera.width * ratio)),
                max(1, round(camera.height * ratio)),
            )
            size = (width, height)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            # Nearest-exact follows the same pixel-centre convention as resized K.
            depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
            points = cv2.resize(points, size, interpolation=cv2.INTER_NEAREST_EXACT)
            # RGB area filtering must not blend masked source pixels into a
            # supposedly valid target pixel. Require its entire area to be valid.
            image_mask = (
                cv2.resize(image_mask.astype("float32"), size, interpolation=cv2.INTER_AREA)
                >= 1 - 1e-6
            )
            valid = cv2.resize(
                valid.astype("uint8"), size, interpolation=cv2.INTER_NEAREST_EXACT
            ).astype(bool)
            valid &= image_mask
            camera = camera.resized(width, height)
        if min(camera.width, camera.height) < 11 or not valid.any():
            raise ValueError(f"{camera.frame_id}: image too small or no geometry after resizing")

        result = FrameData(camera, rgb.astype(np.float32) / 255, depth, points, image_mask, valid)
        self._cache[index] = result
        while len(self._cache) > self.config.cache_frames:
            self._cache.popitem(last=False)
        return result


def load_scene(manifest: str | Path, config: DataConfig | None = None) -> RefinedScene:
    return RefinedScene(manifest, config or DataConfig())
