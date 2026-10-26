"""Read-only access to a section-alignment directory and per-frame global models."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .contracts import FrameModel

_KEYS = (
    "images",
    "valid_mask",
    "depth",
    "depth_confidence",
    "depth_world_points",
    "intrinsics",
    "camera_to_world",
)


class SectionCache:
    """Section predictions keyed by section index; never writes to the section directory."""

    def __init__(self, sections_dir: str | Path, max_cached: int = 3):
        self.root = Path(sections_dir).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.frame_ids: list[str] = list(self.manifest["frame_ids"])
        self.index = {fid: i for i, fid in enumerate(self.frame_ids)}
        self.preprocessing_dir = Path(self.manifest["preprocessing_dir"])
        self.image_size = int(self.manifest["image_size"])
        self.sections: list[list[int]] = [
            [self.index[fid] for fid in section["frame_ids"]]
            for section in self.manifest["sections"]
        ]
        self.timestamps = np.zeros(len(self.frame_ids))
        for provenance in self.manifest["frame_provenance"]:
            self.timestamps[self.index[provenance["frame_id"]]] = provenance["timestamp_s"]
        self.image_paths = {
            self.index[p["frame_id"]]: p["image_path"] for p in self.manifest["frame_provenance"]
        }
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._max_cached = max_cached

    def arrays(self, section: int) -> dict:
        if section not in self._cache:
            with np.load(self.root / f"section_{section:02d}" / "predictions.npz") as archive:
                self._cache[section] = {key: archive[key] for key in _KEYS}
            if len(self._cache) > self._max_cached:
                self._cache.popitem(last=False)
        self._cache.move_to_end(section)
        return self._cache[section]

    def local(self, section: int, frame: int) -> int:
        return self.sections[section].index(frame)

    def sections_with(self, frames: list[int]) -> list[int]:
        """Sections containing every listed frame, most central first."""
        found = [s for s, members in enumerate(self.sections) if set(frames) <= set(members)]

        def margin(section):
            members = self.sections[section]
            return min(min(members.index(f), len(members) - 1 - members.index(f)) for f in frames)

        return sorted(found, key=lambda s: -margin(s))

    def default_owners(self) -> list[int]:
        """Owner rule used by `run_section_alignment`: farthest from a section boundary."""
        owners = []
        for frame in range(len(self.frame_ids)):
            items = [
                (s, members.index(frame), len(members))
                for s, members in enumerate(self.sections)
                if frame in members
            ]
            best = max(items, key=lambda item: (min(item[1], item[2] - 1 - item[1]), -item[0]))
            owners.append(best[0])
        return owners


def _valid(arrays: dict, local: int) -> np.ndarray:
    depth = arrays["depth"][local]
    confidence = arrays["depth_confidence"][local]
    valid = arrays["valid_mask"][local] & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if valid.any():
        valid &= confidence >= np.percentile(confidence[valid], 25)
    return valid


def load_graph_cameras(sections_dir: str | Path, graph_dir: str | Path) -> dict:
    """Per-frame global cameras implied by a section graph export (baseline or method A).

    Returns arrays keyed like `cameras.npz`: owners, camera_to_world, depth_scale, intrinsics.
    """
    cache = SectionCache(sections_dir)
    graph_dir = Path(graph_dir)
    matrices = np.load(graph_dir / "graph.npz")["optimized_matrices"]
    report = json.loads((graph_dir / "alignment_report.json").read_text())
    owners = report.get("owner_section_by_frame") or cache.default_owners()
    count = len(cache.frame_ids)
    camera_to_world = np.zeros((count, 4, 4))
    intrinsics = np.zeros((count, 3, 3))
    depth_scale = np.zeros(count)
    for section in sorted(set(owners)):
        arrays = cache.arrays(section)
        matrix = matrices[section]
        scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
        rotation = matrix[:3, :3] / scale
        for frame in [f for f, o in enumerate(owners) if o == section]:
            local = cache.local(section, frame)
            pose = arrays["camera_to_world"][local].astype(np.float64)
            global_pose = np.eye(4)
            global_pose[:3, :3] = rotation @ pose[:3, :3]
            global_pose[:3, 3] = scale * rotation @ pose[:3, 3] + matrix[:3, 3]
            camera_to_world[frame] = global_pose
            intrinsics[frame] = arrays["intrinsics"][local]
            depth_scale[frame] = scale
    return {
        "frame_ids": np.asarray(cache.frame_ids),
        "owners": np.asarray(owners, dtype=np.int32),
        "camera_to_world": camera_to_world,
        "depth_scale": depth_scale,
        "intrinsics": intrinsics,
    }


def save_cameras(path: str | Path, cameras: dict) -> None:
    np.savez_compressed(path, **cameras)


def build_frame_models(sections_dir: str | Path, cameras: dict) -> list[FrameModel]:
    """Attach each frame's owner-section dense depth to the given global cameras."""
    cache = SectionCache(sections_dir)
    owners = [int(o) for o in cameras["owners"]]
    models: list[FrameModel | None] = [None] * len(owners)
    for section in sorted(set(owners)):
        arrays = cache.arrays(section)
        for frame in [f for f, o in enumerate(owners) if o == section]:
            local = cache.local(section, frame)
            models[frame] = FrameModel(
                frame_id=cache.frame_ids[frame],
                timestamp_s=float(cache.timestamps[frame]),
                owner_section=section,
                intrinsics=np.asarray(cameras["intrinsics"][frame], dtype=np.float64),
                camera_to_world=np.asarray(cameras["camera_to_world"][frame], dtype=np.float64),
                depth=arrays["depth"][local],
                depth_scale=float(cameras["depth_scale"][frame]),
                confidence=arrays["depth_confidence"][local],
                valid=_valid(arrays, local),
                image=arrays["images"][local],
            )
    return models  # type: ignore[return-value]


def export_cloud(
    models: list[FrameModel], output_dir: str | Path, max_points: int = 1_000_000
) -> dict:
    """Write `point_cloud.npz|ply` in the section-alignment schema (no fusion, q25 filter)."""
    from ..reconstruction.contracts import PointCloud
    from ..reconstruction.visualization import save_point_cloud

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    budget = max_points // len(models)
    parts = {k: [] for k in ("points", "colors", "confidence", "pixel_xy", "depths", "frames")}
    for frame, model in enumerate(models):
        indices = np.flatnonzero(model.valid)
        if len(indices) > budget:
            indices = indices[np.linspace(0, len(indices) - 1, budget, dtype=int)]
        width = model.valid.shape[1]
        xy = np.column_stack((indices % width, indices // width)).astype(np.int32)
        parts["points"].append(model.unproject(xy).astype(np.float32))
        parts["colors"].append(model.image.reshape(-1, 3)[indices])
        parts["confidence"].append(model.confidence.reshape(-1)[indices].astype(np.float32))
        parts["pixel_xy"].append(xy)
        parts["depths"].append(
            (model.depth.reshape(-1)[indices] * model.depth_scale).astype(np.float32)
        )
        parts["frames"].append(np.full(len(indices), frame, np.int32))
    camera_to_world = np.stack([m.camera_to_world for m in models])
    world_to_camera = np.stack([np.linalg.inv(pose)[:3] for pose in camera_to_world])
    arrays = {
        "points": np.concatenate(parts["points"]),
        "colors": np.concatenate(parts["colors"]),
        "confidence": np.concatenate(parts["confidence"]),
        "frame_indices": np.concatenate(parts["frames"]),
        "pixel_xy": np.concatenate(parts["pixel_xy"]),
        "section_indices": np.concatenate(
            [np.full(len(p), m.owner_section, np.int16) for p, m in zip(parts["points"], models)]
        ),
        "camera_centers": camera_to_world[:, :3, 3],
        "camera_rotations": camera_to_world[:, :3, :3],
        "camera_translations": world_to_camera,
        "depths": np.concatenate(parts["depths"]),
        "frame_ids": np.asarray([m.frame_id for m in models]),
    }
    np.savez_compressed(output / "point_cloud.npz", **arrays)
    save_point_cloud(
        PointCloud(
            arrays["points"],
            arrays["colors"],
            arrays["confidence"],
            arrays["frame_indices"],
            arrays["pixel_xy"],
        ),
        output / "point_cloud.ply",
    )
    return {"point_count": len(arrays["points"]), "frame_count": len(models)}
