"""Floor-plane estimation and explicit rigid transforms into a Z-up scene frame."""

from dataclasses import dataclass

import numpy as np

from .visualization import _open3d

FLOOR_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"


@dataclass(frozen=True)
class FloorAlignment:
    scene_to_aligned: np.ndarray  # 4,4; original scene -> Z-up, floor Z=0
    frame_to_aligned: np.ndarray  # S,4,4; original world observations -> aligned world
    floor_planes: np.ndarray  # S,4; interpolated unit-normal planes in original world
    reliable_frames: np.ndarray  # S bool; false means no accepted local plane
    report: dict


def segment_floor_images(
    images: np.ndarray, valid_mask: np.ndarray, *, device: str = "cuda"
) -> np.ndarray:
    """Return S,H,W bool floor/rug masks on the unchanged model pixel grid.

    Uses a real pretrained semantic model, not a pixel-position or color heuristic.
    Boundary erosion and a 0.6 probability threshold reduce object-edge leakage.
    """
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise ValueError("images must be S,H,W,3 uint8 RGB")
    if valid_mask.shape != images.shape[:3] or valid_mask.dtype != bool:
        raise ValueError("valid_mask must be S,H,W bool")

    processor = AutoImageProcessor.from_pretrained(FLOOR_MODEL)
    model = AutoModelForSemanticSegmentation.from_pretrained(FLOOR_MODEL).eval().to(device)
    floor_ids = [
        i for i, label in model.config.id2label.items() if label.strip() in ("floor", "rug")
    ]
    if len(floor_ids) != 2:
        raise ValueError("The semantic model must expose floor and rug labels")

    masks = []
    for image, valid in zip(images, valid_mask):
        inputs = processor(images=Image.fromarray(image), return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=image.shape[:2], mode="bilinear", align_corners=False
            )
            probability = logits.softmax(dim=1)[0]
            labels = probability.argmax(dim=0)
            floor = (labels == floor_ids[0]) | (labels == floor_ids[1])
            floor &= probability[floor_ids].sum(dim=0) > 0.6

        mask = floor.cpu().numpy() & valid
        masks.append(cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    return np.asarray(masks)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cosine = float(np.clip(source @ target, -1, 1))
    if cosine < -1 + 1e-8:
        axis = np.eye(3)[np.argmin(np.abs(source))]
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    x, y, z = np.cross(source, target)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def _fit_plane(points: np.ndarray, threshold: float, seed: int):
    o3d = _open3d()
    o3d.utility.random.seed(seed)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane, inliers = cloud.segment_plane(threshold, 3, 1000)
    return np.asarray(plane), np.asarray(inliers)


def estimate_floor_alignment(
    points: np.ndarray,
    floor_mask: np.ndarray,
    confidence: np.ndarray,
    camera_to_world: np.ndarray,
    timestamps: np.ndarray,
    *,
    correct_per_view: bool = False,
) -> FloorAlignment:
    """Fit floor planes and return rigid transforms without flattening individual points.

    Inputs share the S,H,W model grid and original arbitrary-scale world frame.
    Normal directions point from the floor towards the camera. Reliable local
    planes need broad spatial support and >=60% RANSAC inliers. Per-view correction
    assumes one flat floor: missing local planes are interpolated in source time,
    with nearest-anchor extrapolation at the ends. No independent yaw or scale
    correction is estimated. Mark padding/invalid confidence as nonfinite.
    """
    points = np.asarray(points)
    size = len(points)
    if points.ndim != 4 or points.shape[-1] != 3 or size < 1:
        raise ValueError("points must be S,H,W,3")
    if floor_mask.shape != points.shape[:3] or floor_mask.dtype != bool:
        raise ValueError("floor_mask must be S,H,W bool")
    if confidence.shape != floor_mask.shape:
        raise ValueError("confidence must share the floor-mask grid")
    if camera_to_world.shape != (size, 4, 4) or not np.isfinite(camera_to_world).all():
        raise ValueError("camera_to_world must be finite S,4,4")
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.shape != (size,) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite S")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Use one continuous clip with strictly increasing timestamps")

    rng = np.random.default_rng(42)
    samples = []
    for xyz, mask, scores in zip(points, floor_mask, confidence):
        valid = np.isfinite(xyz).all(axis=-1) & np.isfinite(scores)
        cutoff = np.percentile(scores[valid], 40) if valid.any() else np.inf
        floor = xyz[mask & valid & (scores >= cutoff)]
        if len(floor) > 4000:
            floor = floor[rng.choice(len(floor), 4000, replace=False)]
        samples.append(floor.astype(np.float64))

    supported = [i for i, xyz in enumerate(samples) if len(xyz) >= 200]
    if len(supported) < 3:
        raise ValueError("Need at least three views with >=200 confident floor observations")
    scale = float(
        np.median(
            np.concatenate(
                [np.linalg.norm(samples[i] - camera_to_world[i, :3, 3], axis=1) for i in supported]
            )
        )
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Floor observations have no usable scene scale")
    threshold = scale * 0.005
    balanced = np.concatenate([samples[i][:: max(1, len(samples[i]) // 1000)] for i in supported])
    reference, _ = _fit_plane(balanced, threshold, 42)
    centres = camera_to_world[:, :3, 3]
    if np.median(centres @ reference[:3] + reference[3]) < 0:
        reference *= -1

    anchors = []
    records = []
    for i, xyz in enumerate(samples):
        record = {"frame_index": i, "timestamp_s": float(timestamps[i]), "samples": len(xyz)}
        record["reliable"] = False
        if len(xyz) >= 200:
            plane, inliers = _fit_plane(xyz, threshold, 42 + i)
            if plane[:3] @ centres[i] + plane[3] < 0:
                plane *= -1
            spread = np.linalg.svd(xyz[inliers] - xyz[inliers].mean(axis=0), compute_uv=False)
            spread /= np.sqrt(len(inliers))
            angle = float(np.degrees(np.arccos(np.clip(plane[:3] @ reference[:3], -1, 1))))
            fraction = len(inliers) / len(xyz)
            record.update(plane=plane.tolist(), inlier_fraction=fraction, angle_deg=angle)
            record["reliable"] = bool(
                fraction >= 0.6
                and spread[1] >= scale * 0.02
                and spread[1] >= spread[0] * 0.1
                and angle <= 25
            )
            if record["reliable"]:
                anchors.append(i)
        records.append(record)
    if len(anchors) < 3:
        raise ValueError("Fewer than three reliable floor planes; inspect the masks/geometry")

    anchor_planes = np.asarray([records[i]["plane"] for i in anchors])
    planes = np.column_stack(
        [np.interp(timestamps, timestamps[anchors], anchor_planes[:, j]) for j in range(4)]
    )
    planes /= np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    reliable = np.zeros(size, dtype=bool)
    reliable[anchors] = True

    base = np.eye(4)
    base[:3, :3] = _rotation_between(reference[:3], np.array([0.0, 0.0, 1.0]))
    base[2, 3] = reference[3]
    transforms = np.repeat(base[None], size, axis=0)
    if correct_per_view:
        for i, (normal, offset) in enumerate(zip(planes[:, :3], planes[:, 3])):
            rotation = _rotation_between(normal, reference[:3])
            height_shift = (
                normal @ centres[i] + offset - (reference[:3] @ centres[i] + reference[3])
            )
            correction = np.eye(4)
            correction[:3, :3] = rotation
            correction[:3, 3] = centres[i] - rotation @ centres[i] + height_shift * reference[:3]
            transforms[i] = base @ correction

    before, after = [], []
    for i in anchors:
        xyz = samples[i]
        before.extend(np.abs(xyz @ reference[:3] + reference[3]))
        after.extend(np.abs(xyz @ transforms[i, 2, :3] + transforms[i, 2, 3]))
    report = {
        "mode": "per_view" if correct_per_view else "rigid",
        "coordinate_convention": "Z up, shared assumed floor Z=0; arbitrary scene units",
        "transform_convention": "X_aligned = T[:3,:3] @ X_original + T[:3,3]",
        "assumption": "One flat floor; independent venues do not share XY origin or metric scale",
        "reference_plane": reference.tolist(),
        "scene_scale": scale,
        "ransac_distance_threshold": threshold,
        "reliable_floor_views": len(anchors),
        "interpolated_or_extrapolated_views": size - len(anchors) if correct_per_view else 0,
        "residual_population": "confident floor/rug samples in reliable views, including RANSAC outliers",
        "absolute_floor_residual_before_p50_p90_p95": np.quantile(
            before, [0.5, 0.9, 0.95]
        ).tolist(),
        "absolute_floor_residual_after_p50_p90_p95": np.quantile(after, [0.5, 0.9, 0.95]).tolist(),
        "frames": records,
    }
    return FloorAlignment(base, transforms, planes, reliable, report)


def transform_observations(
    points: np.ndarray, frame_indices: np.ndarray, transforms: np.ndarray
) -> np.ndarray:
    """Transform each cloud observation by its source frame's rigid transform."""
    if points.ndim != 2 or points.shape[1] != 3 or frame_indices.shape != (len(points),):
        raise ValueError("Expected N,3 points and N frame indices")
    if not np.issubdtype(frame_indices.dtype, np.integer):
        raise ValueError("frame_indices must be integers")
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("transforms must be S,4,4")
    if not np.isfinite(points).all() or not np.isfinite(transforms).all():
        raise ValueError("Points and transforms must be finite")
    rotation = transforms[:, :3, :3]
    if (
        not np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6)
    ):
        raise ValueError("transforms must be proper homogeneous rigid transforms")
    if frame_indices.size and (frame_indices.min() < 0 or frame_indices.max() >= len(transforms)):
        raise ValueError("Cloud frame index exceeds the transform count")

    result = np.empty_like(points, dtype=np.float32)
    for i, transform in enumerate(transforms):
        selected = frame_indices == i
        result[selected] = points[selected] @ transform[:3, :3].T + transform[:3, 3]
    return result
