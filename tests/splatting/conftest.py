"""Analytic calibrated-plane fixtures, using actual images and numeric map files."""

import json

import cv2
import numpy as np
import pytest


@pytest.fixture
def plane_scene(tmp_path):
    root = tmp_path / "refined"
    root.mkdir()
    frames = []
    height, width = 32, 40
    k = np.array([[45, 0, 19.5], [0, 45, 15.5], [0, 0, 1]], dtype=float)
    y, x = np.mgrid[:height, :width]
    for index in range(5):
        center = np.array([0.03 * index, 0, 0])
        transform = np.eye(4)
        transform[:3, 3] = -center
        depth = np.full((height, width), 2, dtype=np.float32)
        points = np.stack(
            ((x - k[0, 2]) * depth / k[0, 0], (y - k[1, 2]) * depth / k[1, 1], depth), axis=-1
        )
        points = (points + center).astype(np.float32)
        rgb = np.stack(
            (
                np.clip(points[..., 0] + 0.5, 0, 1),
                np.clip(points[..., 1] + 0.5, 0, 1),
                np.full_like(depth, 0.5),
            ),
            axis=-1,
        )
        image = (rgb * 255).round().astype(np.uint8)
        assert cv2.imwrite(str(root / f"{index}.png"), image[..., ::-1])
        np.save(root / f"{index}_depth.npy", depth)
        np.save(root / f"{index}_points.npy", points)
        frames.append(
            {
                "frame_id": f"clip:{index * 100}",
                "timestamp_s": index / 10,
                "width": width,
                "height": height,
                "image": f"{index}.png",
                "intrinsics": k.tolist(),
                "world_to_camera": transform.tolist(),
                "depth": f"{index}_depth.npy",
                "points": f"{index}_points.npy",
                "source_to_image": np.eye(3).tolist(),
                "split": "val" if index == 4 else "train",
            }
        )
    manifest = {
        "schema_version": 1,
        "camera_convention": "opencv_world_to_camera",
        "depth_convention": "camera_z",
        "point_convention": "world",
        "camera_model": "pinhole",
        "pixel_convention": "integer_centers",
        "scene_units": "arbitrary",
        "frames": frames,
    }
    path = root / "scene.json"
    path.write_text(json.dumps(manifest))
    return path
