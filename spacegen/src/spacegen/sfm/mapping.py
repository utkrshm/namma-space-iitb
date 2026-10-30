"""COLMAP database, two-view verification, mapping and undistortion through pycolmap.

Conventions: COLMAP puts the centre of the top-left pixel at (0.5, 0.5), so integer-centre
ALIKED keypoints are shifted by +0.5 when written to the database.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..config import SfmConfig


def camera_params(width: int, height: int, model: str) -> str:
    """COLMAP's default focal guess (1.2 × the long side), centred principal point."""
    focal = 1.2 * max(width, height)
    if model == "SIMPLE_RADIAL":
        return f"{focal},{width / 2},{height / 2},0"
    if model in ("OPENCV", "OPENCV_FISHEYE"):
        return f"{focal},{focal},{width / 2},{height / 2},0,0,0,0"
    if model == "PINHOLE":
        return f"{focal},{focal},{width / 2},{height / 2}"
    raise ValueError(f"Unsupported camera model {model}")


def build_database(
    folder: Path,
    images: Path,
    frames: list[dict],
    features: list[dict],
    matches: dict[tuple[int, int], np.ndarray],
    config: SfmConfig,
) -> Path:
    """One camera per image size; keypoints, raw matches, then two-view verification.

    A database built from identical inputs is reused (``database.json`` records them).
    """
    import pycolmap

    folder.mkdir(parents=True, exist_ok=True)
    database_path, stamp_path = folder / "database.db", folder / "database.json"
    stamp = {
        "camera_model": config.camera_model,
        "pairs": sorted(f"{i}-{j}:{len(m)}" for (i, j), m in matches.items()),
        "min_inliers": config.min_inliers,
        "ransac_px": config.ransac_px,
    }
    if database_path.exists() and stamp_path.exists():
        if json.loads(stamp_path.read_text()) == stamp:
            return database_path
    stamp_path.unlink(missing_ok=True)
    database_path.unlink(missing_ok=True)

    names = [frame["name"] for frame in frames]
    sizes = {(frame["width"], frame["height"]) for frame in frames}
    pycolmap.Database.open(database_path).close()
    for width, height in sorted(sizes):
        subset = [f["name"] for f in frames if (f["width"], f["height"]) == (width, height)]
        options = pycolmap.ImageReaderOptions(
            camera_model=config.camera_model,
            camera_params=camera_params(width, height, config.camera_model),
        )
        pycolmap.import_images(database_path, images, pycolmap.CameraMode.SINGLE, subset, options)

    with pycolmap.Database.open(database_path) as database:
        ids = {image.name: image.image_id for image in database.read_all_images()}
        for name, feature in zip(names, features, strict=True):
            database.write_keypoints(ids[name], feature["keypoints"] + 0.5)
        for (i, j), pair_matches in matches.items():
            database.write_matches(ids[names[i]], ids[names[j]], pair_matches)

    pairs_path = folder / "pairs.txt"
    pairs_path.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in matches))
    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = config.min_inliers
    verification.ransac.max_error = 2 * config.ransac_px
    pycolmap.verify_matches(database_path, pairs_path, verification)
    stamp_path.write_text(json.dumps(stamp))
    return database_path


def run_mapper(folder: Path, database_path: Path, images: Path, mapper: str) -> list:
    """Reconstructions sorted by registered images, written to ``<folder>/sparse/<k>``."""
    import pycolmap

    sparse = folder / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    if mapper == "global":
        models = pycolmap.global_mapping(database_path, images, sparse)
    elif mapper == "incremental":
        options = pycolmap.IncrementalPipelineOptions()
        options.ba_global_function_tolerance = 1e-6
        models = pycolmap.incremental_mapping(database_path, images, sparse, options)
    else:
        raise ValueError(f"Unknown mapper {mapper}")
    ordered = sorted(models.values(), key=lambda m: m.num_reg_images(), reverse=True)
    for leftover in sparse.iterdir():
        shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
    for index, model in enumerate(ordered):
        target = sparse / str(index)
        target.mkdir()
        model.write(target)
    return ordered


def undistort(folder: Path, model_dir: Path, images: Path) -> Path:
    """Undistort to PINHOLE cameras (COLMAP layout ``images/`` and ``sparse/``)."""
    import pycolmap

    target = folder / "undistorted"
    if target.exists():
        shutil.rmtree(target)
    pycolmap.undistort_images(target, model_dir, images, jpeg_quality=95)
    return target
