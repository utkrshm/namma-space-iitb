"""COLMAP database, geometric verification, mapping and undistortion through pycolmap."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .contracts import SfmConfig


def initial_camera_params(frames: list[dict], vggt_cameras: Path | None, config: SfmConfig):
    """SIMPLE_RADIAL ``f,cx,cy,k`` from VGGT's model-grid focal, or COLMAP's 1.2·max(w,h).

    VGGT predicts intrinsics on a square 518 grid whose long side is the full image's long
    side, so the focal scales by ``long_side / 518``.
    """
    width, height = frames[0]["width"], frames[0]["height"]
    focal = 1.2 * max(width, height)
    if vggt_cameras is not None and Path(vggt_cameras).exists():
        with np.load(vggt_cameras) as data:
            model_focal = float(np.median(data["intrinsics"][:, [0, 1], [0, 1]]))
        focal = model_focal * max(width, height) / 518.0
    if config.camera_model == "SIMPLE_RADIAL":
        return f"{focal},{width / 2},{height / 2},0"
    if config.camera_model == "OPENCV":
        return f"{focal},{focal},{width / 2},{height / 2},0,0,0,0"
    raise ValueError(f"Unsupported camera model {config.camera_model}")


def build_database(
    output_dir: Path,
    frames: list[dict],
    features: list[dict],
    matches: dict[tuple[int, int], np.ndarray],
    camera_params: str,
    config: SfmConfig,
) -> Path:
    """Write one shared camera, keypoints and raw matches, then verify every pair.

    Verification is the slow part, so a database built with identical inputs is reused
    (``database.json`` records them).
    """
    import pycolmap

    database_path = output_dir / "database.db"
    stamp_path = output_dir / "database.json"
    stamp = {
        "camera_model": config.camera_model,
        "camera_params": camera_params,
        "pairs": sorted(f"{i}-{j}:{len(m)}" for (i, j), m in matches.items()),
        "min_inliers": config.min_inliers,
        "ransac_px": config.ransac_px,
    }
    reusable = database_path.exists() and stamp_path.exists()
    if reusable and json.loads(stamp_path.read_text()) == stamp:
        return database_path
    stamp_path.unlink(missing_ok=True)
    database_path.unlink(missing_ok=True)
    pycolmap.Database.open(database_path).close()  # import_images needs an existing file
    names = [frame["name"] for frame in frames]
    options = pycolmap.ImageReaderOptions(
        camera_model=config.camera_model, camera_params=camera_params
    )
    pycolmap.import_images(
        database_path, output_dir / "images", pycolmap.CameraMode.SINGLE, names, options
    )

    with pycolmap.Database.open(database_path) as database:
        ids = {image.name: image.image_id for image in database.read_all_images()}
        for name, feature in zip(names, features):
            # COLMAP puts the centre of the top-left pixel at (0.5, 0.5).
            database.write_keypoints(ids[name], feature["keypoints"] + 0.5)
        for (i, j), pair_matches in matches.items():
            database.write_matches(ids[names[i]], ids[names[j]], pair_matches)

    pairs_path = output_dir / "pairs.txt"
    pairs_path.write_text("".join(f"{names[i]} {names[j]}\n" for i, j in matches))
    verification = pycolmap.TwoViewGeometryOptions()
    verification.min_num_inliers = config.min_inliers
    verification.ransac.max_error = 2 * config.ransac_px
    pycolmap.verify_matches(database_path, pairs_path, verification)
    stamp_path.write_text(json.dumps(stamp))
    return database_path


def run_mapper(output_dir: Path, database_path: Path, config: SfmConfig):
    """Run the chosen mapper; returns reconstructions sorted by registered images."""
    import pycolmap

    sparse = output_dir / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    if config.mapper == "global":
        models = pycolmap.global_mapping(database_path, output_dir / "images", sparse)
    elif config.mapper == "incremental":
        options = pycolmap.IncrementalPipelineOptions()
        options.ba_global_function_tolerance = 1e-6
        models = pycolmap.incremental_mapping(database_path, output_dir / "images", sparse, options)
    else:
        raise ValueError(f"Unknown mapper {config.mapper}")
    ordered = sorted(models.values(), key=lambda m: m.num_reg_images(), reverse=True)
    for index, model in enumerate(ordered):
        folder = sparse / str(index)
        folder.mkdir(exist_ok=True)
        model.write(folder)
    return ordered


def undistort(output_dir: Path, model_dir: Path) -> Path:
    """Undistort to one PINHOLE camera (COLMAP layout ``images/`` and ``sparse/``)."""
    import pycolmap

    target = output_dir / "undistorted"
    if target.exists():
        shutil.rmtree(target)
    pycolmap.undistort_images(target, model_dir, output_dir / "images", jpeg_quality=95)
    return target
