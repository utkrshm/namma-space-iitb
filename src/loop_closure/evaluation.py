"""Compare camera solutions on identical image correspondences and render inspection plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import FrameMatches, FrameModel


def correspondence_disagreement(models: list[FrameModel], matches: list[FrameMatches]) -> dict:
    """3D distance between the two depth-unprojected ends of each match.

    Each distance is divided by the mean camera-to-point distance of its two observations,
    so the metric is invariant to global similarity scale. Only matches with valid
    confident depth at both ends count, so every method is scored on the same pixels
    when they share owners and depth masks.
    """
    distances, pair_medians, per_pair = [], [], []
    for item in matches:
        if not len(item.xy_a):
            continue
        a = models[item.frame_a].unproject(item.xy_a)
        b = models[item.frame_b].unproject(item.xy_b)
        valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if not valid.any():
            continue
        ca = models[item.frame_a].camera_to_world[:3, 3]
        cb = models[item.frame_b].camera_to_world[:3, 3]
        scale = (np.linalg.norm(a[valid] - ca, axis=1) + np.linalg.norm(b[valid] - cb, axis=1)) / 2
        value = np.linalg.norm(a[valid] - b[valid], axis=1) / scale
        distances.append(value)
        pair_medians.append(float(np.median(value)))
        per_pair.append(
            {
                "frames": [item.frame_a, item.frame_b],
                "times_s": [models[item.frame_a].timestamp_s, models[item.frame_b].timestamp_s],
                "owners": [models[item.frame_a].owner_section, models[item.frame_b].owner_section],
                "count": len(value),
                "median": float(np.median(value)),
            }
        )
    if not distances:
        return {"pairs": 0, "count": 0}
    values = np.concatenate(distances)
    return {
        "pairs": len(pair_medians),
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "fraction_below_0_05": float(np.mean(values < 0.05)),
        "pair_median_of_medians": float(np.median(pair_medians)),
        "pairs_with_median_above_0_10": int(np.sum(np.asarray(pair_medians) > 0.10)),
        "per_pair": per_pair,
    }


def _up_axis(models: list[FrameModel]) -> np.ndarray:
    """Phones are held upright: image -y averaged over cameras approximates world up."""
    up = -np.mean([m.camera_to_world[:3, :3] @ np.array([0.0, 1.0, 0.0]) for m in models], axis=0)
    return up / np.linalg.norm(up)


def render_comparison(
    clouds: dict[str, Path], models: dict[str, list[FrameModel]], output: Path
) -> list[Path]:
    """Top-down wall-height slices coloured by source time, plus RGB, for each method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    names = list(clouds)
    figure, axes = plt.subplots(len(names), 2, figsize=(16, 8 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        data = np.load(clouds[name])
        points, frames = data["points"], data["frame_indices"]
        frame_models = models[name]
        times = np.asarray([m.timestamp_s for m in frame_models])
        up = _up_axis(frame_models)
        a = np.cross(up, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(up, a)
        height = points @ up
        low, high = np.percentile(height, [2, 98])
        band = (height > low + 0.35 * (high - low)) & (height < low + 0.6 * (high - low))
        chosen = np.flatnonzero(band)
        if len(chosen) > 400_000:
            chosen = np.random.default_rng(0).choice(chosen, 400_000, replace=False)
        centers = np.stack([m.camera_to_world[:3, 3] for m in frame_models])
        for column, colours in enumerate((times[frames[chosen]], data["colors"][chosen] / 255.0)):
            axis = axes[row, column]
            if column == 0:
                axis.scatter(
                    points[chosen] @ a, points[chosen] @ b, c=colours, cmap="turbo", s=0.05
                )
            else:
                axis.scatter(points[chosen] @ a, points[chosen] @ b, c=colours, s=0.05)
            axis.plot(centers @ a, centers @ b, "k-", lw=0.6)
            axis.set_aspect("equal")
            axis.set_title(
                f"{name}: wall-height slice, " + ("colour = source time" if column == 0 else "RGB")
            )
    figure.tight_layout()
    path = output / "topdown_comparison.png"
    figure.savefig(path, dpi=70)
    plt.close(figure)
    return [path]


def render_camera_slices(
    clouds: dict[str, Path], models: dict[str, list[FrameModel]], times_s: list[float], output: Path
) -> list[Path]:
    """Horizontal slice through each chosen source camera, coloured by source time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths = []
    loaded = {name: np.load(path) for name, path in clouds.items()}
    for target in times_s:
        figure, axes = plt.subplots(1, len(clouds), figsize=(7 * len(clouds), 7), squeeze=False)
        for column, (name, data) in enumerate(loaded.items()):
            frame_models = models[name]
            times = np.asarray([m.timestamp_s for m in frame_models])
            k = int(np.argmin(np.abs(times - target)))
            world_to_camera = np.linalg.inv(frame_models[k].camera_to_world)
            points, frames = data["points"], data["frame_indices"]
            camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            own = camera[frames == k]
            with np.errstate(divide="ignore", invalid="ignore"):
                u, v = camera[:, 0] / camera[:, 2], camera[:, 1] / camera[:, 2]
            own_u, own_v = own[:, 0] / own[:, 2], own[:, 1] / own[:, 2]
            far = np.percentile(own[:, 2], 98) * 1.6
            band = (
                (camera[:, 2] > 0.02)
                & (camera[:, 2] < far)
                & (u > own_u.min())
                & (u < own_u.max())
                & (np.abs(v - np.median(own_v)) < 0.08)
            )
            axis = axes[0, column]
            axis.scatter(
                camera[band, 0],
                camera[band, 2],
                c=times[frames[band]],
                cmap="turbo",
                s=0.6,
                vmin=0,
                vmax=times.max(),
            )
            axis.scatter([0], [0], c="k", marker="^")
            axis.set_aspect("equal")
            axis.set_title(f"{name} @ {times[k]:.1f}s (frame {k}): slice coloured by source time")
        figure.tight_layout()
        path = output / f"camera_slice_{target:06.1f}s.png"
        figure.savefig(path, dpi=70)
        plt.close(figure)
        paths.append(path)
    return paths
