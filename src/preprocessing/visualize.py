"""Non-interactive debug plots and paginated contact sheets for real inspection."""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

from .storage import image_name


def visualize(output_dir, clips, scores, selected, measurements, matches, config):
    debug = output_dir / "debug"
    debug.mkdir(exist_ok=True)
    paths = []
    for clip in clips:
        rows = [r for r in scores if r.clip_id == clip.clip_id]
        times = [r.t_s for r in rows]
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        axes[0].plot(times, [r.sharpness for r in rows], linewidth=0.5)
        axes[0].set_ylabel("raw sharpness")
        axes[1].plot(times, [r.sharpness_rel_short for r in rows], lw=0.5, label="short")
        axes[1].plot(times, [r.sharpness_rel_long for r in rows], lw=0.5, label="long")
        axes[1].legend()
        axes[1].set_ylabel("relative sharpness")
        axes[2].plot(times, [r.flow_px if r.flow_px is not None else np.nan for r in rows], lw=0.5)
        axes[2].set_ylabel("flow (analysis px)")
        measured = [m for m in measurements if m["frame_id"].startswith(clip.clip_id + ":")]
        axes[3].plot(
            [m["t_s"] for m in measured],
            [m["overlap"] if m["overlap"] is not None else np.nan for m in measured],
            lw=0.5,
        )
        axes[3].set_ylabel("tracked overlap proxy")
        lookup = {r.frame_id: r for r in rows}
        for record in selected:
            if record.pool == "geometry" and record.frame_id in lookup:
                for ax in axes:
                    ax.axvline(lookup[record.frame_id].t_s, color="green", alpha=0.2, lw=0.7)
        flagged = [r for r in rows if "suspected_blur" in r.flags]
        axes[0].scatter(
            [r.t_s for r in flagged],
            [r.sharpness for r in flagged],
            s=5,
            color="red",
            label="blur/detail flags",
        )
        axes[0].legend()
        axes[-1].set_xlabel("Seconds from stream start; green = geometry selection")
        fig.suptitle(clip.clip_id)
        fig.tight_layout()
        path = debug / f"{clip.clip_id}_timeline.png"
        fig.savefig(path, dpi=120)
        paths.append(path)
        plt.close(fig)
        geometry = [f for f in selected if f.pool == "geometry" and f.frame_id in lookup]
        page_size, columns = config.export.contact_page_size, config.export.contact_columns
        for page, offset in enumerate(range(0, len(geometry), page_size)):
            group = geometry[offset : offset + page_size]
            nrows = (len(group) + columns - 1) // columns
            fig, axes = plt.subplots(
                nrows, columns, figsize=(columns * 2.3, nrows * 3.9), squeeze=False
            )
            for ax in axes.ravel():
                ax.axis("off")
            for ax, record in zip(axes.ravel(), group):
                bgr = cv2.imread(str(output_dir / record.image_path))
                size = config.export.thumbnail_long_side / max(bgr.shape[:2])
                thumb = cv2.resize(bgr, None, fx=size, fy=size)
                ax.imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
                row = lookup[record.frame_id]
                ax.set_title(
                    f"{row.t_s:.2f}s | S={row.sharpness:.0f}\n{', '.join(row.flags)}", fontsize=7
                )
            fig.tight_layout()
            path = debug / f"{clip.clip_id}_keyframes_{page:02d}.jpg"
            fig.savefig(path, dpi=110)
            paths.append(path)
            plt.close(fig)
    for match in matches:
        images = [
            cv2.imread(str(output_dir / ".cache/revisits" / image_name(match[key], "png")))
            for key in ("a", "b")
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8, 7))
        for ax, image, key in zip(axes, images, ("a", "b")):
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            points = np.asarray(match[f"points_{key}"])
            ax.scatter(points[:, 0], points[:, 1], s=4, c=np.arange(len(points)), cmap="turbo")
            ax.set_title(match[key], fontsize=8)
            ax.axis("off")
        for a, b in zip(match["points_a"], match["points_b"]):
            fig.add_artist(
                ConnectionPatch(
                    a,
                    b,
                    "data",
                    "data",
                    axesA=axes[0],
                    axesB=axes[1],
                    color="lime",
                    alpha=0.25,
                    linewidth=0.4,
                )
            )
        fig.tight_layout()
        path = output_dir / match["image_path"]
        fig.savefig(path, dpi=120)
        paths.append(path)
        plt.close(fig)
    return paths
