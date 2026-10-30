"""Benchmarks: run the pipeline on public indoor scenes and score the held-out views.

Scenes (under ``--data``):

| Scene    | Source           | Images used                 |
| -------- | ---------------- | --------------------------- |
| room     | Mip-NeRF 360     | ``360_v2/room/images_2``    |
| counter  | Mip-NeRF 360     | ``360_v2/counter/images_2`` |
| kitchen  | Mip-NeRF 360     | ``360_v2/kitchen/images_2`` |
| bonsai   | Mip-NeRF 360     | ``360_v2/bonsai/images_2``  |
| playroom | Deep Blending    | ``db/playroom/images``      |
| drjohnson| Deep Blending    | ``db/drjohnson/images``     |

The protocol is the one 3DGS reports: every 8th image (sorted by name) is held out,
indoor Mip-NeRF 360 scenes use the half-resolution images, and scores are PSNR, SSIM
and LPIPS (VGG) without colour correction. Unlike published numbers, the cameras come
from our own SfM, *not* the dataset's COLMAP poses: the benchmark tests the whole
pipeline, not only the splat stage.

Output: ``<out>/<scene>/`` (a normal run directory) and ``<out>/results.json`` / ``.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run

SCENES = {
    "room": "360_v2/room/images_2",
    "counter": "360_v2/counter/images_2",
    "kitchen": "360_v2/kitchen/images_2",
    "bonsai": "360_v2/bonsai/images_2",
    "playroom": "db/playroom/images",
    "drjohnson": "db/drjohnson/images",
}

# 3DGS (Kerbl et al. 2023), 30k iterations, per-scene supplementary tables: PSNR, SSIM, LPIPS.
PUBLISHED_3DGS = {
    "room": (30.63, 0.914, 0.220),
    "counter": (28.70, 0.905, 0.204),
    "kitchen": (30.32, 0.922, 0.129),
    "bonsai": (31.98, 0.938, 0.205),
    "playroom": (30.04, 0.906, 0.241),
    "drjohnson": (28.77, 0.899, 0.244),
}


def run_benchmarks(data: Path, output: Path, config: PipelineConfig, scenes=None, device="cuda:0"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for scene in scenes or list(SCENES):
        source = Path(data) / SCENES[scene]
        if not source.exists():
            print(f"skip {scene}: {source} missing")
            continue
        report = run(source, output / scene, config, until="splat", device=device)
        val = report["splat"]["val"]
        results[scene] = {
            "psnr": val["psnr"],
            "ssim": val["ssim"],
            "lpips": val["lpips"],
            "psnr_cc": val["psnr_cc"],
            "registered": report["sfm"]["largest_model"]["registered"],
            "frames": report["sfm"]["frames"],
            "gaussians": report["splat"]["gaussians"],
            "train_minutes": report["splat"]["train_seconds"] / 60,
            "published_3dgs": PUBLISHED_3DGS.get(scene),
        }
        results_path.write_text(json.dumps(results, indent=1) + "\n")
        (output / "results.md").write_text(table(results))
    print(table(results))
    return results


def table(results: dict) -> str:
    lines = [
        "| Scene | PSNR | SSIM | LPIPS | 3DGS PSNR / SSIM / LPIPS | Registered | Gaussians "
        "| Minutes |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for scene, r in results.items():
        published = " / ".join(str(v) for v in r["published_3dgs"]) if r["published_3dgs"] else "–"
        lines.append(
            f"| {scene} | {r['psnr']:.2f} | {r['ssim']:.3f} | {r['lpips']:.3f} | {published} | "
            f"{r['registered']}/{r['frames']} | {r['gaussians']:,} | {r['train_minutes']:.0f} |"
        )
    return "\n".join(lines) + "\n"
