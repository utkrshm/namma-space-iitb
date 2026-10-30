# spacegen: any indoor video → a photoreal 3D space

`spacegen` turns a walkthrough video (or a folder of photos) into a Gaussian-splat 3D
space, fills what the camera never saw, and renders a tour video. Each gap it fills is
marked as generated, not measured.

```sh
cd spacegen
uv sync --extra fill
source scripts/env.sh           # CUDA toolkit for gsplat's kernels (compiled once)
uv run spacegen run my_walkthrough.mp4 --out runs/my_home
```

The run directory then holds:

| Path | What it is |
| --- | --- |
| `splat/scene.ply` | The splat in the standard 3DGS PLY format, made only from the capture. It opens in SuperSplat and other viewers. |
| `fill/scene.ply` | The same splat with the unseen gaps filled. Each Gaussian has a `filled` flag (0 = measured, 1 = generated) and a `support` direction mask. |
| `tour/tour.mp4` | Overview orbit, a 360° look-around in every area, and a walkthrough. The quality checks for each section are in `tour/tour.json`. |
| `demo/demo.mp4` | Demo: a sped-up top-down build-up of the splat, camera pose by camera pose, then a highlight reel of the best views from inside. |
| `report.json` | Every stage's numbers: frames, registration, held-out PSNR / SSIM / LPIPS, filled share. |
| `log.txt` | What happened, with timings. |

Every stage resumes. Rerunning the same command continues where it stopped. To change a
setting, use `--set section.key=value` (for example `--set splat.steps=7000`) or a TOML
file passed with `--config` (see `configs/default.toml`). `--until splat` stops before
the fill stage.

## Stages

1. `ingest`: chooses sharp frames by camera motion.
2. `sfm`: solves the cameras with ALIKED + LightGlue and COLMAP/GLOMAP, checking loops.
3. `depth`: monocular depth prior.
4. `splat`: MCMC Gaussian splatting with per-image exposure and pose correction. Large
   captures are trained in automatic blocks, then merged.
5. `fill`: finds unseen directions at each area's standing point, inpaints them with
   SDXL, distils the result into the splat, and flags it.
6. `tour`: renders the mp4 and checks every 10th frame for unrendered, unmeasured and
   near-camera pixels.
7. `demo`: renders the top-down build-up and the highlight reel.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the contracts,
[docs/LEARNINGS.md](docs/LEARNINGS.md) for the problems each stage handles, and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) for measured results.

## Benchmarks

```sh
uv run spacegen bench --data data/benchmarks --out runs/bench
```

This runs the Mip-NeRF 360 indoor scenes and Deep Blending with the standard
every-8th-image held-out split, using cameras from our own SfM.

## Tests

```sh
uv run pytest -q              # the CUDA tests need a GPU
uv run ruff check src tests
```
