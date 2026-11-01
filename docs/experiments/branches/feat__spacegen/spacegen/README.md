# spacegen: any indoor video → a photoreal 3D space

The run directory then holds:

| Path | What it is |
| --- | --- |
| `splat/scene.ply` | The splat in the standard 3DGS PLY format, made only from the capture. It opens in SuperSplat and other viewers. |
| `tour/tour.mp4` | Overview orbit, a 360° look-around in every area, and a walkthrough. The quality checks for each section are in `tour/tour.json`. |
| `demo/demo.mp4` | Demo: a sped-up top-down build-up of the splat, camera pose by camera pose, then a highlight reel of the best views from inside. |
| `log.txt` | What happened, with timings. |

## Stages

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
