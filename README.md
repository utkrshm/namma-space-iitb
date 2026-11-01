# Namma Space

**Turn a phone walkthrough into a 3D view of your space.**

Namma Space turns an indoor video or a folder of photos into a visual digital twin.
It also produces a walkthrough and a short highlight video.

**[Explore the three scenes in the live web viewer](https://gausswalk.pages.dev)**

Use **WASD** to move and **Q/E** to move down or up. Click the view once, then move the
mouse to look around without holding a button; press **Esc** to get the cursor back. On
touch screens, drag to look. The viewer also lets you follow the capture and compare the
reconstruction with the source video.

## What you get

- A **3D scene**, saved as a `.ply` file for viewers that support Gaussian splats.
- A **walkthrough video** with an overview and views along the captured route.
- A **demo video** showing the scene building up, followed by selected highlights.
- **Saved progress**, so an interrupted run can reuse completed work.

Record slowly, use good lighting, and film objects from several positions with
overlapping views. Areas the camera never saw may have gaps. Accurate real-world
measurements and destination-based route planning are future work.

## Run locally

### Requirements

- Python **3.11** and [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU with a working driver; Linux is the recommended environment.
- A CUDA **12.x development toolkit** compatible with the project's CUDA 12.6
  PyTorch build, plus a C++ compiler.
- Internet access for the first dependency and model downloads.

Memory and processing time depend on the capture size and settings. GPU components
compile on first use, so the first run can take longer.

### Setup

From the repository root:

```sh
uv sync --locked
export CUDA_HOME=/PATH/TO/CUDA
source scripts/env.sh
```

Replace `/PATH/TO/CUDA` with your toolkit location. If its `nvcc` compiler is already
on `PATH`, `source scripts/env.sh` can discover it without the export.

### Reconstruct a video

Put your capture in `data/`, then run:

```sh
uv run --locked namma-reconstruct run data/walkthrough.mp4 --out outputs/my_home
```

An ordered photo folder can replace the video path. Use names such as `0001.jpg`,
`0002.jpg`, and so on. Rerun the identical command to resume; choose a new output
folder when changing the capture or settings.

| File under `outputs/my_home/` | Purpose |
| --- | --- |
| `splat/scene.ply` | The captured 3D scene |
| `tour/tour.mp4` | The walkthrough |
| `demo/demo.mp4` | The build-up and highlights |
| `report.json` | Stage summaries and measurements |
| `log.txt` | Progress and diagnostic messages |

### Try the included capture

Sample footage is checked in under `tests/fixtures/`:

```sh
uv run --locked namma-reconstruct run tests/fixtures/room_short.mp4 --out outputs/sample_room
```

This five-second excerpt shows only part of a room. For a complete space, provide
a longer, connected walkthrough. A ready-made house scene is in
[`example/house/`](example/house/README.md).

## Development

```sh
uv run --locked namma-reconstruct run --help
uv run --locked pytest -q
uv run --locked ruff check
uv run --locked ruff format --check
```

Load `scripts/env.sh` before GPU tests. Add `-m 'not cuda'` to pytest to select
tests that do not require CUDA.

Defaults are in `configs/default.toml`. Use `--config YOUR_CONFIG.toml` to load
overrides, or `--set SECTION.KEY=VALUE` for individual settings. `--until splat`
stops after producing the 3D scene.

A separate preprocessing command is available for inspecting and preparing footage:

```sh
uv run --locked namma-preprocess data/walkthrough.mp4
```

It is not required before reconstruction.

| Folder | Contents |
| --- | --- |
| `src/` | Current pipeline and independent preprocessing |
| `configs/` | Current settings |
| `tests/` | Relevant tests and sample footage |
| `example/house/` | Published house scene and viewing instructions |
| `docs/` | Technical guides, decisions and experiment records |
| `experiments/` | Index of earlier experiment documentation |
| `data/`, `outputs/` | Original captures and generated files, excluded from Git |

## Learn more

- [Architecture](docs/ARCHITECTURE.md)
- [Pipeline contracts](docs/PIPELINE_ARCHITECTURE.md)
- [Learnings](docs/PIPELINE_LEARNINGS.md) and [benchmarks](docs/PIPELINE_BENCHMARKS.md)
- [Project state and experiment ledger](docs/STATE.md)
- [Design decisions](docs/PLANS.md)
- [Earlier experiments](experiments/README.md)
- [Competition goals](docs/GOALS.md)

Developed for the Namma Space challenge at IIT Bombay Techfest '26. Round 1 asks
for one 3–5 minute demonstration video covering scope, capture and the web walkthrough,
alongside the live viewer link, technical abstract and code repository.
