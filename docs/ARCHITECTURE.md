# Submission architecture

`namma-reconstruct` calls `src.cli:main`. One root project and lockfile define the
dependencies. The live free-roaming viewer is https://gausswalk.pages.dev; its web
application is maintained separately from this reconstruction repository.

| Package | Input | Output |
| --- | --- | --- |
| `src.ingest` | Video or ordered photos | Selected upright images and source indices/timestamps |
| `src.sfm` | Selected images | Verified matches, cameras, undistorted images and sparse points |
| `src.depth` | Registered images and SfM observations | Monocular inverse-depth priors fitted to scene units |
| `src.splat` | Cameras, RGB, points and depth priors | Trained Gaussian model, camera split and held-out metrics |
| `src.tour` | Captured model and cameras | Overview, supported look-arounds and walkthrough video |
| `src.demo` | Captured model and cameras | Camera-by-camera build-up and checked highlight reel |

`src.pipeline.run` coordinates these stages. `src.partition` trains larger captures
in balanced spatial blocks, merges them by visibility and jointly fine-tunes the
result. `src.splat_stage` preserves the source-order held-out split.

`src.contracts.Camera` owns pose and pixel-grid conversions. Cameras use OpenCV
axes and world-to-camera transforms. Scene scale is arbitrary; `src.scene_frame`
provides a reversible Z-up display frame normalized by estimated camera height.
Area indices refer to the complete saved camera list. `src.support` records the
directions from which training cameras observed each Gaussian.

Every stage owns a folder under `outputs/<run>/`. Identical runs reuse completed
artifacts. `run.json` checks source path, file sizes/modification times and effective
settings; changed inputs or settings require a new output folder. `report.json`
persists completed-stage summaries, including early `--until` returns.

`src.preprocessing` is independently callable through `namma-preprocess`; the main
pipeline has its own ingest stage. Original captures remain read-only. See
[PREPROCESSING.md](PREPROCESSING.md) for its detailed provenance contract.

Only current implementation code and relevant tests are included. Earlier methods
remain documented in [experiments/README.md](../experiments/README.md); their source
implementations and upstream submodules are not part of this submission snapshot.
The published house example is under [`example/house/`](../example/house/README.md).
