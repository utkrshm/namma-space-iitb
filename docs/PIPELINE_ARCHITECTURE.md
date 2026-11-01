# Pipeline contracts

The runtime stages are `ingest → sfm → depth → splat → tour → demo`.

| Stage | Module | Artifacts under the run directory |
| --- | --- | --- |
| Ingest | `src.ingest` | `frames/frames.json`, upright `frames/images/`, `selection.png` |
| Camera solve | `src.sfm` | Features/matches, verified camera model, `sfm/undistorted/{images,sparse}`, `sfm/report.json` |
| Depth | `src.depth` | Per-image relative inverse depth in `depth/*.npy` and fitted scales/shifts in `depth/alignment.json` |
| Training | `src.splat`, `src.splat_stage`, `src.partition` | `splat/scene.ply`, `cameras.json`, `metrics.json`, checkpoint and RGB comparisons; optional spatial `blocks/` |
| Tour | `src.tour` | `tour/tour.mp4` and `tour/tour.json`, including view-quality checks and skipped arcs |
| Demo | `src.demo` | `demo/demo.mp4` and `demo/demo.json`, including selected clips and scores |

## Coordinates and input provenance

- World coordinates use arbitrary SfM units, not metres.
- Camera transforms are 4×4 world-to-camera matrices with OpenCV axes: X right,
  Y down and Z forward. Intrinsics refer to the specified image dimensions.
- Pixel centres use integer coordinates. Reading COLMAP cameras and rendering
  through gsplat explicitly convert their half-integer conventions.
- The reversible scene-frame transform supplies Z-up display coordinates, an
  estimated floor and camera-height units. It does not establish metric accuracy.
- Ingest records source frame indices, capture order, timestamps, display rotation
  and exported image sizes. Independent preprocessing retains additional PTS and
  image-transformation metadata.

## Training and evaluation

Every eighth source-position frame is held out by default. Missing registrations do
not shift that split. Held-out views are excluded from photometric training and
directional support estimation. Cameras and RGB remain aligned after undistortion.

Depth Anything V2 Small produces inverse-depth priors. Robust scale/shift fits align
them to observed SfM points. Training uses alpha-weighted inverse depth and normals
on textureless regions, with exponentially decaying weights. Images without a
reliable metric fit use a correlation-based fallback. View-dependent colour uses
spherical harmonics; exports retain the standard 3DGS PLY layout.

Large captures use balanced spatial partitions, overlap cameras, visibility-weighted
ownership and joint fine-tuning. The measured model is always `splat/scene.ply`.
Extra `support_0/1` PLY properties record a 64-direction observation mask.

## Presentation and resumption

Tours contain a cutaway overview, qualifying filmed-arc look-arounds and a smoothed
walkthrough. Look-arounds are checked before rendering; insufficient support skips
the arc. Hole checks use alpha below 0.5. Supersampling reduces flicker.

The demo shows the scene building up from capture observations and follows it with
highlight clips scored over seven moments. Neither video replaces the interactive
free-roaming viewer linked in the README.

Use identical inputs/settings to resume. Changed captures or configurations require
a new output folder. Effective settings, progress and summaries are saved as
`config.json`, `run.json`, `log.txt` and `report.json`.
