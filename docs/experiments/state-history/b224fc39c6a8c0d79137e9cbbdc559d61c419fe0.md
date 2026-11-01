# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `26-10-26 10:23:01 IST`

## Current Focus

Current work is on `feat/reconstruction` at `b5878d2`, focused on stabilizing base
reconstruction. Both full captures have real reconstructed point clouds and separate
floor-aligned exports. The interactive viewer remains unexecuted by the agent.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud export | Verified on both captures; interactive qualitative review pending |
| Floor-plane alignment | Implemented and verified on real outputs; Coorg floor residuals reduced, cross-view drift remains |
| Experimental multi-view refinement | Deferred; explored too early on separate branch `fix/recon-mv-refinement` (not implemented in this branch) |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `26-10-26 10:23:01 IST`
- **Decision:** Focus `feat/reconstruction` on base reconstruction; defer multi-view refinement experiments as a promising direction explored too early.
- **Reference:** “Focus base reconstruction; defer refinement experiments” in [PLANS.md](PLANS.md).

- **Timestamp:** `21-10-26 15:58:08 IST`
- **Decision:** Export separate semantic/RANSAC floor-aligned clouds; preserve Room with one rigid transform, use explicit single-floor per-view constraints for Coorg, and retain camera/point provenance.
- **Reference:** “Floor-constrained point-cloud postprocessing” in [PLANS.md](PLANS.md).

## Current Progress

### Deferred refinement experiments

- User-directed status: per-view camera/depth refinement is a **promising direction,
  but explored too early**. It improved measured correspondence and some living-wall
  appearance, while unsupported components and entrance layering remain unresolved.
- Revisit only after base reconstruction is stabilized. The experiment is preserved
  on `fix/recon-mv-refinement` at commit `6c7d168`; it is not implemented in this
  branch. Its targeted suite reported **88 passed** on that experimental branch only.
- The experimental run reported dense components **8 → 3**, but persistent entrance
  quality issues; these are exploratory results, not a validated claim that this
  branch or method is superior. Generated artifacts remain in ignored local
  `outputs/coorg_bnb/multiview_refinement/` and
  `outputs/room_m_block/multiview_refinement/`.

### Floor alignment after user review

- **Recorded `21-10-26 16:05:39 IST`:** Real SegFormer-B2 ADE20K floor/rug masks
  were inferred for all 197 views. Visually inspected both
  `outputs/<capture>/floor_diagnostics/floor_masks.jpg` sheets: visible floor/tiles
  are generally detected, but masks can omit rug areas or contain boundary errors.
  Coorg's original local floor normals differed by up to 18 degrees.
- Added optional `floor` dependencies, public floor-estimation/transform APIs and
  root `align_floor.py`. All original reconstruction artifacts are retained.
- Saved `outputs/room_m_block/floor_aligned/point_cloud.ply` using one global rigid
  transform, and `outputs/coorg_bnb/floor_aligned/point_cloud.ply` using per-view
  constraints. Each still has **1,000,000** RGB points and unchanged per-point
  confidence, source frame index and pixel identity. Both use Z-up, floor Z=0,
  arbitrary scene units; unrelated scenes do not share an XY origin or metric scale.
- Room supplied **7/36** reliable local planes, used only for diagnosis; its global
  rigid transform leaves relative geometry unchanged. Coorg supplied **90/161**
  reliable planes; the remaining 71 use source-time interpolation/extrapolation.
  The longest gap between reliable Coorg anchors is **27.18 s**.
- On Coorg's reliable-view fitting samples, median / p95 floor residual fell
  **0.01693 / 0.06824 → 0.000840 / 0.005185**. On all **736,706** confident semantic
  floor pixels (including rejected/interpolated views), median / p90 / p95 fell
  **0.01708 / 0.05843 / 0.07504 → 0.001017 / 0.009548 / 0.04554** scene units.
- Real ALIKED/LightGlue correspondence diagnostic: **36** adjacent-frame pairs,
  **7,773** matched confident 3D observations; median / p90 disagreement was
  **0.12430 / 0.54203 → 0.12699 / 0.54342**. Floor consistency improves, but general
  registration accuracy does not; per-view corrections can move walls/objects.
- Verified proper rotations, exact source colors/confidence/frame/pixel provenance,
  camera/extrinsic consistency, transformed plane equations and real PLY readback.
  Maximum sampled within-view distance change is **1.20e-7** scene units (float32).
  Inspected both `floor_aligned/floor_comparison.png` headless plots: Coorg's
  dominant floor band is much tighter, with residual outliers. Room's leveled
  geometry is identical before/after. No GUI viewer or tests were run.
- Full process, reproduction commands, assumptions and artifact contracts are in
  [the reconstruction guide](RECONSTRUCTION.md#floor-detection-and-alignment).
- Ruff lint/format checks on the floor module, public exports and both root scripts,
  `uv lock --check` and `git diff --check` pass. The public transform function exactly
  reproduces both saved aligned clouds. Real inference through the new segmentation
  function exactly reproduces the inspected Coorg view-0 floor mask.

### GPU capture execution

- **Recorded `21-10-26 12:54:56 IST`:** `git pull origin feat/reconstruction`
  fast-forwarded `6018f86` to `175fb6f`; active branch is `feat/reconstruction`.
- Installed preprocessing dependencies with `uv sync --extra revisits --group dev`,
  then ran `uv run --extra revisits namma-preprocess videos/room_m_block.mp4` and
  `uv run --extra revisits namma-preprocess videos/coorg_bnb.mp4` separately.
  Outputs reproduce the 36/161 geometry and 101/399 appearance counts below.
  Revisit inference used CPU because the initial CUDA 13 PyTorch build was
  incompatible with driver 535.309.01; both passes completed successfully.
- Visually inspected `debug/room_m_block_keyframes_00.jpg` and
  `debug/coorg_bnb_keyframes_00.jpg` under the corresponding preprocessing outputs.
  Exports are upright; later Room views visibly retain physical phone roll.
  Coorg's first sheet shows the door/window/floor sweep and its low-texture labels.
- `uv sync --extra revisits --extra reconstruction --extra viewer --group dev`
  now installs the official Linux CUDA 12.6 builds, PyTorch **2.14.1+cu126** and
  torchvision **0.29.1+cu126**, plus Open3D **0.20.0**. A real CUDA matrix
  multiplication passes on the **NVIDIA RTX A4000 (16 GB)**.
- Room: all **36 views** jointly inferred at **518×518**, **bfloat16**.
  Coorg: default 518 resolution failed with CUDA out-of-memory; retry with
  `--image-size 252` jointly inferred all **161 views** at **252×252**, **bfloat16**.
  Neither run drops frames or merges independently inferred chunks.
- Each capture's `outputs/<capture>/reconstruction/` contains `predictions.npz`,
  `metadata.json`, `preview.json`, `point_cloud.npz` and `point_cloud.ply`.
  Both clouds have **1,000,000 RGB points**, use depth-derived geometry, discard
  the bottom confidence quartile, and include observations from every input view.
  Each PLY is **25.75 MiB**. Units are arbitrary scene units, not metres.
- Artifact validation: source frame IDs/order exactly match the preprocessing
  geometry pool; depth shapes are `[36, 518, 518]` / `[161, 252, 252]`; camera
  matrices and cloud XYZ/confidence are finite. Real Open3D PLY readback matches
  saved NPZ XYZ and RGB exactly for both captures.
- Added root `view_point_cloud.py`, a thin CLI around the existing Open3D viewer.
  Per user instruction, no tests are added or run and the viewer is not executed.
  Interactive reconstructed-scene quality remains unassessed.
- `uv run --no-sync ruff check view_point_cloud.py`,
  `uv run --no-sync ruff format --check view_point_cloud.py`, `uv lock --check`
  and `git diff --check`: passed.

### Reconstruction interface (initial verification, 21-10-26 12:08:13 IST)

- **Recorded `21-10-26 12:08:13 IST`:** `VGGTInference` loads lazily and exposes
  reusable `load()`, `predict()` and `unload()`. It supports one image or ordered
  multi-view input, CUDA precision selection and optional first-image tracking queries.
- CPU outputs include depth/confidence, direct point maps/confidence, depth-derived
  world points, camera encodings/iterations, model/original intrinsics, inverse poses,
  optional tracks and source provenance. Coordinates and arbitrary scene units are
  explicit in [the reconstruction contract](RECONSTRUCTION.md).
- The preprocessing adapter consumes `load_result()`, defaults to geometry frames,
  preserves clip/PTS order and source timing/rotation, and records model-specific
  resize/padding transforms. The preprocessing source and existing capture artifacts
  are unchanged. All selected views run jointly; there is no unaligned chunk merging.
- `get_recon.py` / `namma-reconstruct` export processed arrays, metadata, preview
  settings and coloured point clouds. `--no-view` supports headless export;
  `--view existing.ply` loads only Open3D, without Torch or model weights.
- `uv lock --check`: passed. `uv sync --extra revisits --extra reconstruction`:
  passed, retaining existing revisit dependencies. Actual upstream VGGT model-class
  and camera-decoder imports pass without instantiating a model.
- `uv run --no-sync pytest tests/reconstruction -q -ra`: **8 passed, 1 skipped in
  13.69 s**. Real short-video preprocessing/handoff, relocatability, image transforms,
  analytic camera/depth reprojection, confidence filtering, NumPy/JSON serialization,
  query validation, CLI validation and lazy imports are covered without model inference.
- `uv run --no-sync ruff check src tests get_recon.py preprocess_capture.py --exclude src/reconstruction/deps`:
  passed. Formatting checks on new modules/tests/public exports and root script pass.
  Root CLI help and `git diff --check` pass.
- The real Open3D RGB PLY round-trip test was initially skipped: the 382.2 MiB Open3D wheel
  download exceeded both the 120 s and 300 s installation timeouts. No fake viewer
  substitutes for that check. Interactive GUI viewing and visual reconstructed-scene
  inspection were not run at that stage. Pretrained CUDA inference was deferred
  until this GPU-machine execution request; see the newer results above.

### Frozen preprocessing results

- **Recorded `21-10-26 03:09:09 IST`:** Round-three corrections are implemented and both
  full captures rerun. See [review results and every accepted pair](PREPROCESSING_REVIEW.md)
  for before/after metrics, trigger histograms, labels, timings and debug image paths.
- Ingest was independently checked by the user: exact source PTS for all 2,426 /
  12,907 observations and four geometry exports pixel-identical to FFmpeg. Those
  decoding conventions remain unchanged. Source videos remain unchanged and GPS tags are not copied.
- Whole-frame eigenvalue floors exclude noise corners before cell occupancy;
  reliability uses eight consistent tracks over three cells, not a survival gate.
  Schema 2 adds per-frame edge fraction and re-blur loss. `low_texture` is informational,
  never a selection quality failure, and has separate info intervals.
- `uv sync --extra revisits` installed real inference dependencies and retained dev
  dependencies. Cached weights ran on the 4 GB RTX 3050 using **CUDA**, without fallback.
- `HF_HUB_OFFLINE=1 uv run --extra revisits pytest -q -ra`: **28 passed, 1 xfailed,
  0 skipped, 1 upstream warning in 56.52 s**. The expected failure transparently
  records the ambiguous pure per-frame blur label at 49.5 s; the full pipeline
  flags it with temporal context. Real revisit and cross-building negative tests ran.
- Additional focused cache/splice-report assertions passed: **2 tests in 22.27 s**.
- `uv run --extra revisits ruff check src tests/preprocessing preprocess_capture.py`: passed.
- New fixtures include the metadata-stripped 480-frame ceiling cut at 72–80 s,
  and eight upright analysis PNGs with source timestamps listed in their README.
  Tests now enforce exact 256-frame decoding, positive ~66.6 ms timing flags and
  spatial rejection of >=50 clustered real correspondences versus an equal spread set.
- Added five metadata-free 576×1024 revisit PNGs. The real door positive accepts
  >=50 unambiguous inliers; both reviewed window controls reject. The real splice
  produces exactly one unsupported link and no supported cross-boundary edge.
- The self-match guard uses normalized ALIKED L2 distances, a **0.8 ratio** and a
  **32 px** exclusion radius on both images. Only unambiguous matches contribute
  geometric inliers/spread. Each edge records `ambiguous_matches`; the report has
  `revisits.rejected_repetitive` and per-pair raw/filtered diagnostics.
- Probe, score and select resumed on both videos; their artifact hashes are
  unchanged. Legacy migration used source provenance verified before editing;
  future invalidation uses pass-specific module/record hashes. Only link/export ran.

### Current capture results

| Capture | CSV rows | Geometry | Appearance | Sequential | Revisit | Components |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `room_m_block` | 2,426 | 36 | 101 | 35 | 0 | 1 |
| `coorg_bnb` | 12,907 | 161 | 399 | 160 | 4 | 1 |

- Both sequential graphs are connected before adding revisit edges. Coorg tracking
  breaks dropped **19 → 0**, and its 72–79 s geometry count **20 → 10**. Room's
  8.8–14 s geometry count dropped **15 → 5**. Neither capture has weak-link items.
- All four reviewed false-blur observations now receive informational low texture.
  Coorg has four low-quality intervals and seven separate low-texture info intervals;
  Room has no low-quality intervals. The 5 s sharp observation also gets low texture
  information, not a quality failure. The 25 s and 140 s examples remain unflagged.
- Real DINOv2 retrieval found zero Room candidates above the configured similarity
  threshold, and 28 Coorg candidates; filtered geometry verification accepted **4**
  rather than 18. Fourteen previously accepted pairs are now rejected: all thirteen
  grille links and one luggage link with only 49 unambiguous matches. Thresholds
  were not loosened to preserve it.
  Every accepted pair is outside the 10 s temporal exclusion and has a debug image.

### Inspection and measured timings

- Inspected all four current match images: two shared-door/key/sconce pairs and
  two shared-luggage/table/trim pairs. All previously accepted grille matches were
  rejected. Pair IDs, before/after inlier and ambiguity counts, and current image
  paths are in the review document. Rejected stale match images were cleaned up.
- The user-labelled PNGs are upright and preserve the source observations. The
  49.5 s door cannot be reliably separated from sharp doors using the chosen
  per-frame re-blur measure at 640 px; this is explicitly documented, not overfit.

| Capture | Probe / score / select | Link (CUDA) | Export |
| --- | --- | ---: | ---: |
| Room | Resumed, unchanged | 37.88 s | 46.37 s |
| Coorg | Resumed, unchanged | 182.19 s | 240.26 s |

## Blockers

- Floor constraints assume one flat storey. Semantics/depth errors remain, 71 Coorg
  views lack an accepted local plane, and per-view leveling is not bundle adjustment.
  Further consistency requires actual multi-view optimization, not more flattening.
- Interactive point-cloud quality review is pending the user's return; the standalone
  viewer has intentionally not been run.
- Coorg's full 161-view geometry pool exceeds the A4000's memory at 518 resolution.
  A complete joint reconstruction is available at 252 resolution; higher-resolution
  full-pool inference needs more GPU memory or an explicitly adopted memory strategy.
- No preprocessing execution or dependency blockers remain for the supplied SDR videos.
- Per-frame blur classification is ambiguous for the labelled 49.5 s door at
  analysis resolution; the full pipeline uses context. Conservative self-similarity
  filtering can discard true loop links, as observed for one luggage pair.
- HDR conversion remains outside the implementation; HDR inputs fail explicitly.

## Next Actions

1. Use `uv run --no-sync python view_point_cloud.py outputs/coorg_bnb/floor_aligned/point_cloud.ply`
   and the corresponding Room path to compare floor-aligned and original clouds.
2. Record visual reconstruction quality; consider higher-resolution Coorg inference
   on a larger-memory GPU if needed. Peak GPU memory and inference timings were not
   measured during these successful runs.
3. Use verified scene geometry to plan fusion, surface generation and downstream
   navigation. Keep preprocessing frozen and preserve source/candidate provenance.
