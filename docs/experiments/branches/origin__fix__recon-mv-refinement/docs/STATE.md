# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `26-10-26 10:20:08 IST`

## Current Focus

Multi-view refinement is a **promising direction, but explored too early**. Preserve
its executed dense-anchor Coorg run, constrained-pose export, joint-input M Block run,
real matcher controls and export audits as experimental evidence; defer further
refinement until base reconstruction is stabilized, then reassess its interaction
with that stage. Active work is moving to `feat/reconstruction`; no evidence here
establishes that branch as superior. Current refinement results improve some held-out
metrics and visible structure but **do not meet the scene-quality goal**: entrance/
upper-wall layering remains and the cloud is not navigation-ready.
The latest execution checklist, JSON-derived measurements, exact commands and
remaining checks are in [REFINEMENT.md](REFINEMENT.md). Results recorded here are
the current machine's actual outputs; older results below remain historical and are
not substituted for these runs. See [PLANS.md](PLANS.md) for the adopted design.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud export | Verified on both captures; Coorg original/section clouds also reviewed headlessly |
| Floor-plane alignment | Implemented and verified on real outputs; Coorg floor residuals reduced, cross-view drift remains |
| Coorg high-resolution section alignment | Verified on real Coorg outputs: connected global graph, all seven primary handoffs pass, improved consistency and visual structure; residual distortion remains |
| Multi-view camera/depth refinement | Implemented and exercised on Coorg and M Block; all execution checks and audits completed, scene-quality goal not met |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `26-10-26 10:20:08 IST`
- **Decision:** Preserve multi-view refinement as promising but premature; defer it until base reconstruction is stabilized, reassess its interaction then, and make `feat/reconstruction` active work. No comparative superiority claim is supported.
- **Reference:** “Defer multi-view refinement and return to reconstruction” in [PLANS.md](PLANS.md).

## Current Progress

### Multi-view refinement execution (current machine)

- **Recorded `26-10-26 10:12:33 IST`:** Main execution regenerated the Coorg baseline
  because prior multi-view outputs were not transferred. Its deterministic base tracks
  remain **29,409 tracks / 78,861 observations**, 660 candidates and 252 supported
  pairs. The baseline test median/p90 is **0.014908441925989 / 0.055694211609**.
- Coorg `dense_links_v2` added **363 tracks / 726 observations**. Accepted pair counts:
  23/24 **104**, 24/25 **70**, 36/37 **83**, 37/38 **59**, 140/141 **47**. The graph
  changed from **8 to 3 components**, with anchors **0, 51, 140**; there is still no
  link to isolated views 51–57 or the dining group 140–160. Main inspected two dense
  match sheets from five accepted pair groups: floor/shoe anchors look plausible, while the repetitive grille
  remains ambiguous and there is no independent accuracy proof. The dense result
  improves the living wall, but entrance/upper-wall layering and green floor patches
  remain; it is rejected as a successful whole-scene reconstruction.
- Baseline refinement test median/p90 changes **0.014908441925989 / 0.055694211609 →
  0.002817065372553119 / 0.01319972200159**. Dense-anchor test median/p90 changes
  **0.0149222511 / 0.0557457992 → 0.00289809773 / 0.01325093729**; its denominator
  changes with the added observations. The constrained-pose test used
  `--field-steps 0 --geometry-prior-weight 0.2` and exact effective dense tracks
  `tracks_c53000b9c55f7c84.npz`; its selected initial state is unchanged, before and
  after **0.0149222511 / 0.0557457992**. Coorg dense output contains **1,673,762 points**.
- The joint-input M Block route completed for all **36 frames** after an OpenCV remap
  batch fix and resumption with `--export-only`. On **2,254** test pairs, median/p90
  changed **0.024690008895 / 0.070772898169 → 0.003012746528 / 0.009096651970**.
  It has one training component anchored at view 0 and **1,395,236 points**. Main's
  visual review sees slight bed/room improvement, but holes remain; it is not
  navigation-ready.
- All four `audit_report.json` reports passed: regenerated baseline, dense Coorg,
  constrained-pose Coorg and M Block. Dense Coorg also passed the enhanced
  exact-metadata audit. Dense
  held-out protection compared baseline
  **78,861** to augmented **79,587** observations: **726** additions across five
  pairs, original arrays/pairs unchanged, all additions training-only, **0** held-out
  mask hits, nearest held-out distance **10.1782547 px** (median **46.5522 px**).
  The real matcher controls accepted 400 positive warped matches (24×16 px), with median/p90 residual
  **0.76069 / 1.19252**; blank control returned zero and cross-capture control
  returned zero. Flow added 850 training-only observations with zero held-out mask
  hits. Its minimum integer-rounded Euclidean distance was **9.393 px**; do not
  describe this as satisfying a continuous 10 px radius.
- Main inspected PLY-backed `geometry_comparison.jpg` (three angles), `supported_comparison`,
  and `camera_comparison_00.jpg` for dense Coorg; component views 51 and 140; M Block
  `geometry_comparison.jpg` and `camera_comparison_04`; and constrained-pose geometry
  (initial/identity state). Dense-match sheets `dense_matches_00.jpg` and
  `dense_matches_04.jpg` were inspected (five accepted pair groups, not five sheets).
  Sampled matches look plausible, but
  repeated grille ambiguity remains and there is no independent truth validation.
  These checks support the observations above, not ground-truth geometric accuracy.
- **Confirmed checks:** `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync pytest tests/test_refinement.py tests/test_refinement_cache.py tests/test_refinement_controls.py tests/test_refinement_inspection.py tests/test_refinement_validation.py tests/reconstruction -q` — **88 passed in 53.87 s**. Targeted Ruff check on `src/refinement`, these five tests, `src/__init__.py` and four refinement/validation scripts passed; formatting on 23 Python files passed; `uv lock --check` and `git diff --check` passed. Unscoped `uv run --no-sync ruff check` returned **1,943 pre-existing findings** under vendored `src/reconstruction/deps`; full-repository Ruff is not clean.

### Earlier multi-view refinement session (historical machine)

- Created `fix/recon-mv-refinement`, implemented separate `src/refinement/` APIs,
  `refine_multiview.py` / `namma-refine`, actual-PLY inspection, documentation and
  meaningful analytic optimizer/geometry tests. Full scene inference artifacts remain
  separate. No source captures, preprocessing selections or section predictions changed.
- Historical real Coorg learned graph: 660 candidates, 252 supported pairs, 29,409 tracks and
  78,861 observations. Whole tracks are split train/validation/test. First-pass test
  median/p90 disagreement improves 0.014908/0.055694 → 0.002722/0.012388; reprojection
  median/p90 improves 5.399/23.305 → 0.827/2.272 model pixels. Actual source-pixel
  clouds show significant entrance regression: these numerical improvements do not
  establish a good scene. User confirms severe multiple entrance layers and improved
  wall behind the person.
- Historical first-pass graph had eight anchored groups, including three unsupported views.
  Independent entrance window/ceiling groups remain a central weakness. Sparse
  texture anchors and a dense initialization-displacement prior were tested, exported
  and inspected, with persistent entrance layering. A first real LoFTR experiment
  at 0.9 confidence accepted no additional observations. A revised 0.5-confidence,
  reciprocal, broader cross-group matcher was not yet run on that historical machine;
  it has since run on the current machine as documented above.
- Historical outputs were not transferred; the current baseline was regenerated.
- The historical machine ran `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync pytest tests/test_refinement.py tests/reconstruction -q`:
  **57 passed**. Current-machine checks are listed above.

### Fresh Coorg execution and full-duration capture review

- **Recorded `25-10-26 23:17:26 IST`:** Recreated the missing local section outputs
  using the current four 36-view + four 24-view configuration at padded 518×518,
  bfloat16. The first command timed out after four completed sections; resumed
  their checked caches and completed the tail. Installed the already-declared
  alignment extra after a missing-SciPy error, then reran real alignment, revisit
  verification, ICP evaluation, export and benchmarking successfully.
- Final cloud again has **999,971 points from all 161 frames**. All seven primary
  handoffs pass. Independent identical-support median/p90 disagreement is
  **0.231584/0.816671 → 0.014350/0.061621** (93.8% median reduction). Actual PLY/NPZ
  equality, connected graph and camera/depth integrity checks pass.
- Visually inspected actual-PLY multi-angle RGB/section views and orthogonal/local
  diagnostics. Decoded all **12,907** original frames and inspected **216** upright
  one-second samples across the complete **215.44 s** video, all keyframe sheets,
  **20** camera comparisons (also excluding each query frame), and **16**
  confidence-removal/export-pixel comparisons. This is full-duration sampling,
  not viewing every source frame or interactive GUI inspection.
- **Newly localized issues:** TV, backing panel and console are badly incomplete
  despite clear 115–119 s capture; confidence filtering removes connected portions.
  Glass dining tabletop and thin furniture are also heavily lost. Entrance wall,
  door/window and upper trim remain bowed/layered. Floors/shell remain incomplete;
  global display tilt alone is not evidence of physical floor distortion.
- **Internal error:** Same-owner-section frames 112/113 (**160.192/162.207 s**)
  have **366** independent matched observations with normalized median/p90
  **0.2361/0.3550**. This local failure remains despite good pooled/handoff metrics.
- **Separate sources of sparsity:** Median/max keyframe gap is **1.516/2.016 s**;
  selected temporal coverage is complete but surface coverage/baseline is uneven.
  Each frame exports **6,211 pixels**, only **4.08%** of its content grid. Flattened
  evenly spaced pixel sampling produces visible patterned gaps independently of
  keyframe extraction. No denser-keyframe causal ablation was performed.
- Detailed findings, source-time references and diagnostic reproduction script:
  `outputs/coorg_bnb/section_alignment/inspection/REVIEW.md`. Start with
  `missing_surfaces_evidence.jpg`, `section_colored_views.jpg`,
  `camera_comparison_08.jpg` and the six `full_video_*.jpg` sheets in that directory.
  No application code changed and no tests were added/run for this execution review.

### Coorg section alignment after inspection of the initial outputs

- **Recorded `25-10-26 14:23:56 IST`:** Completed on the user-created
  `fix/recon-section-alignment` branch. The motivation was the initial Coorg BnB
  cloud's warped/duplicated surfaces, detached dining fragments and inconsistent
  camera/depth estimates when compared with the source video frames.
- All six initial **36-view, 518×518, bfloat16** sections ran on the A4000. Peak
  allocation was **9.96 GiB**, with **12.21 GiB** reserved. Their last connection
  failed independent support checks; neither ICP nor simply shortening the last
  window repaired the already-distorted preceding section. Superseded predictions
  remain under `section_alignment/experiments/`.
- The final configuration preserves 36-view starts **0,25,50,75**, followed by
  24-view starts **100,113,126,137**, covering all **161** source frames. The smaller
  windows peaked at **9.37 GiB allocated / 13.30 GiB reserved**. These are measured
  PyTorch allocator peaks, not estimates of minimum GPU capacity. Actual image
  content is **294×518** within each padded square.
- Saved `outputs/coorg_bnb/section_alignment/point_cloud.ply`: **999,971 RGB points**,
  exactly **6,211 per source frame**, with matching NPZ confidence, source pixel,
  frame/owner identity, cameras and scaled z-depth. Coordinates use the first
  section's arbitrary scene frame/scale; no floor correction or fusion is applied.
- Registration uses exact shared source pixels, frame/cell-balanced evidence,
  depth-scaled RANSAC/weighted Umeyama and robust refinement. Independent whole
  interior frames validate owner handoffs; shared endpoints are separate stress
  observations and are never used for fitting. Four conservative image revisits
  were reverified using real CPU ALIKED/LightGlue and broad depth-valid/3D support,
  producing **six section-pair measurements** (some source frames belong to two
  sections), including two non-adjacent 0–2 constraints.
- The final **8-node, 13-edge** graph is connected (7 adjacent + 6 revisit
  measurements). Robust objective fell **0.014445 → 0.003561**. All **seven**
  post-optimization primary handoffs pass: normalized median **0.0078–0.0497**,
  p90 **0.0136–0.0965**. This is section-level consistency, not ground-truth accuracy
  or proof that every surface/loop closes perfectly.
- **ICP decision:** Tested robust multiscale point-to-plane ICP on every adjacent
  overlap. **0/7** candidates were retained: six failed the required held-out
  improvement and one exceeded the allowed translation correction. The final
  graph/export therefore uses robust Sim(3) registration without ICP corrections.
- **Independent comparison:** Real ALIKED/LightGlue tested 46 adjacent/owner-boundary
  pairs; 43 supplied **6,937 identical confidence-eligible feature correspondences**
  in both old and new reconstructions. Distances divide by one camera-to-observation
  median per method (**0.52693 old / 0.91638 new** arbitrary units), preserving
  similarity-scale invariance. Pooled normalized median/p90 fell
  **0.23158/0.81667 → 0.01435/0.06162**: **93.8% median reduction**. On **1,101**
  owner-boundary observations, median/p90 fell
  **0.18310/0.35371 → 0.01220/0.05437**. One boundary pair (frames 55–56) has only
  two eligible feature matches; dense held-out shared-frame evidence is recorded
  separately rather than claiming strong independent feature evidence there.
- **Visual inspection:** Read the actual final and original PLYs. Inspected RGB and
  section-colored renders from four angles, common top/side plots, all seven
  boundary correspondence plots, per-section geometry and the camera trajectory.
  Entrance/living/dining geometry and furniture are substantially more recognizable;
  the earlier exploded/smeared footprint is much more coherent. Bowed/doubled wall
  layers and offsets persist around the large wall/door region. At boundary 4/5,
  endpoint-stress median/p90 remain **0.215/0.312**; several globally evaluated
  revisit held-out medians remain **0.085–0.130**. These are retained diagnostics,
  not hidden by observation fusion. The common-display similarity has only
  **113/1,800** supporting calibration samples and is visualization-only; it does
  not enter graph optimization, export or the numerical benchmark.
- **Integrity and review:** Exact actual-PLY XYZ/RGB readback; every source frame
  contributes; proper rotations/positive section scales; camera inverses and
  scaled camera-z depths agree. Independent reviewer additionally checked every
  exported point/color/depth against its original owner-section pixel. No averaging,
  hidden fusion or cross-owner mixing was found. Parallel implementation agents
  and an independent reviewer were used; identified direction, unit, support and
  cache-provenance defects were fixed before the final export.
- **Checks:** `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync pytest tests/reconstruction -q`:
  **52 passed**. Ruff checks/formatting, `uv lock --check` and `git diff --check`
  pass. Real inference, feature verification, Open3D ICP and headless actual-PLY
  inspection ran; no interactive GUI was opened. Commands/contracts are in
  [the section-alignment guide](RECONSTRUCTION.md#high-resolution-section-alignment).

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

- Whole-scene multi-view refinement is not qualitatively acceptable: distinct
  entrance/upper-wall layers and holes persist. Follow the checklist and visual
  acceptance criteria in [REFINEMENT.md](REFINEMENT.md); improved held-out metrics
  are insufficient for acceptance.
- Coorg's initial section-alignment benchmark is met, but residual within-section
  warping, endpoint pose/depth outliers and conflicting revisit measurements remain.
  A single similarity per section cannot remove these. Do not advance to observation
  fusion or multi-view geometry correction until the user explicitly requests it.
- Floor constraints assume one flat storey. Semantics/depth errors remain, 71 Coorg
  views lack an accepted local plane, and per-view leveling is not bundle adjustment.
  Further consistency requires actual multi-view optimization, not more flattening.
- Agent headless actual-PLY review is complete for the new section export. User
  interactive review remains pending; the standalone GUI viewer has not been run.
- Coorg's full 161-view geometry pool exceeds the A4000's memory at 518 resolution.
  A complete joint reconstruction is available at 252 resolution; the implemented
  overlapping-section strategy now also supplies a complete 518-resolution export.
- No preprocessing execution or dependency blockers remain for the supplied SDR videos.
- Per-frame blur classification is ambiguous for the labelled 49.5 s door at
  analysis resolution; the full pipeline uses context. Conservative self-similarity
  filtering can discard true loop links, as observed for one luggage pair.
- HDR conversion remains outside the implementation; HDR inputs fail explicitly.

## Next Actions

1. Review `uv run --no-sync python view_point_cloud.py outputs/coorg_bnb/section_alignment/point_cloud.ply`.
   RGB/section-colored renders and the detailed inspection are under
   `outputs/coorg_bnb/section_alignment/inspection/`.
2. Use `section_benchmark.json`, graph/held-out/stress diagnostics and the source
   video to review remaining registration and within-section errors.
3. Await explicit user direction before multi-view observation fusion, surface
   generation or navigation work. Preserve frozen preprocessing and provenance.
