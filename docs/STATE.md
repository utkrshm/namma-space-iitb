# Project state — R1 submission

## Current submission

- Runtime: one `src/` package, one root project and lockfile, and
  `namma-reconstruct run SOURCE --out outputs/NAME`.
- Stages: ingest → verified cameras → depth priors → Gaussian splatting → tour → demo.
- The independent `namma-preprocess` command remains available.
- The four-scene web viewer is https://gausswalk.pages.dev.
- The bundled published reconstruction is `example/house/`.
- Earlier implementations, their tests and upstream submodules are excluded. Their
  experiments, outcomes and rationale remain documented below and in `docs/`.
- This branch is a standalone submission snapshot; earlier source commits remain
  on their original branches, not in this branch's rewritten history.

## Experiment ledger

“Passed” means the stated check or measured goal passed, not that a complete
navigation-ready reconstruction was established. “Failed/not adopted” includes
executed variants that lost quality or failed the intended scene goal.

| Experiment | Outcome and evidence | Disposition |
| --- | --- | --- |
| Smartphone preprocessing, decoding and repeated-structure guards | **Passed** on Room/Coorg; 36/161 geometry frames, 101/399 appearance frames; source timing and upright export checked | Retained as independent preprocessing |
| Pure per-frame blur classification at 49.5 s | **Failed/known ambiguity** at 640 px; temporal context flags the interval | Existing expected failure retained |
| VGGT joint inference | **Passed** at 518 for 36 Room views; 161 Coorg views required 252 after 518 OOM | Archived; arbitrary scale, missing/warped surfaces remain |
| Semantic floor alignment | **Passed floor-residual goal**, not general registration; Coorg matched-view disagreement did not improve | Archived single-floor assumption |
| High-resolution overlapping sections and Sim(3) graph | **Passed** all seven handoffs; independent normalized median disagreement reduced 93.8% | Archived; residual wall layering/pose distortion |
| ICP corrections to section graph | **Failed/not adopted**: 0/7 retained (six lacked held-out improvement; one excessive correction) | Diagnostics retained |
| COLMAP Room reconstruction trial | **Failed** experiment, recorded on section-alignment branch | Preserved in historical STATE/PLANS/RECONSTRUCTION records |
| Loop chunks (A), bundle adjustment (B), square-wall prior (M) | **B passed a measured improvement**: sequential −35%, held-out loops −10%; A/M did not improve held-out metrics | Archived; dining loop unverified and wall layering persists |
| Dense/flow multi-view refinement | **Execution/audit checks passed; scene-quality goal failed**; Coorg graph components 8 → 3, entrance layers remain; M Block holes remain | **Deferred**, promising but premature |
| Camera-path point-cloud viewer | **Focused tests passed**; interactive playback not established by historical records | Historical experiment documented |
| Unfiltered global SfM retrieval loops | **Failed**: repeated grilles/doors folded trajectory | Replaced by rotation/epipolar-gated loops |
| Dense full-resolution SfM with gated loops | **Passed** on Coorg: 927/927 frames, 1.31 px | Earlier implementation archived; lessons adopted |
| Single whole-house splat | **Inferior** to blocks; best historical single model 19.05/21.6 dB, SSIM 0.80 | Block pipeline adopted |
| Nearest-camera vs visibility-weighted block merge | **Visibility merge passed**: 20.6 → 24.4 dB colour-corrected at half resolution | Adopted with joint fine-tuning |
| Joint fine-tuning and SH degree | **Passed**: second fine-tune 24.6/27.6 dB, SSIM 0.91; SH2/SH0 lost quality | Full-SH final model retained |
| Motion-blur training-frame exclusion | **No gain/not adopted**: tied baseline; capture-limited weak spots remain | Experiment documented |
| Larger per-block Gaussian budgets | **Not adopted**: worse entrance/living, only +0.2 dB dining | Experiment documented |
| Earlier monocular depth prior | **Mixed**: −0.4 dB held-out, offset-view floaters 2.65 → 2.35%; worse entrance | Later dense-frame/depth and lens refinements superseded it |
| PnP densification to 8 frames/s | **Passed**: 834/842 extra frames, median 1814 inliers | Historical dense-depth model retained |
| OPENCV lens solve, splat transfer and fine-tune | **Passed**: floor spread 0.144 → 0.059 camera heights; final 25.39/27.37 dB, SSIM 0.904 | Published model retained in `example/house/` |
| Incremental/floor-BA/piecewise-similarity/split-focal alternatives | **Failed/not adopted** for entrance floor correction | Details in SFM experiment guide |
| Invisible-Gaussian pruning | **No visual improvement** despite 88% never visible | Experiment documented |
| Random-background/flatness/depth “solid” model and MCMC regrowth | **Not adopted**: won only 7/20 views; regrowth lost 2–3 dB; one relocation attempt produced NaN | Non-finite-loss stop retained |
| Full 360° room turns | **Failed scene-quality goal** in unfilmed directions; smears and dark gaps | Filmed-arc tours adopted |
| Playroom depth-seeded initialization | **Failed**: 11.9 dB vs 21.9 dB; seeds grew into foreground blobs | Depth seeds off by default |
| Robust SfM radius and pointless-camera removal | **Passed**: playroom 24.0 dB/0.39 LPIPS at 6k, closing earlier own-SfM gap | Adopted |
| Playroom 30k baseline vs classic rasterization/bigger cap | **Baseline retained**: 30.24 dB, SSIM 0.909, LPIPS 0.305; alternatives 29.97/29.42 dB | Detailed benchmark guide retained |
| Focal estimation, continuity checks, balanced spatial blocks | **Adopted** from latest Spacegen commits; focused geometry tests retained | Production defaults |
| Metric depth/normal priors and clipped-highlight loss | **Measured improvement in wall normals**, PSNR trade-off above; full reruns pending in source record | Latest production defaults preserved |
| Surface mesh, semantic scene graph, metric navigation and AR | **Deferred/planned**, not implemented by a Gaussian PLY and rendered tour | Challenge scope still open |

## Documentation archive

The archive retains original branch and revision identifiers and the recorded outcomes
of earlier experiments. Their executable source is not included in this snapshot.

## Remaining limits and deferred work

Capture blur, reflective surfaces, sparse texture and unobserved directions can produce
gaps or distorted surfaces. Scene scale remains arbitrary. Destination-based navigation
and metric measurements are not implemented by this reconstruction pipeline.

## Historical branch STATE records

Historical records below are curated for the submission scope and grouped by source.
The source ref(s) appear before each record. Older commands use their original
paths/dependency extras; see the [experiment index](../experiments/README.md)
for their scope. These historical “current” sections do not override
the current ledger above. Earlier intermediate revisions are available in
[state-history/](experiments/state-history/) with exact Git blob IDs.

## Historical branch snapshot: feat/dense-sfm-splat, feat/spacegen, run/spacegen-new-home-runpod-20261031, run/spacegen-room-runpod-20261031, origin/feat/dense-sfm-splat

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `30-10-26 11:10:00 IST`

## Current Focus

Work is on `feat/dense-sfm-splat`. It adds full-resolution cameras (`src/sfm/`, see
[SFM.md](SFM.md)) and photoreal Gaussian splatting of the whole Coorg house
(`train_sfm_splat.py` → block training → visibility merge → joint fine-tune; see
[SPLATTING.md](SPLATTING.md#splatting-from-dense-sfm-cameras-train_sfm_splatpy--verified-on-coorg)).
Final model: `outputs/coorg_bnb/splat_runs/opencv/finetune_2` (published in
`example/house/`): merged blocks, 8 frames/s and a depth prior, moved onto the
OPENCV-lens cameras and fine-tuned. On the 20 held-out frames it scores 25.39 dB raw /
27.37 dB colour-corrected, SSIM 0.90; at the 518 grid it scores 26.3 dB raw, against
17.07 dB for the earlier VGGT splat. The tour video now includes an in-room look-around.

Earlier focus (`feat/LoopClosure`):
Work is on `feat/LoopClosure` (from `fix/recon-section-alignment`): loop closure for the
global camera drift seen in the Coorg section-aligned cloud. Two methods are compared,
VGGT-Long-style loop chunks (A) and bundle adjustment (B), plus an optional square-walls
arm (M). See [LOOP_CLOSURE.md](LOOP_CLOSURE.md).

Previous focus (`fix/recon-section-alignment`) was motivated by inspecting the initial
Coorg BnB reconstruction against its source frames: overlapping/warped surfaces,
detached fragments and inconsistent camera/depth geometry persist after floor
leveling. All 161 selected Coorg frames now have a 518-resolution section-aligned
export: four 36-view sections plus four 24-view tail sections, with an optimized
eight-node/13-edge Sim(3) graph. Independent matched-observation consistency and
actual-PLY visual inspection improve substantially over the initial 252 run.
Residual wall/pose distortion and imperfect loop closure remain. Observation fusion
is deferred pending the user's explicit instruction.
See the detailed adopted decision in [PLANS.md](PLANS.md).

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud export | Verified on both captures; Coorg original/section clouds also reviewed headlessly |
| Floor-plane alignment | Implemented and verified on real outputs; Coorg floor residuals reduced, cross-view drift remains |
| Coorg high-resolution section alignment | Verified on real Coorg outputs: connected global graph, all seven primary handoffs pass, improved consistency and visual structure; residual distortion remains |
| Loop closure (`src/loop_closure/`) | Implemented and run on real Coorg outputs; B gives a measurable gain (sequential −35%, held-out loops −10%); dining-wing drift and wall layering remain (no verifiable dining loop in this capture) |
| Gaussian splatting (`src/splatting`) | Implemented (awaiting verification on development server) |
| Shared rendering and Python/Viser viewer | Implemented (awaiting verification on development server) |
| Dense full-resolution SfM (`src/sfm/`) | Verified on Coorg: 927/927 frames, 1.31 px, loops checked by rotation and epipolar fit |
| SfM-input splatting, block merge, web export, fly-through, tour video (`src/splatting/sfm_*`, `merge_blocks`, `web_export`, `flythrough`, `tour`) | Verified on Coorg: held-out 23.9/27.0 dB, SSIM 0.91; visually inspected |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `30-10-26 10:30:00 IST` (outcome 11:10)
- **Decision:** Show each room from inside with a look-around that turns only through
  directions filmed from its standing point. Supersample tour renders. Keep the
  free-viewpoint training terms as options; the final model is unchanged.
- **Reference:** “Free-viewpoint quality: solid surfaces and an in-room look-around” in
  [PLANS.md](PLANS.md).

### Earlier decision (29 Oct)

- **Timestamp:** `29-10-26 17:20:00 IST` (amended 21:50)
- **Decision:** Solve Coorg's cameras with the OPENCV lens model; move the trained splat
  onto re-solved cameras (`transfer_splat.py`) and fine-tune; tour video with a local
  floor.
- **Reference:** “Flat-floor check, corrected cameras for the entrance, and a robust tour
  video” in [PLANS.md](PLANS.md).

### Earlier decision

- **Timestamp:** `26-10-26 19:11:38 IST`
- **Decision:** Add a separate read-only loop-closure stage and compare VGGT loop chunks (A) with bundle adjustment (B) on identical held-out correspondences.
- **Reference:** “Loop closure for global camera drift (two compared methods)” in [PLANS.md](PLANS.md).

### Previous decision

- **Timestamp:** `25-10-26 14:12:41 IST`
- **Decision:** Use depth-scaled robust section registration with independent interior handoff validation, explicit endpoint stress reporting, verified revisit constraints and a global Sim(3) solve. Keep the first four 36-view sections and use four 24-view tail sections. Reject ICP corrections that do not improve held-out agreement.
- **Reference:** “Depth-scaled fitting and independent interior handoff validation” in [PLANS.md](PLANS.md).

## Current Progress

### Dense SfM and photoreal splatting on Coorg (`feat/dense-sfm-splat`)

- **Recorded `28-10-26 21:13:39 IST`:** 927 frames (4 per second plus the 161 geometry frames) were
  solved at 1080×1920.
  - GLOMAP with every retrieval pair folded the trajectory: false loops on repeated
    grilles and doors, 10–20% of the pairs, with 100–180° rotation error.
  - A three-pass solve fixed it: sequential → loops checked by rotation → loops checked
    by epipolar fit. The last pass also removed translated-grille false loops at the
    entrance.
  - Final cameras (`sfm_loops2`): 1.31 px, 3 small trajectory breaks, true loops fit
    to 0.97 px median.
- **Splat training (`train_sfm_splat.py`):** MCMC, per-image pose and colour corrections,
  coarse-to-fine resolution. Root causes found and fixed:
  - MCMC noise was not scale-normalized (≈ radius² too strong), which caused fog.
  - Gaussian sizes were unbounded; there was no screen-size pruning (near-camera
    floaters, out-of-memory errors).
  - Budgets above 1.6M were unstable for the whole house.
  - The best single model (`full_loops2_v6b`) scores 19.05/21.6 dB, SSIM 0.80.
- **Blocks:** entrance 0–74.5 s, living room 74.5–180 s, dining 180–216 s, each trained on
  the same cameras with the global holdout (25.8–29.3 dB colour-corrected on their own
  views).
  - Nearest-camera merge: 20.6 dB. Visibility-weighted merge: 24.4 dB (colour-corrected,
    at half resolution).
  - A 6k-step joint fine-tune gives 23.9 / 27.0 dB; a second 10k-step pass gives the final
    24.6 / 27.6 dB, SSIM 0.91. SH degree 3 is needed (SH2 −0.9 dB, SH0 −3.7 dB).
- **Blur filter experiment:** a third fine-tune without motion-blurred training frames tied
  (24.65 / 27.48 dB, SSIM 0.90), so `merged_finetune2` stays final. The 200 s and
  72–80 s weak spots are capture-limited (blurred or textureless source frames).
- **Further experiments (29-10-26):**
  - Bigger per-block budgets: worse for the entrance and living room, +0.2 dB for dining.
  - Monocular depth prior: −0.4 dB on held-out views, fewer floaters in offset views
    (2.65 → 2.35% of pixels), worse at the entrance.
  - **Densification (`densify_sfm.py`):** 834/842 extra frames (8 per second) registered by
    PnP into `sfm_loops2`, with a median of 1814 inliers. The previous model renders them at
    28.87 dB, the same as its training frames.
  - Fine-tuning on the dense frames plus the depth prior gives the new final model
    `merged_dense8_depth`. Further gains need a recapture: slower passes at 199–201 s and
    72–80 s, and a loop back from the dining room.
- **In-room look-around and free-viewpoint tests (30-10-26, recorded 11:10 IST):**
  - *Request:* the user asked for a room without black spots, smears or merged objects,
    and for a view from inside each room before the walkthrough.
  - *Diagnosis:* eye-level renders along the path match the video. A level 360° turn from
    a room's middle smears walls and leaves dark gaps under the ceiling. 88% of the
    Gaussians never show in a training view, but removing them
    (`prune_invisible.py`) changes nothing. The smears are visible Gaussians in the wrong
    place, in directions nobody filmed from there.
  - *Training tests (not adopted):*
    - Random background plus a flatness term plus depth prior 0.2 (`solid.toml`) scores
      25.46 / 27.22 dB and wins only 7 of 20 views.
    - MCMC regrowth lost 2–3 dB; relocating all dead Gaussians at once gave NaN. The
      trainer now stops on a non-finite loss.
  - *Tour:* part 4 is a look-around per room (the living hall gets two). It picks the
    standing point by its filmed arc and a render score that penalises close surfaces.
    It follows the phone's tilt and trims 20° off each end of the arc. Renders are
    supersampled 1.5×.
  - *Visual inspection:* contact sheets every 2 s of the new tour. All four look-arounds
    are clean: the entrance bay window and crocs; the sofa and coffee table; the TV wall
    with the dining room through the door; the dining table, laptop and chairs. The
    build-up no longer shows a hole in the hall floor (reveal distance 1.6 → 3 camera
    heights). Published as `example/house/coorg_tour.mp4` and
    `look_around_frames.jpg`.
  - *Checks:* `uv run --no-sync pytest -q tests/splatting tests/sfm`: 47 passed (CUDA
    tests on the A4000). Ruff passes on the changed files.
- **Entrance floor and tour video (29-10-26, recorded 22:20 IST):**
  - *Problem:* the user found a black hole in the tour's plan view at 1:09. The
    single-coefficient lens solve had sunk the foyer and mat-corridor floor
    0.25–0.3 camera heights, and the tour's one global floor cut it away.
  - *Tour fix:* the tour measures the floor locally and filters glossy-floor reflections
    and floaters.
  - *Camera fix:* cameras re-solved with the OPENCV lens model (`sfm_opencv`, 1.27 px;
    floor spread 0.144 → 0.059 camera heights), then densified (`sfm_opencv_dense8`,
    806/826 extra frames).
  - *New final model `opencv/finetune_2`:* the splat transferred and fine-tuned. Held-out
    25.39 / 27.37 dB, SSIM 0.904; worst view 20.3 → 21.2 dB (colour-corrected, half
    resolution); floaters 2.48%.
  - *Visual inspection:* `example/house/heldout_overview.jpg` (real | previous final |
    new final) and tour frames at the entrance, orbit and walkthrough. The entrance floor,
    mat and crocs show in the plan, and the 50 s door floater is gone.
  - *Failed attempts:* an incremental solve, floor-constrained BA, a piecewise similarity
    and a split focal length (see [SFM.md](SFM.md)).
- **Visual inspection (earlier):** `outputs/coorg_bnb/final/heldout_overview.jpg` (real frame |
  VGGT splat | new splat) and `flythrough_frames.jpg` from
  `outputs/coorg_bnb/final/coorg_flythrough.mp4` (1387 frames).
  - Sharp: furniture, curtains, inlays, grilles.
  - Remaining weak spots: a floater at the left edge of the 50 s door view, a soft shelf
    at 200 s, and the blank-ceiling pan at 72–80 s. The 113 s and 200 s held-out views
    are not better than the VGGT splat.
- **Web export:** pruning at opacity < 0.02 leaves 2.08M Gaussians with no PSNR change.
  The full-SH PLY is 491 MB; the 32-byte `.splat` is 63 MB (it loses view-dependent
  colour, −3 dB).
- **Checks:** `uv run --no-sync pytest tests -q`: 117 passed, 1 skipped, 1 xfailed (CUDA
  tests ran on the A4000). Ruff passes on the changed files.

### Gaussian splatting — authored, not executed

- **Recorded `27-10-26 02:46:18 IST`:** Added `src/splatting`, `src/rendering`,
  `configs/gaussian_splatting.toml`, an example scene manifest, root training/rendering/
  viewing scripts, CLI entry points, optional dependency groups and stage exports.
- The input contract requires aligned RGB, decoded VGGT camera-head world-to-camera
  extrinsics/intrinsics, refined camera-Z depth and world-space point maps. It
  records source frame IDs/timestamps, image transforms, scale and optional masks/
  confidence. Shapes, camera transforms and own-camera depth/reprojection agreement
  are checked explicitly. Paths support per-frame NPY and named/indexed NPZ arrays.
- Training-only bounded point sampling, multiview depth support, voxel fusion,
  outlier removal and nearest-neighbor scales seed fixed-camera RGB/SSIM/depth
  optimization. Budgeted adaptive refinement retains optimizer alignment. Checkpoints
  save parameters, optimizer/refinement/RNG state, input fingerprints and versions.
- Export includes standard full-SH Gaussian PLY, metadata, numeric metrics and
  reference/render previews. The shared renderer produces RGB/alpha/camera-Z depth;
  the server-side Python/Viser viewer uses it for view-dependent appearance.
  The integer-centre VGGT to half-integer gsplat raster convention is converted
  explicitly at the rendering boundary.
- Tests are written for file/camera contracts, geometry filtering, holdout exclusion,
  budgets, SH PLY and Adam checkpoint continuation, NaN-safe losses, topology state,
  real CUDA training/rendering and raster pixel-centre alignment. **All are not run.**
  No fake renderer or substitute model was used.
- [SPLATTING.md](SPLATTING.md) documents the exact contract, API, server commands,
  outputs, limitations and verification steps. `uv.lock` is unchanged because
  dependency resolution must be performed on the development server.

### Loop closure on Coorg (`feat/LoopClosure`)

- **Recorded `26-10-26 20:25:16 IST`:** `close_loops.py outputs/coorg_bnb/section_alignment
  --output-dir outputs/coorg_bnb/loop_closure` ran all arms on the A4000. The section
  directory was read-only; every artifact is under `outputs/coorg_bnb/loop_closure/`.
- 494 DINOv2 candidates, 24 verified loops (5 chunk centres / 13 train / 6 held out).
  The repeated-structure guard rejected every true wide-baseline living-room loop, so a
  non-planar override was added. Measured planar fraction was 0.30–0.38 for true loops
  and 0.79–0.98 for grille/inlay false positives. Match images were inspected.
- No verifiable dining→living loop exists. An exhaustive scan of 178–214 s against
  77–146 s found 0 epipolar inliers, and LoFTR indoor found no matches.
- Method A accepted 3/5 loop edges (0↔2, 2↔3, 2↔4); all adjacent held-out checks still
  pass after optimization. B from baseline: inlier reprojection 2.53→1.07 px.
  Normalized held-out loop disagreement 0.0458→0.0413; sequential 0.0166→0.0107.
  A alone and M did not improve the held-out metrics. Full table in
  [LOOP_CLOSURE.md](LOOP_CLOSURE.md#coorg-results-26-10-26-2025-ist-a4000).
- Square-walls diagnosis: frames 0–160 s lie within ±10° of the house axes, while
  frames from 162 s sit at +31–35°. The 160.2→162.2 s transition has 407 feature
  matches, so the turn is image-supported, and the prior does not override it.
- Actual-PLY inspection (`inspection/topdown_comparison.png`, seven camera slices):
  the arms are visually almost identical. The doubled 171 s wall, the dining geometry
  crossing the TV wall and the 158 s pillar layers persist.
- Checks: `uv run --no-sync pytest tests/reconstruction tests/loop_closure -q`
  **60 passed**. Ruff check/format and `uv lock --check` pass. Real VGGT, DINOv2 and
  ALIKED/LightGlue inference ran. No GUI viewer was opened.

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

- Coorg's dining wing (≈162–215 s) has no image-verifiable loop to earlier rooms, so its
  ≈30° yaw offset and the resulting TV-wall interpenetration cannot be closed from this
  capture. A recapture that walks back from the dining room into the living room is needed,
  or the user can confirm whether that wing is genuinely angled.
- Remaining wall layering is partly per-frame VGGT depth error on plain walls and
  reflective/translucent regions; camera-only optimization cannot remove it.

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
- The development server and actual refined VGGT files are inaccessible. Dependency
  resolution/CUDA compilation, lint/tests, actual file integration, viewer operation,
  memory/performance and real-capture splat quality remain unverified. No visual
  inspection of splatting output has been performed. Resolve the lockfile on the
  development server before using locked dependency workflows.
- No execution or dependency blockers remain for the supplied SDR videos.
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
1. On the development server, resolve splatting dependencies and run the documented
   lint, CPU and real-CUDA tests; record failures and actual versions.
2. Map the actual refined VGGT output layout to the documented manifest. Train a
   representative capture, inspect held-out previews and the viewer, and record
   quality/memory findings before declaring this stage verified.
3. Retain preprocessing follow-ups: review accepted revisit images, consider richer
   blur evidence if needed, and integrate reconstruction through `load_result(output_dir)`
   while preserving source/candidate provenance.

## Historical branch snapshot: feat/preprocessing, master, origin/master

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `21-10-26 03:09:09 IST`

## Current Focus

Round-three review corrections applied: descriptor self-similarity filtering rejects
repeated-window ambiguity. Real positive/negative revisit controls and a real
spliced-input tracking-break control pass. User commit `4a6a0a3` is retained.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| Reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `21-10-26 02:39:40 IST`
- **Decision:** Reject repeated-structure descriptor ambiguity before geometric verification; use pass-specific cache provenance.
- **Reference:** “Reject descriptor-ambiguous repeated structures” in [PLANS.md](PLANS.md).

## Current Progress

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

- No execution or dependency blockers remain for the supplied SDR videos.
- Per-frame blur classification is ambiguous for the labelled 49.5 s door at
  analysis resolution; the full pipeline uses context. Conservative self-similarity
  filtering can discard true loop links, as observed for one luggage pair.
- HDR conversion remains outside the implementation; HDR inputs fail explicitly.

## Next Actions

1. User review of the four accepted match images and reported conservative rejections.
2. Consider higher-resolution or richer blur evidence if a context-free decision is required.
3. Integrate reconstruction through `load_result(output_dir)` and use global geometry
   to evaluate tentative loop links. Preserve source and candidate provenance.

## Historical branch snapshot: feat/reconstruction

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `21-10-26 12:08:13 IST`

## Current Focus

Preprocessing is frozen. The VGGT wrapper, preprocessed-frame adapter and Open3D
point-cloud script are implemented on `feat/reconstruction`. CPU processing and
handoff checks pass. Pretrained inference is reserved for a larger CUDA GPU; no
model was instantiated or run on this laptop.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud preview | Implemented (awaiting larger-GPU inference and Open3D runtime verification) |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `21-10-26 11:48:44 IST`
- **Decision:** Joint CUDA-only vanilla VGGT inference with CPU NumPy outputs and Open3D export/viewing; keep preprocessing frozen and do not run model inference on the laptop.
- **Reference:** “CUDA VGGT inference and Open3D reconstruction preview” in [PLANS.md](PLANS.md).

## Current Progress

### Reconstruction interface

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
- The real Open3D RGB PLY round-trip test is skipped: the 382.2 MiB Open3D wheel
  download exceeded both the 120 s and 300 s installation timeouts. No fake viewer
  substitutes for that check. Interactive GUI viewing and visual reconstructed-scene
  inspection are not run. Pretrained CUDA inference is not run per user instruction.

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

- Larger-GPU pretrained inference remains to be verified externally. There is no
  claimed reconstructed-capture result or visual inspection from this laptop.
- Open3D runtime/PLY round-trip verification awaits a completed viewer dependency
  installation; its large wheel download timed out locally.
- No preprocessing execution or dependency blockers remain for the supplied SDR videos.
- Per-frame blur classification is ambiguous for the labelled 49.5 s door at
  analysis resolution; the full pipeline uses context. Conservative self-similarity
  filtering can discard true loop links, as observed for one luggage pair.
- HDR conversion remains outside the implementation; HDR inputs fail explicitly.

## Next Actions

1. On the larger GPU machine, install reconstruction/viewer extras and run `get_recon.py`
   on a complete geometry pool, following [the reconstruction guide](RECONSTRUCTION.md).
2. Complete the real Open3D PLY check, open an exported capture cloud, and record visual
   reconstruction quality, GPU memory usage and inference timing.
3. Use verified scene geometry to plan fusion, surface generation and downstream
   navigation. Keep preprocessing frozen and preserve source/candidate provenance.

## Historical branch snapshot: feat/splatting, origin/feat/splatting

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `27-10-26 02:46:18 IST`

## Current Focus

Gaussian-splatting code authored on `feat/splatting` for execution exclusively on
the development server. Upstream refined VGGT files and that server are currently
unavailable. No dependencies, lint, tests, training, rendering or viewer runtime
checks were executed on the authoring machine. Earlier verified preprocessing
results are retained below.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| Gaussian splatting (`src/splatting`) | Implemented (awaiting verification on development server) |
| Shared rendering and Python/Viser viewer | Implemented (awaiting verification on development server) |
| Upstream VGGT/refinement integration, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `27-10-26 02:30:17 IST`
- **Decision:** Geometry-initialized fixed-camera Gaussian splatting, with a versioned refined-VGGT manifest and shared rendering/viewing; code authoring only on this machine.
- **Reference:** “Geometry-initialized Gaussian splatting” in [PLANS.md](PLANS.md).

## Current Progress

### Gaussian splatting — authored, not executed

- **Recorded `27-10-26 02:46:18 IST`:** Added `src/splatting`, `src/rendering`,
  `configs/gaussian_splatting.toml`, an example scene manifest, root training/rendering/
  viewing scripts, CLI entry points, optional dependency groups and stage exports.
- The input contract requires aligned RGB, decoded VGGT camera-head world-to-camera
  extrinsics/intrinsics, refined camera-Z depth and world-space point maps. It
  records source frame IDs/timestamps, image transforms, scale and optional masks/
  confidence. Shapes, camera transforms and own-camera depth/reprojection agreement
  are checked explicitly. Paths support per-frame NPY and named/indexed NPZ arrays.
- Training-only bounded point sampling, multiview depth support, voxel fusion,
  outlier removal and nearest-neighbor scales seed fixed-camera RGB/SSIM/depth
  optimization. Budgeted adaptive refinement retains optimizer alignment. Checkpoints
  save parameters, optimizer/refinement/RNG state, input fingerprints and versions.
- Export includes standard full-SH Gaussian PLY, metadata, numeric metrics and
  reference/render previews. The shared renderer produces RGB/alpha/camera-Z depth;
  the server-side Python/Viser viewer uses it for view-dependent appearance.
  The integer-centre VGGT to half-integer gsplat raster convention is converted
  explicitly at the rendering boundary.
- Tests are written for file/camera contracts, geometry filtering, holdout exclusion,
  budgets, SH PLY and Adam checkpoint continuation, NaN-safe losses, topology state,
  real CUDA training/rendering and raster pixel-centre alignment. **All are not run.**
  No fake renderer or substitute model was used.
- [SPLATTING.md](SPLATTING.md) documents the exact contract, API, server commands,
  outputs, limitations and verification steps. `uv.lock` is unchanged because
  dependency resolution must be performed on the development server.

### Preprocessing — previous verified results

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

- The development server and actual refined VGGT files are inaccessible. Dependency
  resolution/CUDA compilation, lint/tests, actual file integration, viewer operation,
  memory/performance and real-capture splat quality remain unverified. No visual
  inspection of splatting output has been performed. Resolve the lockfile on the
  development server before using locked dependency workflows.
- No execution or dependency blockers remain for the supplied SDR videos.
- Per-frame blur classification is ambiguous for the labelled 49.5 s door at
  analysis resolution; the full pipeline uses context. Conservative self-similarity
  filtering can discard true loop links, as observed for one luggage pair.
- HDR conversion remains outside the implementation; HDR inputs fail explicitly.

## Next Actions

1. On the development server, resolve splatting dependencies and run the documented
   lint, CPU and real-CUDA tests; record failures and actual versions.
2. Map the actual refined VGGT output layout to the documented manifest. Train a
   representative capture, inspect held-out previews and the viewer, and record
   quality/memory findings before declaring this stage verified.
3. Retain preprocessing follow-ups: review accepted revisit images, consider richer
   blur evidence if needed, and integrate reconstruction through `load_result(output_dir)`
   while preserving source/candidate provenance.

## Historical branch snapshot: research/home-video-preprocessing

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `20-10-26 13:21:45 IST`

## Current Focus

Preprocessing: video ingestion, frame extraction, quality filtering and keyframe selection.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Planned |
| Reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `20-10-26 13:18:20 IST`
- **Decision:** Adopt the minimal documentation structure and add component explanations; defer detailed module contracts.
- **Reference:** “Establish documentation scaffolding” in [PLANS.md](PLANS.md).

## Current Progress

- The two-line README, component explanations, and state/plan documents are in place. Documentation links, section structure and timestamp formatting have been checked.
- `main.py` contains only the project greeting. Pipeline-stage packages and tests are not present yet.

## Blockers

- None recorded.

## Next Actions

1. Record the preprocessing implementation plan in `docs/PLANS.md`.
2. Implement preprocessing under `src/`, with tests under `tests/` and a runnable demonstration at the repository root.
3. Run the relevant checks and visually inspect outputs from a representative capture; record the results under Current Progress.

## Historical branch snapshot: origin/feat/LoopClosure

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `26-10-26 20:25:16 IST`

## Current Focus

Work is on `feat/LoopClosure` (from `fix/recon-section-alignment`): loop closure for the
global camera drift seen in the Coorg section-aligned cloud. Two methods are compared,
VGGT-Long-style loop chunks (A) and bundle adjustment (B), plus an optional square-walls
arm (M). See [LOOP_CLOSURE.md](LOOP_CLOSURE.md).

Previous focus (`fix/recon-section-alignment`) was motivated by inspecting the initial
Coorg BnB reconstruction against its source frames: overlapping/warped surfaces,
detached fragments and inconsistent camera/depth geometry persist after floor
leveling. All 161 selected Coorg frames now have a 518-resolution section-aligned
export: four 36-view sections plus four 24-view tail sections, with an optimized
eight-node/13-edge Sim(3) graph. Independent matched-observation consistency and
actual-PLY visual inspection improve substantially over the initial 252 run.
Residual wall/pose distortion and imperfect loop closure remain. Observation fusion
is deferred pending the user's explicit instruction.
See the detailed adopted decision in [PLANS.md](PLANS.md).

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud export | Verified on both captures; Coorg original/section clouds also reviewed headlessly |
| Floor-plane alignment | Implemented and verified on real outputs; Coorg floor residuals reduced, cross-view drift remains |
| Coorg high-resolution section alignment | Verified on real Coorg outputs: connected global graph, all seven primary handoffs pass, improved consistency and visual structure; residual distortion remains |
| Loop closure (`src/loop_closure/`) | Implemented and run on real Coorg outputs; B gives a measurable gain (sequential −35%, held-out loops −10%); dining-wing drift and wall layering remain (no verifiable dining loop in this capture) |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `26-10-26 19:11:38 IST`
- **Decision:** Add a separate read-only loop-closure stage and compare VGGT loop chunks (A) with bundle adjustment (B) on identical held-out correspondences.
- **Reference:** “Loop closure for global camera drift (two compared methods)” in [PLANS.md](PLANS.md).

### Previous decision

- **Timestamp:** `25-10-26 14:12:41 IST`
- **Decision:** Use depth-scaled robust section registration with independent interior handoff validation, explicit endpoint stress reporting, verified revisit constraints and a global Sim(3) solve. Keep the first four 36-view sections and use four 24-view tail sections. Reject ICP corrections that do not improve held-out agreement.
- **Reference:** “Depth-scaled fitting and independent interior handoff validation” in [PLANS.md](PLANS.md).

## Current Progress

### Loop closure on Coorg (`feat/LoopClosure`)

- **Recorded `26-10-26 20:25:16 IST`:** `close_loops.py outputs/coorg_bnb/section_alignment
  --output-dir outputs/coorg_bnb/loop_closure` ran all arms on the A4000. The section
  directory was read-only; every artifact is under `outputs/coorg_bnb/loop_closure/`.
- 494 DINOv2 candidates, 24 verified loops (5 chunk centres / 13 train / 6 held out).
  The repeated-structure guard rejected every true wide-baseline living-room loop, so a
  non-planar override was added. Measured planar fraction was 0.30–0.38 for true loops
  and 0.79–0.98 for grille/inlay false positives. Match images were inspected.
- No verifiable dining→living loop exists. An exhaustive scan of 178–214 s against
  77–146 s found 0 epipolar inliers, and LoFTR indoor found no matches.
- Method A accepted 3/5 loop edges (0↔2, 2↔3, 2↔4); all adjacent held-out checks still
  pass after optimization. B from baseline: inlier reprojection 2.53→1.07 px.
  Normalized held-out loop disagreement 0.0458→0.0413; sequential 0.0166→0.0107.
  A alone and M did not improve the held-out metrics. Full table in
  [LOOP_CLOSURE.md](LOOP_CLOSURE.md#coorg-results-26-10-26-2025-ist-a4000).
- Square-walls diagnosis: frames 0–160 s lie within ±10° of the house axes, while
  frames from 162 s sit at +31–35°. The 160.2→162.2 s transition has 407 feature
  matches, so the turn is image-supported, and the prior does not override it.
- Actual-PLY inspection (`inspection/topdown_comparison.png`, seven camera slices):
  the arms are visually almost identical. The doubled 171 s wall, the dining geometry
  crossing the TV wall and the 158 s pillar layers persist.
- Checks: `uv run --no-sync pytest tests/reconstruction tests/loop_closure -q`
  **60 passed**. Ruff check/format and `uv lock --check` pass. Real VGGT, DINOv2 and
  ALIKED/LightGlue inference ran. No GUI viewer was opened.

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

- Coorg's dining wing (≈162–215 s) has no image-verifiable loop to earlier rooms, so its
  ≈30° yaw offset and the resulting TV-wall interpenetration cannot be closed from this
  capture. A recapture that walks back from the dining room into the living room is needed,
  or the user can confirm whether that wing is genuinely angled.
- Remaining wall layering is partly per-frame VGGT depth error on plain walls and
  reflective/translucent regions; camera-only optimization cannot remove it.

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

## Historical branch snapshot: origin/feat/floor

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `21-10-26 16:05:39 IST`

## Current Focus

Both full captures have real reconstructed point clouds. User review found uneven
Coorg floors; separate floor-aligned exports now provide a common Z-up/floor Z=0
convention. Room uses one global rigid transform; Coorg uses per-view floor
constraints. Work is on `feat/floor`, branched from `feat/reconstruction` at
`175fb6f`, per user request. The interactive viewer remains unexecuted by the agent.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud export | Verified on both captures; interactive qualitative review pending |
| Floor-plane alignment | Implemented and verified on real outputs; Coorg floor residuals reduced, cross-view drift remains |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `21-10-26 15:58:08 IST`
- **Decision:** Export separate semantic/RANSAC floor-aligned clouds; preserve Room with one rigid transform, use explicit single-floor per-view constraints for Coorg, and retain camera/point provenance.
- **Reference:** “Floor-constrained point-cloud postprocessing” in [PLANS.md](PLANS.md).

## Current Progress

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

## Historical branch snapshot: origin/feat/reconstruction

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `26-10-26 11:20:28 IST`

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

### Camera-traversal viewer (26-10-26 11:20:28 IST)

- Added `--follow-cameras` to `view_point_cloud.py` to play saved clouds along
  VGGT, section-aligned or refined camera poses at 3 fps. Camera artifacts are
  auto-discovered beside the cloud or can be passed explicitly. Focused tests pass
  (**6 passed**); interactive playback remains for the user to confirm.

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

## Historical branch snapshot: origin/feat/spacegen

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `31-10-26 15:20:00 IST`

## Current Focus

Work is on `feat/spacegen`: `spacegen/`, a standalone pipeline from any indoor video to
a Gaussian splat with tour and demo videos (see `spacegen/docs/ARCHITECTURE.md`,
`spacegen/docs/LEARNINGS.md`, `spacegen/docs/BENCHMARKS.md`). Latest change: metric depth and
normal priors for blank walls and glass, and tours that only show what was filmed (see
[PLANS.md](PLANS.md), 31-10-26 13:04:01). On `vyas-room` at 10k steps:
- held-out normal error on blank walls fell from 37.5° to 17.9°;
- held-out PSNR went from 23.06 dB to 22.86 dB (colour-corrected 28.06 → 28.22).

In progress: full reruns of `vyas-room` and Coorg with these defaults. Paused: the
benchmark queue (`outputs/spacegen/queue`).

Earlier focus (`feat/dense-sfm-splat`):

Work is on `feat/dense-sfm-splat`. It adds full-resolution cameras (`src/sfm/`, see
[SFM.md](SFM.md)) and photoreal Gaussian splatting of the whole Coorg house
(`train_sfm_splat.py` → block training → visibility merge → joint fine-tune; see
[SPLATTING.md](SPLATTING.md#splatting-from-dense-sfm-cameras-train_sfm_splatpy--verified-on-coorg)).
Final model: `outputs/coorg_bnb/splat_runs/opencv/finetune_2` (published in
`example/house/`): merged blocks, 8 frames/s and a depth prior, moved onto the
OPENCV-lens cameras and fine-tuned. On the 20 held-out frames it scores 25.39 dB raw /
27.37 dB colour-corrected, SSIM 0.90; at the 518 grid it scores 26.3 dB raw, against
17.07 dB for the earlier VGGT splat. The tour video now includes an in-room look-around.

Earlier focus (`feat/LoopClosure`):
Work is on `feat/LoopClosure` (from `fix/recon-section-alignment`): loop closure for the
global camera drift seen in the Coorg section-aligned cloud. Two methods are compared,
VGGT-Long-style loop chunks (A) and bundle adjustment (B), plus an optional square-walls
arm (M). See [LOOP_CLOSURE.md](LOOP_CLOSURE.md).

Previous focus (`fix/recon-section-alignment`) was motivated by inspecting the initial
Coorg BnB reconstruction against its source frames: overlapping/warped surfaces,
detached fragments and inconsistent camera/depth geometry persist after floor
leveling. All 161 selected Coorg frames now have a 518-resolution section-aligned
export: four 36-view sections plus four 24-view tail sections, with an optimized
eight-node/13-edge Sim(3) graph. Independent matched-observation consistency and
actual-PLY visual inspection improve substantially over the initial 252 run.
Residual wall/pose distortion and imperfect loop closure remain. Observation fusion
is deferred pending the user's explicit instruction.
See the detailed adopted decision in [PLANS.md](PLANS.md).

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud export | Verified on both captures; Coorg original/section clouds also reviewed headlessly |
| Floor-plane alignment | Implemented and verified on real outputs; Coorg floor residuals reduced, cross-view drift remains |
| Coorg high-resolution section alignment | Verified on real Coorg outputs: connected global graph, all seven primary handoffs pass, improved consistency and visual structure; residual distortion remains |
| Loop closure (`src/loop_closure/`) | Implemented and run on real Coorg outputs; B gives a measurable gain (sequential −35%, held-out loops −10%); dining-wing drift and wall layering remain (no verifiable dining loop in this capture) |
| Gaussian splatting (`src/splatting`) | Implemented (awaiting verification on development server) |
| Shared rendering and Python/Viser viewer | Implemented (awaiting verification on development server) |
| Dense full-resolution SfM (`src/sfm/`) | Verified on Coorg: 927/927 frames, 1.31 px, loops checked by rotation and epipolar fit |
| SfM-input splatting, block merge, web export, fly-through, tour video (`src/splatting/sfm_*`, `merge_blocks`, `web_export`, `flythrough`, `tour`) | Verified on Coorg: held-out 23.9/27.0 dB, SSIM 0.91; visually inspected |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `30-10-26 10:30:00 IST` (outcome 11:10)
- **Decision:** Show each room from inside with a look-around that turns only through
  directions filmed from its standing point. Supersample tour renders. Keep the
  free-viewpoint training terms as options; the final model is unchanged.
- **Reference:** “Free-viewpoint quality: solid surfaces and an in-room look-around” in
  [PLANS.md](PLANS.md).

### Earlier decision (29 Oct)

- **Timestamp:** `29-10-26 17:20:00 IST` (amended 21:50)
- **Decision:** Solve Coorg's cameras with the OPENCV lens model; move the trained splat
  onto re-solved cameras (`transfer_splat.py`) and fine-tune; tour video with a local
  floor.
- **Reference:** “Flat-floor check, corrected cameras for the entrance, and a robust tour
  video” in [PLANS.md](PLANS.md).

### Earlier decision

- **Timestamp:** `26-10-26 19:11:38 IST`
- **Decision:** Add a separate read-only loop-closure stage and compare VGGT loop chunks (A) with bundle adjustment (B) on identical held-out correspondences.
- **Reference:** “Loop closure for global camera drift (two compared methods)” in [PLANS.md](PLANS.md).

### Previous decision

- **Timestamp:** `25-10-26 14:12:41 IST`
- **Decision:** Use depth-scaled robust section registration with independent interior handoff validation, explicit endpoint stress reporting, verified revisit constraints and a global Sim(3) solve. Keep the first four 36-view sections and use four 24-view tail sections. Reject ICP corrections that do not improve held-out agreement.
- **Reference:** “Depth-scaled fitting and independent interior handoff validation” in [PLANS.md](PLANS.md).

## Current Progress

### Dense SfM and photoreal splatting on Coorg (`feat/dense-sfm-splat`)

- **Recorded `28-10-26 21:13:39 IST`:** 927 frames (4 per second plus the 161 geometry frames) were
  solved at 1080×1920.
  - GLOMAP with every retrieval pair folded the trajectory: false loops on repeated
    grilles and doors, 10–20% of the pairs, with 100–180° rotation error.
  - A three-pass solve fixed it: sequential → loops checked by rotation → loops checked
    by epipolar fit. The last pass also removed translated-grille false loops at the
    entrance.
  - Final cameras (`sfm_loops2`): 1.31 px, 3 small trajectory breaks, true loops fit
    to 0.97 px median.
- **Splat training (`train_sfm_splat.py`):** MCMC, per-image pose and colour corrections,
  coarse-to-fine resolution. Root causes found and fixed:
  - MCMC noise was not scale-normalized (≈ radius² too strong), which caused fog.
  - Gaussian sizes were unbounded; there was no screen-size pruning (near-camera
    floaters, out-of-memory errors).
  - Budgets above 1.6M were unstable for the whole house.
  - The best single model (`full_loops2_v6b`) scores 19.05/21.6 dB, SSIM 0.80.
- **Blocks:** entrance 0–74.5 s, living room 74.5–180 s, dining 180–216 s, each trained on
  the same cameras with the global holdout (25.8–29.3 dB colour-corrected on their own
  views).
  - Nearest-camera merge: 20.6 dB. Visibility-weighted merge: 24.4 dB (colour-corrected,
    at half resolution).
  - A 6k-step joint fine-tune gives 23.9 / 27.0 dB; a second 10k-step pass gives the final
    24.6 / 27.6 dB, SSIM 0.91. SH degree 3 is needed (SH2 −0.9 dB, SH0 −3.7 dB).
- **Blur filter experiment:** a third fine-tune without motion-blurred training frames tied
  (24.65 / 27.48 dB, SSIM 0.90), so `merged_finetune2` stays final. The 200 s and
  72–80 s weak spots are capture-limited (blurred or textureless source frames).
- **Further experiments (29-10-26):**
  - Bigger per-block budgets: worse for the entrance and living room, +0.2 dB for dining.
  - Monocular depth prior: −0.4 dB on held-out views, fewer floaters in offset views
    (2.65 → 2.35% of pixels), worse at the entrance.
  - **Densification (`densify_sfm.py`):** 834/842 extra frames (8 per second) registered by
    PnP into `sfm_loops2`, with a median of 1814 inliers. The previous model renders them at
    28.87 dB, the same as its training frames.
  - Fine-tuning on the dense frames plus the depth prior gives the new final model
    `merged_dense8_depth`. Further gains need a recapture: slower passes at 199–201 s and
    72–80 s, and a loop back from the dining room.
- **In-room look-around and free-viewpoint tests (30-10-26, recorded 11:10 IST):**
  - *Request:* the user asked for a room without black spots, smears or merged objects,
    and for a view from inside each room before the walkthrough.
  - *Diagnosis:* eye-level renders along the path match the video. A level 360° turn from
    a room's middle smears walls and leaves dark gaps under the ceiling. 88% of the
    Gaussians never show in a training view, but removing them
    (`prune_invisible.py`) changes nothing. The smears are visible Gaussians in the wrong
    place, in directions nobody filmed from there.
  - *Training tests (not adopted):*
    - Random background plus a flatness term plus depth prior 0.2 (`solid.toml`) scores
      25.46 / 27.22 dB and wins only 7 of 20 views.
    - MCMC regrowth lost 2–3 dB; relocating all dead Gaussians at once gave NaN. The
      trainer now stops on a non-finite loss.
  - *Tour:* part 4 is a look-around per room (the living hall gets two). It picks the
    standing point by its filmed arc and a render score that penalises close surfaces.
    It follows the phone's tilt and trims 20° off each end of the arc. Renders are
    supersampled 1.5×.
  - *Visual inspection:* contact sheets every 2 s of the new tour. All four look-arounds
    are clean: the entrance bay window and crocs; the sofa and coffee table; the TV wall
    with the dining room through the door; the dining table, laptop and chairs. The
    build-up no longer shows a hole in the hall floor (reveal distance 1.6 → 3 camera
    heights). Published as `example/house/coorg_tour.mp4` and
    `look_around_frames.jpg`.
  - *Checks:* `uv run --no-sync pytest -q tests/splatting tests/sfm`: 47 passed (CUDA
    tests on the A4000). Ruff passes on the changed files.
- **Entrance floor and tour video (29-10-26, recorded 22:20 IST):**
  - *Problem:* the user found a black hole in the tour's plan view at 1:09. The
    single-coefficient lens solve had sunk the foyer and mat-corridor floor
    0.25–0.3 camera heights, and the tour's one global floor cut it away.
  - *Tour fix:* the tour measures the floor locally and filters glossy-floor reflections
    and floaters.
  - *Camera fix:* cameras re-solved with the OPENCV lens model (`sfm_opencv`, 1.27 px;
    floor spread 0.144 → 0.059 camera heights), then densified (`sfm_opencv_dense8`,
    806/826 extra frames).
  - *New final model `opencv/finetune_2`:* the splat transferred and fine-tuned. Held-out
    25.39 / 27.37 dB, SSIM 0.904; worst view 20.3 → 21.2 dB (colour-corrected, half
    resolution); floaters 2.48%.
  - *Visual inspection:* `example/house/heldout_overview.jpg` (real | previous final |
    new final) and tour frames at the entrance, orbit and walkthrough. The entrance floor,
    mat and crocs show in the plan, and the 50 s door floater is gone.
  - *Failed attempts:* an incremental solve, floor-constrained BA, a piecewise similarity
    and a split focal length (see [SFM.md](SFM.md)).
- **Visual inspection (earlier):** `outputs/coorg_bnb/final/heldout_overview.jpg` (real frame |
  VGGT splat | new splat) and `flythrough_frames.jpg` from
  `outputs/coorg_bnb/final/coorg_flythrough.mp4` (1387 frames).
  - Sharp: furniture, curtains, inlays, grilles.
  - Remaining weak spots: a floater at the left edge of the 50 s door view, a soft shelf
    at 200 s, and the blank-ceiling pan at 72–80 s. The 113 s and 200 s held-out views
    are not better than the VGGT splat.
- **Web export:** pruning at opacity < 0.02 leaves 2.08M Gaussians with no PSNR change.
  The full-SH PLY is 491 MB; the 32-byte `.splat` is 63 MB (it loses view-dependent
  colour, −3 dB).
- **Checks:** `uv run --no-sync pytest tests -q`: 117 passed, 1 skipped, 1 xfailed (CUDA
  tests ran on the A4000). Ruff passes on the changed files.

### Gaussian splatting — authored, not executed

- **Recorded `27-10-26 02:46:18 IST`:** Added `src/splatting`, `src/rendering`,
  `configs/gaussian_splatting.toml`, an example scene manifest, root training/rendering/
  viewing scripts, CLI entry points, optional dependency groups and stage exports.
- The input contract requires aligned RGB, decoded VGGT camera-head world-to-camera
  extrinsics/intrinsics, refined camera-Z depth and world-space point maps. It
  records source frame IDs/timestamps, image transforms, scale and optional masks/
  confidence. Shapes, camera transforms and own-camera depth/reprojection agreement
  are checked explicitly. Paths support per-frame NPY and named/indexed NPZ arrays.
- Training-only bounded point sampling, multiview depth support, voxel fusion,
  outlier removal and nearest-neighbor scales seed fixed-camera RGB/SSIM/depth
  optimization. Budgeted adaptive refinement retains optimizer alignment. Checkpoints
  save parameters, optimizer/refinement/RNG state, input fingerprints and versions.
- Export includes standard full-SH Gaussian PLY, metadata, numeric metrics and
  reference/render previews. The shared renderer produces RGB/alpha/camera-Z depth;
  the server-side Python/Viser viewer uses it for view-dependent appearance.
  The integer-centre VGGT to half-integer gsplat raster convention is converted
  explicitly at the rendering boundary.
- Tests are written for file/camera contracts, geometry filtering, holdout exclusion,
  budgets, SH PLY and Adam checkpoint continuation, NaN-safe losses, topology state,
  real CUDA training/rendering and raster pixel-centre alignment. **All are not run.**
  No fake renderer or substitute model was used.
- [SPLATTING.md](SPLATTING.md) documents the exact contract, API, server commands,
  outputs, limitations and verification steps. `uv.lock` is unchanged because
  dependency resolution must be performed on the development server.

### Loop closure on Coorg (`feat/LoopClosure`)

- **Recorded `26-10-26 20:25:16 IST`:** `close_loops.py outputs/coorg_bnb/section_alignment
  --output-dir outputs/coorg_bnb/loop_closure` ran all arms on the A4000. The section
  directory was read-only; every artifact is under `outputs/coorg_bnb/loop_closure/`.
- 494 DINOv2 candidates, 24 verified loops (5 chunk centres / 13 train / 6 held out).
  The repeated-structure guard rejected every true wide-baseline living-room loop, so a
  non-planar override was added. Measured planar fraction was 0.30–0.38 for true loops
  and 0.79–0.98 for grille/inlay false positives. Match images were inspected.
- No verifiable dining→living loop exists. An exhaustive scan of 178–214 s against
  77–146 s found 0 epipolar inliers, and LoFTR indoor found no matches.
- Method A accepted 3/5 loop edges (0↔2, 2↔3, 2↔4); all adjacent held-out checks still
  pass after optimization. B from baseline: inlier reprojection 2.53→1.07 px.
  Normalized held-out loop disagreement 0.0458→0.0413; sequential 0.0166→0.0107.
  A alone and M did not improve the held-out metrics. Full table in
  [LOOP_CLOSURE.md](LOOP_CLOSURE.md#coorg-results-26-10-26-2025-ist-a4000).
- Square-walls diagnosis: frames 0–160 s lie within ±10° of the house axes, while
  frames from 162 s sit at +31–35°. The 160.2→162.2 s transition has 407 feature
  matches, so the turn is image-supported, and the prior does not override it.
- Actual-PLY inspection (`inspection/topdown_comparison.png`, seven camera slices):
  the arms are visually almost identical. The doubled 171 s wall, the dining geometry
  crossing the TV wall and the 158 s pillar layers persist.
- Checks: `uv run --no-sync pytest tests/reconstruction tests/loop_closure -q`
  **60 passed**. Ruff check/format and `uv lock --check` pass. Real VGGT, DINOv2 and
  ALIKED/LightGlue inference ran. No GUI viewer was opened.

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

- Coorg's dining wing (≈162–215 s) has no image-verifiable loop to earlier rooms, so its
  ≈30° yaw offset and the resulting TV-wall interpenetration cannot be closed from this
  capture. A recapture that walks back from the dining room into the living room is needed,
  or the user can confirm whether that wing is genuinely angled.
- Remaining wall layering is partly per-frame VGGT depth error on plain walls and
  reflective/translucent regions; camera-only optimization cannot remove it.

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
- The development server and actual refined VGGT files are inaccessible. Dependency
  resolution/CUDA compilation, lint/tests, actual file integration, viewer operation,
  memory/performance and real-capture splat quality remain unverified. No visual
  inspection of splatting output has been performed. Resolve the lockfile on the
  development server before using locked dependency workflows.
- No execution or dependency blockers remain for the supplied SDR videos.
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
1. On the development server, resolve splatting dependencies and run the documented
   lint, CPU and real-CUDA tests; record failures and actual versions.
2. Map the actual refined VGGT output layout to the documented manifest. Train a
   representative capture, inspect held-out previews and the viewer, and record
   quality/memory findings before declaring this stage verified.
3. Retain preprocessing follow-ups: review accepted revisit images, consider richer
   blur evidence if needed, and integrate reconstruction through `load_result(output_dir)`
   while preserving source/candidate provenance.

## Historical branch snapshot: origin/fix/recon-mv-refinement

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

## Historical branch snapshot: origin/fix/recon-section-alignment

# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `25-10-26 14:23:56 IST`

## Current Focus

Work is on `fix/recon-section-alignment`, motivated by inspecting the initial
Coorg BnB reconstruction against its source frames: overlapping/warped surfaces,
detached fragments and inconsistent camera/depth geometry persist after floor
leveling. All 161 selected Coorg frames now have a 518-resolution section-aligned
export: four 36-view sections plus four 24-view tail sections, with an optimized
eight-node/13-edge Sim(3) graph. Independent matched-observation consistency and
actual-PLY visual inspection improve substantially over the initial 252 run.
Residual wall/pose distortion and imperfect loop closure remain. Observation fusion
is deferred pending the user's explicit instruction.
See the detailed adopted decision in [PLANS.md](PLANS.md).

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures; conservative repeated-structure filtering, known per-frame blur ambiguity |
| VGGT inference and point-cloud export | Verified on both captures; Coorg original/section clouds also reviewed headlessly |
| Floor-plane alignment | Implemented and verified on real outputs; Coorg floor residuals reduced, cross-view drift remains |
| Coorg high-resolution section alignment | Verified on real Coorg outputs: connected global graph, all seven primary handoffs pass, improved consistency and visual structure; residual distortion remains |
| Surface reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `25-10-26 14:12:41 IST`
- **Decision:** Use depth-scaled robust section registration with independent interior handoff validation, explicit endpoint stress reporting, verified revisit constraints and a global Sim(3) solve. Keep the first four 36-view sections and use four 24-view tail sections. Reject ICP corrections that do not improve held-out agreement.
- **Reference:** “Depth-scaled fitting and independent interior handoff validation” in [PLANS.md](PLANS.md).

## Current Progress

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
