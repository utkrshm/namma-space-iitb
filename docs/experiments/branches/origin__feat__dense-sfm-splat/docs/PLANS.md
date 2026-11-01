# Project plans

Record substantive decisions and their implementation steps here before execution. Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Latest entries get recoreded first, stale entries come later.

## Entry format

```text
## DD-MM-YY HH:MM:SS IST — Title
- Status: proposed / adopted / superseded
- Decision:
- Rationale and trade-offs:
- Affected components:
- Implementation steps:
- Completion criteria:
```

## 30-10-26 10:30:00 IST — Free-viewpoint quality: solid surfaces and an in-room look-around

- **Status:** Partly adopted (30-10-26 11:10 IST). The look-around, supersampled tour
  renders and `prune_invisible.py` are adopted. The training terms remain options; the
  final model is unchanged.
- **Outcome:**
  - *Training terms:* the fine-tune (`configs/coorg_solid/solid.toml`) held the held-out
    score: 25.46 / 27.22 dB against 25.39 / 27.37. It was better on only 7 of 20 views,
    and in-room views were nearly unchanged.
  - *Regrowth:* MCMC regrowth from the visible Gaussians lost 2–3 dB with or without
    the new terms. Relocating all dead Gaussians at once gave NaN.
  - *Why:* this capture's unfilmed directions cannot be recovered by training.
  - *The fix:* the look-around now turns only through directions filmed from where it
    stands, and follows the phone's tilt, as the problem statement below required.
- **Problem:** Eye-level renders along the capture path match the video, but views from
  the middle of a room, turned away from where the phone pointed, show smeared walls and
  furniture and dark holes near the ceiling. Two findings:
  - 88% of the 4.04M Gaussians never contribute even 1 px in any training view, but
    removing them leaves these views unchanged.
  - Rendering without view-dependent colour also leaves them unchanged.

  So the smears are visible Gaussians that sit in the wrong place: semi-transparent,
  thick layers that only reproduce the filmed angles.
- **Decision:** Add three optional training terms and compare fine-tunes of
  `opencv/finetune_2` on held-out views and on fixed in-room look-around views:
  1. A random background colour per step (`training.random_background`), as in
     Instant-NGP and Nerfstudio. Any see-through region then shows a different colour each
     step, so training must make surfaces opaque instead of fog over black.
  2. A flatness term (`training.flatten_regularization`): the mean ratio of each
     Gaussian's smallest to largest scale. Thin, disc-like Gaussians lie on surfaces, as
     in 2DGS, Gaussian Surfels and PGSR.
  3. A stronger monocular depth prior (0.05 → 0.2), since free roaming now matters more
     than the last few tenths of a dB near the path.

  The tour gains an in-room look-around before the walkthrough. For each room it stands
  at the eye-level point whose surroundings were filmed from the widest range of
  directions, then turns through the headings that were actually filmed from nearby.
- **Completion criteria:** held-out PSNR within about 0.5 dB of the current final model;
  visibly cleaner look-around views; the new tour checked frame by frame.

## 29-10-26 17:20:00 IST — Flat-floor check, corrected cameras for the entrance, and a robust tour video

- **Status:** Adopted. The tour's floor handling is in place. The cameras are now solved
  with the OPENCV lens model (amended 29-10-26 21:50 IST; see below).
- **Amendment (21:50 IST):**
  - *Approaches that failed:*
    - The incremental mapper split into three models.
    - `floor_bundle_adjust.py` flattened the floor only at 2–3 px reprojection error, and
      plain bundle adjustment returned to the step.
    - A piecewise shrink of the entrance, followed by bundle adjustment, also returned to
      the step.
    - A separate focal length for 0–96 s found the same focal and left the step.
  - *The approach that worked:* the same GLOMAP pass with `--camera-model OPENCV`. The
    floor spread falls from 0.144 to 0.059 camera heights. The camera height above the
    floor at 24–60 s and 96–110 s falls to 1.05× and 1.01× the hall value. Reprojection
    improves from 1.31 to 1.27 px. The fitted lens has k1 0.082 and k2 −0.216, so the
    single radial term was the cause.
  - *Adopted pipeline:*
    1. `sfm_opencv`.
    2. `densify_sfm.py --holdout-every 8 --holdout-exclusion-s 0.13`, giving
       `sfm_opencv_dense8`.
    3. `transfer_splat.py`: point-based local similarities move the final splat.
    4. A 12k-step fine-tune with the depth prior
       (`configs/coorg_opencv/finetune.toml`).
  - `floor_bundle_adjust.py` stays as a diagnostic: it measures how much reprojection
    error flattening a floor costs.
- **Decision:**
  - The tour's cut-away measures the floor locally on a plan grid instead of using one
    floor height.
  - Re-solve the cameras with COLMAP's incremental mapper, on the same verified matches as
    `sfm_loops2`.
  - If its floor is flat, move the trained splat onto the new cameras
    (`src/splatting/transfer.py`) and fine-tune, instead of retraining from scratch.
- **Rationale and trade-offs:**
  - *Symptom:* the user saw a black hole in the tour's plan at the entrance (1:09), where
    the eye-level view shows the mat and crocs.
  - *Measured cause:* the rendered floor of the bay-window foyer and the mat corridor
    (24–72 s) lies about 0.4 units (≈0.3 camera heights) below the rest of the house. The
    global floor cut removed it.
  - *Why it is a solve error:* the video shows one flat floor. The main door's bottom
    sits at hall level, and the same mat runs on continuously. Two door heights agree to
    5% (2.13 vs 2.02 units).
  - *Size of the error:* camera height above the local sparse floor is 1.2–1.36× the
    hall value in GLOMAP pass 3, against 2.4× in pass 2 and 3× sequential-only. VGGT shows
    1.09–1.22×.
  - *Why the incremental mapper:* it triangulates and bundle-adjusts as it registers each
    frame, so scale passes through 3D points rather than only through translation
    directions. That is the usual cure for sag along a weakly looped stretch.
  - *Why transfer instead of retraining:* retraining the three blocks would take about 6
    hours. Transfer uses the shared keypoints of both solves: blended per-camera
    similarities plus point residuals, then a fine-tune.
- **Affected components:** `src/splatting/tour.py` (`local_floor`, reflection and floater
  filters), new `src/splatting/transfer.py` and `transfer_splat.py`, and the
  `outputs/coorg_bnb/sfm_inc` run.
- **Completion criteria:**
  - The sparse floor-height ratio is ≈1 in every time window.
  - Held-out PSNR is not worse than `merged_dense8_depth`.
  - The tour plan shows the entrance floor.

## 27-10-26 15:28:36 IST — Dense full-resolution SfM cameras and quality-focused splat training

- **Status:** Adopted (on `feat/dense-sfm-splat`, which merges `feat/LoopClosure` and the
  committed `origin/feat/splatting`).
- **Decision:** Add `src/sfm/`. It selects ~4 sharp frames per second from the source
  video (always including the 161 geometry frames) and exports them at full 1080×1920
  resolution. It matches them with ALIKED/LightGlue on sequential pairs and on verified
  DINOv2 retrieval pairs, then solves one shared camera with pycolmap (global mapper,
  incremental mapper as an alternative) and undistorts to a pinhole COLMAP model.
  Add a COLMAP-input trainer to `src/splatting` (new modules only). It uses gsplat MCMC
  densification, per-image camera refinement, per-image colour (exposure/white-balance)
  compensation and progressive resolution. It writes the existing PLY/checkpoint
  layout, so `view_gaussian_splat.py` and `render_gaussian_splat.py` keep working.
- **Rationale and trade-offs:** The existing Coorg splat (161 VGGT views at 294×518) scores
  30.6 dB on training views but 17.0 dB / SSIM 0.59 on held-out views, and its held-out
  renders are smeared and offset. That gap points to too few views (0.75 per second) and
  VGGT cameras that are only accurate to about one 518-grid pixel (≈4 px at full
  resolution), not to the trainer. Dense video frames give the overlap that SfM and 3DGS
  need. Learned features on dense sequential pairs avoid the failure of the earlier
  SIFT/exhaustive COLMAP test on 36 sparse room views. VGGT cameras stay the independent
  check: SfM cameras on the shared geometry frames are compared with them after a Sim(3).
  Phone video has auto-exposure, so per-image colour compensation stops exposure changes
  from turning into floaters. Camera refinement absorbs remaining sub-pixel pose error.
  The new trainer does not edit utkarsh's in-progress `src/splatting` files; it adds
  modules beside them.
- **Affected components:** new `src/sfm/`; new `src/splatting/sfm_*` modules; root scripts
  `build_sfm.py` and `train_sfm_splat.py`; `sfm` extra (`pycolmap`); tests; SFM/SPLATTING/
  ARCHITECTURE/STATE docs.
- **Implementation steps:**
  1. Select and export frames, extract features, match pairs, map, undistort.
  2. Write a report and inspection plots, and compare with VGGT cameras on shared frames.
  3. Train with held-out views that are exactly utkarsh's 20 validation frames, for a
     like-for-like comparison.
  4. Inspect renders visually.
  5. Iterate on the dominant error, e.g. a depth prior for plain walls, or more frames.
- **Amendments (28-10-26):** Retrieval loops are checked against a sequential-only
  trajectory (rotation), then against the loop-closed model (epipolar fit). The whole
  house is trained as three time blocks, merged by visibility-weighted ownership and
  jointly fine-tuned, because a single model was capacity-limited and became unstable
  with larger budgets. MCMC noise is scale-normalized; screen-size pruning was added.
- **Completed `28-10-26 21:13:39 IST`:** 927/927 frames registered. The final splat scores
  23.9 dB raw / 27.0 dB colour-corrected, SSIM 0.91, on the shared held-out frames
  (VGGT splat: 17.07 dB / SSIM 0.59 at 518). Details are in STATE.md, SFM.md and
  SPLATTING.md.
- **Completion criteria:** Most frames are registered in one model with sub-pixel reprojection
  error. Held-out PSNR/SSIM on the shared validation frames is clearly above 17.0 dB /
  0.59. Held-out renders and novel views are inspected visually. Tests on real data pass.

## 26-10-26 19:11:38 IST — Loop closure for global camera drift (two compared methods)

- **Status:** Adopted for implementation on `feat/LoopClosure` (based on `fix/recon-section-alignment`).
- **Evidence motivating the change:** Camera-relative and top-down slices of the actual
  `section_alignment/point_cloud.ply` (Coorg) colored by source time show that the late
  dining sections (≈180–215 s) are rotated by roughly 30° relative to the living room and
  pass through the TV wall. Plain walls appear twice (≈170 s vs ≈200 s copies), and the
  wood pillar appears in three to four places. The four preprocessing revisits connect only
  entrance/luggage views, so nothing constrains the end of the capture. The video does
  look back from the dining area into the living room (≈207–211 s), and from the living
  room into the dining area (≈88–113 s).
- **Decision:** Add a separate `src/loop_closure/` stage that reads the cached section
  predictions **read-only** and writes only to a new output directory. It shares one
  loop-detection step and offers two methods for comparison:
  - **A — VGGT-Long-style loop chunks.** Retrieve far-apart frame pairs with DINOv2 and
    verify them with ALIKED/LightGlue. Jointly infer a new VGGT "loop chunk" built from
    the frames around both ends. Register the chunk to each owner section using exact
    shared pixels (the same contract as adjacent overlaps), then derive a section-to-section
    Sim(3) loop edge. Add it to the existing global section graph. Held-out whole frames
    and cross-side feature agreement inside the chunk gate every loop edge. This follows
    the algorithm of the vendored `deps/VGGT-Long` (chunk → loop chunk → Sim(3) graph),
    but stays inside our VGGT wrapper, preprocessing contract and validation gates. It
    avoids VGGT-Long's separate image-directory pipeline and its extra dependencies
    (faiss, numba, pytorch-lightning, SALAD weights).
  - **B — Bundle adjustment.** Start from a section-graph export (baseline or A).
    Refine one camera pose and one depth scale per frame, plus 3D track points, using
    ALIKED/LightGlue tracks over sequential neighbours and verified loop pairs.
    Reprojection residuals are robust (Huber), and a weak log-depth prior ties each track
    to its VGGT depth. This prior fixes scale and keeps dense depth exportable. Intrinsics
    stay at the owner-section VGGT values. Some loop pairs are held out of the solve
    entirely for evaluation.
- **Rationale and trade-offs:** One Sim(3) per section cannot remove drift *inside* a
  section, so A alone may leave bends; B can bend every frame but depends on sparse
  feature coverage (weak on blank walls). Comparing both on identical held-out evidence
  shows which actually closes the loop. Neither method fuses observations, edits source
  inputs, or overwrites `section_alignment/`.
- **Affected components:** new `src/loop_closure/` package (contracts, detection, loop
  chunks, graph export, bundle adjustment, evaluation, CLI), a keyword-only `export_dir` /
  `strict_post_graph` option on `run_section_alignment` (defaults unchanged), root
  script `close_loops.py`, `[project.scripts] namma-loop-closure`, `tests/loop_closure/`,
  `docs/LOOP_CLOSURE.md`, `docs/ARCHITECTURE.md`, `docs/STATE.md`, `README.md`.
- **Completion criteria:** Real Coorg run of both methods. The same held-out loop-pair
  feature correspondences are evaluated for baseline, A and B (normalized 3D disagreement),
  along with adjacent-pair consistency and top-down/camera-slice renders of the actual
  exported clouds. The adopted method is recorded with its failures; no gate is loosened
  to make a result pass.
- **Amendments during execution (`26-10-26 20:25:16 IST`):** (1) the retrieval threshold
  dropped to 0.40/top-8, because at 0.70 the repeated entrance grilles monopolised
  candidates. (2) A non-planar override admits repetition-flagged pairs whose epipolar
  inliers are not explained by one or two homographies; this was measured on real true
  and false pairs before adoption. (3) Bundle adjustment uses explicit parameter scales
  plus a 30 px initial gate; with Jacobian-based scaling the cameras did not move.
  (4) An optional square-walls arm (M) was added to test the ≈30° dining yaw. It had no
  effect because image evidence supports the 162 s turn.
- **Completed:** B (from baseline) is adopted as the camera solution. A and M are retained
  as compared, documented arms. Results and limitations are in
  [LOOP_CLOSURE.md](LOOP_CLOSURE.md) and [STATE.md](STATE.md).

## 25-10-26 18:53:36 IST — Failed COLMAP reconstruction experiment on the M Block room

- **Status:** Failed experiment; further work in this direction discontinued.
- **Plan tried:** Evaluate COLMAP-based reconstruction as an alternative to VGGT, using real CPU SIFT extraction, exhaustive feature matching and incremental sparse Structure-from-Motion through `pycolmap`. Implementation and tests were carried out on the separate `feat/colmap-reconstruction` branch; this entry records the outcome on the current `fix/recon-section-alignment` branch.
- **Evaluation:** Tested on `videos/room_m_block.mp4` (the M Block room), using its existing 36 geometry keyframes with shared per-clip `SIMPLE_RADIAL` intrinsics, a 1600-pixel maximum feature-extraction image size, an 8192-feature setting and four CPU threads. The completed run took 259.761 seconds and registered 27/36 distinct images across three overlapping, independently framed submodels. Their image/point counts were 2/479, 21/7,329 and 26/11,970; mean reprojection errors were approximately 0.345, 0.992 and 0.696 pixels respectively. Submodel coverage overlaps and must not be summed.
- **Qualitative findings:** Personally inspected multiple orthogonal and perspective views of every submodel, compared them with chronological source frames, and rendered the largest exported PLY directly in Open3D. The largest model recovered recognizable bedding colors, bed-frame edges and furniture fragments, but most geometry remained thin, disconnected patches. Blank walls and broad floor regions had large holes, some structures were stretched or poorly aligned, and camera positions clustered tightly relative to the reconstructed scene extent. The smaller 21-image model was less coherent; the two-image model was an almost planar patch rather than a room. Nine keyframes remained unregistered, and the low reprojection errors did not establish convincing room shape or metric accuracy.
- **Decision and rationale:** The constructed point cloud was not convincing enough to warrant further work in this direction. Discontinue this COLMAP-based sparse reconstruction approach for the project based on the incomplete, geometrically unreliable M Block room result. This experiment did not evaluate dense stereo or meshing.
- **Evidence retained:** Native sparse models, colored `model_0.ply` / `model_1.ply` / `model_2.ply`, `report.json` and inspection screenshots remain under `outputs/room_m_block/colmap_geometry/`. The separate COLMAP branch retains its implementation and evaluation documentation.

## 25-10-26 14:12:41 IST — Depth-scaled fitting and independent interior handoff validation

- **Status:** Adopted after real overlap investigations; supersedes the tail configuration and endpoint-based primary validation in the preceding execution entry.
- **Evidence:** Retrying only the last window, including an earlier 24-view bridge, did not repair distortion already present at the end of original section 4. Replacing the last two original sections with four 24-view windows gives strong adjacent support. A separate probe showed the generic cloud-extent RANSAC threshold was too tight on wall patches: one overlap's target extent was 0.166 while camera distance was 0.514. Whole-section endpoints also contain per-view pose/depth outliers and are not the views chosen at the owner handoff. The failed original connection remained bad even after excluding endpoints; it was not rescued by relabeling validation.
- **Decision:** Keep four initial 36-view windows (`0,25,50,75`) and reconstruct four 24-view tail windows (`100,113,126,137`) at 518 resolution. Use `--tail-section-size 24 --tail-start-section 4`; preserve all superseded artifacts. Scale section-fitting tolerance by target median camera-to-observation distance (`0.03 × distance`), consistent with depth-dependent errors, rather than visible patch extent.
- **Validation contract:** For sufficiently large overlaps, hold out three whole interior frames near overlap quartiles (positions 2/5/8 of 11) and reserve the first/last shared views as separate endpoint stress observations. Neither primary validation nor stress observations enter fitting. For five shared frames, use three training/two validation frames without a separate stress split. Preserve the existing primary absolute limits and minimum three-frame/two-cell training consensus. Save all stress points/errors, including after optimization; do not remove observations or disguise within-section outliers as registration success.
- **Probe outcome:** All seven candidate boundaries pass independently held-out interior checks. Normalized median/p90 pairs are approximately `0.023/0.067`, `0.045/0.098`, `0.033/0.079`, `0.013/0.029`, `0.050/0.093`, `0.011/0.020`, `0.008/0.014`; training consensus spans 4–8 frames. Full independent feature benchmarking, actual-PLY inspection and final state recording follow the joint graph solve.
- **Completed `25-10-26 14:23:56 IST`:** Exported 999,971 RGB observations from all 161 frames with a connected eight-node/13-edge graph. All seven post-graph primary checks pass; all seven ICP candidates were rejected. On 6,937 identical eligible image-feature observations, normalized median/p90 disagreement improves from `0.23158/0.81667` to `0.01435/0.06162` (93.8% median reduction). Actual-PLY inspection finds markedly improved recognizable structure, with residual wall distortion, endpoint stress and imperfect loop agreement explicitly retained. Full results, limitations and viewing commands are recorded in [STATE.md](STATE.md) and [RECONSTRUCTION.md](RECONSTRUCTION.md#high-resolution-section-alignment). Observation fusion remains deferred.

## 25-10-26 13:45:47 IST — Repair unsupported dining-section overlap

- **Status:** Adopted during real execution of the section-alignment experiment.
- **Evidence:** All six original 36-view sections ran successfully at 518 resolution, peaking at approximately 9.96 GiB allocated / 12.21 GiB reserved. The last overlap had consensus in only one training frame; all three held-out frame medians exceeded 0.10 of camera-to-scene distance. Its ICP candidate reduced the median only from 0.218 to 0.207 and left p90 near 2.96, so that connection is unsupported. Replacing only the last window with two 24-view windows did not establish sufficient shared-frame consensus.
- **Decision:** Preserve failed section artifacts and retain the first five 36-view windows. Add earlier context to the smaller tail windows: zero-based starts 114, 127 and 137, each with 24 frames at unchanged 518 resolution. The active candidate graph therefore has eight nodes and full 161-frame coverage. Expose this explicitly as `--tail-section-size 24 --tail-context-frames 11`, never as an automatic resolution or frame-count fallback.
- **Gate clarification:** Section alignment must have consensus across at least three training frames and two spatial cells, a pooled held-out normalized median at most 0.10, and at least two held-out frames with median at most 0.10. Report upper tails and failing individual views without deleting their observations. Boundary 2–3's large p90 is concentrated in one inconsistent held-out camera/depth view; two independent held-out frames agree closely. A p90 above 0.35 is a diagnostic warning rather than a replacement for frame-level support. ICP must pass these absolute gates in addition to relative improvement. Recheck the same gates after global optimization.
- **Additional verification:** Reverified image revisits must retain broad image and 3D support after depth/confidence filtering and within both training consensus and held-out subsets. Independent benchmarking uses the same source features eligible at both resolutions, one characteristic camera-to-point distance per method, and separately reported owner-boundary pairs. Original clouds and all source observations remain unchanged; no observation fusion is performed.

## 25-10-26 12:41:11 IST — Coorg high-resolution section alignment

- **Status:** Adopted at the user's request; work on the already-created `fix/recon-section-alignment` branch.
- **Motivation:** Inspection of the initial Coorg BnB 161-view, 252-square reconstruction and source frames found ghosted/curved surfaces, detached dining-area geometry, and inconsistent camera/depth estimates. Floor leveling did not repair registration. The immediate benchmark is a demonstrable improvement over that vanilla reconstruction.
- **Scope:** Process only Coorg BnB's existing 161 geometry frames, preserving the complete image field of view, source identities, timestamps and frozen preprocessing. All new generated artifacts, including section PLYs, intermediate transforms, diagnostics and final PLYs, belong under `outputs/coorg_bnb/section_alignment/`. Multi-view observation filtering/fusion, surface reconstruction and further floor correction are deferred until the user requests them.
- **Section inference:** Six chronological 36-frame sections start at zero-based positions 0, 25, 50, 75, 100 and 125, giving 11 shared frames at every adjacent boundary and complete coverage. Infer at padded 518×518 (portrait content approximately 294×518), bfloat16, sequentially on the 16 GB A4000. Record elapsed time and actual CUDA peak allocation/reservation. Persist resumable section predictions and provenance; never silently lower resolution or drop frames on OOM.
- **Registration contract:** A section-to-global similarity acts as `X_global = s * R @ X_section + t`, with proper rotation, positive scale, and arbitrary scene units. The first section fixes the global gauge. Transform camera centers with scale and camera orientations with rotation; retain valid inverse extrinsics and per-observation provenance. Shared source-frame/pixel identities supply direct 3D correspondences between independent section predictions.
- **Pairwise optimization:** Balance samples over shared frames and spatial cells, excluding padding/nonfinite/nonpositive depths and the lowest-confidence observations. Initialize positive-scale Sim(3) with RANSAC/weighted Umeyama, refine using a robust loss, and evaluate on whole held-out shared frames. Record support, normalized median/p90 errors, scale and camera agreement. Reject collinear/insufficient support rather than inventing a connecting edge.
- **Global optimization:** Construct a connected graph of adjacent section measurements and geometrically verified cross-section revisit measurements. Reverify existing conservative preprocessing revisit candidates using actual image features and reconstructed 3D support; never force a false loop. Jointly optimize robust relative Sim(3) constraints with section 0 fixed. A chain alone is reported honestly; additional independent overlap measurements may provide redundant constraints, but do not masquerade as nonlocal loop closure.
- **ICP experiment:** After scale-aware initialization, evaluate robust, multiscale point-to-plane ICP on overlapping geometry. ICP estimates a rigid correction at the established scale. Keep a candidate only if independent held-out shared-frame correspondences improve without unacceptable upper-tail/camera regression; nearest-neighbor fitness alone cannot establish success. Save rejected experiment metrics as well as accepted decisions.
- **Validation:** Use real pretrained inference and real feature correspondences. Compare the old 252 reconstruction and the new scene on identical source observations with source-to-model pixel transforms, confidence eligibility, a common similarity gauge and normalized errors. Inspect RGB, section-colored, top/side and boundary plots of the actual saved PLYs; check PLY/NPZ equality, all 161 source IDs, finite cameras, proper rotations, positive scales and graph connectivity. Report remaining internal section distortion rather than treating section registration as bundle adjustment.
- **Implementation structure:** Reusable section inference/export, numerical registration/graph optimization and evaluation/ICP modules under `src/reconstruction/`, public exports, and a thin runnable root script. Independent code-writing agents own disjoint modules; a separate reviewer examines correctness and tests. Dependencies remain in `pyproject.toml`, managed with `uv`.
- **Commit sequence:** (1) Section inference and robust Sim(3) registration plus this adopted plan; (2) graph/ICP integration and any corrections required by real Coorg execution; (3) final verification improvements and consistent results/state/architecture/usage documentation. Target three commits total; initial implementation must remain within four code commits/five including documentation. Generated captures/artifacts remain ignored, not committed.
- **Completion criteria:** A real 518-resolution, full-161-frame Coorg export; checked and saved global section transforms/graph; an evidence-based ICP decision; quantitative and visual improvement over the initial cloud, or an explicit unresolved blocker; final reproduction/viewing commands and final `STATE.md` update.

## 21-10-26 15:58:08 IST — Floor-constrained point-cloud postprocessing

- **Status:** Adopted following user review of uneven Coorg floors.
- **Decision:** Preserve original VGGT artifacts and export separate `floor_aligned/` clouds. Use pretrained ADE20K SegFormer floor/rug masks, confidence-filtered 3D observations and robust plane fitting. Give Room one global rigid leveling transform. For the single-storey Coorg capture, also align reliable per-view floor planes by minimal rotations about each camera and vertical translations; interpolate plane estimates in source time where the floor is unseen. Export all transforms, adjusted camera poses, mask provenance and before/after residuals.
- **Rationale and trade-offs:** Diagnostics show Coorg's local floor normals vary by up to 18 degrees; global leveling alone cannot address this. Transform whole observations rigidly rather than projecting floor pixels onto a fabricated flat surface. Preserve per-view shape, scale and provenance, but explicitly treat this as a single-floor prior, not bundle adjustment or a correction of all cross-view drift. Interpolation across floor-free spans is an estimate, and genuine steps/multiple storeys require a different model. Both scenes can share Z-up/floor Z=0 without claiming a common XY origin or metric scale.
- **Affected components:** `src/reconstruction/floor.py`, public exports, root `align_floor.py`, optional floor dependencies and reconstruction/architecture/state documentation; generated diagnostics and corrected artifacts under `outputs/`.
- **Implementation steps:** Inspect semantic masks and per-view plane evidence; implement robust floor estimation and transform serialization; generate Room rigid-level and Coorg floor-constrained clouds; verify residual reduction, proper rotations, rigid distance preservation and exact cloud/camera provenance. Retain the unexecuted interactive viewer and add no tests per the user's earlier instruction.
- **Completion criteria:** Separate usable colored clouds, a documented coordinate contract, explicit inferred/reliable floor-frame status, measurable floor-residual improvement for Coorg, and original reconstructions retained.

## 21-10-26 12:38:58 IST — Execute capture reconstruction and provide a standalone viewer

- **Status:** Adopted following user request.
- **Decision:** Run each full capture's preprocessing separately, pull `feat/reconstruction`, and export joint VGGT reconstructions under `outputs/<capture>/reconstruction/`. Source Linux PyTorch and torchvision from the official CUDA 12.6 index, retaining the declared package versions. Add root `view_point_cloud.py` using the existing Open3D visualization module; do not execute the viewer or add tests.
- **Rationale and trade-offs:** This GPU machine has an RTX A4000 (16 GB) and driver 535.309.01. The default CUDA 13 PyTorch wheel cannot initialize CUDA; CUDA 12.x minor-version compatibility permits using a CUDA 12.6 build without changing the system driver. Full geometry pools remain the initial inference target; any memory-limited run must explicitly record reduced resolution or frame selection rather than silently merging independent chunks.
- **Affected components:** `pyproject.toml`, `uv.lock`, `view_point_cloud.py`, reconstruction usage/state documentation and generated artifacts under `outputs/`.
- **Implementation steps:** Install reconstruction/viewer extras with compatible CUDA wheels; verify real CUDA initialization; run both capture reconstructions headlessly; inspect saved array/cloud metadata; document commands and results; add a thin standalone viewer CLI accepting a point-cloud path.
- **Completion criteria:** Both captures have real exported point clouds and recorded inference settings, or an explicit hardware/runtime blocker. The standalone viewer is documented and linted but is not executed; interactive quality review is left to the user.

## 21-10-26 11:48:44 IST — CUDA VGGT inference and Open3D reconstruction preview

- **Status:** Adopted following user clarification.
- **Decision:** Add a reusable vanilla VGGT wrapper under `src/reconstruction/`, an adapter consuming the frozen preprocessing `load_result()` contract, and root script `get_recon.py`. Run all selected views jointly on a larger CUDA GPU. Use Open3D for coloured point-cloud export and interactive viewing. Do not run model inference on the development laptop.
- **Rationale and trade-offs:** Joint inference gives one scene coordinate system; independently inferred chunks cannot be concatenated without alignment. Default to full-field-of-view, patch-aligned resize and padding, with explicit pixel transforms and valid masks. Preserve frame identity, timestamps and source rotation. Depth and geometry remain in arbitrary VGGT scene units, not metres. The preview concatenates filtered observations; it does not claim surface fusion, a mesh, or a navigation-ready digital twin.
- **Affected components:** New reconstruction modules and tests, stage public exports, optional `reconstruction`/`viewer` dependencies, `get_recon.py`, architecture/state and reconstruction usage documentation. The preprocessing implementation and its artifacts are frozen.
- **Implementation steps:** Define CPU NumPy output contracts; implement lazy model loading, CUDA precision selection, camera decoding and depth unprojection; adapt ordered geometry/appearance records; save arrays/provenance plus PLY; provide Open3D viewing of new or existing PLY files.
- **Dependency boundary:** Load the checked-in VGGT submodule lazily and declare its inference runtime dependencies explicitly. Avoid installing its upstream demo stack or its conflicting `numpy<2`/GUI OpenCV requirements into the frozen preprocessing environment. Keep Open3D separately installable for CPU-only viewing of an exported cloud.
- **Completion criteria:** Focused image-transform, geometry, filtering, preprocessing-handoff and serialization tests pass without a fake model; lint passes; CLI help and import checks run locally. Pretrained CUDA inference and interactive reconstructed-scene inspection are explicitly recorded as not run on this laptop.

## 27-10-26 02:30:17 IST — Geometry-initialized Gaussian splatting

- **Status:** Adopted; code authoring only, runtime verification deferred to the development server.
- **Decision:** Implement `src/splatting` on `feat/splatting`, consuming a versioned JSON manifest of RGB images, decoded VGGT camera-head extrinsics/intrinsics, refined camera-Z depth and refined world-space point maps. Use PyTorch and gsplat 1.5.3 for geometry-initialized, RGB/depth-supervised 3D Gaussian optimization. Keep cameras fixed. Provide `src/rendering`, offline rendering, and a Python-launched Viser viewer using the same full spherical-harmonic renderer.
- **Rationale and trade-offs:** The upstream files are on an inaccessible development server, so define the handoff explicitly rather than guessing its filenames. Preserve source identities, image transforms, world coordinates and arbitrary reconstruction scale. Initialization uses training views only, confidence/validity filtering, reprojection and multiview depth checks, bounded sampling, voxel fusion and nearest-neighbor scales. A temporal holdout measures novel-view appearance without feeding its geometry into initialization. CUDA execution belongs exclusively to the development server; no third-party compute is involved.
- **Affected components:** `src/splatting/`, `src/rendering/`, root training/rendering/viewing scripts, `configs/gaussian_splatting.toml`, optional dependencies, tests, and splatting/architecture/state documentation.
- **Implementation steps:** Define strict contracts/configuration; implement memory-bounded geometry loading and initialization; add RGB/SSIM and decaying relative depth losses, capped adaptive refinement, resumable checkpoints and standard SH PLY; share differentiable rendering with the viewer; write real-implementation tests and server commands.
- **Completion criteria:** Code and tests are written, statically reviewed, and documented. Dependency resolution, lint, tests, CUDA compilation, actual VGGT integration, reconstruction quality and visual inspection remain explicitly unverified until run on the development server. Do not execute project code, install dependencies, train or render on the authoring machine.

## 21-10-26 02:39:40 IST — Reject descriptor-ambiguous repeated structures

- **Status:** Adopted following round-three user review.
- **Decision:** Before geometric verification, compare each cross-image ALIKED match against distant same-image descriptor alternatives in both images. Only matches whose cross-image descriptor distance beats both self alternatives by a configured ratio may contribute inliers or spatial spread. Record removed-match counts on accepted edges and rejected-repetitive pair diagnostics in the report. Do not use a homography-fraction veto or time-specific exceptions.
- **Rationale and trade-offs:** Repeated window motifs can produce shifted correspondences that satisfy fundamental-matrix and coverage checks. Prefer missed loop links over ambiguous ones; do not relax the guard to retain doors or luggage. Keep the geometric matcher shared between the standalone pair API and pipeline.
- **Affected components:** Revisit verification/config/edge diagnostics, real fixtures/tests and pass-specific cache provenance. Preserve ingest, scoring and selection behavior, including user commit `4a6a0a3`.
- **Implementation steps:** Add self-alternative filtering and real door/window controls; add a real spliced-input tracking-break control; rerun link/export only with validated upstream cache reuse; record all prior accepted pairs' before/after counts and images.
- **Cache detail:** Invalidation now uses pass-specific module/record fingerprints plus config, schema, library versions, inputs and artifact checksums. The round-two global source hash was verified against the unchanged source before editing; per-pass provenance was recorded then for a validated legacy-key migration. Old runs without that provenance conservatively recompute. Optional edge diagnostics remain backward compatible with cached sequential edges.
- **Completion criteria:** Door positive accepts >= configured inliers, at most one of the two reviewed different-window pairs accepts, cross-building and temporal controls still pass, splice has one break without a cross-boundary edge. Record any lost true links rather than tuning them back in.

## 21-10-26 00:51:44 IST — Texture-aware tracking and blur evidence

- **Status:** Adopted following user review.
- **Decision:** Apply a whole-frame relative and absolute minimum-eigenvalue floor before distributing corners into cells. Only supported cells enter the overlap denominator. Tracking reliability depends on track count and spatial coverage; survival remains diagnostic. Separately, require sufficient edge content for a per-frame blur decision, use edge re-blur evidence rather than a cross-scene sharpness reference as the deciding signal, and report low texture as informational rather than a selection failure.
- **Rationale and trade-offs:** Cell-local corner thresholds admit compression noise in blank walls, causing false breaks. Sharpness relative to a textured neighboring scene confounds texture with blur. Real-frame regression fixtures will exercise both failures; ambiguous blur labels will be reported rather than fitted with time-specific rules.
- **Affected components:** Tracking, scoring/config/contracts, quality reporting, fixtures/tests and documentation. Existing PTS, orientation and color conversion remain unchanged.
- **Implementation steps:** Fix tracking first; then evaluate texture/blur on the eight user-labelled Coorg observations; finally run real optional revisits, export every accepted pair, and rerun both captures.
- **Implementation detail:** The corner floor is `max(1e-5, 0.01 * frame_max_eigenvalue)`. Continuity needs eight forward/backward-consistent tracks over at least three of sixteen cells, initially and currently; survival is never a gate. Low reliability is recorded on triggered selections rather than independently forcing minimum-interval keyframes. Schema 2 adds edge fraction and re-blur loss to CSV scores. At 640 px, the 49.5 s motion-blurred door cannot be reliably classified from re-blur evidence alone; retain an explicit expected-failure test and use corroborating temporal evidence, rather than fitting a narrow threshold to eight observations.
- **Completion criteria:** Compare components, tracking-break/weak-link counts, interval keyframe counts and reason histograms against the baseline below. Run real revisit, cross-building negative, spread and temporal-exclusion tests; report all accepted pairs and image paths.
- **Baseline:** Room: 1 component, 0 breaks/weak links, 49 geometry, 15 in 8.8–14 s; reasons `clip_start=1, overlap=43, flagged_fallback=5, max_interval=5`. Coorg: 20 components, 19 breaks/0 weak links, 193 geometry, 20 in 72–79 s; reasons `clip_start=1, max_interval=46, tracking_weak=30, flagged_fallback=32, overlap=127`. Reasons may co-occur.

## 20-10-26 23:15:43 IST — Model-independent preprocessing pipeline for continuous smartphone video

- **Status:** Adopted.
- **Decision:** Implement preprocessing as the package `src/preprocessing`. It turns one or more smartphone clips into a model-independent package: full-resolution selected frames, a manifest with provenance and a candidate pool, a covisibility graph and a gap report. The default capture is one long continuous video with deliberate revisits; short patch clips are allowed. Model-specific resizing, reference-frame choice and memory-bounded grouping belong to the reconstruction stage, not preprocessing. Thresholds live in `configs/preprocessing.toml`. The per-frame score table is CSV, not Parquet.
- **Rationale and trade-offs:**
  - The reconstruction model is not decided (VGGT, FastVGGT, VGGT-Ω, VGGT-Long or others), so a model-specific handoff would couple the stages.
  - Continuous capture links rooms through sequential tracking. It cannot redo a single section, so patch clips and revisit detection recover weak connections.
  - Scoring every frame at low resolution is cheap on CPU and fits the 4 GB development laptop. Exporting only the selected frames avoids writing tens of thousands of full-resolution images.
  - Five passes (probe plus four processing passes), each writing a resumable artifact, cost extra decoding but allow on-site re-runs after a threshold change without starting over.
  - Measurements are flags, not verdicts. Blur, exposure, timestamp gaps and weak links are reported for review and do not cause automatic recapture.
- **Affected components:** `src/preprocessing/`, `src/__init__.py`, `configs/preprocessing.toml`, `preprocess_capture.py` (repository root), `tests/preprocessing/`, `pyproject.toml`, `docs/ARCHITECTURE.md` (section 2).
- **Implementation steps:**
  1. **Contracts and config.** `contracts.py` defines frozen dataclasses with JSON round trips and a `schema_version`. `config.py` defines `PreprocessConfig`, loaded from `configs/preprocessing.toml`.
  2. **Probe (pass 0).** `probe.py` reads rotation, time base, frame-rate variation, colour transfer/HDR, size and duration with PyAV and writes `clips.json`.
  3. **Score (pass 1).** `decode.py` yields frames with presentation timestamps at a 640 px long side in greyscale, with rotation applied. `scoring.py` computes:
     - tiled Laplacian sharpness: p75 of the variances on a 4×4 grid;
     - relative sharpness against both a short-window and a long-window reference;
     - dark and clipped fractions, luminance mean and spread, and exposure jumps;
     - timestamp gaps (`Δt` relative to the median) and cut detection from histogram correlation;
     - consecutive-frame KLT motion and survival from `tracking.py`. Unreliable tracking is recorded as unknown motion, not low motion.

     It flags low-quality intervals: severe blur can be flagged without high measured motion, and high motion is supporting evidence. Writes `all_frames.csv`.
  4. **Select (pass 2).** `selection.py` tracks KLT features from the last geometry keyframe:
     - Overlap = surviving cells ÷ cells occupied at the keyframe; initial coverage is recorded separately.
     - A keyframe is triggered by overlap below τ, or by a maximum time interval. Robust track displacement is recorded as a viewpoint-change proxy; homography residual is optional and labelled as such, not "parallax".
     - The keyframe is the sharpest unflagged frame in a short buffer. When an earlier buffered frame is chosen, tracking restarts there and replays the buffer.
     - A denser appearance set is selected with a higher τ, each frame linked to its geometry keyframe.
     - All candidates are kept as IDs in the candidate pool.

     Writes `selection.json`.
  5. **Link (pass 3).** `revisits.py` retrieves non-neighbour candidates with a global descriptor (DINOv2-based), matches local features (ALIKED or DISK with LightGlue; not SuperPoint, whose weights are non-commercial) and verifies with RANSAC on the fundamental matrix. Accepted links need spatially distributed inliers, and the temporal exclusion window is configurable. `graph.py` builds sequential and revisit edges with measured weights; consecutive keyframes do not get an edge without measured tracking support. `report.py` lists low-quality intervals, tracking breaks, weak links, unlinked clips and the loop links found. Revisits are an optional dependency extra; when they are not installed, the report states "revisits: not run".
  6. **Export (pass 4).** One sequential full-resolution decode. `storage.py` resolves output paths and source-frame retrieval. `export.py` writes `geometry_frames/` (PNG), `appearance_frames/` (JPEG q95, configurable), `manifest.json`, `covisibility.json` and `report.json`. `visualize.py` writes `debug/` plots: a sharpness/overlap timeline with selections and flags, keyframe contact sheets and revisit match images.
  7. **Public API.** `preprocess_capture(clips, output_dir, config=None, resume=True) -> PreprocessResult` and `load_result(output_dir)` are re-exported from `src/preprocessing/__init__.py` and `src/__init__.py`. The root script is `preprocess_capture.py`; the `[project.scripts]` entry point is `namma-preprocess`.
- **Completion criteria:**
  - Each pass writes its artifact and resumes when its config is unchanged.
  - The full pipeline runs on the clips in `videos/` and produces every output listed above.
  - Tests use real implementations on a short real clip cut from a capture.
  - The `debug/` outputs of a real capture have been inspected visually and the findings recorded in `docs/STATE.md`.

## 20-10-26 13:18:20 IST — Establish documentation scaffolding

- **Status:** Adopted.
- **Decision:** Keep the README to two lines, explain the architecture's components, and maintain separate state and plan documents. Defer detailed module contracts.
- **Rationale and trade-offs:** Provide a small, usable documentation structure for preprocessing work without prematurely specifying implementation interfaces.
- **Affected components:** `README.md`, `docs/ARCHITECTURE.md`, `docs/STATE.md`, `docs/PLANS.md`, and the README source notes in `docs/GOALS.md`.
- **Implementation steps:** Add the summary and documentation links; explain the existing diagram; initialise the state document with the requested sections and IST timestamps; keep affected references consistent.
- **Completion criteria:** The README has exactly two lines; documentation links resolve; the diagram's components are explained; state tracking includes Latest Decision and Current Progress instead of Latest Verification; timestamps follow the declared format.
