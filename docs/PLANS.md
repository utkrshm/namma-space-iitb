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

## 26-10-26 10:23:01 IST — Focus base reconstruction; defer refinement experiments

- **Status:** Adopted at user request.
- **Decision:** Keep current `feat/reconstruction` work focused on stabilizing base reconstruction. Defer per-view camera/depth refinement, recorded as a promising direction explored too early. Preserve the experiment on `fix/recon-mv-refinement` at commit `6c7d168`; do not copy its implementation into this branch.
- **Rationale and trade-offs:** The experimental branch improved measured correspondence and some living-wall appearance, but unsupported components and entrance layering remain unresolved. Revisit after base reconstruction is stabilized; exploratory results do not establish superiority or readiness for integration.
- **Affected components:** Reconstruction focus and status in `docs/STATE.md` and this plan; brief deferred-direction note in `docs/RECONSTRUCTION.md`. Experimental code and generated outputs remain on their existing branch/local ignored paths.
- **Implementation steps:** Preserve the experiment and its handoff at `git show fix/recon-mv-refinement:docs/REFINEMENT.md`; continue base reconstruction on this branch.
- **Completion criteria:** Current branch documentation distinguishes deferred experiments from implemented work, retains earlier progress history, and makes no unvalidated comparative claim.

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
