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
