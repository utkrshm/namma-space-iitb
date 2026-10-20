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
