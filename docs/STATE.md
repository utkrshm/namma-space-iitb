# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `21-10-26 00:32:17 IST`

## Current Focus

Preprocessing core implemented and exercised on both supplied captures; optional
revisit matching awaits installation and real-model verification.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Implemented (awaiting verification of optional revisits) |
| Reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `20-10-26 23:15:43 IST`
- **Decision:** Adopt model-independent preprocessing for continuous video, with versioned records, CSV scores, resumable passes and optional revisit linking.
- **Reference:** “Model-independent preprocessing pipeline for continuous smartphone video” in [PLANS.md](PLANS.md).

## Current Progress

- **Recorded `21-10-26 00:32:17 IST`:** Implemented `src/preprocessing`, public re-exports,
  `configs/preprocessing.toml`, the root demonstration script and `namma-preprocess` CLI.
  The package includes real KLT selection/replay, candidate provenance, supported-only
  sequential edges, optional DINOv2 → ALIKED/LightGlue → MAGSAC verification, frame
  export, debug plots, `load_result` and hash-validated resume. See [usage](PREPROCESSING.md).
- Real five-second fixture: `tests/fixtures/room_short.mp4`, 316,739 bytes,
  640×360 coded / 360×640 upright. VFR and −90° rotation retained, no location tags.
- `uv run pytest -q -rs`: **14 passed, 1 skipped in 19.96 s**. The skipped test is
  real optional revisit inference because its dependency extra is not installed.
  Tests include real blur, sustained blur without reliable flow, frame timestamps,
  rotation/color versus FFmpeg, JSON/CSV round trips, buffered-reference replay,
  cut boundaries, threshold direction, end-to-end export and resume/corruption recovery.
- `uv run ruff check src tests/preprocessing preprocess_capture.py`: passed.
- Original Room M Block first-frame comparison against FFmpeg autorotation: both
  1080×1920 upright RGB, full range; mean, p99 and maximum pixel error all **0**.
  First source PTS is **3834**, relative time **0 s**. Sources were not modified.

### Full-capture results (CPU, two threads per pipeline)

Final passes were run concurrently across the two captures; timings are measured,
not GPU estimates. Export includes PNG/JPEG encoding and debug plots. Hash validation
and initial input hashing add overhead beyond the action durations below.

| Capture | CSV rows | Geometry | Appearance | Sequential edges | Components |
| --- | ---: | ---: | ---: | ---: | ---: |
| `room_m_block` | 2,426 | 49 | 118 | 48 | 1 |
| `coorg_bnb` | 12,907 | 193 | 497 | 173 | 20 |

| Capture | Probe | Score | Select | Link | Export | Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `room_m_block` | 0.14 s | 47.22 s | 46.00 s | <0.01 s | 53.57 s | 146.93 s |
| `coorg_bnb` | 0.57 s | 245.01 s | 224.84 s | 0.01 s | 253.59 s | 724.02 s |

- CSV counts exactly match supplied reference counts. Median Δt is about 16.66 ms
  for both; there are respectively **77 / 25** steps over 1.5× median and maximum
  steps **133.28 / 50.02 ms**. Ordinary 33 ms steps are not flagged. Only Room M
  Block has ≥67 ms info flags (**11**); Coorg has **0**.
- Revisit status for both: **not run — revisits extra not installed**. Zero computed
  revisit edges must not be read as evidence that the captures contain no revisits.

### Visual inspection

- Opened both timeline PNGs; Room M Block keyframe sheet page 0; Coorg keyframe
  sheets pages 0 and 2; original-resolution exported Room M Block geometry frame
  `room_m_block_3834.png`; and Coorg appearance frame `coorg_bnb_0.jpg`.
- **Room M Block:** display orientation and colors are correct. Selection becomes
  denser during turns around 8–14 s and preserves the room scan. One low-quality
  interval is reported at **10.66–11.78 s**. That section includes a plain wall and
  soft detail, so the suspected-blur flag is not a conclusive motion-blur diagnosis.
  Later camera roll in the original footage remains visible; no stabilization is applied.
- **Coorg:** orientation and sampled export colors are correct. Selected views preserve
  the entry/window scan, living area and doorway transitions, with denser selections
  during fast turns and around 158–170 s. There are **9** low-quality intervals and
  **19** unsupported sequential links, yielding **20** graph components without
  revisit recovery. Plain-wall/low-texture scenes also trigger suspected-blur flags.
  People remain in the images; dynamic masking is not part of this implementation.
- No threshold tuning was justified as a clear correction from these contact sheets:
  flags remain review evidence, including texture confounds. User qualitative review
  and real revisit-match inspection are still pending.

## Blockers

- Optional `revisits` dependencies are declared and locked but not installed in the
  current environment. DINOv2/ALIKED/LightGlue inference, GPU memory use and accepted
  revisit images have not been verified. No fake implementation substitutes for them.
- HDR conversion is not implemented; HDR inputs fail explicitly before scoring.

## Next Actions

1. Install `uv sync --extra revisits`, run the real revisit test and rerun both captures.
2. Inspect true/false revisit matches and whether they recover Coorg's disconnected groups.
3. Complete the user's qualitative review and tune selection/quality thresholds only
   where the footage provides evidence for a change.
4. Integrate the independent reconstruction stage through `load_result(output_dir)`.
