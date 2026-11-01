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
