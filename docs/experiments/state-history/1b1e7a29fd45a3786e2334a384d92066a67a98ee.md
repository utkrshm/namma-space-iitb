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
