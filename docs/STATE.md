# Project state

Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Keep this document as the current snapshot; retain decision history in [PLANS.md](PLANS.md).

**Last updated:** `21-10-26 01:56:29 IST`

## Current Focus

User review corrections applied: supported-corner tracking, low-texture/blur
separation, and real CUDA revisit inference on both captures. Review the accepted
repeated-pattern links and remaining single-frame blur ambiguity.

## Stage Status

Use: Planned, In progress, Implemented (awaiting verification), Verified, or Blocked.

| Stage | Status |
| --- | --- |
| Documentation scaffolding | Verified |
| Preprocessing | Verified on supplied captures, with documented blur/revisit ambiguities |
| Reconstruction, scene understanding and navigation | Planned |

## Latest Decision

- **Timestamp:** `21-10-26 00:51:44 IST`
- **Decision:** Use frame-global corner strength floors and spatial track support; separate low texture from blur evidence.
- **Reference:** “Texture-aware tracking and blur evidence” in [PLANS.md](PLANS.md).

## Current Progress

- **Recorded `21-10-26 01:56:29 IST`:** Review corrections are implemented and both
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
- `HF_HUB_OFFLINE=1 uv run --extra revisits pytest -q -ra`: **25 passed, 1 xfailed,
  0 skipped, 1 upstream warning in 42.87 s**. The expected failure transparently
  records the ambiguous pure per-frame blur label at 49.5 s; the full pipeline
  flags it with temporal context. Real revisit and cross-building negative tests ran.
- `uv run --extra revisits ruff check src tests/preprocessing preprocess_capture.py`: passed.
- New fixtures include the metadata-stripped 480-frame ceiling cut at 72–80 s,
  and eight upright analysis PNGs with source timestamps listed in their README.
  Tests now enforce exact 256-frame decoding, positive ~66.6 ms timing flags and
  spatial rejection of >=50 clustered real correspondences versus an equal spread set.

### Current capture results

| Capture | CSV rows | Geometry | Appearance | Sequential | Revisit | Components |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `room_m_block` | 2,426 | 36 | 101 | 35 | 0 | 1 |
| `coorg_bnb` | 12,907 | 161 | 399 | 160 | 18 | 1 |

- Both sequential graphs are connected before adding revisit edges. Coorg tracking
  breaks dropped **19 → 0**, and its 72–79 s geometry count **20 → 10**. Room's
  8.8–14 s geometry count dropped **15 → 5**. Neither capture has weak-link items.
- All four reviewed false-blur observations now receive informational low texture.
  Coorg has four low-quality intervals and seven separate low-texture info intervals;
  Room has no low-quality intervals. The 5 s sharp observation also gets low texture
  information, not a quality failure. The 25 s and 140 s examples remain unflagged.
- Real DINOv2 retrieval found zero Room candidates above the configured similarity
  threshold, and 28 Coorg candidates; local geometry verification accepted 18.
  Every accepted pair is outside the 10 s temporal exclusion and has a debug image.

### Inspection and measured timings

- Inspected the new Room contact sheet and all 18 accepted Coorg pairs in three
  six-pair sheets. Door/key/sconce and luggage/table context support five pairs;
  thirteen window-bank pairs are dominated by repeated grilles and need human or
  downstream geometric review of motif correspondence. Do not equate acceptance
  with ground-truth loop closure. Pair IDs and image paths are in the review document.
- The user-labelled PNGs are upright and preserve the source observations. The
  49.5 s door cannot be reliably separated from sharp doors using the chosen
  per-frame re-blur measure at 640 px; this is explicitly documented, not overfit.

| Capture | Probe | Score | Select | Link (CUDA) | Export | Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Room | 0.13 s | 55.51 s | 41.27 s | 37.91 s | 47.91 s | 182.73 s |
| Coorg | 0.50 s | 294.25 s | 210.82 s | 181.73 s | 242.99 s | 930.29 s |

## Blockers

- No execution or dependency blockers remain for the supplied SDR videos.
- Per-frame blur classification is ambiguous for the labelled 49.5 s door at
  analysis resolution; the full pipeline uses context. Repeated grilles can also
  produce ambiguous geometrically accepted revisit correspondences.
- HDR conversion remains outside the implementation; HDR inputs fail explicitly.

## Next Actions

1. User review of all accepted revisit match images, especially repeated window grilles.
2. Consider higher-resolution or richer blur evidence if a context-free decision is required.
3. Integrate reconstruction through `load_result(output_dir)` and use global geometry
   to evaluate tentative loop links. Preserve source and candidate provenance.
