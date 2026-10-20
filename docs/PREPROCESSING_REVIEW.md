# Preprocessing review corrections

Recorded `21-10-26 01:56:29 IST` on `feat/preprocessing`. Baseline is the previous
implementation reviewed by the user; after results use schema 2 and real revisits.

## Tracking and selection

| Metric | Room before | Room after | Coorg before | Coorg after |
| --- | ---: | ---: | ---: | ---: |
| Components (sequential graph alone) | 1 | 1 | 20 | 1 |
| `tracking_break` | 0 | 0 | 19 | 0 |
| `weak_link` | 0 | 0 | 0 | 0 |
| Geometry total | 49 | 36 | 193 | 161 |
| Geometry in review interval | 15 | 5 | 20 | 10 |
| Appearance total | 118 | 101 | 497 | 399 |
| Low-quality intervals | 1 | 0 | 9 | 4 |

Review intervals are Room **8.8–14 s** and Coorg **72–79 s**, inclusive.
Revisit edges are not responsible for the repaired sequential connectivity.

Geometry trigger-reason histogram from `selection.json` (reasons may co-occur):

| Reason | Room before | Room after | Coorg before | Coorg after |
| --- | ---: | ---: | ---: | ---: |
| `clip_start` | 1 | 1 | 1 | 1 |
| `overlap` | 43 | 30 | 127 | 105 |
| `max_interval` | 5 | 5 | 46 | 55 |
| `tracking_weak` | 0 | 0 | 30 | 1 |
| `flagged_fallback` | 5 | 0 | 32 | 4 |

The detector uses `cornerMinEigenVal >= max(1e-5, 0.01 * frame_max)` before
per-cell corner selection. Blank unsupported cells do not enter the denominator.
At least eight forward/backward-checked tracks across three of sixteen cells are
required, initially and currently. Survival is still recorded but never gates
reliability. Low reliability alone no longer triggers a new keyframe; overlap,
maximum interval and cuts trigger selections. These are continuity measurements,
not assertions of 3D reconstruction quality.

## Blur and texture

| Requested time | User label | Full-pipeline annotation |
| ---: | --- | --- |
| 5 s | Sharp | Informational `low_texture`, no quality failure |
| 12.5 s | Sharp, low texture | `low_texture`, no quality failure |
| 25 s | Sharp | No flags |
| 49.5 s | Motion blur | `suspected_blur`, with temporal corroboration |
| 103.5 s | Sharp, low texture | `low_texture`, no quality failure |
| 140 s | Sharp | No flags |
| 172 s | Sharp, low texture | `low_texture`, no quality failure |
| 204 s | Sharp, low texture | `low_texture`, no quality failure |

The low-texture gate requires 5% strong Sobel edge pixels at the fixed 640-pixel
analysis scale. Re-blur loss measures additional gradient-energy loss on those
pixels under sigma=1 Gaussian blur. A loss below 0.2 is strong evidence; ambiguous
loss below 0.35 additionally needs the temporal relative-sharpness signal. Flow
is never a prerequisite. Low texture cannot be promoted to suspected blur by a
textured neighboring scene and has its own info-level intervals (seven in Coorg).

**Remaining ambiguity:** at 640 px, the 49.5 s observation has re-blur loss about
0.31, overlapping sharp doors. Its pure per-frame blur assertion remains an explicit
strict expected failure. The pipeline still flags it with context. The sharp 5 s
observation also receives informational low texture; this is not a blur or rejection
label. No time-specific production rules or narrow eight-frame threshold fit were added.

## Real-model execution and tests

`uv sync --extra revisits` retained the dev group. Cached Hugging Face DINOv2 and
Torch ALIKED/LightGlue weights ran on **CUDA, RTX 3050 Laptop GPU, 4 GB**. No CPU
fallback was needed. Original clips still produce exactly **2,426 / 12,907** rows.

```text
HF_HUB_OFFLINE=1 uv run --extra revisits pytest -q -ra
25 passed, 1 xfailed, 1 warning in 42.87s
```

No tests skipped. The expected failure is the single-frame ambiguity above; the
warning is upstream Torch JIT deprecation. Ruff is clean. Regressions include the
real 72–80 s ceiling cut (480 frames, no location tags, no unsupported links, at most
18 geometry keyframes), eight user-labelled PNGs, >=50 clustered real KLT
correspondences versus the same number spread across the image, exact 256-frame
fixture decoding, positive 66.6 ms timing flags, real cross-building rejection,
revisit inference and temporal-exclusion checks.

Measured pass durations (seconds):

| Capture | Probe | Score | Select | Link | Export | Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Room | 0.13 | 55.51 | 41.27 | 37.91 | 47.91 | 182.73 |
| Coorg | 0.50 | 294.25 | 210.82 | 181.73 | 242.99 | 930.29 |

Room has zero eligible descriptor pairs at the configured 0.75 retrieval threshold;
Coorg has 28 verified candidates and **18 accepted** revisit edges. Both sequential
graphs are connected independently. Every accepted edge is outside the same-clip
10 s exclusion window and has a match image.

## Accepted revisit pairs

Room: none. Coorg paths below are relative to
`outputs/coorg_bnb/preprocessing/`. Times are seconds relative to stream start.
All 18 images were inspected in three six-pair review sheets. Pairs 1–2 have the
same door, key and sconce context; 16–18 have shared luggage/table context. Pairs
3–15 rely heavily on repeated window grilles: the shared window-bank context is
recognizable, but motif-level correspondence ambiguity remains for human/downstream
review. Geometric acceptance is not ground-truth validation of a loop closure.

| # | Frame A | Time A | Frame B | Time B | Match image |
| --- | --- | ---: | --- | ---: | --- |
| 1 | `coorg_bnb:701592` | 7.795467 | `coorg_bnb:7302316` | 81.136844 | `debug/revisit_coorg_bnb_701592__coorg_bnb_7302316.jpg` |
| 2 | `coorg_bnb:863504` | 9.594489 | `coorg_bnb:7302316` | 81.136844 | `debug/revisit_coorg_bnb_863504__coorg_bnb_7302316.jpg` |
| 3 | `coorg_bnb:1421182` | 15.790911 | `coorg_bnb:4956159` | 55.068433 | `debug/revisit_coorg_bnb_1421182__coorg_bnb_4956159.jpg` |
| 4 | `coorg_bnb:1421182` | 15.790911 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_1421182__coorg_bnb_5011622.jpg` |
| 5 | `coorg_bnb:1575589` | 17.506544 | `coorg_bnb:4956159` | 55.068433 | `debug/revisit_coorg_bnb_1575589__coorg_bnb_4956159.jpg` |
| 6 | `coorg_bnb:1575589` | 17.506544 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_1575589__coorg_bnb_5011622.jpg` |
| 7 | `coorg_bnb:1737501` | 19.305567 | `coorg_bnb:4956159` | 55.068433 | `debug/revisit_coorg_bnb_1737501__coorg_bnb_4956159.jpg` |
| 8 | `coorg_bnb:1737501` | 19.305567 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_1737501__coorg_bnb_5011622.jpg` |
| 9 | `coorg_bnb:2041828` | 22.686978 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_2041828__coorg_bnb_5011622.jpg` |
| 10 | `coorg_bnb:2199233` | 24.435922 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_2199233__coorg_bnb_5011622.jpg` |
| 11 | `coorg_bnb:2352151` | 26.135011 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_2352151__coorg_bnb_5011622.jpg` |
| 12 | `coorg_bnb:2521548` | 28.017200 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_2521548__coorg_bnb_5011622.jpg` |
| 13 | `coorg_bnb:2621991` | 29.133233 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_2621991__coorg_bnb_5011622.jpg` |
| 14 | `coorg_bnb:2666971` | 29.633011 | `coorg_bnb:5011622` | 55.684689 | `debug/revisit_coorg_bnb_2666971__coorg_bnb_5011622.jpg` |
| 15 | `coorg_bnb:2666971` | 29.633011 | `coorg_bnb:5056602` | 56.184467 | `debug/revisit_coorg_bnb_2666971__coorg_bnb_5056602.jpg` |
| 16 | `coorg_bnb:8239281` | 91.547567 | `coorg_bnb:9802883` | 108.920922 | `debug/revisit_coorg_bnb_8239281__coorg_bnb_9802883.jpg` |
| 17 | `coorg_bnb:8413175` | 93.479722 | `coorg_bnb:9446091` | 104.956567 | `debug/revisit_coorg_bnb_8413175__coorg_bnb_9446091.jpg` |
| 18 | `coorg_bnb:8413175` | 93.479722 | `coorg_bnb:9622983` | 106.922033 | `debug/revisit_coorg_bnb_8413175__coorg_bnb_9622983.jpg` |
