# Loop closure

> Historical experiment guide. The source and commands described here belong to the original branches. Use the root README for the current submission; this branch retains these records as documentation only.
> Earlier experimental implementation, preserved under `experiments/`. The R1 runtime is documented in [ARCHITECTURE.md](ARCHITECTURE.md).
This stage fixes global camera drift in a section-aligned reconstruction. On Coorg, the
dining sections (≈180–215 s) were rotated ≈30° against the living room and passed
through the TV wall; plain walls appeared twice. The stage reads an existing
`section_alignment/` directory **read-only** and writes everything to a new directory.
The decision record is in [PLANS.md](PLANS.md); the contracts are in
[ARCHITECTURE.md](ARCHITECTURE.md#loop-closure-srcloop_closure) and
`experiments/loop_closure/contracts.py`.

## Setup

```sh
uv sync --extra experiments --group dev
```

It needs the cached Hugging Face weights `facebook/VGGT-1B` and `facebook/dinov2-base`,
plus the LightGlue ALIKED weights (all already used by earlier stages). Pass
`--allow-download` on a new machine. VGGT loop chunks need CUDA; ≤26 views at 518 px
fit on the 16 GB A4000.

## Run on the Coorg video

The section-alignment export must exist first (see
[RECONSTRUCTION.md](RECONSTRUCTION.md#high-resolution-section-alignment)):

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_HUB_OFFLINE=1 \
  uv run --no-sync python -m experiments.scripts.close_loops outputs/coorg_bnb/section_alignment \
  --output-dir outputs/coorg_bnb/loop_closure
```

`uv run --no-sync python -m experiments.scripts.close_loops ...` is equivalent. `--methods a` or `--methods b`
runs only one method. Matches, descriptors and loop-chunk predictions are cached and
reused on reruns with identical settings.

View the results:

```sh
uv run --no-sync python -m experiments.scripts.view_point_cloud outputs/coorg_bnb/loop_closure/method_a/point_cloud.ply
uv run --no-sync python -m experiments.scripts.view_point_cloud outputs/coorg_bnb/loop_closure/method_b_from_method_a/point_cloud.ply
```

Compare `outputs/coorg_bnb/loop_closure/inspection/topdown_comparison.png` (wall-height
slice per method, coloured by source time) and the `camera_slice_*.png` sheets (TV wall
at 117 s, white wall at 171 s and others). Numbers are in `comparison.json`.

## Procedure

1. **Detection.** DINOv2 CLS descriptors are computed for every geometry frame. Each
   frame contributes its top-8 pairs more than 20 s away with cosine ≥ 0.40; a high
   threshold only finds the repeated entrance grilles. Each pair is verified with
   ALIKED/LightGlue on the 518-pixel model grid (valid crop) using fundamental-matrix
   RANSAC (≥40 inliers, ≥25% cell coverage on both images) and the existing
   repeated-structure guard. A pair rejected *only* by that guard is accepted when its
   inliers are clearly non-planar (see results).
2. **Roles.** The strongest verified pair per section pair (at most 2 per pair, 12 in
   total, ≥6 s apart) becomes a method-A chunk centre. Of the remaining pairs, every
   third is **held out** from every method; the rest are bundle-adjustment inputs.
3. **Method A.** VGGT jointly infers frames i±6 and j±6. The chunk is registered to the
   section owning each side with `robust_sim3` on identical model pixels. Whole interior
   frames are held out under the same absolute gate as adjacent overlaps (pooled
   normalized median ≤ 0.10). The loop pair's own feature matches must agree inside
   the chunk (normalized median ≤ 0.10). The edge (section B → section A) joins the 7
   adjacent edges in `optimize_section_graph`. A degraded adjacent check after
   optimization is recorded in `graph.json` rather than raised.
4. **Method B.** Sequential pairs (offsets 1–3) plus training and chunk-centre loop pairs
   form tracks (keypoints that conflict within a frame are dropped). `scipy`
   `least_squares` (TRF, Huber, sparse Jacobian) refines a rotation, centre and log
   depth-scale per frame (frame 0 fixed) and every track point. It uses a log-depth
   prior (σ = 0.05) and weak scale/rotation priors. Two outlier rounds follow at 8 px
   and 4 px.
5. **Comparison.** Every solution is scored on the same pixels: matched points are
   unprojected from each frame's own depth and camera, and their distance is divided by
   the mean camera distance.

6. **Arm M (optional, `--methods m`).** Method B plus a per-frame square-walls yaw prior
   (`manhattan.py`): wall normals from each frame's own depth, compared modulo 90° with
   the dominant house direction (σ = 3°, weighted by how clearly the frame shows right
   angles).

## Coorg results (`26-10-26 20:25 IST`, A4000)

Run on `outputs/coorg_bnb/section_alignment` (161 frames, 8 sections), all arms.
The table shows normalized 3D disagreement of matched pixels (median / p90; lower is
better). Held-out loop pairs entered no method.

| Arm | Held-out loops (6 pairs) | Used loops (18 pairs) | Sequential (280 pairs) |
| --- | ---: | ---: | ---: |
| baseline (section graph) | 0.0458 / 0.484 | 0.0309 / 1.30 | 0.0166 / 0.0556 |
| A — VGGT loop chunks | 0.0494 / 0.536 | 0.0337 / 1.36 | 0.0168 / 0.0562 |
| B from baseline | **0.0413** / 0.484 | 0.0310 / 1.29 | **0.0107** / **0.0458** |
| B from A | 0.0421 / 0.535 | 0.0313 / 1.35 | 0.0110 / 0.0470 |
| M from A (square walls) | 0.0422 / 0.535 | 0.0313 / 1.35 | 0.0110 / 0.0471 |

- **Detection.** There were 494 DINOv2 candidates and 24 verified pairs (5 chunk
  centres, 13 training, 6 held out). The repeated-structure guard alone rejected every
  real wide-baseline living-room loop: on Coorg its descriptor ratio cannot separate
  true loops (median ≈1.1–1.3) from grille false positives (≈1.4). The non-planar
  override accepts pairs only when one homography explains ≤50% of fundamental inliers
  and two homographies explain ≤75%. Measured planar fractions were 0.30–0.38 for true
  bench/pillar loops and 0.79–0.98 for grille and floor-inlay false positives.
- **No dining loop exists.** An exhaustive scan of frames 125–160 (178–214 s) against
  55–99 (77–146 s) found no pair with even raw epipolar support (best: 88 raw matches,
  0 inliers). Indoor LoFTR also found none. Nothing in this capture can constrain the
  dining wing back to the living room.
- **Method A.** Three of five loop edges were accepted: 0↔2 (5↔60), 2↔3 (64↔81) and
  2↔4 (76↔106). Held-out side medians were 0.007–0.048 and cross-side agreement
  0.007–0.014. The two entrance-window chunks (11↔45, 19↔45) were rejected: section 1
  held-out medians were 0.47/0.11 and cross-side agreement 1.5. The graph objective fell
  0.0149 → 0.0047, and every adjacent held-out check still passes. A improves the
  held-out entrance loop (4↔60: 0.072 → 0.037) but slightly worsens the others.
  A single Sim(3) per section cannot remove error *inside* a section.
- **Method B.** 18.8k tracks and 50k observations; about 37k inliers remain after the
  30/8/4 px rounds. Inlier reprojection fell 2.53 → 1.07 px (median) and 8.96 → 2.63 px
  (p90). Camera changes: median 0.23°, max 1.5°. Sequential disagreement fell 35%, and
  held-out loops improved 10%. This is the clearest measurable gain.
- **Square walls (M) had no effect, by design of the evidence.** Frames 0–112 (0–160 s)
  lie within about ±10° of the house axes. From frame 113 (162 s) every frame sits at
  +31–35°, and 407 feature matches link 160.2 s to 162.2 s. Image evidence supports
  that turn, so the prior is outweighed. Either the corridor/dining wing really is
  angled, or VGGT's wall slant is wrong over that whole stretch; this capture cannot
  tell which.
- **Visual check of the actual PLYs** (`inspection/topdown_comparison.png`,
  `camera_slice_*.png`): the arms are visually almost identical. The doubled white wall
  at 171 s (≈170 s vs ≈200 s copies plus a far ghost), the dining geometry crossing the
  TV wall and the multi-layer pillar at 158 s all persist. Those observations have no
  verified image link to each other, and part of the layering is per-frame VGGT depth
  error, which camera-only corrections cannot remove.

**Recommendation:** adopt **B from baseline** as the camera solution (best on every
metric, small and well-supported corrections). The remaining wall/TV/curtain layering
needs either a recapture that walks back from the dining room to the living room (an
SOP loop), or depth-level fixes: plane priors on plain walls and masking TV, glass and
curtain regions before fusion.
