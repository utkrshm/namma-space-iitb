# Dense full-resolution structure from motion (`src/sfm/`)

Photoreal Gaussian splatting needs many views with cameras accurate to about a pixel at
the training resolution. The VGGT stages solve 161 geometry frames on a 518-pixel grid.
That is about 0.75 frames per second, with cameras good to about 4 px at 1080×1920. This
stage solves its own cameras for about 4 sharp frames per second at full resolution. It
uses VGGT only as an independent check. The decision record is in
[PLANS.md](PLANS.md); the contract is in `src/sfm/contracts.py` and
[ARCHITECTURE.md](ARCHITECTURE.md#dense-full-resolution-sfm-srcsfm).

## Setup

```sh
uv sync --extra alignment --extra reconstruction --extra viewer --extra revisits --extra floor \
  --extra splatting-viewer --extra sfm --group dev
```

`pycolmap` (CPU wheel, with GLOMAP's global mapper) does the mapping. ALIKED/LightGlue and
DINOv2 run on CUDA and reuse the cached weights of the earlier stages.

## Run on Coorg (recommended three-pass recipe)

Revisit (loop) pairs are essential: without them, every return to a room is offset from the
first pass. False loops are what break a global mapper, though, so the loops are checked
against progressively better trajectories:

```sh
P=outputs/coorg_bnb/preprocessing
V=outputs/coorg_bnb/loop_closure/method_b_from_baseline/cameras.npz
export OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1

# 1. Frames, features, all pairs, sequential-only reference (no loops).
uv run --no-sync python build_sfm.py $P --output-dir outputs/coorg_bnb/sfm_seq \
  --retrieval-top-k 0 --vggt-cameras $V
# 2. Loops whose essential-matrix rotation agrees with pass 1 within 10°.
uv run --no-sync python build_sfm.py $P --output-dir outputs/coorg_bnb/sfm_loops \
  --loop-reference outputs/coorg_bnb/sfm_seq/sparse/0 --vggt-cameras $V
# 3. Loops that also fit pass 2's epipolar geometry within 4 px.
uv run --no-sync python build_sfm.py $P --output-dir outputs/coorg_bnb/sfm_loops2 \
  --loop-reference outputs/coorg_bnb/sfm_loops/sparse/0 --loop-max-epipolar-px 4 --vggt-cameras $V
```

Each directory caches its own steps. On Coorg, the later passes reuse pass 1's frames,
features and matches through symlinks (`images`, `features`, `descriptors.npy`,
`matches.npz`, `pairs.json`, `frames.json`). Only verification and mapping rerun, in about
25 minutes on the CPU. `undistorted/` is the input for
[`train_sfm_splat.py`](SPLATTING.md).

## Adding frames to a solved model (`densify_sfm.py`)

```sh
uv run --no-sync python densify_sfm.py outputs/coorg_bnb/sfm_loops2 outputs/coorg_bnb/preprocessing \
  --output-dir outputs/coorg_bnb/sfm_dense8 --fps 8
```

Frames selected at the higher rate that are not yet solved are matched to their 4 nearest
registered frames. Keypoints that observe a 3D point give 2D–3D correspondences, and each
pose is solved with pycolmap's robust PnP and refinement. The base cameras and points are not
changed. The output is a `src.sfm` hand-off directory: `frames.json`,
`undistorted/{images,sparse}` and `densify.json`. On Coorg, 834/842 frames were registered
with a median of 1814 inliers. The previous splat renders these unseen frames as well as its
training frames (28.87 vs 28.89 dB colour-corrected), so the poses are consistent.

## Procedure

1. **Frames:** the sharpest unflagged frame in every 0.25 s window, plus all 161 geometry
   frames. They are exported upright at 1080×1920 with their `<clip_id>:<pts>` IDs.
   A frame-threaded decoder converts only the wanted frames. It is pixel-identical to the
   preprocessing decoder (tested) and decodes the whole video in about 2 minutes instead
   of about 40.
2. **Features and pairs:** ALIKED (4096 keypoints at 1600 px). Sequential pairs are the
   ±8 neighbouring frames. Retrieval pairs are the DINOv2 top-8, at least 4 s apart, with
   cosine ≥ 0.45, kept only with ≥60 fundamental inliers and a planar fraction ≤ 0.7.
3. **Loop check** (`src/sfm/loops.py`): the pair's own rotation (essential matrix) must
   agree with the reference trajectory, and optionally its matches must fit the
   reference's epipolar geometry.
4. **Mapping:** COLMAP verification, then the GLOMAP global mapper, with one shared
   SIMPLE_RADIAL camera. The result is undistorted to PINHOLE.
5. **Report:** `report.json` records:
   - registration and reprojection error;
   - `continuity`: impossible jumps or turns between consecutive frames;
   - `loop_consistency`: epipolar fit of the loop pairs, per 20 s;
   - `vggt_comparison`: VGGT cameras after a Sim(3).

   `inspection/topdown.png` plots sparse points and the trajectory.

## Coorg results (27–28 Oct 2026)

927 frames (161 geometry). 7,380 sequential pairs and 2,630 accepted retrieval pairs.

| Solve | Registered | Reproj. | Trajectory | Loop fit (true loops, median / >3 px) |
| --- | ---: | ---: | --- | --- |
| GLOMAP, all pairs | 927 | 1.32 px | folded: jumps at 13–74 s, 165–177° flips at 74 s and 181 s | – |
| Sequential only (pass 1) | 927 | 1.24 px | continuous, no loop closure | 6.6 px median over all retrieval pairs |
| + rotation-checked loops (pass 2) | 927 | 1.33 px | 13 small breaks (56–61 s cluster) | 0.97 px / 3.4 % |
| + epipolar-checked loops (pass 3) | 927 | 1.31 px | 3 small breaks | 0.97 px / 3.4 % (60–80 s pairs removed) |

The table's column headings are short; their full meanings:

- **Reproj.** is the mean reprojection error.
- **Loop fit** is the epipolar error of the true loop pairs, given as the median and the
  share over 3 px.

Findings:

- **False loops come in two kinds.**
  - *Rotated:* about 10–20% of retrieval pairs imply a relative rotation 100–180° away from
    the sequential trajectory. These are the repeated grilles and symmetric doorways; they
    folded the first GLOMAP model. Rotation agreement separates them cleanly (median
    1.2°, p90 109°).
  - *Translated:* the identical side-by-side window grilles at the entrance. Their
    rotation agrees, but their position is wrong; 60–70 s pairs fit the model only to
    83–91 px. Pass 3's epipolar gate removes them. In pass 2 they had bent the 56–61 s
    stretch, which is also where splat training fits worst (37–71 s).
- **Loop closure matters for splatting.** A splat trained on the sequential-only cameras
  collapses at full resolution in revisited rooms (see [SPLATTING.md](SPLATTING.md)).
- **VGGT comparison.** VGGT cameras agree with pass 2 to 0.26 of the trajectory radius
  (camera centres), with a median rotation of 10.7°. VGGT's own dining-wing drift (≈30°,
  [LOOP_CLOSURE.md](LOOP_CLOSURE.md)) accounts for most of the tail.
