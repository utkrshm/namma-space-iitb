# Gaussian splatting from refined VGGT geometry

`src/splatting` fits 3D Gaussians to posed RGB images, using refined depth and
world-space point maps for initialization and depth supervision. `src/rendering`
provides the differentiable renderer used by training, offline export and viewing.

**Execution target:** your development server. This implementation was authored
without installing dependencies or executing project code on the authoring machine.
Runtime compatibility, integration with the remote VGGT files and visual quality
remain unverified.

## Splatting from dense SfM cameras (`train_sfm_splat.py`) — verified on Coorg

The recommended path for photoreal output. It uses the full-resolution cameras of
[`src/sfm`](SFM.md) instead of the 161 VGGT views at 294×518. The trainer is
`src/splatting/sfm_training.py`; it writes the same `scene.ply` + sidecar as the trainer
below, so `view_gaussian_splat.py` and `render_gaussian_splat.py` work unchanged.

```sh
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions_namma
export CUDA_HOME=$PWD/outputs/runtime/cuda-12.1 TORCH_CUDA_ARCH_LIST=8.6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# One model for the whole capture.
uv run --no-sync python train_sfm_splat.py outputs/coorg_bnb/sfm_loops2 \
  --config configs/coorg_sfm_splat_v6b.toml --output outputs/coorg_bnb/splat_runs/full

# Or blocks of one SfM model, merged by nearest-camera ownership.
for b in A_entrance B_living C_dining; do
  uv run --no-sync python train_sfm_splat.py outputs/coorg_bnb/sfm_loops2 \
    --config configs/coorg_blocks/$b.toml --output outputs/coorg_bnb/splat_runs/blocks/$b
done
uv run --no-sync python merge_splat_blocks.py outputs/coorg_bnb/sfm_loops2 \
  outputs/coorg_bnb/splat_runs/blocks/{A_entrance,B_living,C_dining} \
  --output-dir outputs/coorg_bnb/splat_runs/blocks/merged

# Diagnose where a splat fails (per 10 s of video, worst frames, Gaussian statistics).
uv run --no-sync python diagnose_splat.py <run>/checkpoint.pt outputs/coorg_bnb/sfm_loops2
# Web export: drop Gaussians below 0.02 opacity; writes scene.splat and scene_pruned.ply.
uv run --no-sync python export_web_splat.py <run>/scene.ply --output-dir <run>/web \
  --sfm-dir outputs/coorg_bnb/sfm_loops2
```

Start GPU jobs at least a minute apart. Two processes that compile gsplat's CUDA
extension at the same moment corrupt ninja's dependency log, and every later process
then fails with `Error building extension 'gsplat_cuda'`. In that case, point
`TORCH_EXTENSIONS_DIR` at a fresh directory.

### What the trainer does

- **Seeds:** SfM points. Each starting Gaussian's scale is the mean distance to its
  3 nearest neighbours.
- **Densification:** gsplat's MCMC strategy, with opacity and scale regularization. MCMC's
  position noise is `lr × covariance`, which has units of length³, and gsplat assumes a
  world of radius ≈ 1. The trainer divides the noise learning rate by the scene radius
  squared; otherwise it is about 40–120× too strong on an unnormalized scene.
- **Size limits:**
  - Gaussian extents are clamped to 10% of the scene extent.
  - On refinement steps, Gaussians whose screen radius exceeds 15% of the image (and
    64 px) are handed to MCMC relocation. This is 3DGS's screen-size pruning; it stops
    near-camera floaters.
- **Per-image corrections:**
  - A pose increment (6D rotation + translation, lr 1e-5) for each image.
  - An affine colour transform for each image, for exposure and white balance. The
    transforms are kept zero-mean, so the canonical colours that viewers render equal the
    average exposure.
- **Schedule:** coarse-to-fine resolution (×4 → ×2 → ×1 at steps 0/2000/6000), 30k steps.
- **Held-out views:** every 8th geometry frame of the whole capture, the same 20 frames
  as the VGGT trainer below, even inside a `time_range_s` block. Each view is reported
  three ways: raw PSNR, colour-fitted PSNR (`psnr_cc_db`), and PSNR on the 518 grid.
- **Resuming:** `--resume` continues from `checkpoint.pt` (Adam moments restart).

### Coorg findings (28 Oct 2026, A4000)

Held-out frames are utkarsh's 20 validation frames. The table's PSNR columns are on the
518 grid, the grid the VGGT splat was evaluated on.

| Model | Raw PSNR | Colour-corrected PSNR | SSIM |
| --- | ---: | ---: | ---: |
| VGGT splat (`section_alignment_60k`, 161 views at 294×518) | 17.07 | – | 0.59 |
| SfM splat, quarter resolution, 5k steps (cameras: sequential or loops) | 21.8–22.1 | 24.6–25.0 | 0.84–0.85 |
| SfM splat, full house, full resolution, 1.6M Gaussians (`full_loops_v2`) | 19.3 | 22.4 | 0.80 |
| + loop-gated cameras and screen-size pruning (`full_loops2_v6b`) | 19.5 | 22.3 | 0.80 |
| Blocks merged by nearest camera | – | 20.6* | – |
| Blocks merged by visibility (`blocks/merged_vis`) | – | 24.4* | – |
| Blocks merged by visibility + 6k-step joint fine-tune (`blocks/merged_finetune`) | 24.6 | 28.3 | 0.91 |
| + second 10k-step joint fine-tune (`blocks/merged_finetune2`) | 25.5 | 29.0 | 0.91 |
| + 8 frames/s (PnP-registered, `densify_sfm.py`) + depth prior 0.05 (`blocks/merged_dense8_depth`) | 25.6 | 28.3 | 0.91 |
| **Moved onto OPENCV-lens cameras (`transfer_splat.py`) + 12k + 8k-step fine-tunes (`opencv/finetune_2`, final)** | **26.3** | **28.6** | **0.90** |

\* Colour-corrected PSNR at half resolution (v6b scores 21.8 on that measure).

The final model is `outputs/coorg_bnb/splat_runs/opencv/finetune_2` (29 Oct 2026).

- *Cameras:* the OPENCV lens model removes most of the entrance floor step of the
  single-coefficient solve (see [SFM.md](SFM.md#lens-model-opencv-removes-most-of-the-entrance-floor-step-29-oct-2026)).
- *How it was trained:* the previous final model was moved onto these cameras by
  point-based local similarities, then fine-tuned for 12k steps (position lr 1.6e-4) and
  8k steps (4e-5) with the depth prior (`configs/coorg_opencv/`).
- *Full resolution:* 25.39 dB raw / 27.37 dB colour-corrected, SSIM 0.904 (previous final
  24.71 / 27.04, 0.907). Off-path floaters 2.48% (2.33%).
- *Per view* (colour-corrected, half resolution): 12 of 20 views improve, worst 20.3 →
  21.2 dB, five weakest 23.4 → 24.7 dB. Four strong views lose 1–1.5 dB.
- *Pruned export:* 491k Gaussians, PSNR unchanged.

The previous final model, `outputs/coorg_bnb/splat_runs/blocks/merged_dense8_depth`. At full
resolution it scores 24.71 dB raw / 27.04 dB colour-corrected, SSIM 0.91. That is +0.08 raw and
−0.5 colour-corrected against `merged_finetune2`. Its worst held-out view improves from 13.7 to 19.5 dB, and
off-path floaters drop from 2.65% to 2.40% of pixels. Dense frames alone gave 24.85 / 27.50 dB with more
floaters (3.70%); the depth prior alone gave 24.24 / 27.13 dB. It has 4.04M
Gaussians; the web export keeps those with opacity ≥ 0.02. The blocks are entrance
0–74.5 s, living room 74.5–180 s and dining 180–216 s (`configs/coorg_blocks/`); each
block alone scores 25.8–29.3 dB colour-corrected on its own held-out frames.

The table covers the same 20 held-out frames for every model. The remaining weak views
are a floater at the left edge of the 50 s door view and a soft shelf at 200 s. These
two views (113 s and 200 s) score about the same as or below the VGGT splat; the other
18 are far better (`outputs/coorg_bnb/final/heldout_*.jpg`).

Lessons from the failed runs:

- **Loop closure is required.** Sequential-only cameras misalign every return to the
  living room. At full resolution that collapses the fit (train PSNR 17 → 13 dB).
- **MCMC noise must be normalized.** See *What the trainer does* above; without it,
  Gaussians grew to a median of 2.2 units (in a ≈10-unit house) at 2% opacity.
- **Budgets above about 1.6M were unstable for the whole house** (2.5M and 4M runs). They
  filled with near-camera fog: depth ≈ 0.11 units in front of every camera, Gaussians at
  the size cap with opacity ≈ 0.005. Memory then blew up in tile intersection.
- **One model for the whole house is capacity-limited.** The 74.5–180 s block alone
  reached 21.2 dB raw / 27.8 dB colour-corrected on its 9 held-out views; that motivated
  block training.
- **Merge by visibility, then fine-tune jointly.** A walk passes through rooms in several
  time windows. Nearest-camera ownership therefore gave rooms to blocks that only walked
  past them (20.6 dB). Ownership by Σ 1/depth² over each block's cameras fixed that
  (24.4 dB). A short joint fine-tune without densification removed the floaters that
  one block left in another's views.
- **Dropping motion-blurred training frames did not help** (`data.min_relative_sharpness = 0.7`,
  54/907 frames, `joint_finetune_3.toml`): 24.65 / 27.48 dB, SSIM 0.90, against
  24.63 / 27.56 / 0.91 for the final model. The 200 s view stays at 13.6 dB; its held-out
  reference frame is itself flagged `suspected_blur` (sharpness 91). The remaining weak
  spots (200 s shelf, 72–80 s blank ceiling) are capture limits: recapture those passes
  more slowly. Photometric training cannot add texture or sharpness that the video lacks.
- **Bigger per-block budgets did not help** (`configs/coorg_blocks_v2`). Own-frame held-out
  PSNR, raw / colour-corrected:

  | Block | Before | After | Result |
  | --- | --- | --- | --- |
  | Entrance (1.6M → 2.4M) | 24.08 / 29.34 dB | 23.26 / 28.82 dB | worse |
  | Living room (1.6M → 2.4M) | 21.65 / 26.45 dB | 21.50 / 25.59 dB | worse |
  | Dining (1.0M → 1.6M) | 21.18 / 25.81 dB | 21.31 / 26.00 dB | slightly better |

  Capacity is not the limit; extra Gaussians mostly add floaters.
- **Monocular depth prior is a trade-off** (`compute_mono_depth.py`, then
  `training.depth_prior_weight`; `joint_finetune_depth*.toml`).
  - *Held-out views near the path:* weight 0.05 gives 24.24 / 27.13 dB and weight 0.02
    gives 24.11 / 27.27 dB, against 24.63 / 27.56 for the final model.
  - *Away from the path:* views offset 0.3 units sideways have fewer near-camera floaters.
    The share of pixels with depth < 0.25 drops 2.65% → 2.35% overall, and 5.1 → 2.5% at
    160–200 s. The blank-ceiling beams come out straight
    (`outputs/coorg_bnb/final/ceiling_novel_view_compare.jpg`).
  - *Entrance:* the proxy rises 4.4 → 7.2% at the entrance. There the camera looks down at
    nearby floor, so the proxy partly counts real surfaces.
  - The final model stays the no-prior one. Use the prior when free roaming far from the
    capture path matters more than accuracy near it.
- **Fly-through video:** `render_flythrough.py <scene.ply> <sfm_dir> --output x.mp4`
  follows the capture path, with smoothed centres and slerped rotations.
- **Whole-model tour video:** `render_tour.py <scene.ply> <run_dir> --output tour.mp4`
  (`src/splatting/tour.py`, H.264 through PyAV) shows the model from outside as well as
  along the path. It has four parts:
  1. A top-down build-up: each Gaussian appears when a capture camera first sees it.
  2. A 360° orbit with the ceiling cut away.
  3. An arc above each room.
  4. An eye-level walkthrough beside the plan, with the current position marked.

  The cut-away views take "up" from a plane fitted to the camera centres.
  - *Floor:* measured locally on a plan grid (`local_floor`), so a floor that is not one
    plane in the reconstruction is kept. The first tour had one global floor height, and
    it cut away the entrance (the black hole at 1:09).
  - *Hidden Gaussians:* those more than 1.25 camera heights above the local floor;
    reflections in glossy marble, more than 0.05 camera heights below it; those far from
    the path; huge, faint or needle-shaped ones; and isolated floaters (8th neighbour
    beyond 0.06 camera heights).
  - The walkthrough renders the full model.
- **Moving a splat onto re-solved cameras:**
  `transfer_splat.py <scene.ply> <old sparse/0> <new sparse/0> --output moved.ply`
  (`src/splatting/transfer.py`).
  - *How:* 3D points that share a keypoint in both solves give correspondences. A robust
    local similarity is fitted around each one (32 neighbours, refitted without
    outliers). Each Gaussian blends its 4 nearest similarities; covariances are mapped
    too.
  - *Then:* a short fine-tune on the new cameras replaces retraining from scratch.
  - *Coorg* (`sfm_loops2` → `sfm_opencv`): held-out points land within 0.16% of the scene
    radius (median), 1.4% at p90.
- **Pruning is free.** Removing Gaussians below 0.02 opacity drops 69% of them with no
  held-out change. The view-independent `.splat` costs about 2.6 dB against the full-SH
  PLY.

## Preparing the development server

Use Python 3.11 or later, a CUDA-capable NVIDIA GPU, and compatible PyTorch,
CUDA toolkit and C++ compiler installations. gsplat can compile its CUDA extension
on first use. GPU memory requirements depend on resolution, visible splats and
refinement growth; the defaults are a starting configuration, not a hardware fit.

Run these commands **on the development server** from the repository root:

```bash
uv sync --extra splatting-viewer
```

The `splatting` extra installs PyTorch, gsplat, SciPy and plyfile. The
`splatting-viewer` extra also installs Viser. Training and rendering do not require
Viser. gsplat is pinned to **1.5.3** because the budgeted refinement strategy uses
version-specific strategy hooks and operations. Its opacity-reset scheduling is
implemented explicitly to avoid that version's bitwise-comparison reset bug.

`pyproject.toml` declares these dependencies. `uv.lock` has deliberately not been
regenerated on the authoring machine; resolve/update it on the development server
with the command above before using locked workflows. Actual installation has not
been verified.

## Input contract

The upstream geometric-refinement stage must produce a JSON scene manifest and
its referenced files. The existing VGGT checkout is not invoked by this module.
Copy [the example manifest](../configs/splatting_scene.example.json) into your
refined output folder, replace its example frame identities/dimensions/references,
and include your actual posed frames. Paths resolve relative to that manifest;
absolute paths are also supported.

Expected layout, with filenames chosen by the producer:

```text
outputs/<capture>/refined/
  scene.json
  cameras.npz           intrinsics [N,3,3], extrinsics [N,3,4] or [N,4,4]
  images/               aligned, upright, undistorted RGB images
  depth/                refined camera-Z .npy maps
  points/               refined world-coordinate .npy point maps
  confidence/           optional .npy maps
  masks/                optional boolean/0–1 .npy masks
```

### Scene-level fields

| Field | Required value or meaning |
| --- | --- |
| `schema_version` | `1` |
| `camera_convention` | `"opencv_world_to_camera"` |
| `depth_convention` | `"camera_z"` |
| `point_convention` | `"world"` |
| `camera_model` | `"pinhole"`; input images must already be undistorted |
| `pixel_convention` | `"integer_centers"` |
| `scene_units` | Nonempty string, e.g. `"arbitrary"` or `"metres"` |
| `frames` | Ordered list of posed observations; at least two training cameras |

Ordering determines automatic holdout selection. Preserve capture order when
using that feature. With explicit splits, supply `"train"` or `"val"` on **every**
frame. Otherwise, every eighth frame is held out by default. Set
`data.holdout_every = 0` to disable automatic holdout. Small datasets can have no
held-out frames; their final `val` report is `null`.

### Per-frame fields

| Field | Shape or meaning |
| --- | --- |
| `frame_id` | Unique source frame ID, preferably preprocessing's `<clip_id>:<pts>` |
| `timestamp_s` | Finite, nonnegative source time in seconds |
| `width`, `height` | Positive integer dimensions of the supplied image/maps |
| `image` | RGB image path; decoded with OpenCV, converted from BGR to RGB internally |
| `intrinsics` | `3×3` K for that exact image grid; no skew |
| `world_to_camera` | `3×4` or homogeneous `4×4` rigid transform |
| `depth` | Refined `[H,W]` or `[H,W,1]` camera-Z depth map |
| `points` | Refined `[H,W,3]` world-space coordinates, indexed by image pixel |
| `source_to_image` | Invertible `3×3` transform from original upright source pixels to this grid |
| `confidence` | Optional `[H,W]` or `[H,W,1]` confidence map |
| `mask` | Optional boolean or 0–1 `[H,W]` validity mask; applies to RGB and geometry |
| `split` | Optional explicit `"train"` or `"val"` |
| `metadata` | Optional producer-specific provenance, e.g. original size, crop/padding and refinement ID |

Camera matrices may be inline nested lists. Arrays can be a `.npy` path or an
object with `path`, optional `key` and optional `index`. For example:

```json
{
  "intrinsics": {"path": "cameras.npz", "key": "intrinsics", "index": 4},
  "world_to_camera": {"path": "cameras.npz", "key": "extrinsics", "index": 4},
  "depth": "depth/00004.npy",
  "points": {"path": "refined_maps.npz", "key": "world_points", "index": 4}
}
```

NPZ references require an explicit `key`; `index` selects an item from the leading
frame axis. Remove VGGT's batch dimension when saving, or save per-frame arrays.
The loader never guesses tensor ordering. NPY maps are memory-mapped; NPZ reads
decompress the selected key's entire array before indexing. For large captures,
prefer per-frame NPY files or uncompressed stacked NPY arrays.

### Coordinate, image and scale conventions

The camera transform follows:

```text
p_camera = R_world_to_camera @ p_world + t_world_to_camera
pixel_homogeneous = K @ p_camera
u = pixel_homogeneous[0] / p_camera.z
v = pixel_homogeneous[1] / p_camera.z
depth[v,u] = p_camera.z
```

Camera axes are x-right, y-down, z-forward. Integer `(u,v)` coordinates denote
pixel centres. VGGT's decoded camera-head outputs use this transform direction;
the compact `pose_enc` tensor itself is **not** an extrinsic matrix. Decode it
upstream with VGGT's camera utility before saving.

Camera translations, point coordinates, depth, clipping planes and exported
Gaussian scales all use the **same scene units**. The module does not infer metres
or perform per-frame depth rescaling. Scale ambiguity must be resolved consistently
upstream if metric outputs are required. No world recentering or rotation is
applied. The scene radius controls learning rates, voxel size and scale limits only.

Images, maps and intrinsics must describe the same resized/cropped/padded grid.
If VGGT inferred at a transformed resolution, either supply that transformed RGB
grid with its matching K/maps, or reproject everything correctly upstream.
`source_to_image` records the producer's transform; the loader does **not** apply
it a second time. For an unchanged source grid, use the identity. Record nonlinear
undistortion details separately in `metadata` when relevant. Mask padding, dynamic
objects and unwanted regions explicitly. Posed appearance frames can be included;
unposed preprocessing appearance frames cannot be optimized here.

Internal downsampling respects OpenCV's half-pixel resize convention:
`u' = sx * (u + 0.5) - 0.5`. RGB uses area resampling; geometry uses nearest-exact
sampling. The RGB validity mask conservatively requires the entire resampled area
to be valid, excluding mixed masked pixels. Intrinsics are adjusted accordingly. RGB is optimized in
the supplied image colour space (normally SDR sRGB-like values); there is no HDR
tone mapping, exposure correction or learned per-camera colour compensation.
At the renderer boundary, the principal point is translated by `(+0.5,+0.5)`
because gsplat 1.5.3 samples at half-integer raster centres. Saved cameras retain
the manifest's integer-centre convention.

NaN/infinite/nonpositive depth and nonfinite points are excluded. Confidence
thresholds use **producer units**, not assumed probabilities: VGGT confidence
scores need not lie in `[0,1]`. Missing confidence maps simply omit that filter.
Before downsampling, points are reprojected into their own camera and compared
with the refined depth. Excess disagreement raises an alignment error rather
than training on incompatible maps.

## Training a splat

```bash
uv run --extra splatting train_gaussian_splat.py \
  outputs/CAPTURE/refined/scene.json \
  --config configs/gaussian_splatting.toml \
  --output outputs/CAPTURE/splatting \
  --device cuda:0
```

The equivalent installed command is `uv run --extra splatting namma-splat` with
the same arguments. Choose a new or empty output folder for a fresh run.

The training sequence is:

1. Validate the manifest and hash the manifest, images and referenced arrays.
2. Sample bounded candidate counts per **training** view. Check agreement with
   nearby other training cameras' depth maps. Out-of-view, masked, occluded and
   inconsistent observations provide no supporting vote.
3. Fuse candidates in voxels, reject isolated points, subsample to the seed budget,
   and initialize scales from three-neighbor distances. Colors initialize the
   degree-zero spherical harmonics; rotations start as identity and can optimize.
4. Fit fixed cameras' RGB using masked L1/SSIM plus decaying relative camera-Z
   depth supervision and scale regularization. Activate SH bands progressively.
5. Clone/split high-gradient Gaussians and prune low-opacity/oversized ones.
   Growth is limited to `refinement.max_gaussians` retained primitives. Temporary
   tensors, optimizer state and split children still require additional memory.
6. Save resumable checkpoints and evaluate held-out views. At completion, export
   the standard SH PLY and evaluate every training and validation view.

`consistency_views` selects nearest camera centres, not a covisibility graph.
Limited overlap can reject useful geometry; inspect initialization diagnostics
before reducing support requirements. Holdout images/maps do not seed the splat or
provide training loss. Upstream joint VGGT/refinement may already have used those
views, so these metrics measure held-out appearance optimization, not an entirely
unseen reconstruction benchmark.

All cameras stay fixed. Poor pose calibration, moving subjects, missing coverage,
exposure changes and reflective/transparent surfaces can still impair quality.
The module reports measurements and previews; it does not label every completed
training run a reliable reconstruction.

### Configuration

[The TOML configuration](../configs/gaussian_splatting.toml) has five sections:

| Section | Controls |
| --- | --- |
| `data` | Image resolution, bounded CPU cache, holdout, confidence/depth and geometric tolerances |
| `initialization` | Candidate/seed budgets, voxel fusion, outlier filtering, scale and opacity initialization |
| `training` | Steps, SH degree, losses, learning rates, logging/checkpoint/evaluation intervals |
| `refinement` | Densification schedule, gradient threshold, opacity/scale pruning, retained Gaussian budget |
| `render` | Scene-unit clipping planes, background RGB, packed rendering and antialiasing mode |

Unknown settings and invalid ranges fail early. Reduce image size, candidate count
and Gaussian budget to fit the development server. Changing clipping limits must
respect the reconstruction's actual scale. Defaults are also available through
`SplattingConfig()` when `--config` is omitted.

### Resuming

```bash
uv run --extra splatting train_gaussian_splat.py \
  outputs/CAPTURE/refined/scene.json \
  --config configs/gaussian_splatting.toml \
  --output outputs/CAPTURE/splatting \
  --resume outputs/CAPTURE/splatting/checkpoint.pt \
  --device cuda:0
```

A checkpoint includes parameters, all Adam states, refinement statistics, step,
sampler and PyTorch RNG states, configuration, versions and input fingerprint.
Resume requires the original output directory, unchanged input contents/paths,
the same configuration and recorded package versions. CUDA accumulation can still
be nondeterministic. Resume continues the original schedule; it does not extend
`training.steps`. JSONL diagnostics may contain repeated steps after recovering
from work performed since the last complete checkpoint.

### Outputs

```text
outputs/<capture>/splatting/
  run.json                  config, conventions, identities, versions, input hash, seed diagnostics
  input_manifest.json       provenance snapshot; paths still refer to the original manifest root
  training.jsonl            sampled losses, SSIM, relative depth, primitive counts
  checkpoint.pt             latest atomically replaced resumable state
  scene.ply                 binary standard 3DGS PLY, all stored SH bands
  scene.ply.json            units, config, active SH degree and initial camera for rendering/viewing
  geometry_checks.json      per-frame own-camera alignment diagnostics
  metrics.json              final train/val metrics; quality_status requires_visual_review
  evaluation/<step>/<split>/
    metrics.json            per-view PSNR, SSIM, relative depth error and alpha coverage
    00000.png               reference RGB (left), rendered RGB (right)
```

PLY stores means in world coordinates, log standard deviations, wxyz quaternions,
opacity logits and channel-major SH coefficients. This is a Gaussian PLY, not a
colored-point-cloud export. Preserve `scene.ply.json` for the viewer's initial
camera, scene units and exact rendering settings. The PLY is inference-only;
training resume uses the checkpoint.

## Rendering cameras offline

The original refined-scene manifest can supply camera views:

```bash
uv run --extra splatting render_gaussian_splat.py \
  outputs/CAPTURE/splatting/scene.ply \
  --cameras outputs/CAPTURE/refined/scene.json \
  --output outputs/CAPTURE/splat_renders \
  --device cuda:0
```

Alternatively, supply a JSON file with `schema_version: 1`,
`camera_convention: "opencv_world_to_camera"` and a `cameras` list. Each camera
requires `frame_id`, `width`, `height`, `intrinsics` and `world_to_camera`, using
the same matrix formats. This supports novel trajectories without geometry maps.
`--max-image-side` optionally rescales the camera grids. Both `.pt` and `.ply`
models are accepted.

Each view produces `*_rgb.png`, `*_alpha.png` and `*_depth.npy`, plus a
`renders.json` mapping files to frame IDs and camera matrices. Alpha PNGs are
8-bit opacity; numeric depth is expected camera-Z in scene units with NaN where
alpha is effectively zero. The renderer itself returns floating alpha and raw
depth, leaving zero-alpha interpretation to its caller.

## Viewing interactively

Run the Python viewer **on the development server**:

```bash
uv run --extra splatting-viewer view_gaussian_splat.py \
  outputs/CAPTURE/splatting/scene.ply \
  --cameras outputs/CAPTURE/refined/scene.json \
  --device cuda:0 --port 8080
```

The Python process renders full spherical-harmonic RGB on the server; Viser sends
images to the browser and supplies orbit/pan/zoom controls and capture presets.
The viewer coalesces camera updates and serializes GPU rendering across clients.
`--max-image-side` and `--fps` bound requested resolution and update rate.
The actual frame rate depends on server load, scene size and network latency.

The default listener is `127.0.0.1:8080`. Use your normal SSH port forwarding to
open it from your browser. `--host` can select an interface on the development
server. No external training service is involved. An exported PLY sidecar or
checkpoint supplies an initial camera when `--cameras` is omitted. A standalone
external Gaussian PLY needs explicit cameras.

Capture presets set pose and vertical field of view. The interactive viewport
uses a centred, square-pixel pinhole camera fitted to its browser aspect ratio;
it does not reproduce off-centre principal points or differing `fx`/`fy` exactly.
Use offline rendering for exact input intrinsics. This is an orbit viewer, not
a collision-aware navigation application.

## Python API

```python
from src import Camera, SplattingConfig, load_model, render_gaussians, train_splat

config = SplattingConfig.from_toml("configs/gaussian_splatting.toml")
result = train_splat(
    "outputs/CAPTURE/refined/scene.json",
    "outputs/CAPTURE/splatting",
    config,
    device="cuda:0",
)
model, metadata = load_model(result.checkpoint, device="cuda:0")
camera = Camera.from_dict(metadata["initial_camera"])
rendered = render_gaussians(model, camera, config.render)
# rendered.rgb: [H,W,3], alpha/depth: [H,W], info: gsplat gradient intermediates
```

`src.splatting` exports configuration, scene loading, training, model loading,
PLY export and viewing. `src.rendering` exports `Camera`, `render_gaussians` and
`render_cameras`; both stage APIs are re-exported through `src`.
Training/viewing dependencies load on demand, so preprocessing does not require
the splatting extras. The renderer preserves autograd for training; use
`torch.no_grad()` for inference in custom callers.

## Verification on the development server

The following commands are supplied for later execution; **none have been run
on the authoring machine**:

```bash
uv run --extra splatting ruff check src/splatting src/rendering src/__init__.py tests/splatting train_gaussian_splat.py render_gaussian_splat.py view_gaussian_splat.py
uv run --extra splatting pytest tests/splatting -m 'not cuda' -q -ra
uv run --extra splatting pytest tests/splatting -m cuda -q -ra
```

Tests exercise actual array/image contracts, geometric reprojection, known-plane
depth support, holdout exclusion, fusion and budgets, SH PLY round trips, Adam
checkpoint continuation, NaN-safe masked losses and topology/optimizer alignment.
The CUDA integration test uses the actual rasterizer and training implementation.
It is skipped with a reason when CUDA/dependencies are unavailable; no fake
renderer or model replaces them. Analytic plane fixtures establish numeric
correctness, not real-capture reconstruction quality.

After numeric checks pass, run a representative actual capture on the server.
Inspect held-out reference/render pairs and the orbit viewer for holes, floaters,
double surfaces, exposure shifts and poor occlusion. Record observations and
runtime versions in [STATE.md](STATE.md). No real-capture splat, viewer smoke test,
performance measurement or visual inspection has yet been performed.

## Implementation references

- [gsplat rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html)
- [gsplat densification strategies](https://docs.gsplat.studio/main/apis/strategy.html)
- [Pinned gsplat 1.5.3 source](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3)
- [Viser examples](https://viser.studio/main/examples/)
